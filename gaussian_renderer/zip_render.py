"""
ZIP Method Renderer: Interval-based Gaussian Splatting with NerfAcc

This module implements the interval splatting approach that combines:
1. NerfAcc occupancy grid for spatial acceleration
2. Depth interval sampling for each ray
3. Binned accumulation of Gaussians into intervals
4. Anti-aliased blending of interval contributions
"""

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

try:
    from nerfacc import OccGridEstimator, ray_marching, rendering
    NERFACC_AVAILABLE = True
except ImportError:
    NERFACC_AVAILABLE = False
    OccGridEstimator = None


# Blending mode mappings
BLENDING_MODE_MAP = {
    "compositing": 0,  # Compositing: front-to-back alpha compositing (original working version)
    "naive": 1,        # Naive: sum RGB and sum opacities (no normalization)
    "gaussian": 2,     # Gaussian: alpha-weighted averaging
    "g-nerf": 3,       # Gaussian-NeRF: alpha-weighted opacity -> density -> NeRF blend
    "mlp-nerf": 4,     # MLP-NeRF: alpha-weighted features -> MLP -> NeRF blend (future)
}


def get_blending_mode_int(blending_mode_str):
    """Convert blending mode string to integer for CUDA kernel."""
    mode_str = blending_mode_str.lower()
    if mode_str not in BLENDING_MODE_MAP:
        print(f"Warning: Unknown blending mode '{blending_mode_str}', defaulting to 'g-nerf'")
        return BLENDING_MODE_MAP["g-nerf"]
    return BLENDING_MODE_MAP[mode_str]


def compute_adaptive_depth_bounds(viewpoint_camera, estimator, pc, default_near=0.1, default_far=100.0):
    """
    Compute adaptive near/far depth bounds using NerfAcc occupancy grid.

    This function samples the occupancy grid along camera rays to find the
    actual depth range where scene content exists, providing tighter bounds
    than fixed near/far planes.

    Args:
        viewpoint_camera: Camera parameters
        estimator: NerfAcc OccGridEstimator with updated occupancy grid
        pc: GaussianModel (used as fallback to compute bounds from Gaussian positions)
        default_near: Default near plane if NerfAcc not available
        default_far: Default far plane if NerfAcc not available

    Returns:
        (near_depth, far_depth) tuple
    """
    if not NERFACC_AVAILABLE or estimator is None:
        # Fallback: compute bounds from Gaussian positions
        with torch.no_grad():
            # Transform Gaussians to camera space
            xyz = pc.get_xyz  # [N, 3]
            w2c = viewpoint_camera.world_view_transform.transpose(0, 1)  # [4, 4]
            xyz_cam = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1) @ w2c  # [N, 4]
            depths = xyz_cam[:, 2]  # Camera-space Z (depth)

            # Filter positive depths (in front of camera)
            valid_depths = depths[depths > 0]
            if len(valid_depths) == 0:
                return default_near, default_far

            # Use percentiles to avoid outliers
            near = max(default_near, valid_depths.quantile(0.01).item())
            far = min(default_far, valid_depths.quantile(0.99).item())
            return near, far

    # Use NerfAcc to sample depth bounds from occupancy grid
    with torch.no_grad():
        # Generate rays for center and corners (cheap sampling)
        H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)

        # Sample a grid of pixels (e.g., 8x8 grid)
        sample_h = torch.linspace(0, H-1, 8, device='cuda').long()
        sample_w = torch.linspace(0, W-1, 8, device='cuda').long()
        grid_h, grid_w = torch.meshgrid(sample_h, sample_w, indexing='ij')
        pixels = torch.stack([grid_w.flatten(), grid_h.flatten()], dim=1)  # [64, 2]

        # Generate rays for sampled pixels
        rays_o, rays_d = generate_rays(viewpoint_camera, pixels)

        # Use NerfAcc to find intervals where occupancy > threshold
        try:
            ray_indices, t_starts, t_ends = estimator.sampling(
                rays_o=rays_o,
                rays_d=rays_d,
                render_step_size=1e-2,
                alpha_thre=1e-2,
                stratified=False,
                cone_angle=0.0,
            )

            if len(t_starts) == 0:
                # No occupied space found, use Gaussian-based fallback
                return compute_adaptive_depth_bounds(viewpoint_camera, None, pc, default_near, default_far)

            # Compute depth bounds from sampled intervals
            near = max(default_near, t_starts.min().item())
            far = min(default_far, t_ends.max().item())

            # Add small margin
            depth_range = far - near
            near = max(default_near, near - 0.1 * depth_range)
            far = min(default_far, far + 0.1 * depth_range)

            return near, far

        except Exception as e:
            # Fallback if NerfAcc sampling fails
            print(f"NerfAcc sampling failed: {e}, using Gaussian-based bounds")
            return compute_adaptive_depth_bounds(viewpoint_camera, None, pc, default_near, default_far)


def generate_rays(viewpoint_camera, pixels):
    """
    Generate ray origins and directions for given pixel coordinates.

    Args:
        viewpoint_camera: Camera parameters
        pixels: [N, 2] tensor of (x, y) pixel coordinates

    Returns:
        rays_o: [N, 3] ray origins in world space
        rays_d: [N, 3] ray directions in world space (normalized)
    """
    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    # Normalize pixel coordinates to [-1, 1]
    x = pixels[:, 0].float()
    y = pixels[:, 1].float()

    # Convert to NDC
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    ndc_x = (x - W/2) / (W/2) * tanfovx
    ndc_y = (y - H/2) / (H/2) * tanfovy

    # Ray direction in camera space (pointing along +Z in camera coords)
    rays_d_cam = torch.stack([ndc_x, -ndc_y, torch.ones_like(ndc_x)], dim=1)  # [N, 3]
    rays_d_cam = rays_d_cam / rays_d_cam.norm(dim=1, keepdim=True)

    # Transform to world space
    # Camera to world: inverse of world_view_transform
    c2w = viewpoint_camera.world_view_transform.transpose(0, 1).inverse()  # [4, 4]

    # Ray origin is camera center
    rays_o = viewpoint_camera.camera_center.unsqueeze(0).expand(len(pixels), -1)  # [N, 3]

    # Transform direction (rotation only, no translation)
    rays_d = (c2w[:3, :3] @ rays_d_cam.T).T  # [N, 3]
    rays_d = rays_d / rays_d.norm(dim=1, keepdim=True)

    return rays_o, rays_d


def compute_ngp_intervals(scene_aabb, num_intervals=None):
    """
    Compute intervals using NGP-style fixed step size.

    From Instant-NGP paper:
    "In synthetic NeRF scenes, which we bound to the unit cube [0,1]³,
     we use a fixed ray marching step size equal to Δt := √3/1024;
     √3 represents the diagonal of the unit cube."

    Args:
        scene_aabb: torch.Tensor [6] as [xmin, ymin, zmin, xmax, ymax, zmax]
        num_intervals: Optional override for number of intervals

    Returns:
        tuple: (num_intervals, near_depth, far_depth, delta_t)
    """
    import math

    # Compute scene diagonal
    scene_min = scene_aabb[:3]
    scene_max = scene_aabb[3:]
    diagonal = torch.norm(scene_max - scene_min).item()

    # NGP-style: Δt = diagonal / 1024
    delta_t = diagonal / 1024.0

    # Compute near/far based on scene bounds
    # Use a small margin to avoid clipping
    near = 0.0
    far = diagonal * 1.1  # Add 10% margin

    # If num_intervals not specified, compute from delta_t
    if num_intervals is None:
        num_intervals = int(math.ceil(far / delta_t))

    return num_intervals, near, far, delta_t


def render_zip(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    estimator=None,
    scaling_modifier=1.0,
    separate_sh=False,
    override_color=None,
    use_trained_exp=False,
    use_intervals=True,  # Default to True for ZIP method
    num_intervals=16,
    near_depth=0.1,
    far_depth=100.0,
    blending_mode="g-nerf",
    use_ngp_intervals=False,
    scene_aabb=None
):
    """
    Render using ZIP method with optional interval splatting.

    Args:
        viewpoint_camera: Camera parameters
        pc: GaussianModel with all Gaussians
        pipe: Pipeline settings
        bg_color: Background color tensor
        estimator: NerfAcc OccGridEstimator (optional, for occupancy grid updates)
        scaling_modifier: Scale modifier for Gaussians
        separate_sh: Whether to separate DC and rest of SH components
        override_color: Optional color override
        use_trained_exp: Whether to use trained exposure
        use_intervals: Whether to use interval-based rendering (default: False)
        num_intervals: Number of depth intervals (default: 16)
        near_depth: Near clipping plane (default: 0.1)
        far_depth: Far clipping plane (default: 100.0)
        blending_mode: Interval blending mode ("compositing", "mlp-nerf", etc.)
        use_ngp_intervals: Use NGP-style adaptive intervals (Δt = √3/1024)
        scene_aabb: Scene bounding box [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        Dictionary with:
        - render: Rendered image [3, H, W]
        - viewspace_points: Screen-space points
        - visibility_filter: Visible Gaussian indices
        - radii: 2D radii of Gaussians
        - depth: Depth map [H, W]
    """

    # Apply NGP-style interval computation if requested
    if use_ngp_intervals and scene_aabb is not None:
        num_intervals, near_depth, far_depth, delta_t = compute_ngp_intervals(
            scene_aabb, num_intervals=num_intervals if num_intervals != 16 else None
        )
        # print(f"[NGP Intervals] num={num_intervals}, near={near_depth:.4f}, far={far_depth:.4f}, Δt={delta_t:.6f}")

    # Create zero tensor for screen-space gradients
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda"
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Covariance computation
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Color computation
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(
                -1, 3, (pc.max_sh_degree + 1) ** 2
            )
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(
                pc.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image
    if use_intervals:
        # Convert blending mode string to integer
        blending_mode_int = get_blending_mode_int(blending_mode)

        # Compute adaptive depth bounds using NerfAcc occupancy grid
        adaptive_near, adaptive_far = compute_adaptive_depth_bounds(
            viewpoint_camera, estimator, pc,
            default_near=near_depth,
            default_far=far_depth
        )

        # Use interval-based rendering with adaptive bounds
        if separate_sh:
            rendered_image, radii, depth_image = rasterizer.forward_intervals(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                num_intervals=num_intervals,
                near_depth=adaptive_near,
                far_depth=adaptive_far,
                blending_mode=blending_mode_int,
            )
        else:
            rendered_image, radii, depth_image = rasterizer.forward_intervals(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
                num_intervals=num_intervals,
                near_depth=adaptive_near,
                far_depth=adaptive_far,
                blending_mode=blending_mode_int,
            )
    else:
        # Use standard rendering
        if separate_sh:
            rendered_image, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                dc=dc,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        else:
            rendered_image, radii, depth_image = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )

    # Apply exposure (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = (
            torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(
                2, 0, 1
            )
            + exposure[:3, 3, None, None]
        )

    # Clamp and return
    rendered_image = rendered_image.clamp(0, 1)

    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
    }

    return out


def render_zip_intervals(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    estimator,
    render_step_size: float = 1e-3,
    scaling_modifier=1.0,
):
    """
    Advanced ZIP rendering with explicit interval sampling.

    This function will:
    1. Generate rays for each pixel
    2. Use NerfAcc to sample depth intervals based on occupancy grid
    3. For each interval, accumulate Gaussians that overlap
    4. Blend intervals front-to-back
    5. Return anti-aliased rendering

    TODO: Implement in next phase
    """
    raise NotImplementedError(
        "Interval-based rendering not yet implemented. Use render_zip() for now."
    )
