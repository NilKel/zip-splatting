#
# MLP-Opacity Mode Renderer
#
# Three-phase rendering pipeline for MLP-opacity mode:
# 1. Aggregate neural features + rasterized opacities into depth intervals (CUDA)
# 2. Decode features to RGB using MLP (Python/tiny-cuda-nn) - NO DENSITY PREDICTION
# 3. Composite intervals using rasterized opacity for density (CUDA)
#
# Key difference from neural_render.py:
# - MLP only predicts RGB (view-dependent color)
# - Density comes from rasterized Gaussian opacities (not MLP)
#

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    rasterize_neural_intervals,
    composite_mlp_opacity
)


def render_mlp_opacity(
    viewpoint_camera,
    pc,
    mlp,
    pipe,
    bg_color: torch.Tensor,
    num_intervals=16,
    near_depth=0.1,
    far_depth=100.0,
    feature_dim=32,
    scaling_modifier=1.0,
    override_color=None,
    debug_iteration=None,
    use_ngp_intervals=False,
    scene_aabb=None
):
    """
    Render a Gaussian Splatting scene using MLP-opacity mode.

    In this mode:
    - Gaussian opacities drive density (rasterized)
    - MLP only predicts RGB color (view-dependent)

    Args:
        viewpoint_camera: Camera object with pose and intrinsics
        pc: GaussianModel with neural features
        mlp: RGB-only MLP (RGBOnlyMLP instance)
        pipe: PipelineParams
        bg_color: [3] Background color tensor
        num_intervals: Number of depth intervals
        near_depth: Near clipping plane
        far_depth: Far clipping plane
        feature_dim: Feature dimension (must match pc.get_features_neural)
        scaling_modifier: Scale modifier for Gaussians
        override_color: Optional color override (not used for neural rendering)
        debug_iteration: Iteration number for debug prints (optional)
        use_ngp_intervals: Use NGP-style adaptive intervals (Δt = √3/1024)
        scene_aabb: Scene bounding box for NGP intervals [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
        dict with keys:
            "render": [3, H, W] Final rendered image
            "viewspace_points": Screenspace points for gradient computation
            "visibility_filter": Boolean mask of visible Gaussians
            "radii": Radii of projected Gaussians
    """

    # Apply NGP-style interval computation if requested
    if use_ngp_intervals and scene_aabb is not None:
        from gaussian_renderer.zip_render import compute_ngp_intervals
        num_intervals, near_depth, far_depth, delta_t = compute_ngp_intervals(
            scene_aabb, num_intervals=num_intervals if num_intervals != 16 else None
        )
        if debug_iteration is not None and debug_iteration % 1000 == 0:
            print(f"[NGP Intervals] num={num_intervals}, near={near_depth:.4f}, far={far_depth:.4f}, Δt={delta_t:.6f}")

    # Create zero tensor for screen-space points (same as baseline render)
    means3D = pc.get_xyz
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
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
        sh_degree=0,  # Not used for neural rendering
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        antialiasing=pipe.antialiasing if hasattr(pipe, 'antialiasing') else False,
        debug=pipe.debug if hasattr(pipe, 'debug') else False
    )

    # Get Gaussian parameters
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    cov3D_precomp = None

    # Get neural features
    features_neural = pc.get_features_neural  # [N, feature_dim]
    features_neural = features_neural.contiguous()

    # Validate feature dimension
    if features_neural.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {feature_dim}, "
            f"got {features_neural.shape[1]}"
        )

    # Phase 1: Aggregate features AND rasterized opacities into depth intervals
    # Output:
    #   interval_features: [H, W, num_intervals, feature_dim]
    #   interval_opacities: [H, W, num_intervals] - averaged Gaussian opacities

    if debug_iteration is not None and debug_iteration % 1000 == 0:
        print(f"\n[MLP-Opacity Mode Iter {debug_iteration}] Starting rasterization...")

    interval_features, interval_opacities, radii = rasterize_neural_intervals(
        means3D=means3D,
        features_neural=features_neural,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=cov3D_precomp,
        raster_settings=raster_settings,
        num_intervals=num_intervals,
        near_depth=near_depth,
        far_depth=far_depth,
        feature_dim=feature_dim
    )

    if debug_iteration is not None and debug_iteration % 1000 == 0:
        print(f"  interval_features.shape: {interval_features.shape}")
        print(f"  interval_opacities.shape: {interval_opacities.shape}")
        print(f"  radii.shape: {radii.shape}")

    # Phase 2: MLP decoding (RGB only - no density prediction)
    H, W, N, F = interval_features.shape

    # Compute view direction for each pixel
    cam_view_transform = viewpoint_camera.world_view_transform.to('cuda')
    view_dir = -cam_view_transform[:3, 2].unsqueeze(0).unsqueeze(0)  # Camera -Z axis
    view_dir = view_dir.expand(H, W, 3).contiguous()
    view_dir = view_dir / (torch.norm(view_dir, dim=-1, keepdim=True) + 1e-6)

    # Expand to all intervals: [H, W, N, 3]
    view_dirs_expanded = view_dir.unsqueeze(2).expand(H, W, N, 3)

    # Reshape for MLP processing
    features_flat = interval_features.reshape(-1, F)
    view_dirs_flat = view_dirs_expanded.reshape(-1, 3)

    # Evaluate RGB-only MLP in batches
    batch_size = 2**18  # ~256K samples per batch
    total_samples = features_flat.shape[0]

    rgb_flat = torch.empty((total_samples, 3), dtype=torch.float32, device='cuda')

    with torch.amp.autocast('cuda', enabled=False):
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)

            # Get batch
            features_batch = features_flat[i:end_idx]
            view_dirs_batch = view_dirs_flat[i:end_idx]

            # Ensure correct dtype (tiny-cuda-nn requires float16)
            if features_batch.dtype != torch.float16:
                features_batch = features_batch.contiguous().to(torch.float16)
            if view_dirs_batch.dtype != torch.float16:
                view_dirs_batch = view_dirs_batch.contiguous().to(torch.float16)

            # RGB-only MLP forward (returns RGB directly, no density)
            rgb_batch = mlp(features_batch, view_dirs_batch)  # [B, 3]

            # Store RGB (convert to FP32)
            rgb_flat[i:end_idx] = rgb_batch.float()

    # Reshape back to image space: [H, W, num_intervals, 3]
    decoded_colors = rgb_flat.reshape(H, W, N, 3)

    if debug_iteration is not None and debug_iteration % 1000 == 0:
        print(f"  decoded_colors.shape: {decoded_colors.shape}")
        print(f"  decoded_colors range: [{decoded_colors.min().item():.4f}, {decoded_colors.max().item():.4f}]")

    # Phase 3: Composite into final image using rasterized opacities for density
    # Input:
    #   - decoded_colors: [H, W, N, 3] from MLP
    #   - interval_opacities: [H, W, N] from rasterization
    # Output: [H, W, 3]
    final_image = composite_mlp_opacity(
        decoded_colors=decoded_colors,
        interval_opacities=interval_opacities,
        background=bg_color,
        raster_settings=raster_settings,
        num_intervals=num_intervals,
        near_depth=near_depth,
        far_depth=far_depth
    )

    # Transpose to [3, H, W] to match expected format
    final_image = final_image.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

    if debug_iteration is not None and debug_iteration % 1000 == 0:
        print(f"  final_image.shape: {final_image.shape}")
        print(f"  final_image range: [{final_image.min().item():.4f}, {final_image.max().item():.4f}]")

    # Return results
    return {
        "render": final_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
