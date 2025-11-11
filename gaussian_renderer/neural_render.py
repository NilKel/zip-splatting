#
# Neural Interval Splatting Renderer
#
# Three-phase rendering pipeline:
# 1. Aggregate neural features into depth intervals (CUDA)
# 2. Decode features to RGB+density using MLP (Python/tiny-cuda-nn)
# 3. Composite intervals into final image (CUDA)
#

import torch
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    rasterize_neural_intervals,
    composite_neural
)


def render_neural_intervals(
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
    Render a Gaussian Splatting scene using neural interval splatting.

    Args:
        viewpoint_camera: Camera object with pose and intrinsics
        pc: GaussianModel with neural features
        mlp: tiny-cuda-nn MLP for decoding features to RGB+density
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

    # TEMP DEBUG: Ensure features are contiguous and properly allocated
    features_neural = features_neural.contiguous()

    # Validate feature dimension
    if features_neural.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: expected {feature_dim}, "
            f"got {features_neural.shape[1]}"
        )

    # Phase 1: Aggregate features into depth intervals
    # Output: [H, W, num_intervals, feature_dim]

    # DEBUG: Print tensor shapes before CUDA call (only after iteration 635)
    if debug_iteration is not None and debug_iteration >= 635:
        print(f"\n[DEBUG Iter {debug_iteration}] Before rasterize_neural_intervals:")
        print(f"  means3D.shape: {means3D.shape}")
        print(f"  features_neural.shape: {features_neural.shape}")
        print(f"  opacity.shape: {opacity.shape}")
        print(f"  scales.shape: {scales.shape}")
        print(f"  rotations.shape: {rotations.shape}")
        print(f"  Image size: {raster_settings.image_height}x{raster_settings.image_width}")
        print(f"  num_intervals: {num_intervals}, feature_dim: {feature_dim}")

        # Check for NaN/Inf values in tensors
        print(f"  Checking for NaN/Inf:")
        print(f"    means3D: NaN={torch.isnan(means3D).any().item()}, Inf={torch.isinf(means3D).any().item()}")
        print(f"    features_neural: NaN={torch.isnan(features_neural).any().item()}, Inf={torch.isinf(features_neural).any().item()}")
        print(f"    opacity: NaN={torch.isnan(opacity).any().item()}, Inf={torch.isinf(opacity).any().item()}, min={opacity.min().item():.6f}, max={opacity.max().item():.6f}")
        print(f"    scales: NaN={torch.isnan(scales).any().item()}, Inf={torch.isinf(scales).any().item()}, min={scales.min().item():.6f}, max={scales.max().item():.6f}")
        print(f"    rotations: NaN={torch.isnan(rotations).any().item()}, Inf={torch.isinf(rotations).any().item()}")

        # Check camera matrices for NaN/Inf
        viewmatrix = viewpoint_camera.world_view_transform
        projmatrix = viewpoint_camera.full_proj_transform
        print(f"  Camera matrices:")
        print(f"    viewmatrix: NaN={torch.isnan(viewmatrix).any().item()}, Inf={torch.isinf(viewmatrix).any().item()}")
        print(f"    projmatrix: NaN={torch.isnan(projmatrix).any().item()}, Inf={torch.isinf(projmatrix).any().item()}")
        print(f"    campos: {viewpoint_camera.camera_center}")

        # Check cov3D_precomp if it exists
        if cov3D_precomp is not None:
            print(f"    cov3D_precomp: NaN={torch.isnan(cov3D_precomp).any().item()}, Inf={torch.isinf(cov3D_precomp).any().item()}")

        # Print actual value ranges to detect subtle issues
        print(f"  Value ranges:")
        print(f"    means3D: min={means3D.min().item():.6f}, max={means3D.max().item():.6f}")
        print(f"    features_neural: min={features_neural.min().item():.6f}, max={features_neural.max().item():.6f}")
        print(f"    scales: [{scales[:, 0].min().item():.6f}, {scales[:, 1].min().item():.6f}, {scales[:, 2].min().item():.6f}] to [{scales[:, 0].max().item():.6f}, {scales[:, 1].max().item():.6f}, {scales[:, 2].max().item():.6f}]")
        print(f"    rotations: min={rotations.min().item():.6f}, max={rotations.max().item():.6f}")

    interval_features, radii = rasterize_neural_intervals(
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

    # Force CUDA synchronization to detect errors immediately
    if debug_iteration is not None and debug_iteration >= 635:
        torch.cuda.synchronize()
        print(f"[DEBUG Iter {debug_iteration}] After rasterize_neural_intervals - CUDA sync OK")
        print(f"  interval_features.shape: {interval_features.shape}")
        print(f"  radii.shape: {radii.shape}")

    # Phase 2: MLP decoding
    # Reshape for MLP: [H*W*num_intervals, feature_dim]
    H, W, N, F = interval_features.shape
    features_flat = interval_features.reshape(-1, F)

    # Evaluate MLP to get RGB + density
    # tiny-cuda-nn expects FP16 input
    with torch.amp.autocast('cuda', enabled=False):
        # Ensure correct dtype (tiny-cuda-nn often requires float16)
        # Use contiguous() and clone() to ensure proper memory layout
        if features_flat.dtype != torch.float16:
            features_flat = features_flat.contiguous().to(torch.float16)

        decoded_flat = mlp(features_flat)  # [H*W*N, 4] - RGB + density

        # Convert back to FP32 for compositing
        if decoded_flat.dtype != torch.float32:
            decoded_flat = decoded_flat.float()

    # Reshape back to image space: [H, W, num_intervals, 4]
    decoded_intervals = decoded_flat.reshape(H, W, N, 4)

    # Apply sigmoid to RGB, softplus to density (NON-INPLACE to avoid autograd issues)
    rgb = torch.sigmoid(decoded_intervals[..., :3])  # RGB in [0, 1]
    density = torch.nn.functional.softplus(decoded_intervals[..., 3:4])  # density >= 0
    decoded_intervals = torch.cat([rgb, density], dim=-1)  # [H, W, N, 4]

    # Phase 3: Composite into final image
    # Output: [H, W, 3]
    final_image = composite_neural(
        decoded_intervals=decoded_intervals,
        background=bg_color,
        raster_settings=raster_settings,
        num_intervals=num_intervals,
        near_depth=near_depth,
        far_depth=far_depth
    )

    # Transpose to [3, H, W] to match expected format
    final_image = final_image.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

    # Return results
    return {
        "render": final_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
