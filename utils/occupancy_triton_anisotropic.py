"""
Improved Triton kernels for Gaussian scattering with anisotropic (full covariance) support.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def scatter_gaussians_anisotropic_kernel(
    # Gaussian parameters
    xyz_ptr,  # [N, 3]
    scales_ptr,  # [N, 3]
    rotations_ptr,  # [N, 4] - quaternions
    opacities_ptr,  # [N, 1]
    # Grid parameters
    grid_aabb_min_ptr,  # [3]
    grid_aabb_max_ptr,  # [3]
    voxel_size_ptr,  # [3]
    # Output
    occupancy_ptr,  # [res_x, res_y, res_z]
    # Dimensions
    N_gaussians,
    res_x, res_y, res_z,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Scatter Gaussians into voxel occupancy grid using full anisotropic covariance.

    Uses the same 3D covariance computation as baseline CUDA:
    Σ = (RS)ᵀ(RS) where R is rotation matrix from quaternion, S is diagonal scale matrix.

    For each Gaussian, computes extent as 3σ where σ = sqrt(max_eigenvalue(Σ)).
    """
    g_idx = tl.program_id(0)
    if g_idx >= N_gaussians:
        return

    # Load Gaussian parameters
    mean_x = tl.load(xyz_ptr + g_idx * 3 + 0)
    mean_y = tl.load(xyz_ptr + g_idx * 3 + 1)
    mean_z = tl.load(xyz_ptr + g_idx * 3 + 2)

    scale_x = tl.load(scales_ptr + g_idx * 3 + 0)
    scale_y = tl.load(scales_ptr + g_idx * 3 + 1)
    scale_z = tl.load(scales_ptr + g_idx * 3 + 2)

    # Load quaternion (w, x, y, z) format
    qr = tl.load(rotations_ptr + g_idx * 4 + 0)  # real part
    qx = tl.load(rotations_ptr + g_idx * 4 + 1)
    qy = tl.load(rotations_ptr + g_idx * 4 + 2)
    qz = tl.load(rotations_ptr + g_idx * 4 + 3)

    opacity = tl.load(opacities_ptr + g_idx)

    # Load grid parameters
    aabb_min_x = tl.load(grid_aabb_min_ptr + 0)
    aabb_min_y = tl.load(grid_aabb_min_ptr + 1)
    aabb_min_z = tl.load(grid_aabb_min_ptr + 2)

    voxel_size_x = tl.load(voxel_size_ptr + 0)
    voxel_size_y = tl.load(voxel_size_ptr + 1)
    voxel_size_z = tl.load(voxel_size_ptr + 2)

    # Build rotation matrix from quaternion (same as CUDA baseline)
    r = qr
    x = qx
    y = qy
    z = qz

    # R matrix (row-major)
    R00 = 1.0 - 2.0 * (y * y + z * z)
    R01 = 2.0 * (x * y - r * z)
    R02 = 2.0 * (x * z + r * y)

    R10 = 2.0 * (x * y + r * z)
    R11 = 1.0 - 2.0 * (x * x + z * z)
    R12 = 2.0 * (y * z - r * x)

    R20 = 2.0 * (x * z - r * y)
    R21 = 2.0 * (y * z + r * x)
    R22 = 1.0 - 2.0 * (x * x + y * y)

    # M = S * R (scale then rotate)
    M00 = scale_x * R00
    M01 = scale_x * R01
    M02 = scale_x * R02

    M10 = scale_y * R10
    M11 = scale_y * R11
    M12 = scale_y * R12

    M20 = scale_z * R20
    M21 = scale_z * R21
    M22 = scale_z * R22

    # Compute 3D covariance: Σ = Mᵀ * M
    # Only need diagonal and upper triangle (symmetric)
    Sigma00 = M00*M00 + M10*M10 + M20*M20
    Sigma01 = M00*M01 + M10*M11 + M20*M21
    Sigma02 = M00*M02 + M10*M12 + M20*M22
    Sigma11 = M01*M01 + M11*M11 + M21*M21
    Sigma12 = M01*M02 + M11*M12 + M21*M22
    Sigma22 = M02*M02 + M12*M12 + M22*M22

    # Compute maximum eigenvalue of Σ using power iteration approximation
    # For better accuracy, could use full eigenvalue decomposition, but this is faster
    # Max eigenvalue ≈ trace(Σ) for well-conditioned matrices
    # Better approximation: Use characteristic polynomial
    trace = Sigma00 + Sigma11 + Sigma22

    # For conservative extent, use max of scales (simpler and safe)
    max_scale = tl.maximum(tl.maximum(scale_x, scale_y), scale_z)
    extent = 3.0 * max_scale  # 3σ extent

    # Alternative: compute actual max eigenvalue (more expensive but accurate)
    # We use conservative max_scale approach for speed

    # Find voxel bounding box
    voxel_min_x = tl.maximum(0, tl.cast((mean_x - extent - aabb_min_x) / voxel_size_x, tl.int32))
    voxel_min_y = tl.maximum(0, tl.cast((mean_y - extent - aabb_min_y) / voxel_size_y, tl.int32))
    voxel_min_z = tl.maximum(0, tl.cast((mean_z - extent - aabb_min_z) / voxel_size_z, tl.int32))

    voxel_max_x = tl.minimum(res_x - 1, tl.cast((mean_x + extent - aabb_min_x) / voxel_size_x + 1.0, tl.int32))
    voxel_max_y = tl.minimum(res_y - 1, tl.cast((mean_y + extent - aabb_min_y) / voxel_size_y + 1.0, tl.int32))
    voxel_max_z = tl.minimum(res_z - 1, tl.cast((mean_z + extent - aabb_min_z) / voxel_size_z + 1.0, tl.int32))

    # Compute inverse of Σ for Mahalanobis distance
    # For 3x3 symmetric matrix: Σ⁻¹ = (1/det) * adj(Σ)
    det = Sigma00 * (Sigma11 * Sigma22 - Sigma12 * Sigma12) - \
          Sigma01 * (Sigma01 * Sigma22 - Sigma02 * Sigma12) + \
          Sigma02 * (Sigma01 * Sigma12 - Sigma02 * Sigma11)

    if tl.abs(det) < 1e-10:
        return  # Degenerate Gaussian, skip

    det_inv = 1.0 / det

    # Inverse covariance (conic) - only upper triangle needed
    Conic00 = (Sigma11 * Sigma22 - Sigma12 * Sigma12) * det_inv
    Conic01 = (Sigma02 * Sigma12 - Sigma01 * Sigma22) * det_inv
    Conic02 = (Sigma01 * Sigma12 - Sigma02 * Sigma11) * det_inv
    Conic11 = (Sigma00 * Sigma22 - Sigma02 * Sigma02) * det_inv
    Conic12 = (Sigma01 * Sigma02 - Sigma00 * Sigma12) * det_inv
    Conic22 = (Sigma00 * Sigma11 - Sigma01 * Sigma01) * det_inv

    # Iterate over overlapping voxels
    # Adaptive range based on extent
    max_range = tl.cast(tl.ceil(extent / tl.minimum(tl.minimum(voxel_size_x, voxel_size_y), voxel_size_z)) + 1, tl.int32)
    max_range = tl.minimum(max_range, 10)  # Cap at 10 to avoid excessive loops

    for dz in range(-10, 11):
        vz = voxel_min_z + dz + 10
        vz_valid = (vz >= voxel_min_z) and (vz <= voxel_max_z) and (vz >= 0) and (vz < res_z)

        if vz_valid:
            voxel_center_z = aabb_min_z + (tl.cast(vz, tl.float32) + 0.5) * voxel_size_z
            diff_z = voxel_center_z - mean_z

            for dy in range(-10, 11):
                vy = voxel_min_y + dy + 10
                vy_valid = (vy >= voxel_min_y) and (vy <= voxel_max_y) and (vy >= 0) and (vy < res_y)

                if vy_valid:
                    voxel_center_y = aabb_min_y + (tl.cast(vy, tl.float32) + 0.5) * voxel_size_y
                    diff_y = voxel_center_y - mean_y

                    for dx in range(-10, 11):
                        vx = voxel_min_x + dx + 10
                        vx_valid = (vx >= voxel_min_x) and (vx <= voxel_max_x) and (vx >= 0) and (vx < res_x)

                        if vx_valid:
                            voxel_center_x = aabb_min_x + (tl.cast(vx, tl.float32) + 0.5) * voxel_size_x
                            diff_x = voxel_center_x - mean_x

                            # Compute Mahalanobis distance: dᵀ Σ⁻¹ d
                            # For symmetric Σ⁻¹, this is:
                            # d = [diff_x, diff_y, diff_z]
                            # dist² = d₀² C₀₀ + d₁² C₁₁ + d₂² C₂₂ + 2(d₀d₁ C₀₁ + d₀d₂ C₀₂ + d₁d₂ C₁₂)
                            dist_sq = diff_x * diff_x * Conic00 + \
                                     diff_y * diff_y * Conic11 + \
                                     diff_z * diff_z * Conic22 + \
                                     2.0 * (diff_x * diff_y * Conic01 + \
                                           diff_x * diff_z * Conic02 + \
                                           diff_y * diff_z * Conic12)

                            # Gaussian density
                            density = opacity * tl.exp(-0.5 * dist_sq)

                            # Atomic add to voxel
                            voxel_idx = vz * res_x * res_y + vy * res_x + vx
                            tl.atomic_add(occupancy_ptr + voxel_idx, density)


def scatter_gaussians_to_occupancy_anisotropic(
    gaussians,
    grid_resolution: torch.Tensor,
    grid_aabb: torch.Tensor,
    levels: int = 1
) -> torch.Tensor:
    """
    Triton-accelerated anisotropic Gaussian scattering using full 3D covariance.

    Args:
        gaussians: GaussianModel with get_xyz(), get_scaling(), get_rotation(), get_opacity()
        grid_resolution: [3] tensor with grid resolution
        grid_aabb: [6] tensor with AABB [xmin, ymin, zmin, xmax, ymax, zmax]
        levels: Number of grid levels (default: 1)

    Returns:
        Occupancy tensor [levels * res_x * res_y * res_z]
    """
    # Get Gaussian parameters
    xyz = gaussians.get_xyz().contiguous() if callable(gaussians.get_xyz) else gaussians.get_xyz.contiguous()
    scales = gaussians.get_scaling().contiguous() if callable(gaussians.get_scaling) else gaussians.get_scaling.contiguous()
    rotations = gaussians.get_rotation().contiguous() if callable(gaussians.get_rotation) else gaussians.get_rotation.contiguous()
    opacities = gaussians.get_opacity().contiguous() if callable(gaussians.get_opacity) else gaussians.get_opacity.contiguous()

    # Handle [N, 1] or [N] opacity shapes
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    device = xyz.device
    N_gaussians = xyz.shape[0]

    # Get grid parameters
    if isinstance(grid_resolution, int):
        res = torch.tensor([grid_resolution] * 3, device=device, dtype=torch.int32)
    else:
        res = grid_resolution.to(device).int()

    res_x, res_y, res_z = int(res[0]), int(res[1]), int(res[2])

    # Compute voxel size
    aabb_min = grid_aabb[:3].to(device)
    aabb_max = grid_aabb[3:].to(device)
    voxel_size = (aabb_max - aabb_min) / res.float()

    # Allocate occupancy grid
    occupancy = torch.zeros(levels * res_x * res_y * res_z, device=device, dtype=torch.float32)

    # Launch kernel
    grid_size = (N_gaussians,)
    BLOCK_SIZE = 1  # Each program handles one Gaussian

    scatter_gaussians_anisotropic_kernel[grid_size](
        xyz, scales, rotations, opacities,
        aabb_min, aabb_max, voxel_size,
        occupancy,
        N_gaussians,
        res_x, res_y, res_z,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return occupancy
