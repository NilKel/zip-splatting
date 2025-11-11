"""
Triton kernels for fast Gaussian scattering into occupancy grids.
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def scatter_gaussians_kernel(
    # Gaussian parameters
    xyz_ptr,  # [N, 3]
    scales_ptr,  # [N, 3]
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
    Scatter Gaussians into voxel occupancy grid.
    Each program processes one Gaussian and accumulates its contribution
    to all overlapping voxels.
    """
    # Get Gaussian index
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

    opacity = tl.load(opacities_ptr + g_idx)

    # Load grid parameters
    aabb_min_x = tl.load(grid_aabb_min_ptr + 0)
    aabb_min_y = tl.load(grid_aabb_min_ptr + 1)
    aabb_min_z = tl.load(grid_aabb_min_ptr + 2)

    voxel_size_x = tl.load(voxel_size_ptr + 0)
    voxel_size_y = tl.load(voxel_size_ptr + 1)
    voxel_size_z = tl.load(voxel_size_ptr + 2)

    # Compute 3σ extent (99.7% of mass)
    extent_x = 3.0 * scale_x
    extent_y = 3.0 * scale_y
    extent_z = 3.0 * scale_z

    # Find voxel bounding box
    voxel_min_x = tl.maximum(0, tl.cast((mean_x - extent_x - aabb_min_x) / voxel_size_x, tl.int32))
    voxel_min_y = tl.maximum(0, tl.cast((mean_y - extent_y - aabb_min_y) / voxel_size_y, tl.int32))
    voxel_min_z = tl.maximum(0, tl.cast((mean_z - extent_z - aabb_min_z) / voxel_size_z, tl.int32))

    voxel_max_x = tl.minimum(res_x - 1, tl.cast((mean_x + extent_x - aabb_min_x) / voxel_size_x + 1.0, tl.int32))
    voxel_max_y = tl.minimum(res_y - 1, tl.cast((mean_y + extent_y - aabb_min_y) / voxel_size_y + 1.0, tl.int32))
    voxel_max_z = tl.minimum(res_z - 1, tl.cast((mean_z + extent_z - aabb_min_z) / voxel_size_z + 1.0, tl.int32))

    # Iterate over overlapping voxels
    # NOTE: Triton doesn't support continue, so we use nested if statements
    # Assume max 5x5x5 voxel neighborhood (should cover 3σ for most cases)
    for dz in range(-2, 3):  # -2, -1, 0, 1, 2
        vz = voxel_min_z + dz + 2

        # Check bounds
        vz_valid = (vz >= voxel_min_z) and (vz <= voxel_max_z) and (vz >= 0) and (vz < res_z)

        if vz_valid:
            voxel_center_z = aabb_min_z + (tl.cast(vz, tl.float32) + 0.5) * voxel_size_z
            diff_z = voxel_center_z - mean_z

            for dy in range(-2, 3):
                vy = voxel_min_y + dy + 2

                vy_valid = (vy >= voxel_min_y) and (vy <= voxel_max_y) and (vy >= 0) and (vy < res_y)

                if vy_valid:
                    voxel_center_y = aabb_min_y + (tl.cast(vy, tl.float32) + 0.5) * voxel_size_y
                    diff_y = voxel_center_y - mean_y

                    for dx in range(-2, 3):
                        vx = voxel_min_x + dx + 2

                        vx_valid = (vx >= voxel_min_x) and (vx <= voxel_max_x) and (vx >= 0) and (vx < res_x)

                        if vx_valid:
                            voxel_center_x = aabb_min_x + (tl.cast(vx, tl.float32) + 0.5) * voxel_size_x
                            diff_x = voxel_center_x - mean_x

                            # Compute isotropic Gaussian density (simplified)
                            # Full version would use rotation, but this is faster approximation
                            dist_sq = (diff_x * diff_x) / (scale_x * scale_x + 1e-8) + \
                                     (diff_y * diff_y) / (scale_y * scale_y + 1e-8) + \
                                     (diff_z * diff_z) / (scale_z * scale_z + 1e-8)

                            density = opacity * tl.exp(-0.5 * dist_sq)

                            # Atomic add to voxel
                            voxel_idx = vz * res_x * res_y + vy * res_x + vx
                            tl.atomic_add(occupancy_ptr + voxel_idx, density)


def scatter_gaussians_to_occupancy_triton(
    gaussians,
    grid_resolution: torch.Tensor,
    grid_aabb: torch.Tensor,
    levels: int = 1
) -> torch.Tensor:
    """
    Triton-accelerated version of scatter_gaussians_to_occupancy.

    Args:
        gaussians: GaussianModel
        grid_resolution: [3] tensor with grid resolution
        grid_aabb: [6] tensor with AABB [xmin, ymin, zmin, xmax, ymax, zmax]
        levels: Number of grid levels (default: 1)

    Returns:
        Occupancy tensor [levels * res_x * res_y * res_z]
    """
    # Get Gaussian parameters
    xyz = gaussians.get_xyz().contiguous() if callable(gaussians.get_xyz) else gaussians.get_xyz.contiguous()  # [N, 3]
    scales = gaussians.get_scaling().contiguous() if callable(gaussians.get_scaling) else gaussians.get_scaling.contiguous()  # [N, 3]
    opacities = gaussians.get_opacity().contiguous() if callable(gaussians.get_opacity) else gaussians.get_opacity.contiguous()  # [N, 1 or N]

    # Handle both [N, 1] and [N] shapes
    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    device = xyz.device

    N_gaussians = xyz.shape[0]

    # Get grid parameters
    res_x, res_y, res_z = grid_resolution.tolist()
    cells_per_lvl = res_x * res_y * res_z

    # Initialize output
    occupancy = torch.zeros(levels * cells_per_lvl, device=device, dtype=torch.float32)

    # Process each level
    for lvl in range(levels):
        # Compute level-specific AABB (NerfAcc enlarges by 2^lvl)
        lvl_scale = 2 ** lvl
        aabb_min = grid_aabb[:3]
        aabb_max = grid_aabb[3:]
        grid_size = aabb_max - aabb_min

        lvl_aabb_min = aabb_min - (grid_size * (lvl_scale - 1) / 2)
        lvl_aabb_max = aabb_max + (grid_size * (lvl_scale - 1) / 2)
        lvl_grid_size = lvl_aabb_max - lvl_aabb_min
        voxel_size = lvl_grid_size / grid_resolution.float()

        # Get level-specific occupancy view
        lvl_occupancy = occupancy[lvl * cells_per_lvl:(lvl + 1) * cells_per_lvl]

        # Launch kernel
        grid = (triton.cdiv(N_gaussians, 256),)
        scatter_gaussians_kernel[grid](
            xyz_ptr=xyz,
            scales_ptr=scales,
            opacities_ptr=opacities,
            grid_aabb_min_ptr=lvl_aabb_min.contiguous(),
            grid_aabb_max_ptr=lvl_aabb_max.contiguous(),
            voxel_size_ptr=voxel_size.contiguous(),
            occupancy_ptr=lvl_occupancy,
            N_gaussians=N_gaussians,
            res_x=res_x,
            res_y=res_y,
            res_z=res_z,
            BLOCK_SIZE=256,
        )

    return occupancy


# Optimized version with better handling of large voxel neighborhoods
@triton.jit
def scatter_gaussians_kernel_v2(
    # Gaussian parameters (flattened)
    xyz_ptr, scales_ptr, opacities_ptr,
    # Grid parameters
    aabb_min_x, aabb_min_y, aabb_min_z,
    voxel_size_x, voxel_size_y, voxel_size_z,
    # Output
    occupancy_ptr,
    # Dimensions
    N_gaussians: tl.constexpr,
    res_x: tl.constexpr,
    res_y: tl.constexpr,
    res_z: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    V2: Process multiple Gaussians per block for better occupancy.
    """
    g_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask for valid Gaussians
    mask = g_idx < N_gaussians

    # Load Gaussian parameters (masked)
    mean_x = tl.load(xyz_ptr + g_idx * 3 + 0, mask=mask, other=0.0)
    mean_y = tl.load(xyz_ptr + g_idx * 3 + 1, mask=mask, other=0.0)
    mean_z = tl.load(xyz_ptr + g_idx * 3 + 2, mask=mask, other=0.0)

    scale_x = tl.load(scales_ptr + g_idx * 3 + 0, mask=mask, other=1.0)
    scale_y = tl.load(scales_ptr + g_idx * 3 + 1, mask=mask, other=1.0)
    scale_z = tl.load(scales_ptr + g_idx * 3 + 2, mask=mask, other=1.0)

    opacity = tl.load(opacities_ptr + g_idx, mask=mask, other=0.0)

    # Note: This version is more complex and would require vectorized voxel iteration
    # For simplicity, sticking with v1 for now
    pass


def benchmark_triton_kernel(N_gaussians=100000, grid_res=128):
    """Quick benchmark to test Triton kernel performance."""
    import time

    device = "cuda"

    # Create dummy data
    xyz = torch.randn(N_gaussians, 3, device=device)
    scales = torch.rand(N_gaussians, 3, device=device) * 0.1
    opacities = torch.rand(N_gaussians, 1, device=device)

    class DummyGaussians:
        def __init__(self):
            self._xyz = xyz
            self._scales = scales
            self._opacities = opacities

        def get_xyz(self):
            return self._xyz

        def get_scaling(self):
            return self._scales

        def get_opacity(self):
            return self._opacities

    gaussians = DummyGaussians()

    grid_resolution = torch.tensor([grid_res, grid_res, grid_res], device=device)
    grid_aabb = torch.tensor([-1, -1, -1, 1, 1, 1], device=device, dtype=torch.float32)

    # Warmup
    _ = scatter_gaussians_to_occupancy_triton(gaussians, grid_resolution, grid_aabb)
    torch.cuda.synchronize()

    # Benchmark
    n_iters = 10
    start = time.time()
    for _ in range(n_iters):
        occupancy = scatter_gaussians_to_occupancy_triton(gaussians, grid_resolution, grid_aabb)
        torch.cuda.synchronize()
    end = time.time()

    avg_time = (end - start) / n_iters
    print(f"Triton kernel: {avg_time*1000:.2f} ms/iter")
    print(f"Throughput: {N_gaussians / avg_time / 1e6:.2f} M Gaussians/s")
    print(f"Occupancy stats: mean={occupancy.mean():.6f}, max={occupancy.max():.6f}, nonzero={(occupancy > 0).sum().item()}/{occupancy.shape[0]}")

    return occupancy


if __name__ == "__main__":
    print("Testing Triton Gaussian scattering kernel...")
    benchmark_triton_kernel(N_gaussians=100000, grid_res=32)
