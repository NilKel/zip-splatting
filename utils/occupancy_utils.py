"""
Occupancy Grid Utilities for ZIP Method

This module provides utilities for scattering Gaussian primitives into
occupancy grid voxels for use with NerfAcc's OccGridEstimator.
"""

import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from typing import Tuple


def scatter_gaussians_to_occupancy(
    gaussians: GaussianModel,
    grid_resolution: torch.Tensor,  # [3] int tensor
    grid_aabb: torch.Tensor,  # [6] float tensor [xmin, ymin, zmin, xmax, ymax, zmax]
    levels: int = 1
) -> torch.Tensor:
    """
    Scatter all Gaussians into occupancy grid voxels.

    For each Gaussian:
    1. Compute its 3σ bounding box in world space
    2. Find all voxels that overlap with this bounding box
    3. Evaluate the Gaussian's density at each voxel center
    4. Accumulate densities into the voxel occupancy grid

    Args:
        gaussians: GaussianModel containing all Gaussians
        grid_resolution: Resolution of the occupancy grid [resx, resy, resz]
        grid_aabb: Axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax]
        levels: Number of grid levels (default: 1)

    Returns:
        Occupancy tensor of shape [levels * resx * resy * resz]
    """
    # Try anisotropic Triton kernel first (most accurate), fallback to isotropic, then Python
    try:
        from utils.occupancy_triton_anisotropic import scatter_gaussians_to_occupancy_anisotropic
        return scatter_gaussians_to_occupancy_anisotropic(gaussians, grid_resolution, grid_aabb, levels)
    except ImportError:
        try:
            from utils.occupancy_triton import scatter_gaussians_to_occupancy_triton
            print("Warning: Using isotropic Triton kernel (anisotropic not available)")
            return scatter_gaussians_to_occupancy_triton(gaussians, grid_resolution, grid_aabb, levels)
        except ImportError:
            print("Warning: Triton not available, falling back to slow Python implementation")
            pass
    except Exception as e:
        print(f"Warning: Anisotropic Triton kernel failed ({e}), trying isotropic")
        try:
            from utils.occupancy_triton import scatter_gaussians_to_occupancy_triton
            return scatter_gaussians_to_occupancy_triton(gaussians, grid_resolution, grid_aabb, levels)
        except Exception as e2:
            print(f"Warning: Isotropic Triton kernel also failed ({e2}), falling back to Python")
            pass

    # Fallback to Python implementation
    device = gaussians.get_xyz.device

    # Get Gaussian parameters
    xyz = gaussians.get_xyz  # [N, 3]
    scales = gaussians.get_scaling  # [N, 3]
    rotations = gaussians.get_rotation  # [N, 4]
    opacities = gaussians.get_opacity  # [N, 1]

    N_gaussians = xyz.shape[0]

    # Initialize occupancy grid
    cells_per_lvl = int(grid_resolution.prod().item())
    total_cells = levels * cells_per_lvl
    occupancy = torch.zeros(total_cells, device=device, dtype=torch.float32)

    occupancy = _scatter_gaussians_python(
        xyz, scales, rotations, opacities,
        grid_resolution, grid_aabb, levels
    )

    return occupancy


def _scatter_gaussians_python(
    xyz: torch.Tensor,  # [N, 3]
    scales: torch.Tensor,  # [N, 3]
    rotations: torch.Tensor,  # [N, 4]
    opacities: torch.Tensor,  # [N, 1]
    grid_resolution: torch.Tensor,  # [3]
    grid_aabb: torch.Tensor,  # [6]
    levels: int
) -> torch.Tensor:
    """
    Python implementation of Gaussian scattering (slow, for initial testing).
    Will be replaced with CUDA kernel.
    """
    device = xyz.device
    N_gaussians = xyz.shape[0]

    # Compute grid properties
    res_x, res_y, res_z = grid_resolution.tolist()
    aabb_min = grid_aabb[:3]  # [3]
    aabb_max = grid_aabb[3:]  # [3]
    grid_size = aabb_max - aabb_min  # [3]
    voxel_size = grid_size / grid_resolution.float()  # [3]

    # Initialize occupancy for all levels
    cells_per_lvl = res_x * res_y * res_z
    occupancy = torch.zeros(levels * cells_per_lvl, device=device, dtype=torch.float32)

    # Process each level
    for lvl in range(levels):
        # For multi-level grids, NerfAcc enlarges the AABB by 2^lvl
        lvl_scale = 2 ** lvl
        lvl_aabb_min = aabb_min - (grid_size * (lvl_scale - 1) / 2)
        lvl_aabb_max = aabb_max + (grid_size * (lvl_scale - 1) / 2)
        lvl_grid_size = lvl_aabb_max - lvl_aabb_min
        lvl_voxel_size = lvl_grid_size / grid_resolution.float()

        # For each Gaussian, scatter to nearby voxels
        for g_idx in range(min(N_gaussians, 10000)):  # Limit for initial testing
            mean = xyz[g_idx]  # [3]
            scale = scales[g_idx]  # [3]
            opacity = opacities[g_idx].item()

            # Compute 3σ extent (99.7% of Gaussian mass)
            extent = 3.0 * scale  # [3]

            # Find voxel bounding box
            voxel_min = ((mean - extent - lvl_aabb_min) / lvl_voxel_size).floor().long()
            voxel_max = ((mean + extent - lvl_aabb_min) / lvl_voxel_size).ceil().long()

            # Clamp to grid bounds (element-wise)
            voxel_min = torch.maximum(voxel_min, torch.zeros_like(voxel_min))
            voxel_min = torch.minimum(voxel_min, grid_resolution.long() - 1)
            voxel_max = torch.maximum(voxel_max, torch.zeros_like(voxel_max))
            voxel_max = torch.minimum(voxel_max, grid_resolution.long() - 1)

            # For each overlapping voxel
            for vz in range(voxel_min[2].item(), voxel_max[2].item() + 1):
                for vy in range(voxel_min[1].item(), voxel_max[1].item() + 1):
                    for vx in range(voxel_min[0].item(), voxel_max[0].item() + 1):
                        # Compute voxel center in world space
                        voxel_coord = torch.tensor([vx, vy, vz], dtype=torch.float32, device=device)
                        voxel_center = lvl_aabb_min + (voxel_coord + 0.5) * lvl_voxel_size

                        # Evaluate Gaussian density at voxel center
                        # For now, use simple isotropic approximation
                        diff = voxel_center - mean  # [3]
                        dist_sq = (diff ** 2 / (scale ** 2 + 1e-8)).sum()
                        density = opacity * torch.exp(-0.5 * dist_sq).item()

                        # Accumulate to voxel
                        voxel_idx = lvl * cells_per_lvl + vz * res_x * res_y + vy * res_x + vx
                        occupancy[voxel_idx] += density

    return occupancy


def lookup_precomputed_occupancy(
    positions: torch.Tensor,  # [N, 3] query positions in world space
    precomputed_occ: torch.Tensor,  # [total_cells] occupancy values
    grid_resolution: torch.Tensor,  # [3]
    grid_aabb: torch.Tensor,  # [6]
    levels: int = 1
) -> torch.Tensor:
    """
    Look up precomputed occupancy values at given 3D positions.

    Uses nearest neighbor lookup for simplicity.

    Args:
        positions: Query positions in world space [N, 3]
        precomputed_occ: Precomputed occupancy grid values
        grid_resolution: Grid resolution [resx, resy, resz]
        grid_aabb: Grid bounding box [xmin, ymin, zmin, xmax, ymax, zmax]
        levels: Number of grid levels

    Returns:
        Occupancy values at query positions [N, 1]
    """
    device = positions.device
    N = positions.shape[0]

    # For simplicity, only use level 0
    lvl = 0
    aabb_min = grid_aabb[:3]
    aabb_max = grid_aabb[3:]
    grid_size = aabb_max - aabb_min

    # Convert positions to voxel coordinates
    voxel_coords = ((positions - aabb_min) / grid_size * grid_resolution.float()).long()

    # Clamp to valid range (element-wise)
    voxel_coords = torch.maximum(voxel_coords, torch.zeros_like(voxel_coords))
    voxel_coords = torch.minimum(voxel_coords, (grid_resolution - 1).long())

    # Compute flat indices
    res_x, res_y, res_z = grid_resolution.tolist()
    cells_per_lvl = res_x * res_y * res_z
    flat_indices = (
        lvl * cells_per_lvl +
        voxel_coords[:, 2] * res_x * res_y +
        voxel_coords[:, 1] * res_x +
        voxel_coords[:, 0]
    )

    # Look up occupancy values
    occupancy_values = precomputed_occ[flat_indices].unsqueeze(-1)  # [N, 1]

    return occupancy_values


def compute_scene_aabb_from_points(points: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """
    Compute axis-aligned bounding box from a set of 3D points.

    Args:
        points: Point cloud [N, 3]
        margin: Relative margin to add (default: 10%)

    Returns:
        AABB tensor [6] = [xmin, ymin, zmin, xmax, ymax, zmax]
    """
    aabb_min = points.min(dim=0)[0]  # [3]
    aabb_max = points.max(dim=0)[0]  # [3]

    # Add margin
    extent = aabb_max - aabb_min
    aabb_min = aabb_min - margin * extent
    aabb_max = aabb_max + margin * extent

    aabb = torch.cat([aabb_min, aabb_max])  # [6]
    return aabb
