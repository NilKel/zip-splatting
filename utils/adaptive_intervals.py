"""
Adaptive interval generation using NerfAcc occupancy grid.

Instead of uniform depth intervals, we create intervals that concentrate
samples in regions where Gaussians are present.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_adaptive_intervals_from_occupancy(
    estimator,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    num_intervals: int = 16,
    near_plane: float = 0.1,
    far_plane: float = 100.0,
    occupancy_threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute adaptive depth intervals per ray using NerfAcc occupancy grid.

    Uses NerfAcc's grid traversal to find occupied regions along each ray,
    then distributes intervals proportionally to occupancy density.

    Args:
        estimator: NerfAcc OccGridEstimator
        rays_o: Ray origins [N_rays, 3]
        rays_d: Ray directions [N_rays, 3] (normalized)
        num_intervals: Number of intervals to generate
        near_plane: Near clipping plane
        far_plane: Far clipping plane
        occupancy_threshold: Minimum occupancy to consider

    Returns:
        interval_bounds: [N_rays, num_intervals + 1] - depth boundaries for each interval
        interval_weights: [N_rays, num_intervals] - relative importance of each interval
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device

    # Use NerfAcc to sample along rays based on occupancy
    # This gives us a variable number of samples per ray
    def sigma_fn(t_starts, t_ends, ray_indices):
        """Dummy sigma function - we just use occupancy grid."""
        # Sample at interval midpoints
        t_mids = (t_starts + t_ends) / 2.0
        positions = rays_o[ray_indices] + rays_d[ray_indices] * t_mids.unsqueeze(-1)

        # Query occupancy grid
        occupancy = estimator.query_occ(positions)
        return occupancy

    # Get samples from NerfAcc
    try:
        from nerfacc import traverse_grids

        # Traverse the occupancy grid to get occupied intervals
        intervals, samples, _ = traverse_grids(
            rays_o,
            rays_d,
            estimator.binaries,  # Binary occupancy grid
            estimator.aabbs,
            near_planes=torch.full((N_rays,), near_plane, device=device),
            far_planes=torch.full((N_rays,), far_plane, device=device),
            step_size=None,  # Use grid resolution
            cone_angle=0.0
        )

        # intervals: (n_intervals,) - which ray each interval belongs to
        # samples: (n_intervals, 2) - (t_start, t_end) for each interval

    except Exception as e:
        print(f"NerfAcc traversal failed: {e}")
        # Fallback to uniform intervals
        return compute_uniform_intervals(rays_o, num_intervals, near_plane, far_plane)

    # Now we have variable number of intervals per ray
    # We need to consolidate into fixed num_intervals per ray

    interval_bounds = torch.zeros(N_rays, num_intervals + 1, device=device)
    interval_weights = torch.zeros(N_rays, num_intervals, device=device)

    for ray_idx in range(N_rays):
        # Get all intervals for this ray
        mask = (intervals == ray_idx)
        ray_samples = samples[mask]  # [K, 2] where K varies per ray

        if len(ray_samples) == 0:
            # No occupied regions, use uniform intervals
            t_vals = torch.linspace(near_plane, far_plane, num_intervals + 1, device=device)
            interval_bounds[ray_idx] = t_vals
            interval_weights[ray_idx] = 1.0 / num_intervals
            continue

        # Get occupied depth range
        t_starts = ray_samples[:, 0]
        t_ends = ray_samples[:, 1]
        occupied_near = t_starts.min().item()
        occupied_far = t_ends.max().item()

        # Sample occupancy at interval centers
        t_centers = (t_starts + t_ends) / 2.0
        positions = rays_o[ray_idx:ray_idx+1].expand(len(t_centers), -1) + \
                   rays_d[ray_idx:ray_idx+1].expand(len(t_centers), -1) * t_centers.unsqueeze(-1)

        # Query occupancy
        occupancy = estimator.query_occ(positions).squeeze()

        # Create adaptive intervals:
        # - More intervals in high-occupancy regions
        # - Fewer intervals in low-occupancy regions

        # Compute cumulative occupancy distribution
        interval_lengths = t_ends - t_starts
        weighted_occupancy = occupancy * interval_lengths
        cumulative = torch.cumsum(weighted_occupancy, dim=0)
        cumulative = cumulative / cumulative[-1]  # Normalize to [0, 1]

        # Sample uniformly in cumulative space
        target_quantiles = torch.linspace(0, 1, num_intervals + 1, device=device)

        # Find corresponding depths
        adaptive_depths = torch.zeros(num_intervals + 1, device=device)
        adaptive_depths[0] = occupied_near
        adaptive_depths[-1] = occupied_far

        for i in range(1, num_intervals):
            # Find interval containing this quantile
            idx = torch.searchsorted(cumulative, target_quantiles[i])
            if idx >= len(t_starts):
                idx = len(t_starts) - 1

            # Interpolate within the interval
            if idx > 0:
                alpha = (target_quantiles[i] - cumulative[idx-1]) / \
                       (cumulative[idx] - cumulative[idx-1] + 1e-8)
            else:
                alpha = target_quantiles[i] / (cumulative[idx] + 1e-8)

            adaptive_depths[i] = t_starts[idx] + alpha * interval_lengths[idx]

        interval_bounds[ray_idx] = adaptive_depths

        # Compute weights (could be uniform or based on occupancy)
        interval_weights[ray_idx] = 1.0 / num_intervals

    return interval_bounds, interval_weights


def compute_uniform_intervals(
    rays_o: torch.Tensor,
    num_intervals: int,
    near_plane: float,
    far_plane: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback: uniform intervals."""
    N_rays = rays_o.shape[0]
    device = rays_o.device

    t_vals = torch.linspace(near_plane, far_plane, num_intervals + 1, device=device)
    interval_bounds = t_vals.unsqueeze(0).expand(N_rays, -1)
    interval_weights = torch.ones(N_rays, num_intervals, device=device) / num_intervals

    return interval_bounds, interval_weights


def compute_global_adaptive_intervals(
    estimator,
    num_intervals: int = 16,
    near_plane: float = 0.1,
    far_plane: float = 100.0
) -> torch.Tensor:
    """
    Compute global adaptive intervals (same for all rays) based on occupancy grid.

    This is simpler than per-ray intervals and works well when scene depth distribution
    is consistent across views.

    Args:
        estimator: NerfAcc OccGridEstimator
        num_intervals: Number of intervals
        near_plane: Near clipping
        far_plane: Far clipping

    Returns:
        interval_bounds: [num_intervals + 1] - global depth boundaries
    """
    device = estimator.binaries[0].device

    # Get AABB
    aabb = torch.tensor(estimator.aabbs[0], device=device)  # [6]
    aabb_min = aabb[:3]
    aabb_max = aabb[3:]

    # Sample depth slices through the scene
    num_depth_samples = 1000
    depth_samples = torch.linspace(near_plane, far_plane, num_depth_samples, device=device)

    # For each depth, compute average occupancy
    # Use a grid of ray origins on a sphere around scene center
    scene_center = (aabb_min + aabb_max) / 2.0
    scene_radius = (aabb_max - aabb_min).norm() / 2.0

    # Sample ray directions
    num_rays_sample = 100
    phi = torch.rand(num_rays_sample, device=device) * 2 * np.pi
    theta = torch.acos(2 * torch.rand(num_rays_sample, device=device) - 1)

    ray_dirs = torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=1)

    # Compute occupancy histogram over depth
    depth_occupancy = torch.zeros(num_depth_samples, device=device)

    for i, depth in enumerate(depth_samples):
        # Sample positions at this depth along various rays
        positions = scene_center.unsqueeze(0) + ray_dirs * depth

        # Query occupancy
        occ = estimator.query_occ(positions)
        depth_occupancy[i] = occ.mean()

    # Smooth the histogram
    kernel_size = 5
    kernel = torch.ones(kernel_size, device=device) / kernel_size
    depth_occupancy_smooth = torch.nn.functional.conv1d(
        depth_occupancy.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        padding=kernel_size//2
    ).squeeze()

    # Compute cumulative distribution
    cumulative = torch.cumsum(depth_occupancy_smooth, dim=0)
    cumulative = cumulative / (cumulative[-1] + 1e-8)

    # Sample uniformly in cumulative space
    target_quantiles = torch.linspace(0, 1, num_intervals + 1, device=device)
    interval_bounds = torch.zeros(num_intervals + 1, device=device)

    for i, q in enumerate(target_quantiles):
        # Find depth corresponding to this quantile
        idx = torch.searchsorted(cumulative, q)
        if idx >= len(depth_samples):
            idx = len(depth_samples) - 1
        interval_bounds[i] = depth_samples[idx]

    # Ensure monotonic and within bounds
    interval_bounds[0] = near_plane
    interval_bounds[-1] = far_plane

    return interval_bounds
