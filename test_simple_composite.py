#!/usr/bin/env python3
"""
Simpler test - just test compositing first since that one should work.
"""

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    composite_neural
)

device = torch.device("cuda:0")
H, W = 128, 128
num_intervals = 8

print("Testing composite_neural only...")

# Create dummy decoded intervals (RGB + density)
decoded_intervals = torch.rand(H, W, num_intervals, 4, device=device)
decoded_intervals[..., :3] = torch.sigmoid(decoded_intervals[..., :3])  # RGB in [0,1]
decoded_intervals[..., 3] = torch.nn.functional.softplus(decoded_intervals[..., 3])  # density >= 0

background = torch.tensor([0.5, 0.5, 0.5], device=device)

# Dummy settings (only need H, W)
viewmatrix = torch.eye(4, device=device)
projmatrix = torch.eye(4, device=device)

raster_settings = GaussianRasterizationSettings(
    image_height=H,
    image_width=W,
    tanfovx=1.0,
    tanfovy=1.0,
    bg=background,
    scale_modifier=1.0,
    viewmatrix=viewmatrix,
    projmatrix=projmatrix,
    sh_degree=0,
    campos=torch.zeros(3, device=device),
    prefiltered=False,
    antialiasing=False,
    debug=False
)

# Call function
out_color = composite_neural(
    decoded_intervals=decoded_intervals,
    background=background,
    raster_settings=raster_settings,
    num_intervals=num_intervals,
    near_depth=0.1,
    far_depth=100.0
)

print(f"✅ Composite test passed!")
print(f"   Output shape: {out_color.shape}")
print(f"   Color range: [{out_color.min():.3f}, {out_color.max():.3f}]")
