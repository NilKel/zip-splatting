# Neural Interval Splatting - Architecture Design Document

## Overview

Hybrid approach combining 3D Gaussian Splatting with NeRF-style volume rendering through interval-based feature aggregation and MLP decoding.

---

## Forward Pass

### Architecture Flow
```
Gaussians → Interval Aggregation → MLP Decode → Volume Rendering → RGB
```

### Step 1: Load Gaussians (3DGS Pattern)
```cuda
// Thread-per-pixel structure
// Block = 16x16 = 256 threads (one tile)
__shared__ int collected_id[BLOCK_SIZE];          // 256 Gaussians
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

// Collectively fetch batch of Gaussians from global to shared memory
collected_id[thread_rank] = point_list[range.x + progress];
collected_xy[thread_rank] = points_xy_image[coll_id];
collected_conic_opacity[thread_rank] = conic_opacity[coll_id];
block.sync();
```

### Step 2: Uniform Depth Intervals
```cuda
// Global parameters
float near_depth = 0.1f;
float far_depth = 100.0f;
int num_intervals = 4;  // or 16, 32, etc.
float interval_step = (far_depth - near_depth) / num_intervals;

// Per-pixel state (in thread-local registers)
float interval_features[MAX_INTERVALS][FEATURE_DIM];  // Accumulated features
float interval_weights[MAX_INTERVALS];                // Accumulated weights
```

**Design Decision:** Uniform intervals across all pixels (same `[near, far]` boundaries).
- Simplifies implementation
- Same depth → same interval index for all pixels
- Easy to debug

### Step 3: Process Gaussians Front-to-Back
```cuda
// Iterate over Gaussians (already depth-sorted)
for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
    int gaussian_id = collected_id[j];
    float gaussian_depth = depths[gaussian_id];

    // Determine which interval this Gaussian belongs to
    int gaussian_interval = (int)((gaussian_depth - near_depth) / interval_step);
    gaussian_interval = max(0, min(gaussian_interval, num_intervals - 1));

    // Check if Gaussian overlaps this pixel
    float2 xy = collected_xy[j];
    float2 d = {xy.x - pixf.x, xy.y - pixf.y};
    float4 con_o = collected_conic_opacity[j];
    float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
    if (power > 0.0f) continue;

    // Compute Gaussian weight and alpha
    float G = exp(power);                           // Gaussian weight
    float alpha = min(0.99f, con_o.w * G);         // opacity * G
    if (alpha < 1.0f / 255.0f) continue;

    // Accumulate into the appropriate interval
    for (int ch = 0; ch < FEATURE_DIM; ch++) {
        interval_features[gaussian_interval][ch] +=
            gaussian_features[gaussian_id * FEATURE_DIM + ch] * alpha;
    }
    interval_weights[gaussian_interval] += alpha;
}
```

**Key: No stack needed!** Gaussians are already sorted by depth, just accumulate into correct interval index.

### Step 4: Normalize and Save State
```cuda
// Normalize aggregated features
for (int i = 0; i < num_intervals; i++) {
    if (interval_weights[i] > 0.0f) {
        for (int ch = 0; ch < FEATURE_DIM; ch++) {
            interval_features[i][ch] /= interval_weights[i];
        }
    }
}

// Save to output (for MLP input)
// Output shape: [H, W, num_intervals, FEATURE_DIM]
for (int i = 0; i < num_intervals; i++) {
    int out_base = (pix_id * num_intervals + i) * FEATURE_DIM;
    for (int ch = 0; ch < FEATURE_DIM; ch++) {
        out_features[out_base + ch] = interval_features[i][ch];
    }
    interval_weights_output[pix_id * num_intervals + i] = interval_weights[i];
}
```

### Step 5: Save State for Backward

**What to save:** Only final per-interval state (no bucket boundaries needed!)

```cuda
// Save final interval weights and features to global memory
// Output shape: [H, W, num_intervals]
for (int i = 0; i < num_intervals; i++) {
    interval_weights_output[pix_id * num_intervals + i] = interval_weights[i];
}

// Output shape: [H, W, num_intervals, FEATURE_DIM]
for (int i = 0; i < num_intervals; i++) {
    int out_base = (pix_id * num_intervals + i) * FEATURE_DIM;
    for (int ch = 0; ch < FEATURE_DIM; ch++) {
        out_features[out_base + ch] = interval_features[i][ch];
    }
}
```

**Why only final state?**
- Intervals are independent (no cross-interval dependencies in aggregation)
- Order within interval doesn't matter (commutative: `sum(alpha*f) / sum(alpha)`)
- Backward needs denominator `sum(alpha)` for chain rule, but can recompute numerator
- Unlike baseline 3DGS which saves transmittance chain (cumulative across depth)

### Step 6: MLP Decoding (PyTorch/Tiny-CUDA-NN)

```python
# Input: interval_features [H, W, num_intervals, FEATURE_DIM]
# For each interval, compute sample location (midpoint)
interval_depths = near + (torch.arange(num_intervals) + 0.5) * interval_step  # [num_intervals]

# MLP input: [aggregated_features, sample_location, viewing_direction]
# For now: just use aggregated features + sample depth
mlp_input = interval_features  # [H, W, num_intervals, FEATURE_DIM]

# MLP forward
mlp_output = mlp(mlp_input)  # [H, W, num_intervals, 4] (RGB + density)
rgb_intervals = mlp_output[..., :3]      # [H, W, num_intervals, 3]
density_intervals = mlp_output[..., 3:]  # [H, W, num_intervals, 1]
```

### Step 7: Volume Rendering (NeRF-style, PyTorch Autograd)

```python
# Compute transmittance for each interval
delta_t = interval_step  # uniform spacing
alpha_intervals = 1 - torch.exp(-density_intervals * delta_t)  # [H, W, num_intervals, 1]

# Compute transmittance (product of (1 - alpha) from front)
T = torch.cumprod(1 - alpha_intervals + 1e-10, dim=2)  # [H, W, num_intervals, 1]
# T[i] = product of (1 - alpha[j]) for j < i

# Alpha compositing
weights = T * alpha_intervals  # [H, W, num_intervals, 1]
rgb_final = (weights * rgb_intervals).sum(dim=2)  # [H, W, 3]

# Add background
T_final = T[:, :, -1] * (1 - alpha_intervals[:, :, -1])  # Final transmittance
rgb_final = rgb_final + T_final * bg_color
```

**Important:** PyTorch autograd handles backward through `cumprod`, `exp`, etc. No need to manually save per-interval transmittance - it's automatically handled by PyTorch's computation graph!

---

## Backward Pass

### Overall Flow
```
dL/d(rgb) → Volume Rendering Backward → dL/d(rgb_i), dL/d(density_i)
          → MLP Backward → dL/d(aggregated_features)
          → Gaussian Aggregation Backward → dL/d(gaussian_features), dL/d(opacity), etc.
```

### Step 1: Volume Rendering + MLP Backward (PyTorch)

```python
# Standard PyTorch autograd handles this
loss = compute_loss(rgb_final, gt_rgb)
loss.backward()

# After backward, we have:
dL_dinterval_features = interval_features.grad  # [H, W, num_intervals, FEATURE_DIM]
```

This gradient flows through:
1. Volume rendering equations (standard NeRF backward)
2. MLP (tiny-cuda-nn handles this)
3. Arrives at `dL/d(aggregated_features)` for each pixel, each interval

### Step 2: Gaussian Aggregation Backward (CUDA Kernel)

**Key Insight:** Use thread-per-Gaussian structure (like baseline 3DGS backward)

#### Thread Structure
```cuda
// Launch config:
// 1 warp = 32 consecutive Gaussians (1 bucket)
// 1 block = multiple warps (e.g., 4 warps = 128 threads)
// Grid dimension = num_buckets / warps_per_block

auto my_warp = cg::tiled_partition<32>(block);
int bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
int gaussian_idx_in_warp = my_warp.thread_rank();  // 0-31
int gaussian_id = point_list[range.x + splat_idx_global];
```

#### Chain Rule Through Normalization

Forward computed:
```
aggregated_feature[i][ch] = sum(gaussian_features[g][ch] * alpha[g]) / sum(alpha[g])
                          = numerator / denominator
                          = numerator / interval_weight[i]
```

Backward (quotient rule):
```
dL/d(gaussian_features[g][ch]) = dL/d(aggregated[ch]) * alpha[g] / interval_weight

dL/d(alpha[g]) = sum_over_ch [
    dL/d(aggregated[ch]) * (gaussian_features[g][ch] - aggregated[ch]) / interval_weight
]
```

#### CUDA Implementation

```cuda
__global__ void backwardNeuralIntervals(
    // Saved from forward:
    const float* interval_weights,            // [H, W, num_intervals] - final weights (denominator)
    const float* interval_features,           // [H, W, num_intervals, FEATURE_DIM] - final normalized features

    // From MLP backward:
    const float* dL_dinterval_features,       // [H, W, num_intervals, FEATURE_DIM]

    // Gaussian data:
    const float* gaussian_features,           // [N, FEATURE_DIM]
    const float2* points_xy_image,            // [N]
    const float4* conic_opacity,              // [N]
    const float* depths,                      // [N]

    // Outputs:
    float* dL_dgaussian_features,             // [N, FEATURE_DIM]
    float* dL_dopacity,                       // [N]
    float* dL_dmean2D,                        // [N, 2]
    float* dL_dconic2D                        // [N, 3]
)
{
    // Setup warp and load MY Gaussian
    auto my_warp = cg::tiled_partition<32>(block);
    int gaussian_id = point_list[splat_idx_global];

    // Load MY Gaussian properties into registers
    float my_depth = depths[gaussian_id];
    float2 my_xy = points_xy_image[gaussian_id];
    float4 my_conic_opacity = conic_opacity[gaussian_id];
    float my_features[FEATURE_DIM];
    for (int ch = 0; ch < FEATURE_DIM; ch++) {
        my_features[ch] = gaussian_features[gaussian_id * FEATURE_DIM + ch];
    }

    // Gradient accumulators (in registers)
    float Register_dL_dfeatures[FEATURE_DIM] = {0};
    float Register_dL_dopacity = 0;
    float Register_dL_dmean2D[2] = {0};
    float Register_dL_dconic2D[3] = {0};

    // Determine which interval MY Gaussian belongs to (constant for all pixels)
    int my_interval = (int)((my_depth - near_depth) / interval_step);
    my_interval = max(0, min(my_interval, num_intervals - 1));

    // === ITERATE OVER ALL PIXELS IN TILE ===
    for (int pixel_idx = 0; pixel_idx < BLOCK_SIZE; pixel_idx++) {

        // Compute pixel coordinates
        uint2 pix = {pix_min.x + pixel_idx % BLOCK_X, pix_min.y + pixel_idx / BLOCK_X};
        uint32_t pix_id = W * pix.y + pix.x;
        bool valid = pix.x < W && pix.y < H;
        if (!valid) continue;

        // Check if MY Gaussian overlaps THIS pixel
        float2 d = {my_xy.x - pix.x, my_xy.y - pix.y};
        float power = -0.5f * (my_conic_opacity.x * d.x * d.x +
                               my_conic_opacity.z * d.y * d.y) -
                               my_conic_opacity.y * d.x * d.y;
        if (power > 0.0f) continue;

        float G = exp(power);
        float alpha = min(0.99f, my_conic_opacity.w * G);
        if (alpha < 1.0f / 255.0f) continue;

        // Load final interval weight (denominator from forward)
        float w_final = interval_weights[pix_id * num_intervals + my_interval];
        if (w_final <= 0.0f) continue;

        // Load gradients for this pixel's interval
        int grad_base = (pix_id * num_intervals + my_interval) * FEATURE_DIM;

        // Load normalized output features (from forward)
        float out_normalized[FEATURE_DIM];
        for (int ch = 0; ch < FEATURE_DIM; ch++) {
            out_normalized[ch] = interval_features[grad_base + ch];
        }

        // Load gradients from MLP backward
        float dL_dout[FEATURE_DIM];
        for (int ch = 0; ch < FEATURE_DIM; ch++) {
            dL_dout[ch] = dL_dinterval_features[grad_base + ch];
        }

        // === COMPUTE GRADIENTS ===

        // 1. Feature gradient: dL/dfeat = dL/dout * alpha / w
        //    Forward: out[ch] = sum(feat[ch] * alpha) / w
        //    Backward: dL/dfeat[ch] = dL/dout[ch] * (alpha / w)
        for (int ch = 0; ch < FEATURE_DIM; ch++) {
            Register_dL_dfeatures[ch] += dL_dout[ch] * alpha / w_final;
        }

        // 2. Alpha gradient (quotient rule)
        //    Forward: out[ch] = numerator[ch] / w
        //    d(out[ch])/d(alpha) = (feat[ch] * w - numerator[ch]) / w^2
        //                        = (feat[ch] - out[ch]) / w
        //    dL/dalpha = sum_ch [ dL/dout[ch] * (feat[ch] - out[ch]) / w ]
        float dL_dalpha = 0.0f;
        for (int ch = 0; ch < FEATURE_DIM; ch++) {
            dL_dalpha += dL_dout[ch] * (my_features[ch] - out_normalized[ch]) / w_final;
        }

        // 3. Chain to opacity: alpha = opacity * G
        Register_dL_dopacity += dL_dalpha * G;

        // 4. Chain to geometric parameters
        //    alpha = opacity * exp(power)
        //    dL/dG = dL/dalpha * opacity
        float dL_dG = my_conic_opacity.w * dL_dalpha;

        //    power = -0.5 * (conic.x * dx^2 + conic.z * dy^2 + 2*conic.y*dx*dy)
        //    G = exp(power), so dG/d(power) = G
        //    dL/d(power) = dL/dG * G
        float gdx = G * d.x;
        float gdy = G * d.y;

        // Gradient w.r.t. mean2D (with NDC to pixel conversion)
        Register_dL_dmean2D[0] += dL_dG * (my_conic_opacity.x * gdx + my_conic_opacity.y * gdy) * 0.5f * W;
        Register_dL_dmean2D[1] += dL_dG * (my_conic_opacity.z * gdy + my_conic_opacity.y * gdx) * 0.5f * H;

        // Gradient w.r.t. conic (inverse covariance)
        Register_dL_dconic2D[0] += -0.5f * dL_dG * d.x * d.x;
        Register_dL_dconic2D[1] += -dL_dG * d.x * d.y;
        Register_dL_dconic2D[2] += -0.5f * dL_dG * d.y * d.y;
    }

    // Write accumulated gradients to global memory
    // (Atomics needed for cross-tile contributions - multiple tiles can see same Gaussian)
    for (int ch = 0; ch < FEATURE_DIM; ch++) {
        atomicAdd(&dL_dgaussian_features[gaussian_id * FEATURE_DIM + ch], Register_dL_dfeatures[ch]);
    }
    atomicAdd(&dL_dopacity[gaussian_id], Register_dL_dopacity);
    atomicAdd(&dL_dmean2D[gaussian_id * 2 + 0], Register_dL_dmean2D[0]);
    atomicAdd(&dL_dmean2D[gaussian_id * 2 + 1], Register_dL_dmean2D[1]);
    atomicAdd(&dL_dconic2D[gaussian_id * 3 + 0], Register_dL_dconic2D[0]);
    atomicAdd(&dL_dconic2D[gaussian_id * 3 + 1], Register_dL_dconic2D[1]);
    atomicAdd(&dL_dconic2D[gaussian_id * 3 + 2], Register_dL_dconic2D[2]);
}
```

### Step 3: Convert 2D Gradients to 3D (Preprocessing Backward)

```cuda
// Use baseline 3DGS preprocessing backward
BACKWARD::preprocess(
    P, D, M,
    means3D,
    radii,
    nullptr,  // dc - not used
    nullptr,  // shs - not used
    geomState.clamped,
    opacities,
    scales,
    rotations,
    scale_modifier,
    cov3D_ptr,
    viewmatrix,
    projmatrix,
    focal_x, focal_y,
    tan_fovx, tan_fovy,
    campos,
    dL_dmean2D,      // 2D gradients from above
    dL_dconic2D,     // 2D gradients from above
    nullptr,         // dL_dinvdepth
    dL_dopacity,     // From above
    dL_dmean3D,      // OUTPUT: 3D position gradients
    nullptr,         // dL_dcolor
    dL_dcov3D,       // OUTPUT
    nullptr,         // dL_ddc
    nullptr,         // dL_dsh
    dL_dscale,       // OUTPUT
    dL_drot,         // OUTPUT
    false            // antialiasing
);
```

---

## Key Design Decisions

### ✅ **Thread-per-Pixel Forward**
- Matches baseline 3DGS structure
- Natural for per-pixel interval accumulation
- Each thread maintains interval state in registers

### ✅ **Thread-per-Gaussian Backward**
- Each Gaussian's gradients accumulated by ONE thread
- No atomic contention within tile
- Accumulation in fast registers
- Warp shuffle for passing pixel state

### ✅ **Uniform Global Intervals**
- All pixels use same `[near, far]` and interval boundaries
- Simplifies implementation
- Same depth → same interval index for all pixels
- Easier to debug

### ✅ **Save Only Final Interval State**
- NO bucket boundaries needed (buckets are threading detail only!)
- Save final `interval_weights` per interval (denominator for backward)
- Save final `interval_features` per interval (for quotient rule)
- Order within interval doesn't matter (commutative aggregation)

### ⚠️ **Normalized Aggregation Formula**
```
aggregated = sum(features * opacity * G) / sum(opacity * G)
           = sum(features * alpha) / interval_weight

Where:
  G = exp(-0.5 * d^T * Sigma^-1 * d)  // Gaussian weight function
  alpha = opacity * G                  // Combined weight
  opacity = learned per-Gaussian parameter
```

---

## Memory Layout

### Forward Pass Outputs
```
interval_features:         [H, W, num_intervals, FEATURE_DIM]  // Normalized aggregated features
interval_weights:          [H, W, num_intervals]               // Sum of alpha per interval (denominator)
```

### Backward Pass Inputs
```
dL_dinterval_features:     [H, W, num_intervals, FEATURE_DIM]  // From MLP backward
interval_features:         [H, W, num_intervals, FEATURE_DIM]  // From forward (for quotient rule)
interval_weights:          [H, W, num_intervals]               // From forward (denominator)
```

### Backward Pass Outputs
```
dL_dgaussian_features:     [N, FEATURE_DIM]
dL_dopacity:               [N, 1]
dL_dmean3D:                [N, 3]
dL_dscales:                [N, 3]
dL_drotations:             [N, 4]
```

---

## Implementation Notes

### Critical for Correctness

1. **Save only FINAL interval state** (no bucket boundaries needed!)
2. **Use saved weights in backward** for correct normalization chain rule (denominator)
3. **Handle empty intervals gracefully** (`if (interval_weight > 0)` checks)
4. **Atomic adds across tiles** in backward (multiple tiles can see same Gaussian)
5. **Buckets are threading detail only** - no semantic significance, don't confuse with intervals!

### Performance Optimizations

1. **Warp shuffling** instead of shared memory for pixel state
2. **Register accumulation** for gradients (no shared memory atomics)
3. **Early termination** in forward (transmittance threshold)
4. **Coalesced memory access** (consecutive Gaussians in warp)

### Debugging Strategy

1. Start with **small num_intervals** (4) and **small feature_dim** (16)
2. Test **forward only** first - visualize aggregated features directly
3. Add **MLP + volume rendering** second
4. Implement **backward** last
5. Compare gradients with **numerical differentiation** on toy example

---

## Open Questions / Future Improvements

### Questions for Review:
1. Should we save `interval_features` in forward or recompute in backward?
   - **Trade-off:** Memory vs computation
   - **Current choice:** Save it (easier to debug, needed for chain rule)

2. Do we need to save MLP outputs (rgb, density) for backward?
   - **Answer:** No, tiny-cuda-nn handles its own backward
   - Only need final `dL/d(aggregated_features)`

3. Should interval boundaries be per-pixel or global?
   - **Current choice:** Global uniform intervals
   - **Future:** Could add per-pixel near/far based on scene bounds

### Future Optimizations:
- [ ] Adaptive intervals based on scene density
- [ ] Variable number of intervals per pixel
- [ ] Learned interval boundaries
- [ ] Aggregate Gaussian depths (not just midpoints for sampling)
- [ ] Add viewing direction to MLP input
- [ ] Position encoding for interval locations

---

## Comparison to Baseline 3DGS

| Aspect | Baseline 3DGS | Neural Intervals |
|--------|---------------|------------------|
| **Forward** | Direct RGB blend | Feature aggregation → MLP → Volume render |
| **Intervals** | None | Depth-based intervals |
| **Features** | SH coefficients | Learned neural features |
| **Backward** | Thread-per-Gaussian | Thread-per-Gaussian (same!) |
| **State Saving** | T every 32 Gaussians (bucket boundaries) | interval_weights per interval (final only) |
| **Complexity** | Simpler | More complex (MLP, intervals) |

## Comparison to ZIP-NeRF / Instant-NGP

| Aspect | ZIP-NeRF | Neural Intervals |
|--------|----------|------------------|
| **Samples** | Ray marching | Interval-based Gaussian aggregation |
| **Features** | Grid interpolation | Gaussian splatting |
| **Rendering** | Volume integration | Same (alpha compositing) |
| **Speed** | Slower (ray marching) | Faster (tile-based rasterization) |

---

## References

- Original 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NeRF: https://www.matthewtancik.com/nerf
- Instant-NGP: https://nvlabs.github.io/instant-ngp/
- tiny-cuda-nn: https://github.com/NVlabs/tiny-cuda-nn
