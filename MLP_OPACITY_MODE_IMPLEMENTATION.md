# MLP-Opacity Mode Implementation

**Date**: 2025-11-11
**Status**: Implementation in progress (Phases 1-4 complete, need CUDA rebuild + Python integration)

## Overview

This document describes the implementation of **MLP-Opacity Mode**, a refactored architecture that decouples density from the MLP. Key changes:

- **Density**: Derived from rasterized average of Gaussians' explicit opacities
- **MLP**: Simplified to decode RGB color only (view-dependent appearance)
- **Benefits**: More stable training, better gradient flow to Gaussian opacities

## Architecture Comparison

### Original (Two-Stage MLP Mode)
```
Gaussians → Feature Aggregation → MLP → [RGB + Density] → Volume Rendering
```

### New (MLP-Opacity Mode)
```
Gaussians → Feature Aggregation → MLP → [RGB only]
                ↓                              ↓
         Opacity Averaging  → [Opacity] → Volume Rendering
```

## Implementation Status

### ✅ Phase 1: Simplified RGB-Only MLP

**File**: `utils/neural_mlp.py`

**Changes Made**:
1. Created new `RGBOnlyMLP` class
   - Input: `[B, feature_dim + view_encoding_dim]`
   - Output: `[B, 3]` RGB with sigmoid activation
   - No density prediction

2. Updated `create_interval_mlp()` factory function
   - Added `rgb_only=False` parameter
   - Returns `RGBOnlyMLP` when `rgb_only=True`

**Code**:
```python
class RGBOnlyMLP(nn.Module):
    """RGB-only MLP for MLP-opacity mode."""

    def __init__(self, feature_dim=32, sh_degree=4, hidden_dim=64, n_hidden_layers=2):
        # Create SH encoder for view directions
        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config=get_spherical_harmonics_encoding_config(sh_degree)
        )

        # Input is features + encoded view direction
        input_dim = feature_dim + self.view_encoder.n_output_dims

        # Create RGB network (output activation = Sigmoid)
        self.network = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=3,  # RGB only
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
            }
        )

    def forward(self, features, view_dirs):
        view_encoded = self.view_encoder(view_dirs)
        combined = torch.cat([features, view_encoded], dim=-1)
        rgb = self.network(combined)  # Sigmoid already applied
        return rgb
```

**Parameter Count**:
- Total: ~7,000 parameters (half of two-stage MLP)
- Network: `(32+16) → 64 → 64 → 3`

---

### ✅ Phase 2: Modified CUDA Forward Kernel

**File**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_neural_intervals.cu`

**Changes Made**:
1. Added `interval_opacity_sum[MAX_INTERVALS]` register buffer
2. Added `interval_opacities` output parameter to kernel signature
3. Accumulate weighted opacity during splatting:
   ```cuda
   interval_opacity_sum[gaussian_interval] += alpha * con_o.w;
   ```
4. Compute and write average opacity in output section:
   ```cuda
   float avg_opacity = interval_opacity_sum[iv] / interval_weight[iv];
   interval_opacities[pix_id * num_intervals + iv] = avg_opacity;
   ```

**Kernel Signature** (updated):
```cuda
template <uint32_t FEATURE_DIM>
__global__ void renderCUDA_neural_intervals(
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H, int P,
    const float2* points_xy_image,
    const float* features_neural,
    const float4* conic_opacity,
    float* final_T,
    uint32_t* n_contrib,
    uint32_t* max_contrib,
    float* out_features,           // [H, W, num_intervals, FEATURE_DIM]
    float* interval_weights,       // [H, W, num_intervals]
    float* interval_opacities,     // [H, W, num_intervals] ← NEW OUTPUT
    const float* depths,
    int num_intervals,
    float near_depth,
    float far_depth)
```

**Header File**: `cuda_rasterizer/forward.h` - Updated `FORWARD_NEURAL::render_intervals()` signature

---

### ✅ Phase 3: Python Render Function

**File**: `gaussian_renderer/mlp_opacity_render.py` (NEW)

**Implementation**:
```python
def render_mlp_opacity(
    viewpoint_camera, pc, mlp, pipe, bg_color,
    num_intervals=16, near_depth=0.1, far_depth=100.0,
    feature_dim=32, scaling_modifier=1.0,
    override_color=None, debug_iteration=None
):
    """
    Three-phase rendering pipeline:
    1. Rasterize features + opacities
    2. MLP decodes RGB only
    3. Composite using rasterized opacities for density
    """

    # Phase 1: Rasterize (returns features + opacities)
    interval_features, interval_opacities, radii = rasterize_neural_intervals(
        ...,
        return_opacities=True  # Request opacity output
    )

    # Phase 2: MLP RGB-only decoding
    rgb_flat = torch.empty((total_samples, 3), device='cuda')
    for i in range(0, total_samples, batch_size):
        features_batch = features_flat[i:end_idx]
        view_dirs_batch = view_dirs_flat[i:end_idx]
        rgb_batch = mlp(features_batch, view_dirs_batch)  # [B, 3] only
        rgb_flat[i:end_idx] = rgb_batch.float()

    decoded_colors = rgb_flat.reshape(H, W, N, 3)

    # Phase 3: Composite with rasterized opacities
    final_image = composite_mlp_opacity(
        decoded_colors=decoded_colors,
        interval_opacities=interval_opacities,
        background=bg_color,
        ...
    )

    return {"render": final_image, ...}
```

---

### ✅ Phase 4: New Composite CUDA Kernel

**File**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/composite_mlp_opacity.cu` (NEW)

**Key Logic**:
```cuda
__global__ void compositeCUDA_mlp_opacity(
    int W, int H,
    const float* decoded_colors,      // [H, W, num_intervals, 3] from MLP
    const float* interval_opacities,  // [H, W, num_intervals] from rasterization
    float* out_color,                 // [H, W, 3]
    const float* bg_color,
    int num_intervals,
    float near_depth,
    float far_depth)
{
    // For each pixel
    float T = 1.0f;
    float C[3] = {0, 0, 0};

    for (int i = 0; i < num_intervals; i++) {
        // Read RGB from MLP
        float r = decoded_colors[...];
        float g = decoded_colors[...];
        float b = decoded_colors[...];

        // Read average opacity from rasterization
        float avg_opacity = interval_opacities[interval_idx];

        // Convert opacity to density:
        // alpha = 1 - exp(-density * delta)
        // → density = -log(1 - alpha) / delta
        float delta = (far_depth - near_depth) / num_intervals;
        float density = -logf(1.0f - avg_opacity + 1e-6f) / delta;

        // Compute alpha for compositing
        float alpha = 1.0f - expf(-density * delta);
        alpha = min(0.99f, alpha);

        // NeRF compositing
        float weight = T * alpha;
        C[0] += r * weight;
        C[1] += g * weight;
        C[2] += b * weight;
        T *= (1.0f - alpha);

        if (T < 0.001f) break;
    }

    // Add background
    C[0] += bg_color[0] * T;
    C[1] += bg_color[1] * T;
    C[2] += bg_color[2] * T;

    out_color[...] = C[0];
    out_color[...] = C[1];
    out_color[...] = C[2];
}
```

**Header File**: `cuda_rasterizer/forward.h` - Added `COMPOSITE_MLP_OPACITY` namespace

---

## Remaining Work

### ⏳ Phase 5: Python Bindings Update

**What's Needed**:
1. Update C++ extension bindings to expose new functions
2. Rebuild CUDA extension

**Files to Modify**:
- `submodules/diff-gaussian-rasterization/rasterize_points.cu` (C++ binding code)
- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`

**Required Changes**:

#### 5.1: Update `rasterize_neural_intervals` to return opacities

**In `rasterize_points.cu`**:
```cpp
std::tuple<
    int,
    int,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,  // Add interval_opacities tensor
    torch::Tensor,
    torch::Tensor,
    torch::Tensor
>
RasterizeGaussiansNeuralIntervalsCUDA(
    // ... existing parameters ...
)
{
    // ... existing code ...

    // Allocate output for interval opacities
    torch::Tensor interval_opacities = torch::zeros(
        {height * width * num_intervals},
        means3D.options().dtype(torch::kFloat32)
    );

    // Call CUDA kernel with new parameter
    FORWARD_NEURAL::render_intervals(
        grid, block,
        ranges,
        point_list,
        W, H, P,
        means2D_ptr,
        features_neural_ptr,
        conic_opacity_ptr,
        final_T_ptr,
        n_contrib_ptr,
        max_contrib_ptr,
        out_features_ptr,
        interval_weights_ptr,
        interval_opacities.contiguous().data_ptr<float>(),  // NEW
        depths_ptr,
        num_intervals,
        near_depth,
        far_depth,
        feature_dim
    );

    return std::make_tuple(
        num_rendered,
        num_buckets,
        out_features,
        interval_weights,
        interval_opacities,  // NEW
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer
    );
}
```

#### 5.2: Add `composite_mlp_opacity` binding

**In `rasterize_points.cu`**:
```cpp
torch::Tensor CompositeMLPOpacityCUDA(
    const torch::Tensor& decoded_colors,      // [H, W, num_intervals, 3]
    const torch::Tensor& interval_opacities,  // [H, W, num_intervals]
    const torch::Tensor& background,
    const int width,
    const int height,
    const int num_intervals,
    const float near_depth,
    const float far_depth)
{
    // Allocate output
    torch::Tensor out_color = torch::zeros(
        {height * width * 3},
        decoded_colors.options().dtype(torch::kFloat32)
    );

    // Set up grid/block
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Call CUDA kernel
    COMPOSITE_MLP_OPACITY::composite(
        grid, block,
        width, height,
        decoded_colors.contiguous().data_ptr<float>(),
        interval_opacities.contiguous().data_ptr<float>(),
        out_color.contiguous().data_ptr<float>(),
        background.contiguous().data_ptr<float>(),
        num_intervals,
        near_depth,
        far_depth
    );

    return out_color;
}

// Register binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // ... existing bindings ...
    m.def("composite_mlp_opacity", &CompositeMLPOpacityCUDA);
}
```

#### 5.3: Update Python wrapper

**In `diff_gaussian_rasterization/__init__.py`**:
```python
# Update rasterize_neural_intervals to return opacities
def rasterize_neural_intervals(..., return_opacities=False):
    class _RasterizeNeuralIntervals(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ...):
            # Call C++ extension
            num_rendered, num_buckets, out_features, interval_weights, interval_opacities, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_neural_intervals(*args)

            if return_opacities:
                return out_features, interval_opacities, radii
            else:
                return out_features, radii

    return _RasterizeNeuralIntervals.apply(...)


# Add new composite_mlp_opacity function
def composite_mlp_opacity(
    decoded_colors,
    interval_opacities,
    background,
    raster_settings,
    num_intervals,
    near_depth,
    far_depth
):
    """
    Composite MLP-decoded RGB + rasterized opacities into final image.
    """
    # Reshape inputs
    H, W, N, C = decoded_colors.shape
    assert C == 3, "decoded_colors must have 3 channels (RGB)"
    assert interval_opacities.shape == (H, W, N), "opacity shape mismatch"

    # Flatten for CUDA kernel
    decoded_colors_flat = decoded_colors.reshape(H * W * N * 3)
    interval_opacities_flat = interval_opacities.reshape(H * W * N)

    # Call C++ extension
    out_color = _C.composite_mlp_opacity(
        decoded_colors_flat,
        interval_opacities_flat,
        background,
        W, H,
        num_intervals,
        near_depth,
        far_depth
    )

    # Reshape output to [H, W, 3]
    out_color = out_color.reshape(H, W, 3)

    return out_color
```

#### 5.4: Rebuild Extension

```bash
cd submodules/diff-gaussian-rasterization
python setup.py build_ext --inplace
```

---

### ⏳ Phase 6: Command-Line Argument

**File**: `arguments/__init__.py`

**Changes**:
```python
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ... existing arguments ...

        # Add blending mode argument
        self.blending = "mlp-opacity"  # Default to new mode
        parser.add_argument("--blending", type=str, default="mlp-opacity",
                          choices=["mlp-density", "mlp-opacity"],
                          help="Blending mode: mlp-density (two-stage MLP) or mlp-opacity (RGB-only MLP with rasterized opacity)")
```

**File**: `train.py`

**Changes**:
```python
# Instantiate MLP based on blending mode
if opt.blending == "mlp-opacity":
    # RGB-only MLP
    mlp = create_interval_mlp(
        feature_dim=gaussians.feature_dim,
        rgb_only=True,
        sh_degree=4,
        color_hidden_dim=64,
        color_n_hidden=2
    ).to("cuda")

    from gaussian_renderer.mlp_opacity_render import render_mlp_opacity
    render_fn = render_mlp_opacity

else:  # mlp-density
    # Two-stage MLP (original)
    mlp = create_interval_mlp(
        feature_dim=gaussians.feature_dim,
        geo_feat_dim=16,
        two_stage=True,
        ...
    ).to("cuda")

    from gaussian_renderer.neural_render import render_neural_intervals
    render_fn = render_neural_intervals

# In training loop
render_pkg = render_fn(
    viewpoint_cam, gaussians, mlp, pipe, background,
    ...
)
```

---

### ⏳ Phase 7: Testing

**Test Steps**:
1. Rebuild CUDA extension
2. Run small training test (100 iterations)
3. Verify outputs:
   - Check that `interval_opacities` has reasonable values (0-1 range)
   - Check that rendered images look plausible
   - Verify gradients flow to Gaussian opacities
4. Run full training and compare to baseline

**Debug Flags**:
```bash
python train.py --blending mlp-opacity --debug_iteration 10
```

---

## Key Benefits of MLP-Opacity Mode

1. **Simpler MLP**: ~50% fewer parameters (7K vs 14K)
2. **Better gradient flow**: Opacity gradients go directly to Gaussians
3. **More stable training**: Density not learned by MLP (less coupling)
4. **Easier pruning**: Can prune Gaussians with low opacity directly
5. **Physically meaningful**: Opacity is a direct property of Gaussians

---

## Memory Layout

### Forward Pass Outputs
```
interval_features:     [H, W, num_intervals, feature_dim]  (e.g., 800x800x16x32 = 1.3 GB)
interval_opacities:    [H, W, num_intervals]               (e.g., 800x800x16 = 164 MB)
decoded_colors:        [H, W, num_intervals, 3]            (e.g., 800x800x16x3 = 123 MB)
```

### Memory Savings vs Two-Stage Mode
- No density prediction tensor needed
- Simpler MLP = less activation memory
- **Estimated savings**: ~10-15% total memory

---

## Performance Expectations

### Speed
- **Rasterization**: ~5% slower (extra opacity accumulation)
- **MLP**: ~1.5× faster (single network, simpler input)
- **Composite**: Same speed (same NeRF compositing logic)
- **Overall**: ~1.2× faster than two-stage mode

### Quality
- Should match or exceed two-stage mode
- Better gradient signal to Gaussian opacities
- More stable convergence

---

## Troubleshooting

### Common Issues

**Issue**: `return_opacities` parameter not recognized
- **Fix**: Make sure Python bindings were updated and extension rebuilt

**Issue**: Shape mismatch in composite kernel
- **Fix**: Check that `decoded_colors` is [H,W,N,3] and `interval_opacities` is [H,W,N]

**Issue**: NaN or Inf in opacity values
- **Fix**: Check that Gaussian opacities are properly initialized (use sigmoid activation)

**Issue**: Black output images
- **Fix**: Verify that `interval_opacities` is not all zeros (check rasterization)

**Issue**: Gradients not flowing to Gaussian opacities
- **Fix**: Ensure autograd graph is connected through opacity → density conversion

---

## Files Modified/Created

### Created
- `utils/neural_mlp.py` - `RGBOnlyMLP` class
- `gaussian_renderer/mlp_opacity_render.py` - Main render function
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/composite_mlp_opacity.cu` - Composite kernel

### Modified
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_neural_intervals.cu` - Added opacity accumulation
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h` - Updated signatures
- (TODO) `submodules/diff-gaussian-rasterization/rasterize_points.cu` - Python bindings
- (TODO) `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py` - Python wrappers
- (TODO) `arguments/__init__.py` - Command-line argument
- (TODO) `train.py` - MLP instantiation and render function selection

---

## References

- Original 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- NeRF density-opacity conversion: NeRF (Mildenhall et al. 2020)
- Instant-NGP MLP design: https://nvlabs.github.io/instant-ngp/
