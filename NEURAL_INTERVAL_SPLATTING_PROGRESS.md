# Neural Interval Splatting Implementation Progress

## Overview
This document tracks the implementation of a two-pass neural interval splatting renderer that combines 3D Gaussian Splatting with tiny-cuda-nn MLPs.

**Architecture**: Pass 1 aggregates neural features into depth intervals → Pass 2 uses MLP to decode features into RGB+density → Final compositing

---

## ✅ Completed Components

### Phase 0: Setup & Dependencies
- [x] Installed tiny-cuda-nn in conda environment `gaussian_splatting_py310`
- [x] Verified installation with test script ([test_tcnn.py](test_tcnn.py))
- [x] Confirmed MLP forward/backward passes work correctly

**Location**: `/home/nilkel/Projects/tiny-cuda-nn` (installed from source)

---

### Phase 1: GaussianModel Feature Storage
**File**: `scene/gaussian_model.py`

**Changes Made**:

1. **Added neural feature storage** (line 50-58):
   ```python
   def __init__(self, sh_degree, optimizer_type="default", feature_dim=32):
       self.feature_dim = feature_dim  # Neural feature dimension
       self._features_neural = torch.empty(0)  # Neural features for interval MLP
       # ... rest of init
   ```

2. **Added getter property** (line 130-132):
   ```python
   @property
   def get_features_neural(self):
       return self._features_neural
   ```

3. **Initialize neural features in `create_from_pcd`** (line 175-177):
   ```python
   # Initialize neural features with small random values
   neural_features = torch.randn((fused_point_cloud.shape[0], self.feature_dim), device="cuda") * 0.1
   self._features_neural = nn.Parameter(neural_features.requires_grad_(True))
   ```

4. **Added to optimizer** (line 196):
   ```python
   {'params': [self._features_neural], 'lr': training_args.feature_lr, "name": "f_neural"},
   ```

5. **Updated save/load** (line 76, 92):
   - Added `self._features_neural` to `capture()` and `restore()`

6. **Updated densification methods**:
   - `densification_postfix()`: Added `new_features_neural` parameter (line 400)
   - `densify_and_clone()`: Clone neural features (line 458)
   - `densify_and_split()`: Repeat neural features (line 441)
   - `prune_points()`: Prune neural features (line 368)

**Status**: ✅ Complete - Neural features are fully integrated into GaussianModel

---

### Phase 3: MLP Wrapper
**File**: `utils/neural_mlp.py`

**Created Classes**:

1. **`IntervalMLP`**: Base class wrapping tiny-cuda-nn's FullyFusedMLP
   - Input: `[B, feature_dim]` features
   - Output: `[B, 4]` RGB + density
   - Architecture: 2-layer, 64 hidden units by default
   - Uses ReLU activation, no output activation

2. **`IntervalMLPWithActivation`**: Extended version with sigmoid/softplus activations
   - Applies sigmoid to RGB → [0, 1]
   - Applies softplus to density → positive values

3. **`create_interval_mlp()`**: Factory function for easy instantiation

**Testing**: Successfully tested with 1024-sample batches, forward/backward passes work correctly.

**Status**: ✅ Complete - MLP wrapper is functional and tested

---

## 🚧 Remaining Work

### Phase 2: Feature Aggregation CUDA Kernel
**File to Create**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_neural_intervals.cu`

**Based On**: `forward_intervals.cu` (already exists and implements uniform interval rendering)

**Required Changes**:

1. **Template for feature dimension**:
   ```cpp
   template <uint32_t CHANNELS, uint32_t FEATURE_DIM>
   __global__ void renderCUDA_neural_intervals(
       const float* features_neural,  // [N, FEATURE_DIM] instead of RGB
       float* out_features,           // [H, W, num_intervals, FEATURE_DIM]
       // ... other params
   )
   ```

2. **Interval buffer changes**:
   - Replace `float interval_color[3]` with `float interval_features[FEATURE_DIM]`
   - Keep `interval_weight` for normalization

3. **Accumulation logic** (modify lines ~246-254):
   ```cpp
   for (int ch = 0; ch < FEATURE_DIM; ch++)
       interval_features[ch] += features_neural[gaussian_id * FEATURE_DIM + ch] * alpha;
   interval_weight += alpha;
   ```

4. **Output to global memory** (when interval completes):
   ```cpp
   // Normalize and write features
   int out_idx = (pix_y * W + pix_x) * num_intervals * FEATURE_DIM + current_interval * FEATURE_DIM;
   for (int ch = 0; ch < FEATURE_DIM; ch++) {
       out_features[out_idx + ch] = interval_features[ch] / (interval_weight + 1e-6f);
   }
   ```

**Key Design Decision**: Output aggregated features to global memory, NOT final RGB. MLP evaluation happens in Python between CUDA passes.

**Estimated Effort**: 1-2 days

---

### Phase 4: Compositing Kernel
**File to Create**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/composite_neural.cu`

**Purpose**: Take MLP-decoded RGB+density and perform NeRF volumetric compositing.

**Kernel Signature**:
```cpp
__global__ void compositeCUDA_neural(
    const float* decoded_intervals,  // [H, W, num_intervals, 4] (RGB+density)
    float* out_color,                // [H, W, 3]
    float near_depth,
    float far_depth,
    int num_intervals,
    int H, int W
)
```

**Algorithm** (per-pixel):
```cpp
float delta = (far_depth - near_depth) / num_intervals;
float T = 1.0f;
float C[3] = {0};

for (int i = 0; i < num_intervals; i++) {
    // Read RGB + density for this interval
    int idx = (pix_y * W + pix_x) * num_intervals * 4 + i * 4;
    float rgb[3] = {decoded_intervals[idx], decoded_intervals[idx+1], decoded_intervals[idx+2]};
    float density = decoded_intervals[idx+3];

    // NeRF volumetric rendering
    float alpha = 1.0f - expf(-density * delta);

    C[0] += T * alpha * rgb[0];
    C[1] += T * alpha * rgb[1];
    C[2] += T * alpha * rgb[2];
    T *= (1.0f - alpha);

    if (T < 0.01f) break;  // Early termination
}

// Write output
out_color[(pix_y * W + pix_x) * 3 + 0] = C[0];
out_color[(pix_y * W + pix_x) * 3 + 1] = C[1];
out_color[(pix_y * W + pix_x) * 3 + 2] = C[2];
```

**Estimated Effort**: 1 day

---

### Phase 5: Python Pipeline Integration
**File to Create**: `gaussian_renderer/neural_render.py`

**Function**: `render_neural_intervals()`

**Architecture**:
```python
def render_neural_intervals(viewpoint_cam, gaussians, mlp, pipe, bg, estimator=None, ...):
    # Pass 1: Aggregate features into intervals
    interval_features, radii = rasterizer.forward_neural_intervals(
        means3D=gaussians.get_xyz,
        features_neural=gaussians.get_features_neural,
        opacities=gaussians.get_opacity,
        # ... other Gaussian params
    )  # Returns [H, W, num_intervals, FEATURE_DIM]

    # Reshape for batch MLP evaluation
    H, W, N, F = interval_features.shape
    features_flat = interval_features.view(-1, F)  # [H*W*N, F]

    # Pass 2a: MLP decode
    with torch.cuda.amp.autocast(enabled=False):  # tcnn requires fp32 or fp16
        decoded_flat = mlp(features_flat.half()).float()  # [H*W*N, 4]

    decoded = decoded_flat.view(H, W, N, 4)

    # Pass 2b: Final composite
    final_image = rasterizer.composite_neural(
        decoded_intervals=decoded,
        near_depth=adaptive_near,
        far_depth=adaptive_far,
        num_intervals=num_intervals
    )  # Returns [H, W, 3]

    return {
        "render": final_image,
        "viewspace_points": ...,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
```

**Estimated Effort**: 1 day

---

### Phase 6: C++ Bindings
**Files to Modify**:

1. **`submodules/diff-gaussian-rasterization/ext.cpp`**:
   ```cpp
   PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
       // ... existing bindings
       m.def("rasterize_neural_intervals", &RasterizeNeuralIntervalsCUDA);
       m.def("composite_neural", &CompositeNeuralCUDA);
   }
   ```

2. **`submodules/diff-gaussian-rasterization/rasterize_points.cu`**:
   Add C++ wrapper functions that call the CUDA kernels.

3. **`submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`**:
   Create autograd functions:
   ```python
   class _RasterizeNeuralIntervals(torch.autograd.Function):
       @staticmethod
       def forward(ctx, means3D, features_neural, ...):
           ...

       @staticmethod
       def backward(ctx, grad_features):
           ...

   class _CompositeNeural(torch.autograd.Function):
       @staticmethod
       def forward(ctx, decoded_intervals, ...):
           ...

       @staticmethod
       def backward(ctx, grad_out_color):
           ...
   ```

4. **`submodules/diff-gaussian-rasterization/setup.py`**:
   ```python
   sources=[
       # ... existing files
       "cuda_rasterizer/forward_neural_intervals.cu",
       "cuda_rasterizer/composite_neural.cu",
       "cuda_rasterizer/backward_neural_intervals.cu",  # Phase 7
   ]
   ```

**Estimated Effort**: 2 days

---

### Phase 7: Backward Pass
**File to Create**: `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward_neural_intervals.cu`

**Based On**: `backward_intervals.cu`

**Required Changes**:
- Propagate gradients to `FEATURE_DIM` channels instead of 3
- Chain rule through feature normalization
- No need to propagate gradients to MLP weights (handled by PyTorch autograd in Python)

**Estimated Effort**: 2-3 days

---

### Phase 8: Training Integration
**File to Modify**: `train.py`

**Changes**:

1. **Add arguments**:
   ```python
   parser.add_argument("--use_neural_intervals", action="store_true")
   parser.add_argument("--feature_dim", type=int, default=32)
   parser.add_argument("--mlp_hidden_dim", type=int, default=64)
   parser.add_argument("--mlp_n_layers", type=int, default=2)
   ```

2. **Initialize MLP**:
   ```python
   if opt.use_neural_intervals:
       from utils.neural_mlp import create_interval_mlp
       mlp = create_interval_mlp(
           feature_dim=opt.feature_dim,
           hidden_dim=opt.mlp_hidden_dim,
           n_hidden_layers=opt.mlp_n_layers
       ).cuda()
       optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)
   ```

3. **Modify training loop**:
   ```python
   if opt.use_neural_intervals:
       from gaussian_renderer.neural_render import render_neural_intervals
       render_pkg = render_neural_intervals(
           viewpoint_cam, gaussians, mlp, pipe, background,
           estimator=estimator
       )

       # ... compute loss ...

       optimizer.step()      # Gaussian parameters
       optimizer_mlp.step()  # MLP parameters
   ```

**Estimated Effort**: 1 day

---

## 📊 Progress Summary

| Phase | Component | Status | Est. Time | Priority |
|-------|-----------|--------|-----------|----------|
| 0 | Setup & Dependencies | ✅ Complete | - | - |
| 1 | GaussianModel Integration | ✅ Complete | - | - |
| 3 | MLP Wrapper | ✅ Complete | - | - |
| 2 | Feature Aggregation Kernel | 🚧 Pending | 1-2 days | **HIGH** |
| 4 | Compositing Kernel | 🚧 Pending | 1 day | **HIGH** |
| 5 | Python Pipeline | 🚧 Pending | 1 day | **MEDIUM** |
| 6 | C++ Bindings | 🚧 Pending | 2 days | **MEDIUM** |
| 7 | Backward Pass | 🚧 Pending | 2-3 days | **LOW** (can test forward-only first) |
| 8 | Training Integration | 🚧 Pending | 1 day | **MEDIUM** |
| 9 | Testing & Validation | 🚧 Pending | 3-5 days | **HIGH** |

**Total Remaining Effort**: 11-15 days

---

## 🔧 Implementation Strategy

### Recommended Order:

1. **Phase 2**: Implement `forward_neural_intervals.cu` kernel
   - Start by copying `forward_intervals.cu` and modifying for features
   - Test that it outputs correct shapes before integrating

2. **Phase 4**: Implement `composite_neural.cu` kernel
   - This is simpler and can be tested independently

3. **Phase 6**: Add C++ bindings
   - Wire up the two new kernels to Python

4. **Phase 5**: Create Python pipeline
   - Test end-to-end forward pass (no training yet)

5. **Phase 8**: Training integration
   - Test forward-only rendering with a small scene
   - Verify gradients don't explode

6. **Phase 7**: Backward pass
   - Only needed for full training
   - Can test with frozen Gaussians initially

7. **Phase 9**: Full testing
   - Train on small scene (100 images)
   - Compare PSNR with baseline
   - Profile performance

---

## 🐛 Known Issues & Considerations

### Memory Requirements
- **Interval features buffer**: `[H, W, num_intervals, FEATURE_DIM]`
- For 1920x1080, 16 intervals, 32 features: ~4.2 GB
- **Solution**: Use FP16 for features, or reduce resolution during training

### Batch Size for MLP
- tiny-cuda-nn requires batch size to be multiple of 128
- `H * W * num_intervals` may not be aligned
- **Solution**: Pad to nearest multiple of 128, discard padded outputs

### Gradient Flow
- MLP gradients flow back through PyTorch autograd automatically
- Feature gradients need custom CUDA backward pass
- **Test strategy**: Start with frozen MLP, train only features

### Existing Infrastructure
- Current `forward_intervals.cu` already implements uniform intervals
- We're essentially replacing RGB accumulation with feature accumulation
- Blending logic remains the same

---

## 📝 Next Steps (Immediate)

1. **Start with Phase 2**: Create `forward_neural_intervals.cu`
   - Copy `forward_intervals.cu` as starting point
   - Modify line-by-line to accumulate features instead of RGB
   - Focus on getting correct output shapes first

2. **Add debug prints**: Print intermediate values to verify correctness
   - Check that features are being accumulated
   - Verify interval boundaries are correct
   - Ensure normalization doesn't cause NaNs

3. **Test incrementally**: Don't try to integrate everything at once
   - First: Test feature aggregation kernel standalone
   - Second: Add MLP evaluation in Python
   - Third: Add compositing kernel
   - Fourth: Full pipeline

---

## 📚 Reference Files

### Existing Code to Study:
- `forward_intervals.cu` (lines 32-354): Current interval rendering
- `forward.cu` (lines 119-153): Baseline covariance computation
- `backward_intervals.cu`: Gradient computation structure
- `zip_render.py` (lines 171-378): Current rendering pipeline

### Key Concepts:
- **Interval assignment**: `int gaussian_interval = min((int)((gaussian_depth - near_depth) / interval_step), num_intervals - 1);`
- **Alpha computation**: `float alpha = min(0.99f, con_o.w * exp(power));`
- **Weighted averaging**: `avg_feature = sum(feature * alpha) / sum(alpha)`

---

## ✅ Verification Checklist (for when complete)

- [ ] Feature aggregation produces correct shapes
- [ ] MLP evaluation runs without errors
- [ ] Compositing produces valid images
- [ ] Gradients don't explode or vanish
- [ ] Training loss decreases
- [ ] PSNR comparable to baseline (within 2-3 dB)
- [ ] Rendering speed reasonable (<10x slower than baseline)
- [ ] Memory usage acceptable (<16 GB for 1080p)
- [ ] No NaNs or Infs in any stage
- [ ] Densification works with neural features

---

**Last Updated**: Session continuation (Phase 0-3 complete)
**Author**: Claude Agent
**Project**: Neural Interval Splatting for 3DGS
