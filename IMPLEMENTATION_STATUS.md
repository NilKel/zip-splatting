# Neural Interval Splatting - Implementation Status

## Session Progress Summary

### ✅ Completed Components

#### 1. Foundation (Previous Session)
- [x] tiny-cuda-nn installed and tested
- [x] Neural features added to GaussianModel
- [x] IntervalMLP Python wrapper created
- [x] instant-ngp architecture analyzed

#### 2. CUDA Kernels (This Session)
- [x] **forward_neural_intervals.cu** - Feature aggregation kernel
  - Aggregates neural features into depth intervals
  - Template-based for multiple feature dimensions (16, 32, 64)
  - Outputs [H, W, num_intervals, FEATURE_DIM]

- [x] **composite_neural.cu** - Compositing kernel
  - Takes MLP-decoded RGB+density
  - Performs NeRF volumetric rendering
  - Outputs final [H, W, 3] image

- [x] **forward.h** - Added namespace declarations
  - FORWARD_NEURAL::render_intervals
  - COMPOSITE_NEURAL::composite

#### 3. C++ Wrappers (This Session - In Progress)
- [x] RasterizeNeuralIntervalsCUDA - Feature aggregation wrapper
- [x] CompositeNeuralCUDA - Compositing wrapper
- ⚠️  **NEEDS FIX**: binningState extraction in rasterize_points.cu line 610

### 🚧 Remaining Work

#### Phase 1: Fix and Complete C++ Integration (1 hour)

**File**: `submodules/diff-gaussian-rasterization/rasterize_points.cu`

**Issue at line 610**:
```cpp
binningState.point_list  // ❌ binningState not declared
```

**Fix**:
```cpp
// After line 604, add:
BinningState binningState = BinningState::fromChunk(binning_ptr, P);
```

**Then update ext.cpp** to register new functions:
```cpp
m.def("rasterize_neural_intervals", &RasterizeNeuralIntervalsCUDA);
m.def("composite_neural", &CompositeNeuralCUDA);
```

#### Phase 2: Python Autograd Bindings (2-3 hours)

**File**: `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`

Create two new autograd functions:

**1. _RasterizeNeuralIntervals**:
```python
class _RasterizeNeuralIntervals(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, features_neural, opacities, scales, rotations,
                cov3D_precomp, raster_settings, num_intervals, near_depth, far_depth, feature_dim):
        # Call _C.rasterize_neural_intervals
        out_features, radii = _C.rasterize_neural_intervals(...)

        # Save for backward
        ctx.save_for_backward(means3D, features_neural, opacities, scales, rotations, radii, ...)
        ctx.feature_dim = feature_dim

        return out_features, radii

    @staticmethod
    def backward(ctx, grad_out_features, grad_radii):
        # For now, can return None for all gradients (test forward-only first)
        # Later: implement backward_neural_intervals.cu
        return None, None, None, ...
```

**2. _CompositeNeural**:
```python
class _CompositeNeural(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decoded_intervals, background, raster_settings, num_intervals, near_depth, far_depth):
        # Call _C.composite_neural
        out_color = _C.composite_neural(...)

        # Save for backward
        ctx.save_for_backward(decoded_intervals)
        ctx.num_intervals = num_intervals
        ctx.near_depth = near_depth
        ctx.far_depth = far_depth

        return out_color

    @staticmethod
    def backward(ctx, grad_out_color):
        # Gradients flow back through MLP automatically (PyTorch autograd)
        # Need to compute grad_decoded_intervals
        decoded_intervals, = ctx.saved_tensors

        # For now, simple gradient pass-through
        # Later: implement proper volumetric rendering gradients
        grad_decoded = compute_composite_gradients(grad_out_color, ...)

        return grad_decoded, None, None, None, None, None
```

**3. Wrapper in GaussianRasterizer**:
```python
class GaussianRasterizer(nn.Module):
    def forward_neural_intervals(self, means3D, means2D, features_neural, opacities,
                                 scales, rotations, cov3D_precomp,
                                 num_intervals=16, near_depth=0.1, far_depth=100.0, feature_dim=32):
        return _RasterizeNeuralIntervals.apply(
            means3D, means2D, features_neural, opacities, scales, rotations, cov3D_precomp,
            self.raster_settings, num_intervals, near_depth, far_depth, feature_dim
        )

    def composite_neural(self, decoded_intervals, background,
                        num_intervals=16, near_depth=0.1, far_depth=100.0):
        return _CompositeNeural.apply(
            decoded_intervals, background,
            self.raster_settings, num_intervals, near_depth, far_depth
        )
```

#### Phase 3: Python Rendering Pipeline (2 hours)

**File**: `gaussian_renderer/neural_render.py`

```python
def render_neural_intervals(viewpoint_cam, gaussians, mlp, pipe, bg, estimator=None,
                            num_intervals=16, near_depth=0.1, far_depth=100.0, feature_dim=32):
    # Setup
    screenspace_points = torch.zeros_like(gaussians.get_xyz, requires_grad=True)

    # Rasterizer
    rasterizer = GaussianRasterizer(raster_settings=...)

    # Compute adaptive depth bounds (if estimator available)
    adaptive_near, adaptive_far = compute_adaptive_depth_bounds(
        viewpoint_cam, estimator, gaussians,
        default_near=near_depth,
        default_far=far_depth
    )

    # Pass 1: Aggregate features into intervals
    interval_features, radii = rasterizer.forward_neural_intervals(
        means3D=gaussians.get_xyz,
        means2D=screenspace_points,
        features_neural=gaussians.get_features_neural,
        opacities=gaussians.get_opacity,
        scales=gaussians.get_scaling,
        rotations=gaussians.get_rotation,
        cov3D_precomp=None,
        num_intervals=num_intervals,
        near_depth=adaptive_near,
        far_depth=adaptive_far,
        feature_dim=feature_dim
    )  # Returns [H, W, num_intervals, feature_dim]

    # Pass 2: MLP evaluation
    H, W, N, F = interval_features.shape
    features_flat = interval_features.reshape(-1, F)  # [H*W*N, F]

    # Evaluate MLP (tiny-cuda-nn)
    with torch.cuda.amp.autocast(enabled=False):  # Force FP32 or FP16
        decoded_flat = mlp(features_flat.half()).float()  # [H*W*N, 4]

    decoded = decoded_flat.reshape(H, W, N, 4)  # [H, W, N, 4]

    # Pass 3: Composite into final image
    final_image = rasterizer.composite_neural(
        decoded_intervals=decoded,
        background=bg,
        num_intervals=num_intervals,
        near_depth=adaptive_near,
        far_depth=adaptive_far
    )  # Returns [H, W, 3]

    return {
        "render": final_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
```

#### Phase 4: Build System Updates (30 mins)

**File**: `submodules/diff-gaussian-rasterization/setup.py`

```python
sources=[
    # ... existing files
    "cuda_rasterizer/forward_neural_intervals.cu",
    "cuda_rasterizer/composite_neural.cu",
]
```

#### Phase 5: Training Integration (1 hour)

**File**: `train.py`

```python
# Add arguments
parser.add_argument("--use_neural_intervals", action="store_true")
parser.add_argument("--feature_dim", type=int, default=32)

# Initialize MLP
if opt.use_neural_intervals:
    from utils.neural_mlp import create_interval_mlp
    mlp = create_interval_mlp(
        feature_dim=opt.feature_dim,
        hidden_dim=64,
        n_hidden_layers=2
    ).cuda()
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

# In training loop
if opt.use_neural_intervals:
    from gaussian_renderer.neural_render import render_neural_intervals
    render_pkg = render_neural_intervals(
        viewpoint_cam, gaussians, mlp, pipe, background, estimator
    )
    # ... compute loss ...
    optimizer.step()      # Gaussian parameters
    optimizer_mlp.step()  # MLP parameters
else:
    # Standard rendering
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
```

#### Phase 6: Testing (2-3 hours)

**Test Script**: `test_neural_rendering.py`

```python
# 1. Test feature aggregation
print("Testing feature aggregation...")
features = rasterizer.forward_neural_intervals(...)
assert features.shape == (H, W, num_intervals, feature_dim)

# 2. Test MLP evaluation
print("Testing MLP...")
decoded = mlp(features.view(-1, feature_dim))
assert decoded.shape == (H*W*num_intervals, 4)

# 3. Test compositing
print("Testing compositing...")
final_image = rasterizer.composite_neural(...)
assert final_image.shape == (H, W, 3)

# 4. Test full pipeline
print("Testing full pipeline...")
output = render_neural_intervals(...)
assert "render" in output
```

---

## Quick Start to Resume

### Immediate Next Steps:

1. **Fix binningState bug** (5 minutes):
```bash
cd /home/nilkel/Projects/gaussian-splatting/submodules/diff-gaussian-rasterization
# Edit rasterize_points.cu line 604, add:
# BinningState binningState = BinningState::fromChunk(binning_ptr, P);
```

2. **Update ext.cpp** (10 minutes):
```bash
# Add function bindings at end of PYBIND11_MODULE
```

3. **Rebuild** (5 minutes):
```bash
cd submodules/diff-gaussian-rasterization
python setup.py build_ext --inplace
```

4. **Create Python bindings** (1-2 hours):
```bash
# Edit diff_gaussian_rasterization/__init__.py
# Add _RasterizeNeuralIntervals and _CompositeNeural classes
```

5. **Create rendering pipeline** (1 hour):
```bash
# Create gaussian_renderer/neural_render.py
```

6. **Test** (1 hour):
```bash
python test_neural_rendering.py
```

---

## File Checklist

### ✅ Complete
- [x] utils/neural_mlp.py
- [x] scene/gaussian_model.py
- [x] cuda_rasterizer/forward_neural_intervals.cu
- [x] cuda_rasterizer/composite_neural.cu
- [x] cuda_rasterizer/forward.h

### ⚠️  Needs Minor Fixes
- [ ] rasterize_points.cu (line 604 - add binningState extraction)

### 🚧 In Progress
- [ ] rasterize_points.cu (C++ wrappers - 95% done)

### ❌ Not Started
- [ ] ext.cpp (function bindings)
- [ ] __init__.py (Python autograd)
- [ ] neural_render.py (pipeline)
- [ ] setup.py (build config)
- [ ] train.py (integration)
- [ ] test_neural_rendering.py (testing)

---

## Estimated Time to Completion

- **Fix binningState**: 5 min
- **Complete C++ integration**: 30 min
- **Python bindings**: 2 hours
- **Rendering pipeline**: 1 hour
- **Build & test**: 2 hours
- **Training integration**: 1 hour

**Total**: ~6-7 hours of focused work

---

## Testing Strategy

### Phase 1: Forward-Only
1. Test feature aggregation kernel
2. Test MLP evaluation (Python)
3. Test compositing kernel
4. Test full pipeline
5. Verify output shapes and ranges

### Phase 2: With Gradients
1. Implement backward_neural_intervals.cu
2. Test gradient flow
3. Verify no NaNs/Infs
4. Train on toy scene (10 images)

### Phase 3: Full Training
1. Train on small scene (100 images)
2. Compare PSNR with baseline
3. Profile performance
4. Optimize bottlenecks

---

## Known Issues

### Issue 1: binningState not declared (rasterize_points.cu:610)
**Status**: Identified
**Fix**: Add `BinningState binningState = BinningState::fromChunk(binning_ptr, P);`
**Priority**: HIGH (blocks compilation)

### Issue 2: Backward pass not implemented
**Status**: Planned
**Fix**: Create backward_neural_intervals.cu
**Priority**: MEDIUM (can test forward-only first)

### Issue 3: Memory usage untested
**Status**: Unknown
**Fix**: Profile with 1080p rendering
**Priority**: LOW (optimize after it works)

---

**Last Updated**: Current session
**Next Session Goal**: Complete C++ integration and Python bindings, test forward pass
