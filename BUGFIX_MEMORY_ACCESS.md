# Bug Fix: CUDA Illegal Memory Access Error

## Issue
The training was failing during the backward pass with:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

## Root Cause
The CUDA kernels were accessing Gaussian feature arrays and gradient arrays without sufficient bounds checking. Even though there were some boundary checks in place, atomic operations on gradient arrays could still access out-of-bounds memory locations if `gaussian_id` indices were invalid or at boundary conditions.

## Files Modified

### 1. `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward_neural_intervals.cu`

Added additional safety checks before all atomic operations:

- **Line 152-153**: Added bounds check before accessing `dL_dfeatures` array
  ```cuda
  if (gaussian_id * FEATURE_DIM + FEATURE_DIM > P * FEATURE_DIM)
      continue;
  ```

- **Line 179**: Added bounds check before `atomicAdd` to `dL_dopacity`
  ```cuda
  if (gaussian_id < P)
      atomicAdd(&dL_dopacity[gaussian_id], dL_dalpha * G);
  ```

- **Lines 204-207**: Added bounds check before `atomicAdd` to `dL_dmean2D`
  ```cuda
  if (gaussian_id * 2 + 1 < P * 2) {
      atomicAdd(&dL_dmean2D[gaussian_id * 2 + 0], dL_dmean_x);
      atomicAdd(&dL_dmean2D[gaussian_id * 2 + 1], dL_dmean_y);
  }
  ```

- **Lines 213-217**: Added bounds check before `atomicAdd` to `dL_dconic2D`
  ```cuda
  if (gaussian_id * 3 + 2 < P * 3) {
      atomicAdd(&dL_dconic2D[gaussian_id * 3 + 0], -0.5f * dL_dG * d.x * d.x);
      atomicAdd(&dL_dconic2D[gaussian_id * 3 + 1], -dL_dG * d.x * d.y);
      atomicAdd(&dL_dconic2D[gaussian_id * 3 + 2], -0.5f * dL_dG * d.y * d.y);
  }
  ```

### 2. `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward_neural_intervals.cu`

Added safety checks in the forward pass:

- **Lines 159-166**: Added bounds check before reading from `features_neural` array
  ```cuda
  if (gaussian_id * FEATURE_DIM + FEATURE_DIM <= P * FEATURE_DIM)
  {
      for (int ch = 0; ch < FEATURE_DIM; ch++)
      {
          interval_features[gaussian_interval][ch] += features_neural[gaussian_id * FEATURE_DIM + ch] * alpha;
      }
      interval_weight[gaussian_interval] += alpha;
  }
  ```

- **Lines 188-191**: Added bounds checks before writing output arrays
  ```cuda
  if (out_base < 0 || out_base + FEATURE_DIM > W * H * num_intervals * FEATURE_DIM)
      continue;
  if (pix_id * num_intervals + iv >= W * H * num_intervals)
      continue;
  ```

## Rebuild Command

```bash
cd /home/nilkel/Projects/gaussian-splatting/submodules/diff-gaussian-rasterization && \
/home/nilkel/miniconda3/envs/gaussian_splatting_py310/bin/python setup.py build_ext --inplace
```

## Testing

To test with device-side assertions enabled (for more detailed error messages):

```bash
TORCH_USE_CUDA_DSA=1 python train.py -s <dataset_path> [other_args...]
```

## Status

✅ Code modified with boundary checks
✅ Rebuild successful (no compilation errors)
⏳ Needs testing to verify the fix resolves the memory access error

## Next Steps

1. Run training with the fixed code
2. If the error persists, enable `TORCH_USE_CUDA_DSA=1` to get exact line numbers
3. Monitor for any performance impact from the additional bounds checks (should be minimal)


