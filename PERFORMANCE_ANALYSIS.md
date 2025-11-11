# Neural Interval Splatting Performance Analysis

**Date**: 2025-11-11
**Status**: Training is slow - needs optimization

## Current Performance Characteristics

### Computational Load Per Frame

- **Image resolution**: 800×800 pixels
- **Number of intervals**: 16
- **Total MLP evaluations**: 10,240,000 (10.24M samples)
- **Batch size**: 262,144 (0.26M per batch)
- **Number of batches**: 40 batches per frame

### Memory Footprint

- **Interval features [H,W,N,F]**: 1310.7 MB (800×800×16×32 float32)
- **Decoded intervals [H,W,N,4]**: 163.8 MB (800×800×16×4 float32)
- **Total per-frame memory**: ~1.5 GB

### Comparison to Standard NeRF

- NeRF samples per image: ~40.96M (640K rays × 64 samples/ray)
- Neural intervals: 10.24M samples
- **Ratio**: 0.25× fewer samples than NeRF (good!)

However, unlike NeRF which only evaluates rays with actual content, we evaluate **every pixel × every interval**, even if intervals are empty.

## Identified Bottlenecks

### 1. Dense Evaluation of All Intervals

**Current behavior**: Every pixel evaluates all 16 intervals, even if:
- The pixel has no Gaussian contributions
- Specific intervals are empty (no Gaussians in that depth range)
- Transmittance has dropped to near-zero (ray terminated early)

**Evidence from CUDA kernel** (forward_neural_intervals.cu:192-223):
```cuda
// Write all intervals after processing all Gaussians
if (inside) {
    for (int iv = 0; iv < num_intervals; iv++) {
        // ... write features even if interval_weight[iv] == 0
        if (interval_weight[iv] > 0.0f) {
            // Normalize and write features
        } else {
            // Empty interval - write zeros
            for (int ch = 0; ch < FEATURE_DIM; ch++) {
                out_features[out_base + ch] = 0.0f;
            }
        }
    }
}
```

**Impact**: Wasting MLP evaluations on empty intervals that will produce near-zero RGB/density.

### 2. Batching Overhead

**Current implementation** (neural_render.py:220-238):
- 40 batches of 262K samples each
- Each batch requires:
  - Slicing tensors (features_batch, view_dirs_batch)
  - dtype conversion (float32 → float16)
  - Contiguous memory allocation
  - MLP forward pass
  - dtype conversion back (float16 → float32)
  - Write to output buffer

**Impact**: Iteration overhead adds up across 40 batches per frame.

### 3. Two-Stage MLP Complexity

**Architecture**:
- Stage 1: Density MLP (32D → 64 → 64 → 16D)
- Stage 2: SH encoding (3D → 16D)
- Stage 3: Color MLP (31D → 64 → 64 → 3D)

**Total forward pass per sample**:
- 1× Density MLP evaluation
- 1× Spherical harmonics encoding (analytical, cheap)
- 1× Color MLP evaluation

**Comparison to single-stage**:
- Single-stage: 1× MLP evaluation (32D → 64 → 64 → 4D)
- Two-stage: 2× MLP evaluations + encoding

**Expected slowdown**: ~1.5-2× slower than single-stage (confirmed by user feedback).

### 4. View Direction Computation

**Current implementation** (neural_render.py:186-203):
- Computes same view direction for **all pixels** (simplified approach)
- Expands to [H, W, 3], then to [H, W, N, 3]
- Results in redundant encoding work (same view direction encoded 10.24M times)

**Note**: This is intentional for simplicity, but could be optimized.

## Optimization Strategies

### Priority 1: Sparse Interval Sampling

**Goal**: Skip MLP evaluation for empty intervals.

**Implementation**: Modify neural_render.py to:
1. Check `interval_weight` (already available from CUDA kernel)
2. Only evaluate MLP for intervals with `weight > threshold`
3. Set RGB/density to zeros for empty intervals

**Expected speedup**: 2-5× (depends on scene sparsity)

**Code changes needed**:
```python
# After rasterize_neural_intervals, check which intervals are occupied
interval_weights_data = interval_weights.reshape(H, W, num_intervals)  # [H, W, N]
valid_mask = interval_weights_data.reshape(-1) > 0.001  # [H*W*N]

# Only evaluate MLP for valid intervals
valid_features = features_flat[valid_mask]  # [M, F] where M <= H*W*N
valid_view_dirs = view_dirs_flat[valid_mask]  # [M, 3]

# Evaluate MLP on sparse set
mlp_output = mlp(valid_features, valid_view_dirs)

# Scatter back to dense output
rgb_flat[valid_mask] = mlp_output['rgb']
density_flat[valid_mask] = mlp_output['density']
# Remaining entries stay at zero (default initialization)
```

### Priority 2: Reduce Number of Intervals

**Current**: 16 intervals
**Proposed**: 8 or 12 intervals

**Rationale**: Gaussian splatting scenes typically have lower depth complexity than NeRF scenes. 16 intervals may be overkill.

**Expected speedup**: 1.33× (12 intervals) or 2× (8 intervals)

**Trade-off**: Slight quality loss if depth variation is high.

**Code change**: One-line change in render call:
```python
num_intervals=8  # or 12
```

### Priority 3: Reduce Feature Dimension

**Current**: 32D neural features
**Proposed**: 16D neural features

**Rationale**:
- Instant-NGP uses 32D for hash-encoded positions
- We're using pre-computed Gaussian features (may not need 32D)
- Smaller features = faster MLP

**Expected speedup**: ~1.2-1.3×

**Code changes needed**:
- Modify GaussianModel feature initialization
- Update MLP input dimension
- Re-train from scratch

### Priority 4: Increase Batch Size

**Current**: 262K samples per batch (2^18)
**Proposed**: 524K (2^19) or 1M (2^20)

**Rationale**: Larger batches = fewer iterations = less overhead.

**Expected speedup**: 1.2-1.5×

**Risk**: May hit OOM again (need to test incrementally).

**Code change**:
```python
batch_size = 2**19  # or 2**20
```

### Priority 5: Per-Pixel View Directions (Future)

**Current**: Same view direction for all pixels
**Proposed**: Compute actual ray direction per pixel

**Rationale**:
- More accurate view-dependent effects
- But: Requires encoding 10.24M unique directions (expensive)

**Expected impact**: May actually slow down, but improves quality.

**Recommendation**: Keep current simplified approach for now.

### Priority 6: Fused MLP Evaluation (Advanced)

**Current**: Separate batches with dtype conversions
**Proposed**: Custom CUDA kernel that fuses interval aggregation + MLP

**Rationale**: Eliminate Python/CUDA transfer overhead.

**Expected speedup**: 2-3×

**Effort**: High (requires custom CUDA kernel for tiny-cuda-nn integration).

## Recommended Action Plan

### Phase 1: Quick Wins (Implement First)

1. **Reduce intervals to 8** (1-line change, 2× speedup)
2. **Increase batch size to 2^19** (1-line change, 1.2× speedup if no OOM)

**Combined expected speedup**: ~2.4×

### Phase 2: Sparse Sampling (Moderate Effort)

3. **Implement sparse interval MLP evaluation** (moderate code change, 2-5× speedup on top of Phase 1)

**Combined expected speedup**: ~5-12× total

### Phase 3: Architecture Tuning (Requires Re-Training)

4. **Reduce feature dimension to 16D** (small code change + re-train)

**Combined expected speedup**: ~6-15× total

### Phase 4: Advanced Optimization (Future Work)

5. **Fused CUDA kernel** (high effort, research project)

## Testing Protocol

After each optimization:
1. Run training for 100 iterations
2. Measure time per iteration
3. Render test image and compute PSNR
4. Compare quality vs. baseline

## Baseline Metrics (To Be Measured)

- [ ] Current time per iteration: ___ seconds
- [ ] Current PSNR at iteration 7000: ___ dB
- [ ] Memory usage: ___ GB

## References

- Instant-NGP: Uses 64 samples/ray, hash encoding reduces MLP input to 32D
- NeRF: Uses 64-192 samples/ray, evaluates sparsely along rays
- Plenoxels: Uses 256³ voxel grid, only evaluates non-empty voxels

## Notes

- The two-stage MLP is **fundamentally slower** than single-stage (2× MLP evaluations)
- This is expected and acceptable for better view-dependent rendering
- Main optimization opportunity is **sparsity** - don't evaluate empty intervals
