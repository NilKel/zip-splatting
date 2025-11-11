# Instant-NGP Architecture Insights

## Key Discovery: JIT-Compiled Fused Kernels

Instant-NGP achieves extreme performance by **fusing the MLP evaluation directly into the CUDA rendering kernel** using Runtime Compilation (RTC).

---

## How It Works

### 1. **Network Device Function Generation**

The network is compiled into a CUDA device function that can be called from other kernels:

```cpp
// Line 2287 in testbed_nerf.cu
fmt::arg("MODEL_BODY", nerf_network->generate_device_function("eval_nerf"))
```

This generates something like:

```cpp
__device__ vec<4> eval_nerf(vec<N_INPUT> input, const network_precision_t* params) {
    // Inline MLP forward pass using tiny-cuda-nn's device functions
    // All matrix multiplications, activations, etc. are inlined
    return output;
}
```

### 2. **JIT Kernel Compilation**

The device function is embedded into the rendering kernel at runtime:

```cpp
// Lines 2280-2293
device.set_fused_render_kernel(
    std::make_unique<CudaRtcKernel>(
        "render_nerf",
        fmt::format(
            "{MODEL_BODY}\n"  // <- MLP device function goes here
            "using GRID_T = {GRID_T};\n"
            "static constexpr uint32_t N_EXTRA_DIMS = {N_EXTRA_DIMS};\n"
            "#include <neural-graphics-primitives/fused_kernels/render_nerf.cuh>\n",
            ...
        ),
        ...
    )
);
```

### 3. **Fused Rendering Kernel**

File: `/home/nilkel/Projects/instant-ngp/include/neural-graphics-primitives/fused_kernels/render_nerf.cuh`

**Key sections:**

#### Network Evaluation (Line 139):
```cpp
auto nerf_out = eval_nerf(nerf_in, params);
```

This is the **inline MLP call** - no separate kernel launch, no data transfer!

#### Volumetric Accumulation (Lines 172-176):
```cpp
// Standard mode: 4D output [R, G, B, density]
alpha = 1.f - __expf(-network_to_density(float(nerf_out[3]), density_activation) * dt);
weight = alpha * (1.0f - color.a);
rgb = network_to_rgb_vec(vec3(nerf_out[0], nerf_out[1], nerf_out[2]), rgb_activation);
color += vec4(rgb * weight, weight);
```

#### Warp Synchronization (Lines 134-136):
```cpp
// Implicit sync is fine here, because the following nerf model wants synchronized warps anyway
if (__all_sync(0xFFFFFFFF, !alive)) {
    break;
}
```

**Critical insight**: All threads in a warp must execute the MLP together for coherence!

---

## Performance Tricks

### 1. **Warp-Coherent Execution**

All 32 threads in a warp evaluate the MLP simultaneously, even if some rays are "dead":

```cpp
// Line 139: ALL threads execute this
auto nerf_out = eval_nerf(nerf_in, params);

// Lines 142-145: Only alive threads continue with accumulation
if (!alive) {
    continue;  // Skip accumulation but MLP was still evaluated
}
```

**Why?** tiny-cuda-nn's MLPs require warp-synchronous execution for maximum throughput.

### 2. **Occupancy Grid Acceleration**

Before evaluating the MLP, skip empty space:

```cpp
// Lines 113-115
t = if_unoccupied_advance_to_next_occupied_voxel(
    t, cone_angle, ray, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local
);
```

Only evaluate the network at occupied voxels!

### 3. **No Data Transfer**

Everything happens on GPU:
- Ray marching
- MLP evaluation
- Compositing
- All in **one kernel launch**

### 4. **Half Precision**

```cpp
using network_precision_t = __half;  // or __nv_bfloat16
```

The MLP runs in FP16 for 2x speed (supported by tiny-cuda-nn).

---

## Architecture Comparison

### Instant-NGP (Fused):
```
GPU Kernel:
  for each ray:
    for each sample:
      pos = march(ray, t)
      features = eval_network_inline(pos)  // <- Inlined!
      color += blend(features, alpha)
```

### Our Current Plan (Two-Pass):
```
Pass 1 - Feature Aggregation (GPU):
  for each pixel:
    for each Gaussian:
      features_aggregated[interval] += gaussian.features * alpha

Pass 2 - MLP Evaluation (CPU + GPU):
  features_flat = features_aggregated.view(-1, F)  // Reshape
  decoded = mlp(features_flat)                      // Python call
  decoded = decoded.view(H, W, N, 4)                // Reshape

Pass 3 - Compositing (GPU):
  for each pixel:
    for each interval:
      color += blend(decoded[interval])
```

**Problem**: Two Python→CUDA round trips, memory transfers, kernel launches.

---

## What We Should Adopt

### Option A: JIT Fusion (Best Performance, Complex)

**Copy instant-ngp's approach exactly:**

1. Generate MLP device function from tiny-cuda-nn:
   ```cpp
   std::string mlp_code = mlp_network->generate_device_function("eval_mlp");
   ```

2. Use NVRTC to compile fused kernel:
   ```cpp
   #include <nvrtc.h>

   std::string kernel_code = fmt::format(
       "{}\n"  // MLP device function
       "#include \"forward_neural_intervals.cuh\"\n",
       mlp_code
   );

   nvrtcProgram prog;
   nvrtcCreateProgram(&prog, kernel_code.c_str(), "fused_kernel.cu", ...);
   nvrtcCompileProgram(prog, ...);
   ```

3. Call fused kernel:
   ```cpp
   __global__ void render_neural_intervals_fused(...) {
       // Feature aggregation
       for (int i = 0; i < num_gaussians; i++) {
           // ... accumulate features ...
       }

       // Inline MLP evaluation!
       auto decoded = eval_mlp(interval_features, mlp_params);

       // Compositing
       color += decoded.rgb * alpha;
   }
   ```

**Pros:**
- Maximum performance (no data transfer)
- Single kernel launch
- Warp-coherent MLP evaluation

**Cons:**
- Requires NVRTC integration
- Complex build system changes
- Harder to debug
- MLP parameters need manual management

### Option B: Three-Kernel (Simpler, Still Fast)

Keep our three-pass approach but optimize:

1. **Pass 1**: Feature aggregation → device memory
2. **Pass 2**: MLP batch evaluation (single kernel)
3. **Pass 3**: Compositing → framebuffer

**Optimization**: Use CUDA streams to overlap:
```cpp
cudaStream_t stream1, stream2;
// Launch aggregation on stream1
aggregate<<<blocks, threads, 0, stream1>>>(...);
// Launch MLP on stream2 (can start as soon as first batch ready)
mlp_network->inference_async(stream2, ...);
// Launch compositing on stream1 (depends on MLP output)
composite<<<blocks, threads, 0, stream1>>>(...);
```

**Pros:**
- Much simpler than JIT
- Still leverages tiny-cuda-nn
- Easier to debug
- Gradual optimization path

**Cons:**
- 2-3x slower than fused (but still fast)
- Extra memory for intermediate buffers

---

## Recommended Approach for Us

### Phase 1: Three-Kernel (Implement Now)

Start with the three-kernel approach:

1. `forward_neural_intervals.cu` - Aggregate features
2. Python: `mlp(features_flat)` - Evaluate MLP
3. `composite_neural.cu` - Final blend

**Why?** Get it working first, profile later.

### Phase 2: Optimize (If Needed)

If performance isn't good enough:

1. **Move MLP to CUDA**: Use tiny-cuda-nn's C++ API directly
   ```cpp
   tcnn::Network<T>* network;
   network->inference(features, output, stream);
   ```

2. **Stream optimization**: Overlap kernels

3. **Memory reduction**: Use FP16 for features

### Phase 3: JIT Fusion (Advanced)

Only if we need instant-ngp-level performance:

1. Add NVRTC dependency
2. Implement JIT compilation infrastructure
3. Fuse all three passes into one kernel

---

## Key Code Patterns to Copy

### 1. Warp Synchronization

```cpp
// Ensure all threads in warp execute MLP together
if (__all_sync(0xFFFFFFFF, !alive)) {
    break;
}

auto nerf_out = eval_nerf(nerf_in, params);

// Now individual threads can diverge
if (!alive) {
    continue;
}
```

### 2. Network Output Processing

```cpp
// Apply activation functions
float density = network_to_density(float(nerf_out[3]), density_activation);
vec3 rgb = network_to_rgb_vec(vec3(nerf_out[0], nerf_out[1], nerf_out[2]), rgb_activation);

// Volumetric rendering
alpha = 1.f - __expf(-density * dt);
weight = alpha * (1.0f - color.a);  // Transmittance
color += vec4(rgb * weight, weight);
```

### 3. Early Termination

```cpp
if (color.a > (1.0f - min_transmittance)) {
    color /= color.a;
    alive = false;  // Stop marching this ray
}
```

### 4. Parameter Passing

```cpp
// MLP parameters are passed as raw pointer
const network_precision_t* __restrict__ params
```

tiny-cuda-nn manages the parameter buffer, we just pass the pointer.

---

## Instant-NGP File Structure

```
instant-ngp/
├── src/
│   ├── testbed_nerf.cu          # Main NeRF logic, JIT compilation
│   └── ...
├── include/neural-graphics-primitives/
│   ├── fused_kernels/
│   │   └── render_nerf.cuh      # ⭐ Main rendering kernel
│   ├── nerf_device.cuh
│   └── ...
└── dependencies/
    └── tiny-cuda-nn/            # Same library we're using
```

**Key file**: `render_nerf.cuh` lines 21-215 - Complete fused rendering kernel

---

## Actionable Next Steps

### For Our Implementation:

1. **Stick with three-kernel approach initially**
   - Get it working end-to-end
   - Measure performance baseline

2. **Copy these patterns from instant-ngp**:
   - Warp synchronization before MLP eval
   - Network output activation functions
   - Early termination on high transmittance
   - FP16 for MLP inputs/outputs

3. **Use tiny-cuda-nn C++ API** (not Python):
   ```cpp
   #include <tiny-cuda-nn/network.h>

   auto network = tcnn::create_network<network_precision_t>(config);
   network->inference(input, output, stream);
   ```

4. **Consider JIT fusion later** if needed:
   - Profile first - three-kernel might be fast enough
   - JIT adds complexity but huge speedup
   - Can copy instant-ngp's NVRTC code directly

---

## Performance Expectations

### Instant-NGP (Fused):
- **1920x1080**: ~30-60 FPS
- **Single kernel**: ~2-5ms

### Our Three-Kernel (Estimated):
- **1920x1080**: ~10-20 FPS
- **Three kernels + transfers**: ~10-20ms

### Baseline 3DGS:
- **1920x1080**: ~100+ FPS
- **Single kernel**: ~1-2ms

**Note**: Our method trades speed for quality via neural features. Expect 3-10x slowdown vs baseline, but much better quality.

---

## Summary

**Instant-NGP's Secret Sauce:**
1. JIT-compile MLP into rendering kernel
2. Warp-synchronous MLP evaluation
3. No data transfers (all on GPU)
4. FP16 precision
5. Occupancy grid skips empty space

**What We Should Do:**
1. Start simple: Three kernels, Python glue
2. Use tiny-cuda-nn C++ API for speed
3. Add JIT fusion if profiling shows bottleneck
4. Copy warp sync and activation patterns

**Expected Outcome:**
- Initial: 10-20 FPS at 1080p (acceptable)
- Optimized: 30-50 FPS at 1080p (great)
- Fully fused: 50-100 FPS at 1080p (instant-ngp level)

---

**Files Referenced:**
- `/home/nilkel/Projects/instant-ngp/src/testbed_nerf.cu` (lines 2236-2350)
- `/home/nilkel/Projects/instant-ngp/include/neural-graphics-primitives/fused_kernels/render_nerf.cuh` (lines 21-215)
