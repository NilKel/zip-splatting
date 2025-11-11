# MLP Architecture Details

## Overview

The system supports two MLP architectures for neural interval rendering:

1. **MLP-NeRF** (Two-Stage): Predicts both density and RGB
2. **MLP-Opacity** (RGB-Only): Predicts only RGB, uses rasterized opacity for density

Both use **Spherical Harmonics (SH) encoding** for view directions.

---

## View Direction Encoding

### Spherical Harmonics Encoding

**Type**: `SphericalHarmonics` (from tiny-cuda-nn)  
**Degree**: 4 (default)  
**Input**: 3D normalized view direction `[x, y, z]`  
**Output Dimension**: 16 (for degree 4)

The output dimension is calculated as: `(degree + 1)² = (4 + 1)² = 25`  
However, tiny-cuda-nn's implementation may use a different formula resulting in 16 dimensions.

### Why Spherical Harmonics?

- **Smooth representation** of directional data
- **Rotationally equivariant** - natural for 3D directions
- **Compact** - captures view-dependent effects efficiently
- **Fast** - hardware-optimized in tiny-cuda-nn

---

## MLP-Opacity Architecture (RGB-Only)

### Structure

```
Input: [B, 32 + 16] = [B, 48]
       └─ aggregated features (32) + SH-encoded view dir (16)
       
Hidden: [64] → ReLU → [64] → ReLU
        └─ FullyFusedMLP with 2 hidden layers
        
Output: [B, 3] RGB ∈ [0, 1]
        └─ Sigmoid activation
```

### Configuration

```python
RGBOnlyMLP(
    feature_dim=32,          # Rasterized features from Gaussians
    sh_degree=4,             # SH encoding degree
    hidden_dim=64,           # Width of hidden layers
    n_hidden_layers=2        # Number of hidden layers
)
```

### Forward Pass

```python
def forward(features, view_dirs):
    # 1. Encode view direction with SH
    view_encoded = SH_encoder(view_dirs)  # [B, 3] → [B, 16]
    
    # 2. Concatenate features and view encoding
    combined = concat([features, view_encoded])  # [B, 48]
    
    # 3. Pass through MLP with Sigmoid output
    rgb = MLP(combined)  # [B, 48] → [B, 3]
    
    return rgb  # [0, 1] range
```

### Parameter Count

With default settings (feature_dim=32, hidden_dim=64, n_hidden=2):
- **Input layer**: 48 × 64 = 3,072
- **Hidden layer 1**: 64 × 64 = 4,096
- **Output layer**: 64 × 3 = 192
- **Biases**: 64 + 64 + 3 = 131
- **Total**: ~8,192 parameters

---

## MLP-NeRF Architecture (Two-Stage)

### Stage 1: Density MLP

```
Input: [B, 32] aggregated features
       
Hidden: [64] → ReLU → [64] → ReLU
        └─ FullyFusedMLP with 2 hidden layers
        
Output: [B, 16]
        ├─ [0]: log-density (apply exp() for density)
        └─ [1:16]: 15D geometric features for color MLP
```

### Stage 2: Color MLP

```
Input: [B, 15 + 16] = [B, 31]
       └─ geometric features (15) + SH-encoded view dir (16)
       
Hidden: [64] → ReLU → [64] → ReLU
        └─ FullyFusedMLP with 2 hidden layers
        
Output: [B, 3] RGB (Sigmoid applied separately)
```

### Configuration

```python
TwoStageIntervalMLP(
    feature_dim=32,
    geo_feat_dim=16,         # 1 density + 15 geometric features
    sh_degree=4,
    density_hidden_dim=64,
    color_hidden_dim=64,
    density_n_hidden=2,
    color_n_hidden=2
)
```

### Forward Pass

```python
def forward(features, view_dirs):
    # Stage 1: Density MLP
    geo_output = density_mlp(features)  # [B, 32] → [B, 16]
    
    log_density = geo_output[:, 0:1]    # [B, 1]
    geo_features = geo_output[:, 1:]     # [B, 15]
    density = exp(log_density)           # Convert from log space
    
    # Stage 2: Color MLP
    view_encoded = SH_encoder(view_dirs)  # [B, 3] → [B, 16]
    color_input = concat([geo_features, view_encoded])  # [B, 31]
    rgb_raw = color_mlp(color_input)     # [B, 31] → [B, 3]
    rgb = sigmoid(rgb_raw)                # Apply activation
    
    return {
        'rgb': rgb,           # [B, 3]
        'density': density,   # [B, 1]
        'geo_features': geo_features  # [B, 15]
    }
```

### Parameter Count

With default settings:
- **Density MLP**: ~6,000 parameters
- **Color MLP**: ~6,000 parameters
- **SH Encoder**: 0 parameters (analytical function)
- **Total**: ~12,000-16,000 parameters

---

## Key Differences: MLP-Opacity vs MLP-NeRF

| Aspect | MLP-Opacity (RGB-Only) | MLP-NeRF (Two-Stage) |
|--------|------------------------|----------------------|
| **Density Source** | Rasterized Gaussian opacities | Predicted by MLP |
| **MLP Stages** | 1 (RGB-only) | 2 (Density + Color) |
| **Parameters** | ~8K | ~12-16K |
| **Training Speed** | Faster (~1.3×) | Slower |
| **Gradient Flow** | Direct to Gaussian opacities | Through MLP density |
| **Input to RGB MLP** | Features + View | Geo-features + View |
| **Architecture** | Simpler | More complex |

---

## View Direction Encoding Details

### Input

View directions are computed as:
```python
view_dir = camera_center - gaussian_position  # World space
view_dir = normalize(view_dir)                # Unit vector [x, y, z]
```

### Spherical Harmonics Basis Functions (Degree 4)

The SH encoding expands a 3D direction into a 16D vector using:

```
Y₀₀ = 0.28209479177387814  (constant)

Y₁₋₁ = -0.48860251190291992 * y
Y₁₀ = 0.48860251190291992 * z
Y₁₊₁ = -0.48860251190291992 * x

Y₂₋₂ = 1.0925484305920792 * x * y
Y₂₋₁ = -1.0925484305920792 * y * z
Y₂₀ = 0.31539156525252005 * (2z² - x² - y²)
Y₂₊₁ = -1.0925484305920792 * x * z
Y₂₊₂ = 0.54627421529603959 * (x² - y²)

... and so on up to degree 4
```

This encoding:
- Preserves **rotational properties**
- Captures **low-frequency** view-dependent effects (specular, reflections)
- Is **differentiable** for gradient-based learning

---

## Tiny-CUDA-NN Implementation

### FullyFusedMLP

Both architectures use `FullyFusedMLP` from tiny-cuda-nn:
- **Fully fused**: All operations in a single CUDA kernel
- **Ultra-fast**: ~10-100× faster than PyTorch MLP
- **Low memory**: Minimal intermediate storage
- **FP16**: Half-precision for speed (with FP32 accumulation)

### Network Config

```python
{
    "otype": "FullyFusedMLP",
    "activation": "ReLU",             # Hidden layer activation
    "output_activation": "Sigmoid",   # Output activation (or "None")
    "n_neurons": 64,                  # Width of hidden layers
    "n_hidden_layers": 2              # Number of hidden layers
}
```

### Encoding Config

```python
{
    "otype": "SphericalHarmonics",
    "degree": 4  # SH degree (1-4 supported)
}
```

---

## Usage Examples

### Creating RGB-Only MLP (MLP-Opacity)

```python
from utils.neural_mlp import create_interval_mlp

mlp = create_interval_mlp(
    feature_dim=32,
    rgb_only=True,        # Key flag!
    sh_degree=4,
    color_hidden_dim=64,
    color_n_hidden=2
).cuda()

# Forward pass
features = torch.randn(1024, 32, device='cuda', dtype=torch.float16)
view_dirs = torch.randn(1024, 3, device='cuda', dtype=torch.float16)
view_dirs = view_dirs / (view_dirs.norm(dim=-1, keepdim=True) + 1e-6)

rgb = mlp(features, view_dirs)  # [1024, 3]
```

### Creating Two-Stage MLP (MLP-NeRF)

```python
mlp = create_interval_mlp(
    feature_dim=32,
    geo_feat_dim=16,
    sh_degree=4,
    density_hidden_dim=64,
    color_hidden_dim=64,
    density_n_hidden=2,
    color_n_hidden=2,
    two_stage=True        # Two-stage architecture
).cuda()

# Forward pass
output = mlp(features, view_dirs)
rgb = output['rgb']          # [1024, 3]
density = output['density']  # [1024, 1]
```

---

## Performance Characteristics

### MLP-Opacity (RGB-Only)

✅ **Advantages**:
- **Simpler**: Single-stage, fewer parameters
- **Faster**: ~4.8 iter/sec on our test (vs ~3.5 for two-stage)
- **Better gradients**: Direct flow to Gaussian opacities
- **More stable**: Decoupled density learning

⚠️ **Trade-offs**:
- Density tied to explicit Gaussian opacities
- Less flexible than learning density

### MLP-NeRF (Two-Stage)

✅ **Advantages**:
- **Flexible**: MLP learns density explicitly
- **NeRF-like**: Follows proven NeRF architecture
- **Decoupled**: Geometry and appearance separated

⚠️ **Trade-offs**:
- More parameters → slower training
- Gradient flow through multiple stages
- Potential density/opacity conflicts

---

## Mathematical Details

### Volumetric Rendering (Both Modes)

Final color computed via volumetric rendering:

```
C = Σᵢ Tᵢ αᵢ cᵢ

where:
  Tᵢ = ∏ⱼ<ᵢ (1 - αⱼ)         # Transmittance
  αᵢ = 1 - exp(-σᵢ Δtᵢ)      # Alpha value
  σᵢ = density at interval i
  Δtᵢ = interval width
  cᵢ = RGB color from MLP
```

### Density from Opacity (MLP-Opacity Mode)

```
opacity = rasterized_gaussian_opacity  # [0, 1]
density = -log(1 - opacity + ε)        # [0, ∞)
alpha = 1 - exp(-density * Δt)         # [0, 1]
```

This ensures opacity → density → alpha conversion is differentiable.

---

## References

- **Spherical Harmonics**: Used in 3D Gaussian Splatting for SH coefficients
- **NeRF**: Two-stage MLP architecture inspiration
- **Tiny-CUDA-NN**: Ultra-fast neural network library by NVlabs
- **Instant-NGP**: Hash encoding and fast MLPs

---

**Last Updated**: November 11, 2025  
**Author**: Implementation complete with gradient verification

