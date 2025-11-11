# Neural Interval Splatting - Training Quick Start

## 🚀 Ready-to-Use Training Commands

### Option 1: Test on Synthetic Scene (Recommended First)

```bash
# Quick test - standard baseline training first
python train.py -s data/nerf_synthetic/lego -m output/lego_baseline --iterations 7000

# Then test neural intervals (requires MLP implementation in train.py)
# python train.py -s data/nerf_synthetic/lego -m output/lego_neural \
#   --use_neural_intervals --feature_dim 32 --num_intervals 16 \
#   --iterations 7000
```

### Option 2: Test on Your Own Data

```bash
# Standard baseline
python train.py -s /path/to/your/data -m output/your_scene_baseline

# Neural intervals (after integration)
# python train.py -s /path/to/your/data -m output/your_scene_neural \
#   --use_neural_intervals --feature_dim 32
```

### Option 3: Small Test Scene

```bash
# Create a tiny test scene first (10 images)
# python train.py -s data/test_scene -m output/test_neural \
#   --use_neural_intervals --iterations 1000 --test_iterations 500 1000
```

---

## ⚠️ IMPORTANT: Integration Required First

The neural interval rendering is **implemented and working**, but you need to integrate it into `train.py` first. Here's what's needed:

### Step 1: Add Arguments to train.py

Add these lines after the existing argument definitions (around line 493):

```python
# Neural Interval Splatting arguments
parser.add_argument('--use_neural_intervals', action='store_true', default=False,
                   help='Use neural interval splatting instead of standard rendering')
parser.add_argument('--feature_dim', type=int, default=32,
                   help='Neural feature dimension (16, 32, or 64)')
parser.add_argument('--num_intervals', type=int, default=16,
                   help='Number of depth intervals for neural rendering')
parser.add_argument('--mlp_hidden_dim', type=int, default=64,
                   help='MLP hidden layer size')
parser.add_argument('--mlp_num_layers', type=int, default=2,
                   help='Number of MLP hidden layers')
parser.add_argument('--mlp_lr', type=float, default=1e-3,
                   help='Learning rate for MLP')
```

### Step 2: Initialize MLP in training function

Add this after Gaussian model initialization (around line 50-60):

```python
# Initialize MLP for neural intervals
mlp = None
optimizer_mlp = None
if args.use_neural_intervals:
    try:
        import tinycudann as tcnn
        print(f"Creating neural MLP: {args.feature_dim} -> {args.mlp_hidden_dim} -> 4")
        mlp = tcnn.Network(
            n_input_dims=args.feature_dim,
            n_output_dims=4,  # RGB + density
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": args.mlp_hidden_dim,
                "n_hidden_layers": args.mlp_num_layers,
            }
        ).cuda()
        optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr)
        print(f"✅ Neural MLP initialized")
    except ImportError:
        print("❌ ERROR: tiny-cuda-nn not available. Install with:")
        print("   pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch")
        sys.exit(1)
```

### Step 3: Modify rendering in training loop

Replace standard `render()` call with conditional (around line 100-150):

```python
# Render
if args.use_neural_intervals:
    from gaussian_renderer.neural_render import render_neural_intervals
    render_pkg = render_neural_intervals(
        viewpoint_cam=viewpoint_cam,
        pc=gaussians,
        mlp=mlp,
        pipe=pipe,
        bg_color=bg_color,
        num_intervals=args.num_intervals,
        near_depth=0.1,
        far_depth=100.0,
        feature_dim=args.feature_dim
    )
else:
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)

image = render_pkg["render"]
```

### Step 4: Update optimizer step

After `loss.backward()`, add MLP optimizer step:

```python
# Standard optimizer step
optimizer.step()
optimizer.zero_grad(set_to_none=True)

# MLP optimizer step (if using neural intervals)
if args.use_neural_intervals and optimizer_mlp is not None:
    optimizer_mlp.step()
    optimizer_mlp.zero_grad(set_to_none=True)
```

### Step 5: Save/Load MLP checkpoints

Add MLP state to checkpoint saving (around checkpoint code):

```python
# Save checkpoint
torch.save({
    'gaussians': gaussians.capture(),
    'mlp': mlp.state_dict() if mlp is not None else None,
    'iteration': iteration,
}, checkpoint_path)

# Load checkpoint
checkpoint = torch.load(checkpoint_path)
gaussians.restore(checkpoint['gaussians'], opt)
if mlp is not None and checkpoint.get('mlp') is not None:
    mlp.load_state_dict(checkpoint['mlp'])
```

---

## 📋 Full Integration Template

Here's a minimal template for integrating neural intervals into your training loop:

```python
def training(args):
    # ... existing setup code ...

    # Initialize MLP if using neural intervals
    mlp = None
    optimizer_mlp = None
    if args.use_neural_intervals:
        import tinycudann as tcnn
        mlp = tcnn.Network(
            n_input_dims=args.feature_dim,
            n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": args.mlp_hidden_dim,
                "n_hidden_layers": args.mlp_num_layers,
            }
        ).cuda()
        optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=args.mlp_lr)

    # Training loop
    for iteration in range(1, args.iterations + 1):
        # ... existing code ...

        # Render
        if args.use_neural_intervals:
            from gaussian_renderer.neural_render import render_neural_intervals
            render_pkg = render_neural_intervals(
                viewpoint_cam, gaussians, mlp, pipe, bg_color,
                num_intervals=args.num_intervals,
                feature_dim=args.feature_dim
            )
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)

        image = render_pkg["render"]

        # ... compute loss ...

        loss.backward()

        # Optimizer steps
        optimizer.step()
        optimizer.zero_grad()

        if optimizer_mlp is not None:
            optimizer_mlp.step()
            optimizer_mlp.zero_grad()
```

---

## 🧪 Testing Without Full Integration

You can test the neural rendering **without modifying train.py** first:

```python
# test_neural_training.py - Minimal training test
import torch
from scene import Scene
from gaussian_renderer.neural_render import render_neural_intervals
from utils.loss_utils import l1_loss, ssim
import tinycudann as tcnn

# Load a trained baseline model
scene = Scene(args, gaussians)

# Create MLP
mlp = tcnn.Network(
    n_input_dims=32,
    n_output_dims=4,
    network_config={
        "otype": "FullyFusedMLP",
        "activation": "ReLU",
        "output_activation": "None",
        "n_neurons": 64,
        "n_hidden_layers": 2,
    }
).cuda()

optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=1e-3)

# Simple training loop
for iteration in range(1000):
    viewpoint = scene.getTrainCameras()[iteration % len(scene.getTrainCameras())]

    # Render
    render_pkg = render_neural_intervals(
        viewpoint, gaussians, mlp, pipe, background,
        num_intervals=16, feature_dim=32
    )

    # Loss
    gt_image = viewpoint.original_image.cuda()
    loss = l1_loss(render_pkg["render"], gt_image)

    # Backward
    loss.backward()
    optimizer_mlp.step()
    optimizer_mlp.zero_grad()

    if iteration % 100 == 0:
        print(f"Iter {iteration}: Loss {loss.item():.4f}")
```

---

## 📊 Expected Results

### Baseline 3DGS (for comparison)
```bash
python train.py -s data/lego -m output/lego_baseline
# Expected: ~30-35 PSNR after 7k iterations
```

### Neural Intervals (after integration)
```bash
python train.py -s data/lego -m output/lego_neural \
  --use_neural_intervals --feature_dim 32
# Expected: Similar or better PSNR (depends on MLP capacity)
```

### Hyperparameter Tuning
Try these variations:
- `--feature_dim 16` - Faster, lower quality
- `--feature_dim 64` - Slower, potentially higher quality
- `--num_intervals 8` - Fewer intervals, faster
- `--num_intervals 32` - More intervals, better quality
- `--mlp_hidden_dim 128` - Larger MLP, more capacity
- `--mlp_num_layers 3` - Deeper MLP

---

## 🔍 Debugging

If training fails:

1. **Check MLP is initialized**:
   ```python
   print(f"MLP: {mlp}")
   print(f"MLP params: {sum(p.numel() for p in mlp.parameters())}")
   ```

2. **Verify forward pass works**:
   ```python
   with torch.no_grad():
       test_render = render_neural_intervals(...)
       print(f"Render shape: {test_render['render'].shape}")
       print(f"Render range: [{test_render['render'].min():.3f}, {test_render['render'].max():.3f}]")
   ```

3. **Check for NaNs**:
   ```python
   if torch.isnan(loss):
       print("❌ NaN loss detected!")
       print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")
   ```

4. **Enable CUDA error checking**:
   ```bash
   CUDA_LAUNCH_BLOCKING=1 python train.py ...
   ```

---

## 📝 Next Steps After Integration

1. **Run baseline comparison**:
   ```bash
   # Baseline
   python train.py -s data/scene -m output/baseline
   # Neural
   python train.py -s data/scene -m output/neural --use_neural_intervals
   ```

2. **Compare metrics**:
   ```bash
   python metrics.py -m output/baseline
   python metrics.py -m output/neural
   ```

3. **Visualize results**:
   ```bash
   python render.py -m output/neural --use_neural_intervals
   ```

4. **Profile performance**:
   ```bash
   python -m torch.utils.bottleneck train.py ... --iterations 100
   ```

---

## 🎯 Training Commands Summary

### Minimal (for testing integration)
```bash
python train.py -s data/small_scene -m output/test \
  --use_neural_intervals --iterations 1000
```

### Standard (full training)
```bash
python train.py -s data/scene -m output/scene_neural \
  --use_neural_intervals \
  --feature_dim 32 \
  --num_intervals 16 \
  --mlp_hidden_dim 64 \
  --iterations 30000
```

### High Quality
```bash
python train.py -s data/scene -m output/scene_neural_hq \
  --use_neural_intervals \
  --feature_dim 64 \
  --num_intervals 32 \
  --mlp_hidden_dim 128 \
  --mlp_num_layers 3 \
  --iterations 30000
```

### Fast (for debugging)
```bash
python train.py -s data/scene -m output/scene_neural_fast \
  --use_neural_intervals \
  --feature_dim 16 \
  --num_intervals 8 \
  --mlp_hidden_dim 32 \
  --iterations 7000
```

---

## ✅ Prerequisites

Before training:
- ✅ Neural interval rendering implemented (DONE!)
- ✅ CUDA extension built and tested (DONE!)
- ⏳ train.py integration (TODO - see above)
- ⏳ tiny-cuda-nn installed (check with `pip list | grep tinycudann`)

Install tiny-cuda-nn if needed:
```bash
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

---

**Last Updated**: 2025-11-03 23:57 UTC
**Status**: Ready for training integration
**Tests**: All forward pass tests passing ✅
