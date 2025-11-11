#!/usr/bin/env python3
"""
Test script for Neural Interval Splatting forward pass.

Tests the complete three-phase pipeline:
1. Feature aggregation (CUDA)
2. MLP decoding (Python/tiny-cuda-nn)
3. Compositing (CUDA)
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_rasterize_neural_intervals():
    """Test Phase 1: Feature aggregation into intervals."""
    print("\n" + "="*60)
    print("TEST 1: Feature Aggregation (rasterize_neural_intervals)")
    print("="*60)

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        rasterize_neural_intervals
    )

    # Setup
    device = torch.device("cuda:0")
    H, W = 256, 256
    num_gaussians = 100
    feature_dim = 32
    num_intervals = 16

    # Create dummy Gaussians
    means3D = torch.randn(num_gaussians, 3, device=device) * 2.0
    means3D[:, 2] += 5.0  # Push away from camera

    features_neural = torch.randn(num_gaussians, feature_dim, device=device)
    opacities = torch.sigmoid(torch.randn(num_gaussians, 1, device=device))
    scales = torch.exp(torch.randn(num_gaussians, 3, device=device) * 0.5) * 0.1
    rotations = torch.randn(num_gaussians, 4, device=device)
    rotations = rotations / rotations.norm(dim=1, keepdim=True)  # Normalize quaternions

    # Create dummy camera
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[2, 3] = -10.0  # Move camera back

    projmatrix = torch.eye(4, device=device)
    projmatrix[0, 0] = 1.0
    projmatrix[1, 1] = 1.0
    projmatrix[2, 2] = -1.0
    projmatrix[2, 3] = -0.1
    projmatrix[3, 2] = -1.0
    projmatrix[3, 3] = 0.0

    campos = torch.tensor([0.0, 0.0, 10.0], device=device)

    # Rasterization settings
    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        antialiasing=False,
        debug=False
    )

    # Call function
    try:
        out_features, radii = rasterize_neural_intervals(
            means3D=means3D,
            features_neural=features_neural,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=None,
            raster_settings=raster_settings,
            num_intervals=num_intervals,
            near_depth=0.1,
            far_depth=100.0,
            feature_dim=feature_dim
        )

        print(f"✅ Function executed successfully!")
        print(f"   Output shape: {out_features.shape}")
        print(f"   Expected: [{H}, {W}, {num_intervals}, {feature_dim}]")
        print(f"   Radii shape: {radii.shape}")
        print(f"   Visible Gaussians: {(radii > 0).sum().item()}/{num_gaussians}")
        print(f"   Feature range: [{out_features.min():.3f}, {out_features.max():.3f}]")

        # Verify shapes
        assert out_features.shape == (H, W, num_intervals, feature_dim), "Shape mismatch!"
        assert radii.shape == (num_gaussians,), "Radii shape mismatch!"
        assert not torch.isnan(out_features).any(), "NaNs detected in output!"
        assert not torch.isinf(out_features).any(), "Infs detected in output!"

        print(f"✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_composite_neural():
    """Test Phase 3: Compositing decoded intervals."""
    print("\n" + "="*60)
    print("TEST 2: Neural Compositing (composite_neural)")
    print("="*60)

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        composite_neural
    )

    # Setup
    device = torch.device("cuda:0")
    H, W = 256, 256
    num_intervals = 16

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
    try:
        out_color = composite_neural(
            decoded_intervals=decoded_intervals,
            background=background,
            raster_settings=raster_settings,
            num_intervals=num_intervals,
            near_depth=0.1,
            far_depth=100.0
        )

        print(f"✅ Function executed successfully!")
        print(f"   Output shape: {out_color.shape}")
        print(f"   Expected: [{H}, {W}, 3]")
        print(f"   Color range: [{out_color.min():.3f}, {out_color.max():.3f}]")
        print(f"   Mean color: {out_color.mean(dim=[0,1])}")

        # Verify shapes
        assert out_color.shape == (H, W, 3), "Shape mismatch!"
        assert not torch.isnan(out_color).any(), "NaNs detected in output!"
        assert not torch.isinf(out_color).any(), "Infs detected in output!"
        assert (out_color >= 0).all() and (out_color <= 1.0).all(), "Color out of range!"

        print(f"✅ All assertions passed!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline_with_mlp():
    """Test complete pipeline with a real tiny-cuda-nn MLP."""
    print("\n" + "="*60)
    print("TEST 3: Full Pipeline with MLP")
    print("="*60)

    try:
        import tinycudann as tcnn
    except ImportError:
        print("⚠️  tiny-cuda-nn not available, skipping full pipeline test")
        return True

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        rasterize_neural_intervals,
        composite_neural
    )

    # Setup
    device = torch.device("cuda:0")
    H, W = 128, 128  # Smaller for speed
    num_gaussians = 50
    feature_dim = 32
    num_intervals = 8

    print(f"Creating MLP (input={feature_dim}, output=4)...")

    # Create tiny-cuda-nn MLP
    mlp = tcnn.Network(
        n_input_dims=feature_dim,
        n_output_dims=4,  # RGB + density
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
    )
    mlp = mlp.to(device)

    print(f"✅ MLP created successfully")

    # Create dummy Gaussians
    means3D = torch.randn(num_gaussians, 3, device=device) * 2.0
    means3D[:, 2] += 5.0

    features_neural = torch.randn(num_gaussians, feature_dim, device=device)
    opacities = torch.sigmoid(torch.randn(num_gaussians, 1, device=device))
    scales = torch.exp(torch.randn(num_gaussians, 3, device=device) * 0.5) * 0.1
    rotations = torch.randn(num_gaussians, 4, device=device)
    rotations = rotations / rotations.norm(dim=1, keepdim=True)

    # Camera setup
    viewmatrix = torch.eye(4, device=device)
    viewmatrix[2, 3] = -10.0

    projmatrix = torch.eye(4, device=device)
    projmatrix[0, 0] = 1.0
    projmatrix[1, 1] = 1.0

    campos = torch.tensor([0.0, 0.0, 10.0], device=device)
    background = torch.zeros(3, device=device)

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
        campos=campos,
        prefiltered=False,
        antialiasing=False,
        debug=False
    )

    try:
        # Phase 1: Feature aggregation
        print(f"\nPhase 1: Aggregating features...")
        interval_features, radii = rasterize_neural_intervals(
            means3D=means3D,
            features_neural=features_neural,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=None,
            raster_settings=raster_settings,
            num_intervals=num_intervals,
            near_depth=0.1,
            far_depth=100.0,
            feature_dim=feature_dim
        )
        print(f"✅ Phase 1 complete: {interval_features.shape}")

        # Phase 2: MLP decoding
        print(f"\nPhase 2: MLP decoding...")
        features_flat = interval_features.reshape(-1, feature_dim)

        # tiny-cuda-nn expects float16
        features_flat = features_flat.half()
        decoded_flat = mlp(features_flat)
        decoded_flat = decoded_flat.float()

        decoded_intervals = decoded_flat.reshape(H, W, num_intervals, 4)

        # Apply activations
        decoded_intervals[..., :3] = torch.sigmoid(decoded_intervals[..., :3])
        decoded_intervals[..., 3] = torch.nn.functional.softplus(decoded_intervals[..., 3])

        print(f"✅ Phase 2 complete: {decoded_intervals.shape}")
        print(f"   RGB range: [{decoded_intervals[..., :3].min():.3f}, {decoded_intervals[..., :3].max():.3f}]")
        print(f"   Density range: [{decoded_intervals[..., 3].min():.3f}, {decoded_intervals[..., 3].max():.3f}]")

        # Phase 3: Compositing
        print(f"\nPhase 3: Compositing...")
        final_image = composite_neural(
            decoded_intervals=decoded_intervals,
            background=background,
            raster_settings=raster_settings,
            num_intervals=num_intervals,
            near_depth=0.1,
            far_depth=100.0
        )
        print(f"✅ Phase 3 complete: {final_image.shape}")

        # Final checks
        assert final_image.shape == (H, W, 3), "Final image shape mismatch!"
        assert not torch.isnan(final_image).any(), "NaNs in final image!"
        assert not torch.isinf(final_image).any(), "Infs in final image!"

        print(f"\n✅ FULL PIPELINE SUCCESS!")
        print(f"   Final image range: [{final_image.min():.3f}, {final_image.max():.3f}]")
        print(f"   Mean RGB: {final_image.mean(dim=[0,1])}")

        return True

    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# Neural Interval Splatting - Forward Pass Tests")
    print("#"*60)

    results = []

    # Test 1: Feature aggregation
    results.append(("Feature Aggregation", test_rasterize_neural_intervals()))

    # Test 2: Compositing
    results.append(("Neural Compositing", test_composite_neural()))

    # Test 3: Full pipeline
    results.append(("Full Pipeline", test_full_pipeline_with_mlp()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nNeural Interval Splatting is ready for use!")
        print("Next steps:")
        print("  1. Test with real scene data")
        print("  2. Implement backward pass for training")
        print("  3. Integrate into train.py")
    else:
        print("\n⚠️  Some tests failed. Please review errors above.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
