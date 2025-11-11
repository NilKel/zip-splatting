"""
Test script to verify tiny-cuda-nn MLP functionality
"""
import torch
import tinycudann as tcnn

print("Testing tiny-cuda-nn MLP...")

# Create a simple MLP: 32 input dims -> 4 output dims (RGB + density)
config = {
    "otype": "FullyFusedMLP",
    "activation": "ReLU",
    "output_activation": "None",
    "n_neurons": 64,
    "n_hidden_layers": 2,
}

mlp = tcnn.Network(
    n_input_dims=32,
    n_output_dims=4,
    network_config=config
).cuda()

print(f"Created MLP: {mlp}")
print(f"Number of parameters: {sum(p.numel() for p in mlp.parameters())}")

# Test forward pass
batch_size = 1024  # Must be multiple of 128 for tiny-cuda-nn
input_features = torch.randn(batch_size, 32, device='cuda', requires_grad=True)

print(f"\nInput shape: {input_features.shape}")

output = mlp(input_features)
print(f"Output shape: {output.shape}")
print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

# Test backward pass
loss = output.sum()
loss.backward()

print(f"\nGradient on input: {input_features.grad is not None}")
print(f"Gradient shape: {input_features.grad.shape if input_features.grad is not None else 'None'}")
print(f"Gradient range: [{input_features.grad.min().item():.4f}, {input_features.grad.max().item():.4f}]")

# Test with different batch sizes
print("\nTesting different batch sizes:")
for bs in [128, 256, 512, 1024, 2048]:
    inp = torch.randn(bs, 32, device='cuda')
    out = mlp(inp)
    assert out.shape == (bs, 4), f"Expected shape ({bs}, 4), got {out.shape}"
    print(f"  Batch size {bs}: ✓")

print("\n✓ All tests passed!")
print("Tiny-cuda-nn is working correctly.")
