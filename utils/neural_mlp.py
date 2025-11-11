"""
Neural MLP wrapper for interval-based feature decoding.

This module provides a wrapper around tiny-cuda-nn's FullyFusedMLP
for decoding aggregated neural features into RGB + density values.
"""

import torch
import torch.nn as nn
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("Warning: tiny-cuda-nn not available. Neural interval rendering will not work.")


class IntervalMLP(nn.Module):
    """
    MLP for decoding aggregated neural features into RGB + density.

    Uses tiny-cuda-nn's FullyFusedMLP for fast inference.

    Args:
        feature_dim: Input feature dimension (e.g., 32)
        hidden_dim: Hidden layer width (e.g., 64)
        n_hidden_layers: Number of hidden layers (e.g., 2)
        output_dim: Output dimension (4 for RGB + density)
    """

    def __init__(self, feature_dim=32, hidden_dim=64, n_hidden_layers=2, output_dim=4):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tiny-cuda-nn is required for IntervalMLP. Please install it.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.output_dim = output_dim

        # Create tiny-cuda-nn network
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
        }

        self.network = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=output_dim,
            network_config=network_config
        )

    def forward(self, features):
        """
        Forward pass through the MLP.

        Args:
            features: Input features [B, feature_dim]

        Returns:
            output: RGB + density [B, 4]
                - output[:, :3]: RGB colors (can be negative, will be activated later)
                - output[:, 3]: Density (will be processed with softplus/exp)
        """
        return self.network(features)

    def get_params(self):
        """Get network parameters for saving."""
        return self.network.parameters()

    def num_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.network.parameters())


class IntervalMLPWithActivation(IntervalMLP):
    """
    Extended version with output activations for RGB and density.

    This version applies sigmoid to RGB and softplus to density,
    which can be useful for stability during training.
    """

    def __init__(self, feature_dim=32, hidden_dim=64, n_hidden_layers=2):
        super().__init__(feature_dim, hidden_dim, n_hidden_layers, output_dim=4)

        self.rgb_activation = nn.Sigmoid()  # Clamp RGB to [0, 1]
        self.density_activation = nn.Softplus()  # Ensure positive density

    def forward(self, features):
        """
        Forward pass with activations.

        Args:
            features: Input features [B, feature_dim]

        Returns:
            output: Activated RGB + density [B, 4]
                - output[:, :3]: RGB colors in [0, 1]
                - output[:, 3]: Positive density
        """
        raw_output = self.network(features)

        rgb = self.rgb_activation(raw_output[:, :3])
        density = self.density_activation(raw_output[:, 3:4])

        return torch.cat([rgb, density], dim=-1)


def create_interval_mlp(feature_dim=32, hidden_dim=64, n_hidden_layers=2, use_activation=False):
    """
    Factory function to create an IntervalMLP.

    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden layer width
        n_hidden_layers: Number of hidden layers
        use_activation: Whether to use output activations

    Returns:
        IntervalMLP or IntervalMLPWithActivation instance
    """
    if use_activation:
        return IntervalMLPWithActivation(feature_dim, hidden_dim, n_hidden_layers)
    else:
        return IntervalMLP(feature_dim, hidden_dim, n_hidden_layers)


if __name__ == "__main__":
    # Test the MLP
    print("Testing IntervalMLP...")

    mlp = create_interval_mlp(feature_dim=32, hidden_dim=64, n_hidden_layers=2)
    mlp = mlp.cuda()

    print(f"Created MLP with {mlp.num_parameters()} parameters")

    # Test forward pass
    batch_size = 1024
    features = torch.randn(batch_size, 32, device='cuda')
    output = mlp(features)

    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"RGB range: [{output[:, :3].min().item():.4f}, {output[:, :3].max().item():.4f}]")
    print(f"Density range: [{output[:, 3].min().item():.4f}, {output[:, 3].max().item():.4f}]")

    # Test backward pass
    loss = output.sum()
    loss.backward()

    print("\nBackward pass successful!")
    print("✓ IntervalMLP is working correctly")
