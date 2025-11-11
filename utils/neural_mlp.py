"""
Neural MLP wrapper for interval-based feature decoding.

Two-stage architecture (NeRF-style):
1. Density MLP: aggregated_features -> [density, geometric_features]
2. Color MLP: [geometric_features, view_direction] -> RGB

This separates view-independent geometry from view-dependent appearance.
"""

import torch
import torch.nn as nn
import math
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("Warning: tiny-cuda-nn not available. Neural interval rendering will not work.")


def get_spherical_harmonics_encoding_config(degree=4):
    """
    Get config for spherical harmonics encoding of view directions.

    Args:
        degree: SH degree (1-4 supported by tiny-cuda-nn)

    Returns:
        Dictionary with encoding config
    """
    return {
        "otype": "SphericalHarmonics",
        "degree": degree
    }


class DensityMLP(nn.Module):
    """
    Density MLP: Decodes aggregated features to density + geometric features.

    Architecture:
        Input: [B, feature_dim] aggregated features from intervals
        Hidden: [64, 64, ...] with ReLU
        Output: [B, 16] where:
            - output[:, 0]: log-density (apply exp() for actual density)
            - output[:, 1:16]: 15D geometric feature vector for color MLP

    Args:
        feature_dim: Input feature dimension (e.g., 32)
        hidden_dim: Hidden layer width (default: 64)
        n_hidden_layers: Number of hidden layers (default: 2)
        geo_feat_dim: Geometric feature output dimension (default: 16)
    """

    def __init__(self, feature_dim=32, hidden_dim=64, n_hidden_layers=2, geo_feat_dim=16):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tiny-cuda-nn is required for DensityMLP. Please install it.")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.geo_feat_dim = geo_feat_dim

        # Create tiny-cuda-nn network
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",  # No activation - density uses exp, features are raw
            "n_neurons": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
        }

        self.network = tcnn.Network(
            n_input_dims=feature_dim,
            n_output_dims=geo_feat_dim,  # density (1) + geometric features (15)
            network_config=network_config
        )

    def forward(self, features):
        """
        Forward pass through the density MLP.

        Args:
            features: Input features [B, feature_dim]

        Returns:
            output: [B, geo_feat_dim] where:
                - output[:, 0]: log-density (apply exp() for actual density)
                - output[:, 1:]: geometric features for color MLP
        """
        return self.network(features)

    def get_params(self):
        """Get network parameters for saving."""
        return self.network.parameters()

    def num_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.network.parameters())


class ColorMLP(nn.Module):
    """
    Color MLP: Decodes geometric features + view direction to RGB.

    Architecture:
        Input: [B, geo_feat_dim + view_encoded_dim]
               = geometric features (15D) + encoded view direction (16D for SH degree 4)
        Hidden: [64, 64, ...] with ReLU
        Output: [B, 3] RGB values (apply sigmoid for final output)

    Args:
        geo_feat_dim: Geometric feature dimension from density MLP (default: 16)
        view_encoding_dim: View direction encoding dimension (default: 16 for SH degree 4)
        hidden_dim: Hidden layer width (default: 64)
        n_hidden_layers: Number of hidden layers (default: 2)
    """

    def __init__(self, geo_feat_dim=16, view_encoding_dim=16, hidden_dim=64, n_hidden_layers=2):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tiny-cuda-nn is required for ColorMLP. Please install it.")

        self.geo_feat_dim = geo_feat_dim
        self.view_encoding_dim = view_encoding_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # Input is geometric features (minus first channel which is density) + view encoding
        input_dim = (geo_feat_dim - 1) + view_encoding_dim

        # Create tiny-cuda-nn network
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",  # Sigmoid applied separately
            "n_neurons": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
        }

        self.network = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=3,  # RGB
            network_config=network_config
        )

    def forward(self, geo_features, view_encoded):
        """
        Forward pass through the color MLP.

        Args:
            geo_features: Geometric features [B, geo_feat_dim-1] (exclude density channel)
            view_encoded: Encoded view direction [B, view_encoding_dim]

        Returns:
            rgb: RGB values [B, 3] (apply sigmoid for final output)
        """
        # Concatenate geometric features and view direction
        combined = torch.cat([geo_features, view_encoded], dim=-1)
        return self.network(combined)

    def get_params(self):
        """Get network parameters for saving."""
        return self.network.parameters()

    def num_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.network.parameters())


class TwoStageIntervalMLP(nn.Module):
    """
    Complete two-stage MLP for neural interval splatting.

    Combines:
    1. DensityMLP: aggregated_features -> [density, geo_features]
    2. ColorMLP: [geo_features, view_dir] -> RGB

    Args:
        feature_dim: Input aggregated feature dimension (e.g., 32)
        geo_feat_dim: Geometric feature dimension (default: 16, includes density)
        sh_degree: Spherical harmonics degree for view encoding (default: 4)
        density_hidden_dim: Hidden dimension for density MLP (default: 64)
        color_hidden_dim: Hidden dimension for color MLP (default: 64)
        density_n_hidden: Number of hidden layers in density MLP (default: 2)
        color_n_hidden: Number of hidden layers in color MLP (default: 2)
    """

    def __init__(
        self,
        feature_dim=32,
        geo_feat_dim=16,
        sh_degree=4,
        density_hidden_dim=64,
        color_hidden_dim=64,
        density_n_hidden=2,
        color_n_hidden=2
    ):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tiny-cuda-nn is required for TwoStageIntervalMLP. Please install it.")

        self.feature_dim = feature_dim
        self.geo_feat_dim = geo_feat_dim
        self.sh_degree = sh_degree

        # Create spherical harmonics encoder for view directions
        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,  # 3D direction
            encoding_config=get_spherical_harmonics_encoding_config(sh_degree)
        )
        view_encoding_dim = self.view_encoder.n_output_dims

        # Create density MLP
        self.density_mlp = DensityMLP(
            feature_dim=feature_dim,
            hidden_dim=density_hidden_dim,
            n_hidden_layers=density_n_hidden,
            geo_feat_dim=geo_feat_dim
        )

        # Create color MLP
        self.color_mlp = ColorMLP(
            geo_feat_dim=geo_feat_dim,
            view_encoding_dim=view_encoding_dim,
            hidden_dim=color_hidden_dim,
            n_hidden_layers=color_n_hidden
        )

    def forward(self, features, view_dirs):
        """
        Forward pass through the two-stage MLP.

        Args:
            features: Aggregated features [B, feature_dim]
            view_dirs: View directions [B, 3] (should be normalized)

        Returns:
            dict with keys:
                - 'rgb': RGB values [B, 3]
                - 'density': Density values [B, 1]
                - 'geo_features': Geometric features [B, geo_feat_dim-1]
        """
        # Stage 1: Density MLP
        geo_output = self.density_mlp(features)  # [B, geo_feat_dim]

        # Extract density (first channel) and geometric features (rest)
        log_density = geo_output[:, 0:1]  # [B, 1]
        geo_features = geo_output[:, 1:]  # [B, geo_feat_dim-1]

        # Apply exponential activation to get density
        density = torch.exp(log_density)  # [B, 1]

        # Stage 2: Encode view direction
        view_encoded = self.view_encoder(view_dirs)  # [B, view_encoding_dim]

        # Stage 3: Color MLP
        rgb_logits = self.color_mlp(geo_features, view_encoded)  # [B, 3]

        # Apply sigmoid to get RGB in [0, 1]
        rgb = torch.sigmoid(rgb_logits)

        return {
            'rgb': rgb,
            'density': density,
            'geo_features': geo_features
        }

    def get_params(self):
        """Get all network parameters for saving."""
        return list(self.density_mlp.get_params()) + list(self.color_mlp.get_params())

    def num_parameters(self):
        """Count total number of parameters."""
        return self.density_mlp.num_parameters() + self.color_mlp.num_parameters()


class RGBOnlyMLP(nn.Module):
    """
    RGB-only MLP for MLP-opacity mode.

    In this mode, density comes from rasterized Gaussian opacities,
    and the MLP only predicts view-dependent RGB color.

    Architecture:
        Input: [B, feature_dim + view_encoding_dim]
        Hidden: [hidden_dim, ...] with ReLU
        Output: [B, 3] RGB with sigmoid activation

    Args:
        feature_dim: Input feature dimension (e.g., 32)
        sh_degree: Spherical harmonics degree for view encoding (default: 4)
        hidden_dim: Hidden layer width (default: 64)
        n_hidden_layers: Number of hidden layers (default: 2)
    """

    def __init__(self, feature_dim=32, sh_degree=4, hidden_dim=64, n_hidden_layers=2):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise ImportError("tiny-cuda-nn is required for RGBOnlyMLP. Please install it.")

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # Create spherical harmonics encoder for view directions
        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,  # 3D direction
            encoding_config=get_spherical_harmonics_encoding_config(sh_degree)
        )
        view_encoding_dim = self.view_encoder.n_output_dims

        # Input is aggregated features + encoded view direction
        input_dim = feature_dim + view_encoding_dim

        # Create RGB network
        network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "Sigmoid",  # RGB in [0, 1]
            "n_neurons": hidden_dim,
            "n_hidden_layers": n_hidden_layers,
        }

        self.network = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=3,  # RGB only
            network_config=network_config
        )

    def forward(self, features, view_dirs):
        """
        Forward pass through RGB-only MLP.

        Args:
            features: Aggregated features [B, feature_dim]
            view_dirs: View directions [B, 3] (should be normalized)

        Returns:
            RGB values [B, 3] in range [0, 1]
        """
        # Encode view direction
        view_encoded = self.view_encoder(view_dirs)  # [B, view_encoding_dim]

        # Concatenate features and view direction
        combined = torch.cat([features, view_encoded], dim=-1)

        # Network already applies sigmoid activation
        rgb = self.network(combined)  # [B, 3]

        return rgb

    def get_params(self):
        """Get network parameters for saving."""
        return self.network.parameters()

    def num_parameters(self):
        """Count total number of parameters."""
        return sum(p.numel() for p in self.network.parameters())


def create_interval_mlp(
    feature_dim=32,
    geo_feat_dim=16,
    sh_degree=4,
    density_hidden_dim=64,
    color_hidden_dim=64,
    density_n_hidden=2,
    color_n_hidden=2,
    two_stage=True,
    rgb_only=False
):
    """
    Factory function to create an interval MLP.

    Args:
        feature_dim: Input feature dimension
        geo_feat_dim: Geometric feature dimension (for two-stage)
        sh_degree: Spherical harmonics degree for view encoding
        density_hidden_dim: Hidden dim for density MLP
        color_hidden_dim: Hidden dim for color MLP
        density_n_hidden: Number of hidden layers in density MLP
        color_n_hidden: Number of hidden layers in color MLP
        two_stage: If True, use two-stage architecture (recommended)
        rgb_only: If True, use RGB-only MLP for MLP-opacity mode

    Returns:
        TwoStageIntervalMLP or RGBOnlyMLP instance
    """
    if rgb_only:
        return RGBOnlyMLP(
            feature_dim=feature_dim,
            sh_degree=sh_degree,
            hidden_dim=color_hidden_dim,
            n_hidden_layers=color_n_hidden
        )
    elif two_stage:
        return TwoStageIntervalMLP(
            feature_dim=feature_dim,
            geo_feat_dim=geo_feat_dim,
            sh_degree=sh_degree,
            density_hidden_dim=density_hidden_dim,
            color_hidden_dim=color_hidden_dim,
            density_n_hidden=density_n_hidden,
            color_n_hidden=color_n_hidden
        )
    else:
        raise NotImplementedError("Single-stage MLP deprecated. Use two_stage=True or rgb_only=True.")


if __name__ == "__main__":
    # Test the two-stage MLP
    print("Testing TwoStageIntervalMLP...")

    mlp = create_interval_mlp(
        feature_dim=32,
        geo_feat_dim=16,
        sh_degree=4,
        density_hidden_dim=64,
        color_hidden_dim=64,
        density_n_hidden=2,
        color_n_hidden=2,
        two_stage=True
    )
    mlp = mlp.cuda()

    print(f"Created MLP with {mlp.num_parameters()} parameters")
    print(f"  Density MLP: {mlp.density_mlp.num_parameters()} params")
    print(f"  Color MLP: {mlp.color_mlp.num_parameters()} params")

    # Test forward pass
    batch_size = 1024
    features = torch.randn(batch_size, 32, device='cuda', dtype=torch.float16)
    view_dirs = torch.randn(batch_size, 3, device='cuda', dtype=torch.float16)
    view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)  # Normalize

    output = mlp(features, view_dirs)

    print(f"\nInput shapes:")
    print(f"  features: {features.shape}")
    print(f"  view_dirs: {view_dirs.shape}")
    print(f"\nOutput shapes:")
    print(f"  rgb: {output['rgb'].shape}")
    print(f"  density: {output['density'].shape}")
    print(f"  geo_features: {output['geo_features'].shape}")

    print(f"\nValue ranges:")
    print(f"  RGB: [{output['rgb'].min().item():.4f}, {output['rgb'].max().item():.4f}]")
    print(f"  Density: [{output['density'].min().item():.4f}, {output['density'].max().item():.4f}]")

    # Test backward pass
    loss = output['rgb'].sum() + output['density'].sum()
    loss.backward()

    print("\nBackward pass successful!")
    print("✓ TwoStageIntervalMLP is working correctly")
