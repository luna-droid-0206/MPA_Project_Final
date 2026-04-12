"""
ResNet-18 Encoder for SimCLR

Removes the final classification layer and returns
the feature vector from the average pooling layer.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class Encoder(nn.Module):
    """
    ResNet-18 encoder that outputs 512-dimensional feature vectors.

    The standard ResNet-18 is modified by removing the final fully
    connected layer (fc). The output is the feature vector from
    the average pooling layer.
    """

    def __init__(self, pretrained=False):
        """
        Initialize the encoder.

        Args:
            pretrained (bool): If True, use ImageNet pretrained weights.
                             If False, randomly initialize.
        """
        super(Encoder, self).__init__()

        # Load ResNet-18
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            base_model = resnet18(weights=weights)
        else:
            base_model = resnet18(weights=None)

        # Remove the final fully connected layer
        # ResNet-18 structure: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )

        # Store output dimension
        self.feature_dim = 512

    def forward(self, x):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W]

        Returns:
            torch.Tensor: Feature vectors of shape [B, 512]
        """
        features = self.encoder(x)
        # Flatten the spatial dimensions
        features = features.view(features.size(0), -1)
        return features


def test_encoder():
    """Test the encoder with a dummy input."""
    encoder = Encoder(pretrained=False)
    print(f"Encoder created: {encoder.__class__.__name__}")
    print(f"Feature dimension: {encoder.feature_dim}")

    # Create dummy input (batch of 4 CIFAR-10 images)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = encoder(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 512), f"Expected (4, 512), got {output.shape}"
    print("[OK] Encoder test passed!")


if __name__ == "__main__":
    test_encoder()
