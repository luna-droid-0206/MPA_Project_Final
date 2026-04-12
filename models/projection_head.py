"""
Projection Head for SimCLR

Implements the MLP projection head that maps encoder features
to a lower-dimensional space (typically 128-D) for contrastive learning.
"""

import torch
import torch.nn as nn
from config import PROJECTION_HIDDEN_DIM, PROJECTION_DIM


class ProjectionHead(nn.Module):
    """
    MLP projection head for SimCLR.

    Architecture: Linear -> ReLU -> Linear

    The first linear layer projects from encoder dimension to hidden dimension.
    The second linear layer projects from hidden to projection dimension.
    No activation is applied to the final output (contrastive learning uses normalized features).
    """

    def __init__(self, input_dim=512, hidden_dim=PROJECTION_HIDDEN_DIM, output_dim=PROJECTION_DIM):
        """
        Initialize the projection head.

        Args:
            input_dim (int): Input feature dimension (encoder output).
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output projection dimension.
        """
        super(ProjectionHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the projection head.

        Args:
            x (torch.Tensor): Input features of shape [B, input_dim]

        Returns:
            torch.Tensor: Projected features of shape [B, output_dim]
        """
        return self.net(x)


def test_projection_head():
    """Test the projection head with a dummy input."""
    head = ProjectionHead(input_dim=512)
    print(f"Projection head created: {head.__class__.__name__}")

    dummy_input = torch.randn(4, 512)
    output = head(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (4, 128), f"Expected (4, 128), got {output.shape}"
    print("[OK] Projection head test passed!")


if __name__ == "__main__":
    test_projection_head()
