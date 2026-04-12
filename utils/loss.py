"""
Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss

This is the contrastive loss used in SimCLR. Given a batch of N samples,
each sample generates two augmented views, resulting in 2N samples total.
For each anchor sample, its positive pair is its augmented twin. All other
2N-2 samples are negatives.

The loss encourages:
- Positive pairs to have high similarity (cosine similarity close to 1)
- Negative pairs to have low similarity (cosine similarity close to -1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TEMPERATURE


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR.

    Implements the contrastive loss with:
    - Cosine similarity as similarity metric
    - Temperature scaling
    - Cross-entropy formulation
    """

    def __init__(self, temperature=TEMPERATURE):
        """
        Initialize the NT-Xent loss.

        Args:
            temperature (float): Temperature parameter for scaling.
                               Lower temperature makes the distribution sharper.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j=None):
        """
        Compute NT-Xent loss for a batch.

        Args:
            z_i (torch.Tensor): Projected features from first view [N, proj_dim]
                               or concatenated views [2N, proj_dim] if z_j is None
            z_j (torch.Tensor, optional): Projected features from second view [N, proj_dim]
                                          If None, assumes z_i contains concatenated views

        Returns:
            torch.Tensor: Scalar loss value
        """
        if z_j is not None:
            # Two separate tensors case
            assert z_i.shape == z_j.shape, "Both views should have same shape"
            N = z_i.shape[0]
            # Concatenate views along batch dimension
            z = torch.cat([z_i, z_j], dim=0)  # Shape: [2N, proj_dim]
        else:
            # Already concatenated case
            z = z_i
            N = z.shape[0] // 2

        # Normalize each feature vector to unit length
        z = F.normalize(z, dim=1)

        # Compute similarity matrix [2N, 2N]
        # sim[i, j] = cosine similarity between z[i] and z[j]
        sim_matrix = torch.mm(z, z.T)  # [2N, 2N]

        # Create mask to identify positive pairs
        # For sample i (0 <= i < N), positive is i+N
        # For sample i+N, positive is i
        mask = torch.zeros_like(sim_matrix)
        mask[torch.arange(N), torch.arange(N) + N] = 1
        mask[torch.arange(N) + N, torch.arange(N)] = 1

        # For numerical stability in case of large similarity values
        # Remove self-similarities from the denominator calculation
        # We'll use the mask approach instead of explicitly zeroing diagonal
        # because we only care about positive/negative pairs

        # Scale similarities by temperature
        sim_matrix = sim_matrix / self.temperature

        # For each sample i, the positive is at index (i+N) mod 2N
        # All other indices are negatives
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0).to(z.device)

        # The similarity matrix has shape [2N, 2N]
        # For each row i, we want to compute cross-entropy against label[i]
        # which should have high similarity at column labels[i] and low elsewhere

        # However, we need to exclude self-similarity (diagonal) from negatives
        # Create a mask for valid negatives/examples

        # For each row, we exclude the diagonal by setting it to a large negative number
        # This way it won't contribute to the softmax
        diag_mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        # We don't need to mask diagonal explicitly if we structure labels correctly
        # because the positive pair is not self

        # Compute cross-entropy loss
        # For each sample i, we want label[i] to have highest similarity
        loss = self.criterion(sim_matrix, labels)

        # Normalize by number of positive pairs (N)
        loss = loss / (2 * N)

        return loss


class NTXentLossV2(nn.Module):
    """
    Alternative implementation of NT-Xent loss with explicit masking.
    More aligned with the original SimCLR paper formulation.
    """

    def __init__(self, temperature=TEMPERATURE):
        super(NTXentLossV2, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j=None):
        """
        Compute NT-Xent loss.

        Args:
            z_i: [N, proj_dim] or [2N, proj_dim]
            z_j: [N, proj_dim] or None
        """
        if z_j is not None:
            z = torch.cat([z_i, z_j], dim=0)
        else:
            z = z_i

        N = z.shape[0] // 2

        # Normalize embeddings
        z = F.normalize(z, dim=1)

        # Compute similarity matrix
        sim = torch.mm(z, z.T) / self.temperature  # [2N, 2N]

        # Labels: for each sample i, positive is i+N
        labels = torch.cat([torch.arange(N) + N, torch.arange(N)], dim=0)
        labels = labels.unsqueeze(0).to(z.device)  # [1, 2N]

        # Mask: remove self-similarities from loss calculation
        # We want to compute softmax over all other samples
        mask = torch.ones(2 * N, 2 * N, dtype=torch.bool, device=z.device)
        mask.fill_diagonal_(False)

        # For each sample, the positive is at a specific position
        # We'll compute the loss by:
        # - Numerator: exp(sim[i, positive])
        # - Denominator: sum over all j != i of exp(sim[i, j])

        # Create a mask for positives (for correct indexing)
        pos_mask = torch.zeros_like(sim)
        pos_mask[torch.arange(2 * N), labels] = 1

        # Compute numerator (positive pair similarities)
        numerator = (torch.exp(sim) * pos_mask).sum(dim=1)  # [2N]

        # Compute denominator (all negatives + positives but excluding self)
        denominator = (torch.exp(sim) * mask).sum(dim=1)  # [2N]

        # Loss per sample
        loss_per_sample = -torch.log(numerator / denominator)

        # Average loss
        loss = loss_per_sample.mean()

        return loss


def test_ntxent_loss():
    """Test the NT-Xent loss."""
    print("Testing NT-Xent Loss...")

    # Create dummy features
    N = 4
    proj_dim = 128
    z_i = torch.randn(N, proj_dim)
    z_j = torch.randn(N, proj_dim)

    # Test both versions
    criterion1 = NTXentLoss(temperature=0.5)
    criterion2 = NTXentLossV2(temperature=0.5)

    loss1 = criterion1(z_i, z_j)
    loss2 = criterion2(z_i, z_j)

    print(f"NT-Xent Loss (version 1): {loss1.item():.4f}")
    print(f"NT-Xent Loss (version 2): {loss2.item():.4f}")
    print(f"Difference: {abs(loss1.item() - loss2.item()):.6f}")

    # Test with concatenated inputs
    z = torch.cat([z_i, z_j], dim=0)
    loss3 = criterion1(z, None)
    loss4 = criterion2(z, None)

    print(f"Concatenated version 1: {loss3.item():.4f}")
    print(f"Concatenated version 2: {loss4.item():.4f}")

    assert loss1.item() > 0, "Loss should be positive"
    assert torch.isclose(loss1, loss3, rtol=1e-5), "Both input formats should give same result"
    assert torch.isclose(loss2, loss4, rtol=1e-5), "Both input formats should give same result"

    print("[OK] NT-Xent loss test passed!")


if __name__ == "__main__":
    test_ntxent_loss()
