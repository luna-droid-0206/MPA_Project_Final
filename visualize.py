"""
t-SNE Visualization of Learned Features

Visualizes the feature space learned by the SimCLR encoder
using t-SNE dimensionality reduction. Points are colored by
their true class labels to show clustering quality.
"""

import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

sys.path.append(str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, CHECKPOINT_DIR,
    PRETRAINED_CHECKPOINT, RESULTS_DIR, NUM_CLASSES
)
from models.encoder import Encoder
from data.dataset import get_supervised_dataloaders

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Colors for plotting
COLORS = plt.cm.tab10(np.arange(10))


def load_encoder(encoder_path=None):
    """
    Load a pretrained encoder.

    Args:
        encoder_path (str, optional): Path to encoder weights.
            Defaults to SimCLR pretrained encoder.

    Returns:
        nn.Module: Loaded encoder.
    """
    if encoder_path is None:
        encoder_path = PRETRAINED_CHECKPOINT.replace('.pth', '_encoder.pth')

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Encoder not found at {encoder_path}. "
            "Please run SimCLR pretraining first (train_simclr.py)."
        )

    encoder = Encoder(pretrained=False).to(DEVICE)
    encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=False))
    encoder.eval()

    print(f"[OK] Loaded encoder from {encoder_path}")
    return encoder


def extract_features(encoder, dataloader, num_samples=None):
    """
    Extract features from the encoder.

    Args:
        encoder (nn.Module): The encoder model.
        dataloader (DataLoader): Data loader.
        num_samples (int, optional): Maximum number of samples to extract.
            If None, use all samples in the dataloader.

    Returns:
        tuple: (features, labels) - numpy arrays.
    """
    print(f"\nExtracting features...")
    features_list = []
    labels_list = []

    total_samples = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            if num_samples and total_samples >= num_samples:
                break

            images = images.to(DEVICE)
            batch_features = encoder(images)

            features_list.append(batch_features.cpu().numpy())
            labels_list.append(labels.numpy())

            total_samples += images.size(0)

            if batch_idx % 20 == 0:
                print(f"  Processed {total_samples} samples...")

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    print(f"Extracted {len(features)} features, dimension: {features.shape[1]}")
    return features, labels


def apply_tsne(features, perplexity=30, n_iter=1000, random_state=42):
    """
    Apply t-SNE to reduce feature dimensions to 2D.

    Args:
        features (np.ndarray): High-dimensional features [N, D].
        perplexity (float): t-SNE perplexity parameter.
        n_iter (int): Number of iterations.
        random_state (int): Random seed.

    Returns:
        np.ndarray: 2D embeddings [N, 2].
    """
    print(f"\nRunning t-SNE (perplexity={perplexity}, n_iter={n_iter})...")
    print("  This may take a few minutes...")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        verbose=1
    )

    embeddings = tsne.fit_transform(features)
    print(f"[OK] t-SNE complete. Embeddings shape: {embeddings.shape}")
    return embeddings


def plot_tsne(embeddings, labels, title="t-SNE Visualization of SimCLR Features", save_path=None):
    """
    Create t-SNE scatter plot colored by class labels.

    Args:
        embeddings (np.ndarray): 2D embeddings [N, 2].
        labels (np.ndarray): Integer class labels [N].
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 10))

    # Plot each class
    for class_idx in range(NUM_CLASSES):
        mask = labels == class_idx
        class_embeddings = embeddings[mask]

        plt.scatter(
            class_embeddings[:, 0],
            class_embeddings[:, 1],
            color=COLORS[class_idx],
            label=CIFAR10_CLASSES[class_idx],
            alpha=0.6,
            s=10
        )

    plt.title(title, fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="CIFAR-10 Classes", loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add info text
    avg_per_class = len(labels) // NUM_CLASSES
    info_text = f"Total samples: {len(labels)}\nAvg per class: {avg_per_class} samples"
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Figure saved to: {save_path}")

    plt.show()


def visualize_features(
    encoder_path=None,
    num_samples=2000,
    perplexity=30,
    save=True
):
    """
    Main visualization pipeline.

    Args:
        encoder_path (str, optional): Path to encoder weights.
        num_samples (int): Number of test samples to visualize.
        perplexity (float): t-SNE perplexity.
        save (bool): Whether to save the figure.

    Returns:
        tuple: (embeddings, labels)
    """
    print("=" * 60)
    print("t-SNE Visualization of Learned Features")
    print("=" * 60)

    # Load encoder
    encoder = load_encoder(encoder_path)

    # Load test dataloader
    _, test_loader = get_supervised_dataloaders(
        batch_size=BATCH_SIZE,
        train_transform=None,
        test_transform=None
    )

    # Extract features
    features, labels = extract_features(encoder, test_loader, num_samples=num_samples)

    # Apply t-SNE
    embeddings = apply_tsne(features, perplexity=perplexity)

    # Plot
    if encoder_path:
        encoder_name = Path(encoder_path).stem
    else:
        encoder_name = "SimCLR_Pretrained"
    title = f"t-SNE Visualization - {encoder_name} (n={len(labels)})"

    if save:
        save_path = f"{RESULTS_DIR}/tsne_{encoder_name}.png"
    else:
        save_path = None

    plot_tsne(embeddings, labels, title=title, save_path=save_path)

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("=" * 60)

    return embeddings, labels


def main():
    """Parse arguments and run visualization."""
    import argparse

    parser = argparse.ArgumentParser(description="t-SNE Visualization of SimCLR Features")
    parser.add_argument('--encoder-path', type=str, default=None,
                        help='Path to encoder weights (default: SimCLR pretrained)')
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of test samples to visualize (default: 2000)')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity parameter (default: 30)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the figure')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Override device if needed
    if args.no_cuda:
        global DEVICE
        DEVICE = torch.device("cpu")

    try:
        embeddings, labels = visualize_features(
            encoder_path=args.encoder_path,
            num_samples=args.num_samples,
            perplexity=args.perplexity,
            save=not args.no_save
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run SimCLR pretraining first: python train_simclr.py")


if __name__ == "__main__":
    main()
