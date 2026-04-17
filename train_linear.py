"""
Linear Evaluation of SimCLR Pretrained Encoder

Protocol:
1. Load a pretrained encoder (from SimCLR pretraining)
2. Freeze encoder weights (no gradient updates)
3. Train a linear classifier on top using labeled data
4. Report accuracy on test set

This evaluates the quality of the learned representations.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, LINEAR_LR, EPOCHS_LINEAR,
    WEIGHT_DECAY, OPTIMIZER, SAVE_FREQUENCY, CHECKPOINT_DIR,
    LINEAR_EVAL_CHECKPOINT, PRETRAINED_CHECKPOINT, RANDOM_SEED, NUM_CLASSES,
    USE_MIXED_PRECISION, USE_LR_SCHEDULER, SCHEDULER_TYPE,
    COSINE_T_MAX, COSINE_ETA_MIN
)
from models.encoder import Encoder
from data.dataset import get_supervised_dataloaders


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LinearClassifier(nn.Module):
    """
    Linear classifier on top of frozen encoder.
    """

    def __init__(self, encoder, num_classes=NUM_CLASSES):
        """
        Initialize linear classifier.

        Args:
            encoder (nn.Module): Pretrained and frozen encoder.
            num_classes (int): Number of classes.
        """
        super(LinearClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.feature_dim, num_classes)

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        print(f"Encoder frozen: {sum(p.numel() for p in self.encoder.parameters()):,} parameters")
        print(f"Linear classifier parameters: {sum(p.numel() for p in self.fc.parameters()):,}")

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Class logits.
        """
        with torch.no_grad():  # Encoder is frozen
            features = self.encoder(x)
        logits = self.fc(features)
        return logits


def evaluate(model, dataloader):
    """
    Evaluate model accuracy.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Evaluation dataloader.

    Returns:
        float: Accuracy percentage.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    model.train()
    return 100 * correct / total


def train_linear(
    pretrained_encoder_path=PRETRAINED_CHECKPOINT.replace('.pth', '_encoder.pth'),
    lr=LINEAR_LR,
    epochs=EPOCHS_LINEAR,
    batch_size=BATCH_SIZE,
    save_freq=SAVE_FREQUENCY,
    checkpoint_path=LINEAR_EVAL_CHECKPOINT,
    resume=False
):
    """
    Main linear evaluation function.

    Args:
        pretrained_encoder_path (str): Path to pretrained encoder weights.
        lr (float): Learning rate for linear classifier.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        save_freq (int): Save checkpoint every N epochs.
        checkpoint_path (str): Path to save/load checkpoint.
        resume (bool): Resume from checkpoint.

    Returns:
        float: Final test accuracy.
    """
    print("=" * 60)
    print("Linear Evaluation of SimCLR Encoder")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Pretrained encoder: {pretrained_encoder_path}")
    print("=" * 60)

    # Set seed
    set_seed()

    # Load pretrained encoder
    print("\nLoading pretrained encoder...")
    if not os.path.exists(pretrained_encoder_path):
        raise FileNotFoundError(
            f"Pretrained encoder not found at {pretrained_encoder_path}. "
            "Please run SimCLR pretraining first (train_simclr.py)."
        )

    encoder = Encoder(pretrained=False).to(DEVICE)
    encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=DEVICE, weights_only=False))
    print(f"[OK] Loaded encoder from {pretrained_encoder_path}")

    # Create model: frozen encoder + linear classifier
    model = LinearClassifier(encoder, num_classes=NUM_CLASSES).to(DEVICE)

    # Data loaders (no special augmentations for linear eval)
    print("\nLoading data...")
    train_loader, test_loader = get_supervised_dataloaders(
        batch_size=batch_size,
        train_transform=None,  # Use default ToTensor
        test_transform=None
    )

    # Optimizer (only for linear classifier)
    if OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.fc.parameters(),  # Only optimize classifier
            lr=lr,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(
            model.fc.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    criterion = nn.CrossEntropyLoss()

    # Initialize learning rate scheduler
    scheduler = None
    if USE_LR_SCHEDULER:
        if SCHEDULER_TYPE.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS_LINEAR,
                eta_min=COSINE_ETA_MIN
            )
            print(f"Using Cosine Annealing LR scheduler (T_max={EPOCHS_LINEAR}, eta_min={COSINE_ETA_MIN})")
        elif SCHEDULER_TYPE.lower() == 'step':
            scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
            print(f"Using Step LR scheduler")

    # Initialize mixed precision training
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    if USE_MIXED_PRECISION:
        print(f"Mixed precision training enabled")

    # Resume from checkpoint if requested
    start_epoch = 0
    best_acc = 0.0
    if resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from epoch {checkpoint['epoch']}, best test acc: {best_acc:.2f}%")

    # Training loop
    print("\nStarting linear evaluation...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward with mixed precision
            if USE_MIXED_PRECISION:
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                
                # Backward pass with scaled loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward
                logits = model(images)
                loss = criterion(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Evaluate on test set
        test_acc = evaluate(model, test_loader)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'best_acc': best_acc
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[OK] Checkpoint saved: {checkpoint_path} (epoch {epoch+1})")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_checkpoint_path = checkpoint_path.replace('.pth', '_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'test_acc': test_acc,
            }, best_checkpoint_path)
            print(f"[OK] Best model saved: {best_checkpoint_path} (acc: {test_acc:.2f}%)")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Linear evaluation completed in {total_time/60:.1f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print("=" * 60)

    # Save final checkpoint
    final_checkpoint = {
        'epoch': epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'best_acc': best_acc
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"[OK] Final model saved: {checkpoint_path}")

    return best_acc


def main():
    """Parse arguments and start linear evaluation."""
    parser = argparse.ArgumentParser(description="Linear Evaluation on SimCLR Encoder")
    parser.add_argument('--encoder-path', type=str, default=PRETRAINED_CHECKPOINT.replace('.pth', '_encoder.pth'),
                        help='Path to pretrained encoder weights')
    parser.add_argument('--lr', type=float, default=LINEAR_LR, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS_LINEAR, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--save-freq', type=int, default=SAVE_FREQUENCY, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', type=str, default=LINEAR_EVAL_CHECKPOINT, help='Checkpoint path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Override device if needed
    if args.no_cuda:
        global DEVICE
        DEVICE = torch.device("cpu")

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # Run linear evaluation
    try:
        best_acc = train_linear(
            pretrained_encoder_path=args.encoder_path,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_freq=args.save_freq,
            checkpoint_path=args.checkpoint,
            resume=args.resume
        )
        print(f"\nLinear evaluation completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run SimCLR pretraining first: python train_simclr.py")


if __name__ == "__main__":
    main()
