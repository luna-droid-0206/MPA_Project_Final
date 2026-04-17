"""
Supervised Baseline Training on CIFAR-10

Trains a ResNet-18 classifier from scratch using full labels.
This serves as a baseline to compare with SimCLR + linear evaluation.
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

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: tensorboard not found. Training will proceed without TensorBoard logging.")

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, SUPERVISED_LR, EPOCHS_SUPERVISED,
    WEIGHT_DECAY, OPTIMIZER, SGD_MOMENTUM, SAVE_FREQUENCY,
    CHECKPOINT_DIR, SUPERVISED_CHECKPOINT, RANDOM_SEED, NUM_CLASSES,
    USE_MIXED_PRECISION, USE_LR_SCHEDULER, SCHEDULER_TYPE,
    COSINE_T_MAX, COSINE_ETA_MIN
)
from models.encoder import Encoder
from utils.augmentations import SimCLRTransform, get_simclr_augmentations
from data.dataset import get_supervised_dataloaders


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SupervisedClassifier(nn.Module):
    """
    Supervised classifier using ResNet-18 encoder + linear classifier.
    """

    def __init__(self, encoder=None, num_classes=NUM_CLASSES, pretrained=False):
        """
        Initialize classifier.

        Args:
            encoder (Encoder, optional): Pre-initialized encoder. If None, create new.
            num_classes (int): Number of output classes.
            pretrained (bool): If True and encoder is None, use ImageNet pretrained encoder.
        """
        super(SupervisedClassifier, self).__init__()

        if encoder is None:
            encoder = Encoder(pretrained=pretrained)

        self.encoder = encoder
        self.classifier = nn.Linear(encoder.feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images [B, C, H, W]

        Returns:
            torch.Tensor: Class logits [B, num_classes]
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def train_supervised(
    batch_size=BATCH_SIZE,
    lr=SUPERVISED_LR,
    epochs=EPOCHS_SUPERVISED,
    save_freq=SAVE_FREQUENCY,
    checkpoint_path=SUPERVISED_CHECKPOINT,
    resume=False,
    use_augmentation=True
):
    """
    Main supervised training function.

    Args:
        batch_size (int): Batch size.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        save_freq (int): Save checkpoint every N epochs.
        checkpoint_path (str): Path to save/load checkpoint.
        resume (bool): Resume from checkpoint if available.
        use_augmentation (bool): If True, use SimCLR augmentations for training.

    Returns:
        tuple: (final_train_acc, final_test_acc)
    """
    print("=" * 60)
    print("Supervised Training Baseline")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Using augmentations: {use_augmentation}")
    print("=" * 60)

    # Set seed
    set_seed()

    # Create data loaders
    print("\nLoading data...")
    if use_augmentation:
        # Use SimCLR-style augmentations for training (single view for supervised)
        train_transform = SimCLRTransform(return_single=True)
    else:
        # Simple transforms
        train_transform = None  # Default ToTensor + no normalization?

    train_loader, test_loader = get_supervised_dataloaders(
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=None  # Default: ToTensor + normalize
    )

    # Initialize model
    print("\nInitializing model...")
    model = SupervisedClassifier(pretrained=False).to(DEVICE)

    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    classifier_params = sum(p.numel() for p in model.classifier.parameters())
    total_params = encoder_params + classifier_params
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Classifier parameters: {classifier_params:,}")
    print(f"Total trainable parameters: {total_params:,}")

    # Initialize optimizer
    if OPTIMIZER.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=WEIGHT_DECAY
        )
    elif OPTIMIZER.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=SGD_MOMENTUM,
            weight_decay=WEIGHT_DECAY
        )
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize learning rate scheduler
    scheduler = None
    if USE_LR_SCHEDULER:
        if SCHEDULER_TYPE.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=EPOCHS_SUPERVISED,
                eta_min=COSINE_ETA_MIN
            )
            print(f"Using Cosine Annealing LR scheduler (T_max={EPOCHS_SUPERVISED}, eta_min={COSINE_ETA_MIN})")
        elif SCHEDULER_TYPE.lower() == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            print(f"Using Step LR scheduler")

    # Initialize mixed precision training
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    if USE_MIXED_PRECISION:
        print(f"Mixed precision training enabled")

    # TensorBoard
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=f"{CHECKPOINT_DIR}/runs/supervised")
    else:
        writer = None

    # Resume if requested
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
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        # Train
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

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader)

        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', test_acc, epoch)
            if scheduler is not None:
                writer.add_scalar('LR/train', scheduler.get_last_lr()[0], epoch)

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
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
    print(f"Training completed in {total_time/60:.1f} minutes")
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

    if writer is not None:
        writer.close()
    return train_acc, test_acc


def evaluate_model(model, dataloader):
    """
    Evaluate model on a dataset.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Data loader.

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


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description="Supervised Training on CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=SUPERVISED_LR, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS_SUPERVISED, help='Number of epochs')
    parser.add_argument('--save-freq', type=int, default=SAVE_FREQUENCY, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', type=str, default=SUPERVISED_CHECKPOINT, help='Checkpoint path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-aug', action='store_true', help='Disable data augmentations')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Override device if needed
    if args.no_cuda:
        global DEVICE
        DEVICE = torch.device("cpu")

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # Run training
    train_acc, test_acc = train_supervised(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        save_freq=args.save_freq,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        use_augmentation=not args.no_aug
    )

    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_acc:.2f}%")
    print(f"  Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
