"""
SimCLR Self-Supervised Pre-training on CIFAR-10

Trains the encoder and projection head using contrastive learning.
Labels are NOT used during training.
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
    DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS_PRETRAIN, TEMPERATURE,
    WEIGHT_DECAY, OPTIMIZER, SGD_MOMENTUM, SAVE_FREQUENCY,
    CHECKPOINT_DIR, PRETRAINED_CHECKPOINT, RANDOM_SEED,
    USE_MIXED_PRECISION, USE_LR_SCHEDULER, SCHEDULER_TYPE,
    COSINE_T_MAX, COSINE_ETA_MIN
)
from models.encoder import Encoder
from models.projection_head import ProjectionHead
from utils.augmentations import SimCLRTransform
from utils.loss import NTXentLoss
from data.dataset import get_simclr_dataloaders


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pretrain_simclr(
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    epochs=EPOCHS_PRETRAIN,
    temperature=TEMPERATURE,
    save_freq=SAVE_FREQUENCY,
    checkpoint_path=PRETRAINED_CHECKPOINT,
    resume=False
):
    """
    Main pretraining function.

    Args:
        batch_size (int): Batch size.
        lr (float): Learning rate.
        epochs (int): Number of epochs.
        temperature (float): Temperature for NT-Xent loss.
        save_freq (int): Save checkpoint every N epochs.
        checkpoint_path (str): Path to save/load checkpoint.
        resume (bool): Resume from checkpoint if available.

    Returns:
        float: Final average loss.
    """
    print("=" * 60)
    print("SimCLR Self-Supervised Pre-training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Temperature: {temperature}")
    print("=" * 60)

    # Set seed
    set_seed()

    # Create data loaders with SimCLR augmentations
    train_loader, _ = get_simclr_dataloaders(
        batch_size=batch_size,
        train_transform=SimCLRTransform(),
        test_transform=None
    )

    # Initialize model: encoder + projection head
    print("\nInitializing model...")
    encoder = Encoder(pretrained=False).to(DEVICE)
    projection_head = ProjectionHead().to(DEVICE)

    model = nn.Sequential(encoder, projection_head)

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    head_params = sum(p.numel() for p in projection_head.parameters())
    total_params = encoder_params + head_params
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Projection head parameters: {head_params:,}")
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

    # Initialize loss
    criterion = NTXentLoss(temperature=temperature).to(DEVICE)

    # Initialize learning rate scheduler
    scheduler = None
    if USE_LR_SCHEDULER:
        if SCHEDULER_TYPE.lower() == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=COSINE_T_MAX,
                eta_min=COSINE_ETA_MIN
            )
            print(f"Using Cosine Annealing LR scheduler (T_max={COSINE_T_MAX}, eta_min={COSINE_ETA_MIN})")
        elif SCHEDULER_TYPE.lower() == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            print(f"Using Step LR scheduler")

    # Initialize mixed precision training
    scaler = GradScaler() if USE_MIXED_PRECISION else None
    if USE_MIXED_PRECISION:
        print(f"Mixed precision training enabled")
    
    # Optional: TensorBoard
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=f"{CHECKPOINT_DIR}/runs/simclr_pretrain")
    else:
        writer = None

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Training loop
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (view1, view2, _) in enumerate(train_loader):
            # Move to device
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            # Forward pass with mixed precision
            if USE_MIXED_PRECISION:
                with autocast():
                    z1 = projection_head(encoder(view1))
                    z2 = projection_head(encoder(view2))
                    loss = criterion(z1, z2)
                
                # Backward pass with scaled loss
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard backward pass
                z1 = projection_head(encoder(view1))
                z2 = projection_head(encoder(view2))
                loss = criterion(z1, z2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log batch loss
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            if scheduler is not None:
                writer.add_scalar('LR/train', scheduler.get_last_lr()[0], epoch)

        # Save checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projection_head_state_dict': projection_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'loss': avg_loss,
                'config': {
                    'batch_size': batch_size,
                    'lr': lr,
                    'temperature': temperature
                }
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"[OK] Checkpoint saved: {checkpoint_path} (epoch {epoch+1})")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time/60:.1f} minutes")
    print("=" * 60)

    # Save final checkpoint
    final_checkpoint = {
        'epoch': epochs - 1,
        'encoder_state_dict': encoder.state_dict(),
        'projection_head_state_dict': projection_head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'loss': avg_loss,
        'config': {
            'batch_size': batch_size,
            'lr': lr,
            'temperature': temperature
        }
    }
    torch.save(final_checkpoint, checkpoint_path)
    print(f"[OK] Final model saved: {checkpoint_path}")

    # Also save encoder separately for linear evaluation
    encoder_path = checkpoint_path.replace('.pth', '_encoder.pth')
    torch.save(encoder.state_dict(), encoder_path)
    print(f"[OK] Encoder saved separately: {encoder_path}")

    if writer is not None:
        writer.close()
    return avg_loss


def main():
    """Parse arguments and start pretraining."""
    parser = argparse.ArgumentParser(description="SimCLR Pretraining on CIFAR-10")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS_PRETRAIN, help='Number of epochs')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='Temperature parameter')
    parser.add_argument('--save-freq', type=int, default=SAVE_FREQUENCY, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint', type=str, default=PRETRAINED_CHECKPOINT, help='Checkpoint path')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()

    # Override device if needed
    if args.no_cuda:
        global DEVICE
        DEVICE = torch.device("cpu")

    # Create checkpoint directory
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # Run pretraining
    pretrain_simclr(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        temperature=args.temperature,
        save_freq=args.save_freq,
        checkpoint_path=args.checkpoint,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
