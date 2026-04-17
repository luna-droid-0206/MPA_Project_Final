"""
Dataset Handling for Tiny ImageNet (with legacy CIFAR-10/100 support)

Provides dataset classes for:
1. SimCLR pretraining: returns two augmented views per image (no labels)
2. Supervised training: returns image and label
3. Linear evaluation: returns image and label (encoder frozen)
"""

import os
import torch
import shutil
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_DIR,
    DATASET_NAME,
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    IMAGE_SIZE
)
from utils.augmentations import get_default_transform


# ============================================================================
# TINY IMAGENET DATASET HANDLING
# ============================================================================

class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet dataset loader.
    
    Expects directory structure:
    data/tiny-imagenet-200/
        ├── train/
        │   ├── n00000001/
        │   │   ├── images/
        │   │   ├── n00000001_boxes.txt
        │   └── ...
        ├── val/
        │   ├── images/
        │   ├── val_annotations.txt
        └── wnids.txt
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Initialize Tiny ImageNet dataset.
        
        Args:
            root (str): Root directory of Tiny ImageNet
            split (str): 'train' or 'val'
            transform (callable): Transform to apply
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        self._setup_classes()
        self._load_samples()
    
    def _setup_classes(self):
        """Load class information from wnids.txt"""
        wnids_file = self.root / 'wnids.txt'
        
        if not wnids_file.exists():
            raise FileNotFoundError(f"wnids.txt not found at {wnids_file}")
        
        with open(wnids_file, 'r') as f:
            wnids = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
        self.idx_to_class = {idx: wnid for wnid, idx in self.class_to_idx.items()}
    
    def _load_samples(self):
        """Load image paths and labels"""
        if self.split == 'train':
            self._load_train_samples()
        elif self.split == 'val':
            self._load_val_samples()
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _load_train_samples(self):
        """Load training samples"""
        train_dir = self.root / 'train'
        
        for class_dir in sorted(train_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]
            
            images_dir = class_dir / 'images'
            if not images_dir.exists():
                continue
            
            for img_file in sorted(images_dir.glob('*.JPEG')):
                self.samples.append((str(img_file), class_idx))
    
    def _load_val_samples(self):
        """Load validation samples"""
        val_dir = self.root / 'val'
        images_dir = val_dir / 'images'
        annotations_file = val_dir / 'val_annotations.txt'
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"val_annotations.txt not found at {annotations_file}")
        
        # Parse annotations
        img_to_class = {}
        with open(annotations_file, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                img_name = parts[0]
                class_name = parts[1]
                img_to_class[img_name] = class_name
        
        # Load samples
        for img_file in sorted(images_dir.glob('*.JPEG')):
            img_name = img_file.name
            if img_name in img_to_class:
                class_name = img_to_class[img_name]
                class_idx = self.class_to_idx[class_name]
                self.samples.append((str(img_file), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        
        from PIL import Image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, class_idx


def download_tiny_imagenet(data_dir=DATA_DIR):
    """
    Verify Tiny ImageNet dataset exists.
    
    Note: Tiny ImageNet requires manual download due to file size.
    Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    
    Args:
        data_dir (str): Directory to store the dataset
        
    Returns:
        Path: Path to Tiny ImageNet directory
    """
    tiny_imagenet_path = Path(data_dir) / 'tiny-imagenet-200'
    
    if tiny_imagenet_path.exists():
        print(f"✓ Tiny ImageNet found at {tiny_imagenet_path}")
        return tiny_imagenet_path
    
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD REQUIRED FOR TINY IMAGENET")
    print("="*70)
    print("\nPlease download Tiny ImageNet from:")
    print("  http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    print("\nThen extract it to: " + str(data_dir))
    print("\nExpected directory structure:")
    print("  ./data/tiny-imagenet-200/")
    print("  ├── train/")
    print("  ├── val/")
    print("  └── wnids.txt")
    print("="*70 + "\n")
    
    raise FileNotFoundError(
        f"Tiny ImageNet not found at {tiny_imagenet_path}\n"
        f"Please download it manually from http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    )


# ============================================================================
# SIMCLR DATASET WRAPPER
# ============================================================================

class SimCLRDataset(Dataset):
    """
    Dataset wrapper for SimCLR training.

    For each image, returns two randomly augmented views.
    Labels are not used during pretraining but are kept for convenience.
    """

    def __init__(self, base_dataset, transform=None):
        """
        Initialize SimCLR dataset.

        Args:
            base_dataset: Base dataset (CIFAR10, TinyImageNet, etc.)
            transform (callable): Transform that returns two augmented views.
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get two augmented views of an image.

        Returns:
            tuple: (view1, view2, label)
        """
        img, label = self.base_dataset[idx]

        if self.transform is not None:
            # Transform returns either (view1, view2) or a single image
            result = self.transform(img)
            if isinstance(result, tuple):
                view1, view2 = result
            else:
                # If transform returns a single image, apply it twice
                view1 = result
                view2 = self.transform(img)
        else:
            view1 = img
            view2 = img

        return view1, view2, label


# ============================================================================
# DATA LOADER CREATION
# ============================================================================

def get_simclr_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    train_transform=None,
    test_transform=None,
    dataset_name=DATASET_NAME
):
    """
    Create dataloaders for SimCLR pretraining.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        train_transform (callable): Augmentation transform for training.
        test_transform (callable): Transform for validation.
        dataset_name (str): 'tiny-imagenet', 'cifar10', or 'cifar100'

    Returns:
        tuple: (train_loader, test_loader)
    """
    print(f"\nCreating {dataset_name} SimCLR dataloaders...")

    if test_transform is None:
        test_transform = get_default_transform()

    # Create datasets based on dataset name
    if dataset_name.lower() == 'tiny-imagenet':
        tiny_imagenet_path = download_tiny_imagenet(DATA_DIR)
        
        train_base = TinyImageNetDataset(
            root=str(tiny_imagenet_path),
            split='train',
            transform=None  # Transform applied by SimCLRDataset
        )
        
        test_base = TinyImageNetDataset(
            root=str(tiny_imagenet_path),
            split='val',
            transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Wrap in SimCLR dataset
    train_dataset = SimCLRDataset(train_base, transform=train_transform)
    test_dataset = test_base  # Don't wrap test in SimCLRDataset

    prefetch = PREFETCH_FACTOR if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Important for contrastive learning
        prefetch_factor=prefetch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_supervised_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    train_transform=None,
    test_transform=None,
    dataset_name=DATASET_NAME
):
    """
    Create dataloaders for supervised training.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        train_transform (callable): Augmentation transform for training.
        test_transform (callable): Transform for validation/testing.
        dataset_name (str): 'tiny-imagenet', 'cifar10', or 'cifar100'

    Returns:
        tuple: (train_loader, test_loader)
    """
    print(f"\nCreating {dataset_name} supervised dataloaders...")

    if train_transform is None:
        train_transform = get_default_transform()
    if test_transform is None:
        test_transform = get_default_transform()

    # Create datasets based on dataset name
    if dataset_name.lower() == 'tiny-imagenet':
        tiny_imagenet_path = download_tiny_imagenet(DATA_DIR)
        
        train_dataset = TinyImageNetDataset(
            root=str(tiny_imagenet_path),
            split='train',
            transform=train_transform
        )
        
        test_dataset = TinyImageNetDataset(
            root=str(tiny_imagenet_path),
            split='val',
            transform=test_transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prefetch = PREFETCH_FACTOR if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_linear_eval_dataloader(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    transform=None,
    dataset_name=DATASET_NAME
):
    """
    Create dataloader for linear evaluation.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        transform (callable): Transform to apply.
        dataset_name (str): 'tiny-imagenet', 'cifar10', or 'cifar100'

    Returns:
        DataLoader: Test dataloader for linear evaluation.
    """
    if transform is None:
        transform = get_default_transform()

    if dataset_name.lower() == 'tiny-imagenet':
        tiny_imagenet_path = download_tiny_imagenet(DATA_DIR)
        dataset = TinyImageNetDataset(
            root=str(tiny_imagenet_path),
            split='val',
            transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    prefetch = PREFETCH_FACTOR if num_workers > 0 else None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch
    )

    return loader


def test_dataloaders():
    """Test the dataloader functions."""
    from utils.augmentations import SimCLRTransform

    print("\n=== Testing Tiny ImageNet Dataloaders ===")
    
    try:
        train_loader, test_loader = get_simclr_dataloaders(
            train_transform=SimCLRTransform(),
            dataset_name='tiny-imagenet'
        )

        # Check a batch from training
        view1, view2, labels = next(iter(train_loader))
        print(f"✓ Train batch - Views: {view1.shape}, {view2.shape}, Labels: {labels.shape}")
        assert view1.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected view1 shape: {view1.shape}"
        assert view2.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE), f"Unexpected view2 shape: {view2.shape}"
        assert labels.shape == (BATCH_SIZE,), f"Unexpected labels shape: {labels.shape}"

        # Check a batch from test
        images, labels = next(iter(test_loader))
        print(f"✓ Test batch - Images: {images.shape}, Labels: {labels.shape}")
        assert images.shape[0] <= BATCH_SIZE, "Unexpected batch size"
        assert images.shape[1] == 3, "Expected 3 channels"

        print("[OK] SimCLR dataloader test passed!")

    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        print("Please download Tiny ImageNet to run dataloader tests")


if __name__ == "__main__":
    test_dataloaders()
