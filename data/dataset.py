"""
CIFAR-10 Dataset for SimCLR and Supervised Training

Provides dataset classes for:
1. SimCLR pretraining: returns two augmented views per image (no labels)
2. Supervised training: returns image and label
3. Linear evaluation: returns image and label (encoder frozen)
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_DIR,
    CIFAR10_MEAN,
    CIFAR10_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY
)
from utils.augmentations import get_default_transform


def get_base_cifar10(data_dir=DATA_DIR, train=True, transform=None):
    """
    Load CIFAR-10 dataset with optional transforms.

    Args:
        data_dir (str): Directory to store/load the dataset.
        train (bool): If True, load training set, else test set.
        transform (callable, optional): Transform to apply to images.

    Returns:
        torchvision.datasets.CIFAR10: The dataset.
    """
    return datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform
    )


class SimCLRDataset(Dataset):
    """
    Dataset wrapper for SimCLR training.

    For each image, returns two randomly augmented views.
    Labels are not used during pretraining but are kept for convenience
    (can be used for analysis).
    """

    def __init__(self, train=True, transform=None, data_dir=DATA_DIR):
        """
        Initialize SimCLR dataset.

        Args:
            train (bool): Use training set if True, else test set.
            transform (callable): Transform that returns two augmented views.
            data_dir (str): CIFAR-10 data directory.
        """
        self.dataset = get_base_cifar10(
            data_dir=data_dir,
            train=train,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get two augmented views of an image.

        Returns:
            tuple: (view1, view2, label)
        """
        img, label = self.dataset[idx]

        # The transform should return two views
        # If the transform is SimCLRTransform, it returns (view1, view2)
        if isinstance(img, tuple):
            # Already transformed
            view1, view2 = img
        else:
            # Manual transform application
            view1 = img
            view2 = img

        return view1, view2, label


class SupervisedDataset(Dataset):
    """
    Dataset wrapper for supervised training/evaluation.

    Returns a single augmented image and its label.
    """

    def __init__(self, train=True, transform=None, data_dir=DATA_DIR):
        """
        Initialize supervised dataset.

        Args:
            train (bool): Use training set if True, else test set.
            transform (callable): Transform to apply to images.
            data_dir (str): CIFAR-10 data directory.
        """
        self.dataset = get_base_cifar10(
            data_dir=data_dir,
            train=train,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Get an augmented image and its label.

        Returns:
            tuple: (image, label)
        """
        return self.dataset[idx]


def get_simclr_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    train_transform=None,
    test_transform=None
):
    """
    Create dataloaders for SimCLR pretraining (using training set only).

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        train_transform (callable): Augmentation transform for training.
        test_transform (callable): Transform for validation (typically just normalize).

    Returns:
        tuple: (train_loader, test_loader) - test_loader returns two views as well for linear eval
    """
    print("Creating SimCLR dataloaders...")

    # Use default transform for test if None is provided
    if test_transform is None:
        test_transform = get_default_transform()

    # Training set with augmentations
    train_dataset = SimCLRDataset(
        train=True,
        transform=train_transform
    )

    # Test set without augmentations (or with simple normalize)
    # For linear evaluation we still need two views? Actually no, for linear eval we just need one
    test_dataset = SupervisedDataset(
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for contrastive learning: all batches same size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_supervised_dataloaders(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    train_transform=None,
    test_transform=None
):
    """
    Create dataloaders for supervised training.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        train_transform (callable): Augmentation transform for training.
        test_transform (callable): Transform for validation/testing.

    Returns:
        tuple: (train_loader, test_loader)
    """
    print("Creating supervised dataloaders...")

    # Use default transform if None is provided
    if train_transform is None:
        train_transform = get_default_transform()
    if test_transform is None:
        test_transform = get_default_transform()

    train_dataset = SupervisedDataset(
        train=True,
        transform=train_transform
    )

    test_dataset = SupervisedDataset(
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    return train_loader, test_loader


def get_linear_eval_dataloader(
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    transform=None
):
    """
    Create dataloader for linear evaluation.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of data loading workers.
        pin_memory (bool): Pin memory for faster GPU transfer.
        transform (callable): Transform to apply.

    Returns:
        DataLoader: Test dataloader for linear evaluation.
    """
    # Use default transform if None is provided
    if transform is None:
        transform = get_default_transform()
    
    dataset = SupervisedDataset(
        train=False,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader


def test_dataloaders():
    """Test the dataloader functions."""
    from utils.augmentations import SimCLRTransform

    print("\n=== Testing SimCLR Dataloaders ===")
    train_loader, test_loader = get_simclr_dataloaders(
        train_transform=SimCLRTransform(),
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])
    )

    # Check a batch from training
    view1, view2, labels = next(iter(train_loader))
    print(f"Train batch - Views: {view1.shape}, {view2.shape}, Labels: {labels.shape}")
    assert view1.shape == (BATCH_SIZE, 3, 32, 32), "Unexpected view1 shape"
    assert view2.shape == (BATCH_SIZE, 3, 32, 32), "Unexpected view2 shape"
    assert labels.shape == (BATCH_SIZE,), "Unexpected labels shape"

    # Check a batch from test
    images, labels = next(iter(test_loader))
    print(f"Test batch - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == (BATCH_SIZE, 3, 32, 32), "Unexpected image shape"

    print("[OK] SimCLR dataloader test passed!")

    print("\n=== Testing Supervised Dataloaders ===")
    sup_train_loader, sup_test_loader = get_supervised_dataloaders(
        train_transform=SimCLRTransform(return_single=True),
        test_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ])
    )

    images, labels = next(iter(sup_train_loader))
    print(f"Supervised train batch - Images: {images.shape}, Labels: {labels.shape}")
    assert images.shape == (BATCH_SIZE, 3, 32, 32), "Unexpected image shape"

    print("[OK] Supervised dataloader test passed!")


if __name__ == "__main__":
    test_dataloaders()
