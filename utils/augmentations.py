"""
SimCLR Data Augmentations

Implements the augmentation pipeline described in the SimCLR paper:
- RandomResizedCrop
- RandomHorizontalFlip
- ColorJitter
- RandomGrayscale
- GaussianBlur

Designed specifically for CIFAR-10 images (32x32).
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from config import (
    TINY_IMAGENET_MEAN,
    TINY_IMAGENET_STD,
    IMAGE_SIZE,
    CROP_SCALE,
    COLOR_JITTER_PARAMS,
    GRAYSCALE_PROB,
    GAUSSIAN_BLUR_KERNEL,
    GAUSSIAN_BLUR_SIGMA,
    AUGMENTATION_STRENGTH
)


def get_default_transform():
    """
    Get default transform: just ToTensor + Normalize (no augmentations).
    
    Used when no augmentations are requested (e.g., for test sets or supervised training without augmentation).
    
    Returns:
        transforms.Compose: Default transform pipeline
    """
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TINY_IMAGENET_MEAN,
            std=TINY_IMAGENET_STD
        )
    ])
    return transform


def get_simclr_augmentations():
    """
    Returns the standard SimCLR augmentation pipeline.

    For Tiny ImageNet (64x64), uses aggressive augmentations for self-supervised learning.
    Augmentation parameters from SimCLR paper.

    Returns:
        transforms.Compose: Composed augmentation pipeline
    """
    brightness, contrast, saturation, hue = COLOR_JITTER_PARAMS

    # Apply jitter strength scaling as in SimCLR
    brightness = brightness * AUGMENTATION_STRENGTH
    contrast = contrast * AUGMENTATION_STRENGTH
    saturation = saturation * AUGMENTATION_STRENGTH
    hue = hue * AUGMENTATION_STRENGTH

    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=CROP_SCALE
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ),
        transforms.RandomGrayscale(p=GRAYSCALE_PROB),
        transforms.GaussianBlur(
            kernel_size=GAUSSIAN_BLUR_KERNEL,
            sigma=GAUSSIAN_BLUR_SIGMA
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=TINY_IMAGENET_MEAN,
            std=TINY_IMAGENET_STD
        )
    ])

    return transform


class SimCLRTransform:
    """
    Custom transform that applies augmentations to an image.

    By default, returns two independent augmented views for contrastive learning.
    Can optionally return a single view for supervised training.
    """

    def __init__(self, augmentation_strength=1.0, return_single=False):
        """
        Initialize with augmentation pipeline.

        Args:
            augmentation_strength (float): Strength multiplier for color jitter.
            return_single (bool): If True, return a single augmented view instead of two.
        """
        self.transform = get_simclr_augmentations()
        self.return_single = return_single

    def __call__(self, x):
        """
        Apply augmentation(s) to the input image.

        Args:
            x (PIL.Image or torch.Tensor): Input image

        Returns:
            torch.Tensor or tuple: Single augmented view if return_single=True,
                                 otherwise (augmented_view1, augmented_view2)
        """
        if self.return_single:
            return self.transform(x)
        else:
            # Generate two independent augmented views
            view1 = self.transform(x)
            view2 = self.transform(x)
            return view1, view2


def test_simclr_transform():
    """Test the SimCLR transform."""
    from PIL import Image
    import numpy as np

    transform = SimCLRTransform()

    # Create dummy image (Tiny ImageNet size)
    dummy_img = Image.fromarray((np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3) * 255).astype(np.uint8))

    view1, view2 = transform(dummy_img)

    print(f"Input image size: {dummy_img.size}")
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")

    # Check that views are different (due to random augmentations)
    assert view1.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Expected (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {view1.shape}"
    assert view2.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Expected (3, {IMAGE_SIZE}, {IMAGE_SIZE}), got {view2.shape}"
    assert not torch.allclose(view1, view2), "Views should be different due to random augmentation"

    print("[OK] SimCLR transform test passed!")


if __name__ == "__main__":
    test_simclr_transform()
