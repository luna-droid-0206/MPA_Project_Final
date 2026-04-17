"""
Configuration for SimCLR Self-Supervised Learning on Tiny ImageNet
"""

import torch

# Dataset
DATA_DIR = "./data"
DATASET_NAME = "tiny-imagenet"  # "cifar10", "cifar100", or "tiny-imagenet"
TINY_IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
TINY_IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 200
IMAGE_SIZE = 64  # Tiny ImageNet images are 64x64

# Model
ENCODER_DIM = 512  # ResNet-18 output dimension (without fc)
PROJECTION_DIM = 128
PROJECTION_HIDDEN_DIM = 512

# Training Hyperparameters
BATCH_SIZE = 512  # Increased for Tiny ImageNet (adjust if GPU OOM)
LEARNING_RATE = 0.003  # SimCLR with SGD on Tiny ImageNet
SUPERVISED_LR = 0.1    # Supervised baseline learning rate (increased)
LINEAR_LR = 0.1        # Linear evaluation learning rate (increased)
TEMPERATURE = 0.5      # NT-Xent temperature parameter
EPOCHS_PRETRAIN = 300  # Increased for more complex dataset
EPOCHS_SUPERVISED = 100  # Increased for better convergence
EPOCHS_LINEAR = 100    # Increased for better convergence

# Optimizer
OPTIMIZER = "sgd"  # "adam" or "sgd" - SGD better for ImageNet-scale
WEIGHT_DECAY = 5e-4  # Adjusted for larger dataset
SGD_MOMENTUM = 0.9
USE_MIXED_PRECISION = True  # Enable for faster training

# Data Augmentation
AUGMENTATION_STRENGTH = 1.0  # SimCLR strength parameter
CROP_SCALE = (0.08, 1.0)  # Adjusted for larger 64x64 images - allow smaller crops
COLOR_JITTER_PARAMS = (0.4, 0.4, 0.4, 0.1)  # brightness, contrast, saturation, hue
GRAYSCALE_PROB = 0.2
GAUSSIAN_BLUR_KERNEL = 5  # Increased for 64x64 images
GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)

# Training
NUM_WORKERS = 8  # Increased for faster data loading
PIN_MEMORY = True
PREFETCH_FACTOR = 2  # Prefetch data in parallel
SAVE_FREQUENCY = 20  # Save less frequently (larger dataset)
GRADIENT_ACCUMULATION_STEPS = 1  # Set >1 if GPU memory is limited

# Paths
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
PRETRAINED_CHECKPOINT = f"{CHECKPOINT_DIR}/simclr_pretrained_tiny_imagenet.pth"
SUPERVISED_CHECKPOINT = f"{CHECKPOINT_DIR}/supervised_tiny_imagenet.pth"
LINEAR_EVAL_CHECKPOINT = f"{CHECKPOINT_DIR}/linear_eval_tiny_imagenet.pth"

# Reproducibility
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning Rate Scheduler
USE_LR_SCHEDULER = True
SCHEDULER_TYPE = "cosine"  # "cosine", "step", or "exponential"
COSINE_T_MAX = EPOCHS_PRETRAIN  # For cosine annealing
COSINE_ETA_MIN = 1e-5  # Minimum LR for cosine annealing
