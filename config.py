"""
Configuration for SimCLR Self-Supervised Learning on CIFAR-10
"""

import torch

# Dataset
DATA_DIR = "./data"
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]
NUM_CLASSES = 10

# Model
ENCODER_DIM = 512  # ResNet-18 output dimension (without fc)
PROJECTION_DIM = 128
PROJECTION_HIDDEN_DIM = 512

# Training Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001  # SimCLR uses higher LR with LARS, using Adam for simplicity
SUPERVISED_LR = 0.01   # Supervised baseline learning rate
LINEAR_LR = 0.01       # Linear evaluation learning rate
TEMPERATURE = 0.5      # NT-Xent temperature parameter
EPOCHS_PRETRAIN = 200
EPOCHS_SUPERVISED = 50
EPOCHS_LINEAR = 50

# Optimizer
OPTIMIZER = "adam"  # "adam" or "sgd"
WEIGHT_DECAY = 1e-4
SGD_MOMENTUM = 0.9

# Data Augmentation
AUGMENTATION_STRENGTH = 1.0  # SimCLR strength parameter
CROP_SCALE = (0.2, 1.0)  # RandomResizedCrop scale range
COLOR_JITTER_PARAMS = (0.4, 0.4, 0.4, 0.1)  # brightness, contrast, saturation, hue
GRAYSCALE_PROB = 0.2
GAUSSIAN_BLUR_KERNEL = 3
GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)

# Training
NUM_WORKERS = 4
PIN_MEMORY = True
SAVE_FREQUENCY = 10

# Paths
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"
PRETRAINED_CHECKPOINT = f"{CHECKPOINT_DIR}/simclr_pretrained.pth"
SUPERVISED_CHECKPOINT = f"{CHECKPOINT_DIR}/supervised.pth"
LINEAR_EVAL_CHECKPOINT = f"{CHECKPOINT_DIR}/linear_eval.pth"

# Reproducibility
RANDOM_SEED = 42

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
