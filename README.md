# Self-Supervised Learning with SimCLR on CIFAR-10

## Project Overview

This project implements **SimCLR** (Simple Contrastive Learning of Visual Representations) for self-supervised learning on CIFAR-10 images. The framework learns meaningful visual representations **without using labels** through contrastive learning, then evaluates these representations on downstream image classification tasks.

## Key Components

- **SimCLR Pretraining**: Contrastive learning with NT-Xent loss
- **Supervised Baseline**: Standard supervised ResNet-18 training
- **Linear Evaluation**: Freeze pretrained encoder, train linear classifier
- **t-SNE Visualization**: Visualize learned feature space clustering

## Project Structure

```
self_supervised_learning/
├── config.py                 # Hyperparameters and configuration
├── models/
│   ├── encoder.py           # ResNet-18 encoder
│   └── projection_head.py   # MLP projection head
├── utils/
│   ├── augmentations.py     # SimCLR augmentations
│   └── loss.py              # NT-Xent contrastive loss
├── data/
│   └── dataset.py           # Data loaders for CIFAR-10
├── train_simclr.py          # Self-supervised pretraining
├── train_supervised.py      # Supervised baseline training
├── train_linear.py          # Linear evaluation protocol
├── evaluate.py              # Compare all models
├── visualize.py             # t-SNE feature visualization
├── main.py                  # Unified CLI
├── checkpoints/             # Saved models (auto-created)
├── results/                 # Visualization outputs (auto-created)
└── README.md                # This file
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision
- numpy
- matplotlib
- scikit-learn
- tensorboard (optional, for logging)

### Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy matplotlib scikit-learn tensorboard
```

## Usage

### 1. SimCLR Pretraining

Train the encoder using contrastive learning:

```bash
cd self_supervised_learning
python train_simclr.py --epochs 200 --batch-size 256 --lr 0.001
```

Options:
- `--epochs N`: Number of training epochs (default: 200)
- `--batch-size N`: Batch size (default: 256)
- `--lr FLOAT`: Learning rate (default: 0.001)
- `--temperature FLOAT`: NT-Xent temperature (default: 0.5)
- `--checkpoint PATH`: Checkpoint save path
- `--resume`: Resume from checkpoint

Output:
- `checkpoints/simclr_pretrained.pth`: Full model (encoder + projection head)
- `checkpoints/simclr_pretrained_encoder.pth`: Encoder only (used for linear eval)

### 2. Supervised Baseline

Train ResNet-18 from scratch with full labels:

```bash
python train_supervised.py --epochs 50 --batch-size 256 --lr 0.01
```

Options:
- `--epochs N`: Number of epochs (default: 50)
- `--batch-size N`: Batch size (default: 256)
- `--lr FLOAT`: Learning rate (default: 0.01)
- `--no-aug`: Disable SimCLR augmentations
- `--resume`: Resume from checkpoint

Output:
- `checkpoints/supervised.pth`: Trained model (encoder + classifier)
- `checkpoints/supervised_best.pth`: Best model by test accuracy

### 3. Linear Evaluation

Evaluate pretrained encoder by training linear classifier (frozen encoder):

```bash
python train_linear.py --epochs 50 --batch-size 256 --lr 0.01
```

Prerequisites: Must have `checkpoints/simclr_pretrained_encoder.pth` from Step 1.

Options:
- `--encoder-path PATH`: Path to pretrained encoder (default: simclr_pretrained_encoder.pth)
- `--epochs N`: Number of epochs (default: 50)
- `--batch-size N`: Batch size (default: 256)
- `--lr FLOAT`: Learning rate (default: 0.01)
- `--resume`: Resume from checkpoint

Output:
- `checkpoints/linear_eval.pth`: Linear classifier model
- `checkpoints/linear_eval_best.pth`: Best linear classifier

### 4. Evaluation and Comparison

Evaluate all trained models and compare accuracy:

```bash
python evaluate.py
```

Output:
- Console: Accuracy comparison table
- `checkpoints/evaluation_results.json`: Detailed results

### 5. t-SNE Visualization

Visualize learned features in 2D:

```bash
python visualize.py --num-samples 2000 --perplexity 30
```

Options:
- `--encoder-path PATH`: Custom encoder weights
- `--num-samples N`: Number of test samples (default: 2000)
- `--perplexity FLOAT`: t-SNE perplexity (default: 30)
- `--no-save`: Don't save figure
- `--no-cuda`: Disable CUDA

Output:
- `results/tsne_SimCLR_Pretrained.png`: Saved visualization (if `--no-save` not set)
- Matplotlib plot window

### 6. Combined Pipeline (Using main.py)

```bash
# Run full pipeline
python main.py --all

# Run specific steps
python main.py --pretrain --supervised --linear --visualize

# Or with arguments
python main.py --pretrain --epochs-pretrain 100 --batch-size 128
```

## Expected Results

After successful training, you should observe:

- **SimCLR Pretraining**: Contrastive loss decreases steadily from ~-log(2N) to lower values
- **Supervised Baseline**: ~90%+ accuracy on CIFAR-10 test set (ResNet-18 standard)
- **SimCLR Linear Eval**: Should achieve competitive accuracy (80-90% depending on hyperparameters)
- **t-SNE Visualization**: Clear clustering of same-class images in feature space

### Result Template

| Model           | Accuracy    |
|-----------------|-------------|
| Supervised      | ~91%        |
| SimCLR + Linear | ~85-89%     |

*Note: Exact numbers vary based on random seed, augmentation strength, and training duration.*

## Configuration

All hyperparameters can be adjusted in `config.py`:

```python
# Model architecture
ENCODER_DIM = 512
PROJECTION_DIM = 128
PROJECTION_HIDDEN_DIM = 512

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS_PRETRAIN = 200
TEMPERATURE = 0.5

# Augmentation parameters
CROP_SCALE = (0.2, 1.0)
COLOR_JITTER_PARAMS = (0.4, 0.4, 0.4, 0.1)
GRAYSCALE_PROB = 0.2
```

## SimCLR Algorithm Summary

1. For each image, generate **two augmented views** (x_i, x_j)
2. Pass both views through encoder and projection head to get z_i, z_j
3. Compute **NT-Xent loss**:
   - Positive pair: (z_i, z_j) from same image
   - Negative pairs: All other samples in batch (2N-2 per anchor)
   - Loss encourages positive similarity >> negative similarity
4. Discard projection head after pretraining
5. Freeze encoder and train linear classifier on labeled data for evaluation

## Important Notes

### Memory Considerations
- CIFAR-10 images are small (32x32), so batch size 256 should fit on most GPUs
- If CUDA out-of-memory, reduce `batch_size` in config.py
- NT-Xent loss has O(N²) complexity in batch size (2N samples)

### Augmentations
Critical for SimCLR success. The augmentations used:
1. RandomResizedCrop (scale 0.2-1.0)
2. RandomHorizontalFlip
3. ColorJitter (brightness, contrast, saturation, hue)
4. RandomGrayscale (20%)
5. GaussianBlur

### Contrastive Loss
- Temperature τ controls the sharpness of the distribution
- Smaller τ (e.g., 0.1) -> harder, more selective
- Larger τ (e.g., 0.5-1.0) -> softer, more uniform

### Linear Evaluation Protocol
- Encoder weights are **frozen** (no gradient updates)
- Only the linear classifier is trained
- This measures the quality of learned representations

## Troubleshooting

**Q: CUDA out of memory**
- Reduce `batch_size` in config.py
- Use `--no-cuda` flag to run on CPU (slow)

**Q: Training is too slow**
- Use GPU (CUDA) if available
- Reduce number of epochs
- Reduce dataset size for quick testing

**Q: t-SNE runs forever**
- Reduce `--num-samples` to 1000 or 500
- Use `--perplexity 5` or 10 for faster computation

**Q: Accuracy lower than supervised baseline**
- SimCLR may need longer pretraining (try 400-500 epochs)
- Try larger batch size if possible
- Adjust temperature parameter (0.1-0.5)
- Ensure encoder isn't frozen during pretraining
- Check that augmentations are being applied correctly

**Q: NT-Xent loss becomes NaN**
- Reduce learning rate
- Check for overflows in similarity computation (add numerical stability)
- Verify normalization is applied correctly

## References

- [SimCLR Paper](https://arxiv.org/abs/2002.05709): "A Simple Framework for Contrastive Learning of Visual Representations"
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This project is for educational and research purposes.

## Author

Claude Code Implementation - Self-Supervised Learning Project
