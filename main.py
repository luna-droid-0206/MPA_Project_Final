"""
Main CLI for SimCLR Self-Supervised Learning Pipeline

Provides a unified command-line interface to run all components:
- Pretraining (SimCLR)
- Supervised baseline training
- Linear evaluation
- Visualization (t-SNE)
- Full pipeline execution
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import (
    EPOCHS_PRETRAIN, EPOCHS_SUPERVISED, EPOCHS_LINEAR,
    BATCH_SIZE, LEARNING_RATE, SUPERVISED_LR, LINEAR_LR,
    PRETRAINED_CHECKPOINT, SUPERVISED_CHECKPOINT, LINEAR_EVAL_CHECKPOINT,
    CHECKPOINT_DIR
)


def run_command(cmd, description):
    """Run a subprocess command and check for errors."""
    print("\n" + "=" * 60)
    print(f"{description}")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"✗ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"[OK] {description} completed")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="SimCLR Self-Supervised Learning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --all

  # Run only pretraining
  python main.py --pretrain --epochs-pretrain 100

  # Run linear evaluation (requires pretrained encoder)
  python main.py --linear

  # Visualize features
  python main.py --visualize --num-samples 1000

  # Compare all available models
  python main.py --evaluate
        """
    )

    # Stage selection
    parser.add_argument('--pretrain', action='store_true',
                        help='Run SimCLR pretraining')
    parser.add_argument('--supervised', action='store_true',
                        help='Run supervised baseline training')
    parser.add_argument('--linear', action='store_true',
                        help='Run linear evaluation')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate and compare models')
    parser.add_argument('--visualize', action='store_true',
                        help='Run t-SNE visualization')
    parser.add_argument('--all', action='store_true',
                        help='Run all stages in order: pretrain -> supervised -> linear -> evaluate -> visualize')

    # Hyperparameters (allow override)
    parser.add_argument('--epochs-pretrain', type=int, default=EPOCHS_PRETRAIN,
                        help=f'Pretraining epochs (default: {EPOCHS_PRETRAIN})')
    parser.add_argument('--epochs-supervised', type=int, default=EPOCHS_SUPERVISED,
                        help=f'Supervised epochs (default: {EPOCHS_SUPERVISED})')
    parser.add_argument('--epochs-linear', type=int, default=EPOCHS_LINEAR,
                        help=f'Linear eval epochs (default: {EPOCHS_LINEAR})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr-pretrain', type=float, default=LEARNING_RATE,
                        help=f'Pretraining LR (default: {LEARNING_RATE})')
    parser.add_argument('--lr-supervised', type=float, default=SUPERVISED_LR,
                        help=f'Supervised LR (default: {SUPERVISED_LR})')
    parser.add_argument('--lr-linear', type=float, default=LINEAR_LR,
                        help=f'Linear eval LR (default: {LINEAR_LR})')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')

    # Visualization args
    parser.add_argument('--num-samples', type=int, default=2000,
                        help='Number of samples for t-SNE (default: 2000)')
    parser.add_argument('--perplexity', type=float, default=30,
                        help='t-SNE perplexity (default: 30)')

    args = parser.parse_args()

    # Determine which stages to run
    stages = []
    if args.all:
        stages = ['pretrain', 'supervised', 'linear', 'evaluate', 'visualize']
    else:
        if args.pretrain:
            stages.append('pretrain')
        if args.supervised:
            stages.append('supervised')
        if args.linear:
            stages.append('linear')
        if args.evaluate:
            stages.append('evaluate')
        if args.visualize:
            stages.append('visualize')

    if not stages:
        parser.print_help()
        print("\nError: No stage selected. Use --all or specify stages.")
        sys.exit(1)

    print("=" * 60)
    print("SimCLR Pipeline")
    print("=" * 60)
    print(f"Stages to run: {', '.join(stages)}")
    print("=" * 60)

    # Build checkpoints directory if needed
    import os
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Execute stages in order
    for stage in stages:
        cwd = Path(__file__).parent

        if stage == 'pretrain':
            cmd = [
                sys.executable, "train_simclr.py",
                f"--epochs={args.epochs_pretrain}",
                f"--batch-size={args.batch_size}",
                f"--lr={args.lr_pretrain}",
            ]
            if args.no_cuda:
                cmd.append("--no-cuda")
            run_command(cmd, "SimCLR Pretraining")

        elif stage == 'supervised':
            cmd = [
                sys.executable, "train_supervised.py",
                f"--epochs={args.epochs_supervised}",
                f"--batch-size={args.batch_size}",
                f"--lr={args.lr_supervised}",
            ]
            if args.no_cuda:
                cmd.append("--no-cuda")
            run_command(cmd, "Supervised Baseline Training")

        elif stage == 'linear':
            cmd = [
                sys.executable, "train_linear.py",
                f"--epochs={args.epochs_linear}",
                f"--batch-size={args.batch_size}",
                f"--lr={args.lr_linear}",
            ]
            if args.no_cuda:
                cmd.append("--no-cuda")
            run_command(cmd, "Linear Evaluation")

        elif stage == 'evaluate':
            cmd = [sys.executable, "evaluate.py"]
            run_command(cmd, "Evaluation and Comparison")

        elif stage == 'visualize':
            cmd = [
                sys.executable, "visualize.py",
                f"--num-samples={args.num_samples}",
                f"--perplexity={args.perplexity}",
            ]
            if args.no_cuda:
                cmd.append("--no-cuda")
            run_command(cmd, "t-SNE Visualization")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
