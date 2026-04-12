"""
Comprehensive Evaluation and Comparison

Loads trained models (supervised and SimCLR-based) and evaluates
them on the CIFAR-10 test set, producing comparison metrics.
"""

import os
import sys
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent))

from config import (
    DEVICE, BATCH_SIZE, CHECKPOINT_DIR,
    SUPERVISED_CHECKPOINT, LINEAR_EVAL_CHECKPOINT,
    NUM_CLASSES
)
from models.encoder import Encoder
from data.dataset import get_supervised_dataloaders


def load_model(checkpoint_path, model_type='supervised'):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path (str): Path to checkpoint.
        model_type (str): 'supervised' or 'linear_eval'.

    Returns:
        nn.Module: Loaded model.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    if model_type == 'supervised':
        from train_supervised import SupervisedClassifier
        model = SupervisedClassifier(pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_type == 'linear_eval':
        from train_linear import LinearClassifier
        encoder = Encoder(pretrained=False)
        encoder_path = checkpoint_path.replace('.pth', '_encoder.pth').replace('linear_eval', 'simclr_pretrained')
        if os.path.exists(encoder_path):
            encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE, weights_only=False))
        model = LinearClassifier(encoder, num_classes=NUM_CLASSES)
        # Load the classifier part specifically
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.to(DEVICE)
    model.eval()

    print(f"[OK] Loaded {model_type} model from {checkpoint_path}")
    return model


def evaluate_model(model, dataloader):
    """
    Evaluate model accuracy.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Test dataloader.

    Returns:
        float: Test accuracy (0-100).
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

    accuracy = 100 * correct / total
    return accuracy


def evaluate_all_models():
    """
    Evaluate all available trained models.

    Returns:
        dict: Results with accuracies for each model.
    """
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)

    # Load test dataloader
    _, test_loader = get_supervised_dataloaders(
        batch_size=BATCH_SIZE,
        train_transform=None,
        test_transform=None
    )
    print(f"\nTest dataset size: {len(test_loader.dataset)}")

    results = {}

    # Evaluate Supervised Baseline
    print("\n--- Supervised Baseline ---")
    try:
        supervised_model = load_model(SUPERVISED_CHECKPOINT, model_type='supervised')
        supervised_acc = evaluate_model(supervised_model, test_loader)
        results['Supervised'] = {
            'accuracy': supervised_acc,
            'checkpoint': SUPERVISED_CHECKPOINT
        }
        print(f"Supervised Test Accuracy: {supervised_acc:.2f}%")
    except FileNotFoundError as e:
        print(f"Supervised model not found: {e}")
        results['Supervised'] = None

    # Evaluate SimCLR + Linear
    print("\n--- SimCLR + Linear Evaluation ---")
    try:
        linear_model = load_model(LINEAR_EVAL_CHECKPOINT, model_type='linear_eval')
        linear_acc = evaluate_model(linear_model, test_loader)
        results['SimCLR_Linear'] = {
            'accuracy': linear_acc,
            'checkpoint': LINEAR_EVAL_CHECKPOINT
        }
        print(f"SimCLR Linear Eval Test Accuracy: {linear_acc:.2f}%")
    except FileNotFoundError as e:
        print(f"SimCLR linear evaluation model not found: {e}")
        results['SimCLR_Linear'] = None

    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    for model_name, info in results.items():
        if info is not None:
            print(f"{model_name}: {info['accuracy']:.2f}%")
        else:
            print(f"{model_name}: NOT AVAILABLE")

    print("=" * 60)

    return results


def save_results(results, output_path=f"{CHECKPOINT_DIR}/evaluation_results.json"):
    """
    Save evaluation results to JSON file.

    Args:
        results (dict): Results dictionary.
        output_path (str): Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to JSON-serializable format
    serializable_results = {}
    for model_name, info in results.items():
        if info is not None:
            serializable_results[model_name] = {
                'accuracy': float(info['accuracy']),
                'checkpoint': info['checkpoint']
            }
        else:
            serializable_results[model_name] = None

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main evaluation function."""
    results = evaluate_all_models()
    save_results(results)

    # Print comparison if both available
    if results.get('Supervised') and results.get('SimCLR_Linear'):
        sup_acc = results['Supervised']['accuracy']
        simclr_acc = results['SimCLR_Linear']['accuracy']
        gap = simclr_acc - sup_acc

        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        print(f"Supervised:  {sup_acc:.2f}%")
        print(f"SimCLR:      {simclr_acc:.2f}%")
        print(f"Difference:  {gap:+.2f}%")
        print("=" * 60)

        if gap > 0:
            print("[OK] SimCLR outperforms supervised baseline!")
        elif gap < 0:
            print("Note: SimCLR is below supervised baseline.")
        else:
            print("→ Both methods perform similarly.")


if __name__ == "__main__":
    main()
