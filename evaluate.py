import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from models.pointnet import PointNetClassifier
from dataset import ModelNet40Dataset
from utils import get_classification_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard 40 category names in alphabetical order
MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
    'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
    'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
    'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
    'wardrobe', 'xbox'
]

def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """
    Evaluates the PointNet classifier on a given dataloader.

    Args:
        model (nn.Module): The PointNetClassifier model.
        dataloader (DataLoader): PyTorch DataLoader for the evaluation set.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (mean_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for points, targets in dataloader:
            points = points.to(device)
            targets = targets.to(device)

            logits, _, trans_feat = model(points)

            loss = get_classification_loss(logits, targets, trans_feat)
            total_loss += loss.item() * points.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += points.size(0)

    mean_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return mean_loss, accuracy


def evaluate_per_class(model: nn.Module, dataloader: DataLoader, device: torch.device, class_names: list):
    """
    Computes accuracy for each individual class.

    Args:
        model (nn.Module): The PointNetClassifier model.
        dataloader (DataLoader): PyTorch DataLoader for the evaluation set.
        device (torch.device): Device to run evaluation on.
        class_names (list): List of strings representing class names.

    Returns:
        dict: A dictionary mapping class name to accuracy (float).
        list: True labels across the whole dataset.
        list: Predicted labels across the whole dataset.
    """
    model.eval()
    num_classes = len(class_names)
    class_correct = [0.0] * num_classes
    class_total = [0.0] * num_classes

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for points, targets in dataloader:
            points = points.to(device)
            targets = targets.to(device)

            logits, _, _ = model(points)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            for pred, target in zip(preds, targets):
                if pred == target:
                    class_correct[target.item()] += 1
                class_total[target.item()] += 1

    per_class_acc = {}
    for i in range(num_classes):
        name = class_names[i]
        if class_total[i] > 0:
            per_class_acc[name] = class_correct[i] / class_total[i]
        else:
            per_class_acc[name] = 0.0

    return per_class_acc, all_targets, all_preds


def plot_confusion_matrix(y_true: list, y_pred: list, class_names: list, output_path: str):
    """
    Plots the confusion matrix using matplotlib and saves it as an image.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) # Replace NaNs (e.g. division by zero for unrepresented classes)

    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Set tick marks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    ax.set_title('Normalized Confusion Matrix for PointNet Classifier', fontsize=16)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PointNet Classifier on ModelNet40")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where dataset is stored")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num_points", type=int, default=1024, help="Number of points to sample")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--mock", action="store_true", help="Run with mock data and model weights for testing")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Running evaluation on device: {device}")

    # Load dataset
    dataset = ModelNet40Dataset(root_dir=args.data_dir, split="test", num_points=args.num_points, augment=False, _mock=args.mock)
    # Use num_workers=0 when mocking to avoid multiprocessing overhead
    num_workers = 0 if args.mock else 2
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model
    model = PointNetClassifier(num_classes=len(MODELNET40_CLASSES))

    # Mock or load checkpoint
    if args.mock:
        logger.info("Mock mode enabled: Using random model weights for evaluation report.")
        # Ensure results directory exists
        os.makedirs("results", exist_ok=True)
    else:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint path does not exist: {args.checkpoint}. Please train the model first.")
            return
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from: {args.checkpoint}")

    model.to(device)

    # Run overall evaluation
    mean_loss, accuracy = evaluate(model, dataloader, device)
    logger.info(f"Overall Test Loss: {mean_loss:.4f}")
    logger.info(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

    # Run per-class evaluation
    per_class_acc, y_true, y_pred = evaluate_per_class(model, dataloader, device, MODELNET40_CLASSES)

    print("\n--- Per-Class Accuracy Report ---")
    for cls_name, acc in per_class_acc.items():
        print(f"{cls_name:<15}: {acc * 100:.2f}%")
    print("---------------------------------\n")

    # Plot and save confusion matrix
    plot_confusion_matrix(y_true, y_pred, MODELNET40_CLASSES, "results/confusion_matrix.png")


if __name__ == "__main__":
    main()
