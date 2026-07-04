"""
PointNet Training Script for Classification on ModelNet40.

This script implements the training loop for PointNet, including parsing hyperparameters,
setting up data loading, initializing the optimizer and StepLR learning rate scheduler,
running epochs, evaluating the model at each epoch, checkpointing the best and latest
model, and logging metrics to a CSV file.

How to Run:
1. Locally:
   python train.py --epochs 100 --batch_size 32 --lr 0.001 --data_dir ./data

2. In Google Colab:
   !git clone <repo_url>
   %cd <repo_name>
   !pip install -r requirements.txt
   !python train.py --epochs 250 --data_dir /content/data --batch_size 32
"""

import os
import argparse
import csv
import logging
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from models.pointnet import PointNetClassifier
from dataset import ModelNet40Dataset
from utils import get_classification_loss
from evaluate import evaluate

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Parse configuration
    config = Config()
    
    parser = argparse.ArgumentParser(description="Train PointNet Classifier on ModelNet40")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--num_points", type=int, default=config.num_points, help="Number of points to sample per model")
    parser.add_argument("--device", type=str, default=config.device, help="Device to use (cpu or cuda)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory where dataset is stored")
    parser.add_argument("--mock", action="store_true", help="Use a small mock dataset for quick end-to-end testing")
    parser.add_argument("--no_feature_transform", action="store_true", help="Disable feature transform network")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Training on device: {device}")
    
    # 2. Set up Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_dataset = ModelNet40Dataset(
        root_dir=args.data_dir, 
        split="train", 
        num_points=args.num_points, 
        augment=True, 
        _mock=args.mock
    )
    test_dataset = ModelNet40Dataset(
        root_dir=args.data_dir, 
        split="test", 
        num_points=args.num_points, 
        augment=False, 
        _mock=args.mock
    )
    
    # Use num_workers=0 when mocking to avoid multiprocessing overhead for small mock sets
    num_workers = 0 if args.mock else 2
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # 3. Initialize PointNetClassifier
    feature_transform_enabled = not args.no_feature_transform
    model = PointNetClassifier(num_classes=config.num_classes, feature_transform=feature_transform_enabled)
    model.to(device)
    
    # 4. Initialize Optimizer and Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Decay learning rate by 0.5 every 20 epochs (based on standard PointNet scheduling)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Create output directories for checkpoints and logging results
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize the training log CSV file
    log_file_path = "results/training_log.csv"
    with open(log_file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
        
    best_test_acc = 0.0
    logger.info("Starting training loop...")
    
    # 5. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for points, targets in pbar:
            points = points.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: logits, input_transform, feature_transform
            logits, trans, trans_feat = model(points)
            
            # Compute cross-entropy loss + feature transform regularization loss
            loss = get_classification_loss(logits, targets, trans_feat)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * points.size(0)
            
            # Compute training metrics
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += points.size(0)
            
            # Update tqdm progress bar
            running_loss = total_loss / total
            running_acc = correct / total
            pbar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{100 * running_acc:.2f}%")
            
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # 6. Evaluation after each epoch
        test_loss, test_acc = evaluate(model, test_loader, device)
        
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%"
        )
        
        # 8. Log metrics to CSV
        with open(log_file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
            
        # 7. Save best and last checkpoints
        torch.save(model.state_dict(), "checkpoints/last_model.pth")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            logger.info(f"Saving new best model checkpoint (Test Acc: {best_test_acc*100:.2f}%)")

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
