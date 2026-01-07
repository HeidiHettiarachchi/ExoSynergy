#!/usr/bin/env python3
"""
Model Training Pipeline for CRISM Hyperspectral Mineral Segmentation.

This script implements the complete training pipeline for the U-Net model:
- Loads and preprocesses CRISM data
- Initializes the U-Net architecture  
- Runs training loop with validation
- Implements early stopping and model checkpointing
- Saves the best model for inference

Usage:
    python pipelines/model_pipeline.py
"""

import sys
import os
import time
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src import config
from src.dataset import get_dataloaders
from src.model import create_model
from src.utils import (
    calculate_class_weights, save_model_checkpoint, 
    plot_training_history, get_class_names
)
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.Training")


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss stops improving.
    """
    
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                
    def save_checkpoint(self, model):
        """Save the best model weights."""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def calculate_accuracy(outputs, targets, ignore_index=None):
    """Calculate pixel-level accuracy with proper handling for multi-class classification."""
    predictions = torch.argmax(outputs, dim=1)
    
    # Calculate overall accuracy
    overall_correct = (predictions == targets).sum().item()
    total = targets.numel()
    overall_acc = overall_correct / total if total > 0 else 0.0
    
    # Calculate accuracy excluding background (class 0) for meaningful mineral assessment
    if ignore_index is not None:
        non_bg_mask = targets != ignore_index
        if non_bg_mask.sum() > 0:
            non_bg_predictions = predictions[non_bg_mask]
            non_bg_targets = targets[non_bg_mask]
            non_bg_correct = (non_bg_predictions == non_bg_targets).sum().item()
            non_bg_total = non_bg_targets.numel()
            # Weight the accuracy: 70% overall + 30% mineral-only for balanced assessment
            mineral_acc = non_bg_correct / non_bg_total if non_bg_total > 0 else 0.0
            return overall_acc * 0.7 + mineral_acc * 0.3
    
    return overall_acc


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate metrics
        batch_loss = loss.item()
        batch_acc = calculate_accuracy(outputs, targets, ignore_index=config.BACKGROUND_CLASS)
        
        running_loss += batch_loss
        running_accuracy += batch_acc
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{batch_loss:.4f}',
            'Acc': f'{batch_acc:.3f}',
            'Avg_Loss': f'{running_loss/num_batches:.4f}',
            'Avg_Acc': f'{running_accuracy/num_batches:.3f}'
        })
    
    avg_loss = running_loss / num_batches
    avg_accuracy = running_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate_epoch(model, val_loader, criterion, device, epoch):
    """
    Validate the model for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]')
    
    with torch.no_grad():
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Calculate metrics
            batch_loss = loss.item()
            batch_acc = calculate_accuracy(outputs, targets, ignore_index=config.BACKGROUND_CLASS)
            
            running_loss += batch_loss
            running_accuracy += batch_acc
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{batch_loss:.4f}',
                'Acc': f'{batch_acc:.3f}',
                'Avg_Loss': f'{running_loss/num_batches:.4f}',
                'Avg_Acc': f'{running_accuracy/num_batches:.3f}'
            })
    
    avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
    avg_accuracy = running_accuracy / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_accuracy


def main():
    """
    Main training pipeline execution.
    """
    print("=" * 70)
    print("CRISM Hyperspectral Mineral Segmentation - Model Training")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
    
    try:
        # 1. Load data
        print(f"\n1. Loading and preparing data...")
        print(f"   - Batch size: {config.BATCH_SIZE}")
        print(f"   - Train/Val/Test split: {config.TRAIN_RATIO:.1f}/{config.VAL_RATIO:.1f}/{config.TEST_RATIO:.1f}")
        
        train_loader, val_loader, test_loader, label_mapping, original_labels = get_dataloaders()
        
        logger.info(f"🏷️  Label Information:")
        logger.info(f"   Original mineral labels: {original_labels[:5]}..." if len(original_labels) > 5 else f"   Original mineral labels: {original_labels}")
        logger.info(f"   Continuous mapping range: 0 to {len(original_labels)-1}")
        logger.info(f"   Total classes in dataset: {len(original_labels)}")
        
        logger.info(f"📦 DataLoader Information:")
        logger.info(f"   Training batches: {len(train_loader)} (samples: {len(train_loader.dataset)})")
        logger.info(f"   Validation batches: {len(val_loader)} (samples: {len(val_loader.dataset)})")
        logger.info(f"   Test batches: {len(test_loader)} (samples: {len(test_loader.dataset)})")
        
        # 2. Create model
        logger.info(f"\n🧠 Initializing U-Net model...")
        model = create_model()
        
        # Log model details using logger
        logger.log_model_info(model)
        
        # 3. Setup training components
        print(f"\n3. Setting up training components...")
        
        # Calculate class weights if needed
        if config.CLASS_WEIGHTS is None:
            print("   - Calculating class weights for balanced training...")
            class_weights = calculate_class_weights(train_loader)
            class_weights = class_weights.to(config.DEVICE)
        else:
            class_weights = config.CLASS_WEIGHTS
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            betas=(config.BETA1, config.BETA2),
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.PATIENCE, 
            min_delta=config.MIN_DELTA
        )
        
        print(f"   ✓ Loss function: CrossEntropyLoss with class weights")
        print(f"   ✓ Optimizer: Adam (lr={config.LEARNING_RATE})")
        print(f"   ✓ Scheduler: ReduceLROnPlateau")
        print(f"   ✓ Early stopping: patience={config.PATIENCE}")
        
        # 4. Training loop
        print(f"\n4. Starting training...")
        print(f"   - Device: {config.DEVICE}")
        print(f"   - Maximum epochs: {config.NUM_EPOCHS}")
        
        # Training history tracking
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(config.NUM_EPOCHS):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, config.DEVICE, epoch
            )
            
            # Validation phase
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, config.DEVICE, epoch
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.3f}")
            print(f"  LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  🎉 New best validation loss! Saving model...")
                
                save_model_checkpoint(
                    model, optimizer, epoch, val_loss, config.MODEL_SAVE_PATH,
                    additional_info={
                        'train_loss': train_loss,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'epoch': epoch,
                        'class_names': get_class_names()
                    }
                )
            
            # Early stopping check
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered after epoch {epoch+1}")
                print(f"   No improvement in validation loss for {config.PATIENCE} epochs")
                break
            
            print("-" * 70)
        
        total_time = time.time() - start_time
        
        # 5. Training completed
        print(f"\n5. Training completed!")
        print(f"   - Total training time: {total_time/60:.1f} minutes")
        print(f"   - Epochs completed: {len(train_losses)}")
        print(f"   - Best validation loss: {best_val_loss:.4f}")
        print(f"   - Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"   - Model saved to: {config.MODEL_SAVE_PATH}")
        
        # 6. Generate training plots
        print(f"\n6. Generating training history plots...")
        plot_save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
        plot_training_history(
            train_losses, val_losses, 
            train_accuracies, val_accuracies,
            save_path=plot_save_path
        )
        
        # 7. Final model evaluation on validation set
        print(f"\n7. Final model evaluation...")
        
        # Load best model for final evaluation
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        final_val_loss, final_val_acc = validate_epoch(
            model, val_loader, criterion, config.DEVICE, epoch=-1
        )
        
        print(f"   - Final validation loss: {final_val_loss:.4f}")
        print(f"   - Final validation accuracy: {final_val_acc:.3f}")
        
        # 8. Save training summary
        summary_path = os.path.join(config.OUTPUT_DIR, 'training_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"CRISM Hyperspectral Mineral Segmentation - Training Summary\n")
            f.write(f"=" * 60 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  - Model: U-Net\n")
            f.write(f"  - Input bands: {config.NUM_BANDS}\n")
            f.write(f"  - Output classes: {config.NUM_CLASSES}\n")
            f.write(f"  - Batch size: {config.BATCH_SIZE}\n")
            f.write(f"  - Learning rate: {config.LEARNING_RATE}\n")
            f.write(f"  - Max epochs: {config.NUM_EPOCHS}\n")
            f.write(f"  - Device: {config.DEVICE}\n\n")
            
            f.write(f"Training Results:\n")
            f.write(f"  - Epochs completed: {len(train_losses)}\n")
            f.write(f"  - Total training time: {total_time/60:.1f} minutes\n")
            f.write(f"  - Best validation loss: {best_val_loss:.4f}\n")
            f.write(f"  - Final validation accuracy: {final_val_acc:.3f}\n")
            f.write(f"  - Early stopping: {'Yes' if early_stopping.early_stop else 'No'}\n\n")
            
            f.write(f"Class Names:\n")
            for i, name in enumerate(get_class_names()):
                f.write(f"  {i}: {name}\n")
        
        print(f"   - Training summary saved: {summary_path}")
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"   Next step: Run 'python pipelines/inference_pipeline.py' to evaluate the model")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Training interrupted by user")
        print(f"   Current model state saved (if any)")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR in training pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
