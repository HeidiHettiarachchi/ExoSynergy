"""
Utility functions for CRISM hyperspectral mineral segmentation.
Includes model evaluation, visualization, and helper functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional
import os
from tqdm import tqdm

from . import config


def evaluate_model_performance(model: torch.nn.Module, 
                             test_loader: torch.utils.data.DataLoader,
                             device: torch.device = None,
                             class_names: List[str] = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of model performance on test dataset.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary containing evaluation metrics
    """
    device = device or config.DEVICE
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    print("Evaluating model performance...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(test_loader, desc="Evaluating")):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs, dim=1)
            
            # Flatten and collect results
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Filter out background/ignore class for meaningful metrics
    valid_mask = all_targets != config.BACKGROUND_CLASS
    valid_predictions = all_predictions[valid_mask]
    valid_targets = all_targets[valid_mask]
    
    # Calculate metrics
    avg_loss = total_loss / len(test_loader)
    overall_accuracy = accuracy_score(all_targets, all_predictions)
    valid_accuracy = accuracy_score(valid_targets, valid_predictions) if len(valid_targets) > 0 else 0.0
    
    # Unique classes in the data
    unique_classes = np.unique(np.concatenate([all_targets, all_predictions]))
    
    print(f"\n=== Model Performance Evaluation ===")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Valid Class Accuracy (excluding background): {valid_accuracy:.4f}")
    print(f"Total test samples: {len(all_targets)}")
    print(f"Valid samples (non-background): {len(valid_targets)}")
    
    # Detailed classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in unique_classes]
    else:
        # Ensure we have names for all classes present in data
        class_names = class_names[:len(unique_classes)]
        while len(class_names) < len(unique_classes):
            class_names.append(f"Class_{len(class_names)}")
    
    # Only evaluate classes that are actually present in the test data
    # This fixes the confusion matrix mismatch issue
    present_classes = np.unique(np.concatenate([all_targets, all_predictions]))
    present_class_names = []
    
    for cls in present_classes:
        if cls < len(class_names):
            present_class_names.append(class_names[cls])
        else:
            present_class_names.append(f"Class_{cls}")
    
    print(f"\n=== Detailed Classification Report ===")
    print(f"Evaluating {len(present_classes)} classes present in test data:")
    print(f"Present classes: {present_classes.tolist()}")
    
    report = classification_report(all_targets, all_predictions, 
                                 labels=present_classes,
                                 target_names=present_class_names,
                                 zero_division=0)
    print(report)
    
    # Class-wise accuracy
    print(f"\n=== Class Distribution ===")
    unique_targets, target_counts = np.unique(all_targets, return_counts=True)
    unique_preds, pred_counts = np.unique(all_predictions, return_counts=True)
    
    # Only show classes that are actually present to avoid confusion
    for cls_id in present_classes:
        target_count = target_counts[unique_targets == cls_id][0] if cls_id in unique_targets else 0
        pred_count = pred_counts[unique_preds == cls_id][0] if cls_id in unique_preds else 0
        cls_name = present_class_names[np.where(present_classes == cls_id)[0][0]]
        print(f"{cls_name:15} - Target: {target_count:6d}, Predicted: {pred_count:6d}")
    
    # Return metrics dictionary
    metrics = {
        'test_loss': avg_loss,
        'overall_accuracy': overall_accuracy,
        'valid_accuracy': valid_accuracy,
        'total_samples': len(all_targets),
        'valid_samples': len(valid_targets),
        'predictions': all_predictions,
        'targets': all_targets
    }
    
    return metrics


def visualize_inference_results(model: torch.nn.Module,
                               test_loader: torch.utils.data.DataLoader,
                               save_dir: str = None,
                               num_samples: int = 5,
                               device: torch.device = None) -> None:
    """
    Generate side-by-side visualizations of ground truth vs model predictions.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader containing test data
        save_dir: Directory to save visualization files
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    device = device or config.DEVICE
    save_dir = save_dir or config.OUTPUT_DIR
    
    model.eval()
    
    # Create custom colormap for mineral classes
    colors_list = plt.cm.tab10(np.linspace(0, 1, config.NUM_CLASSES))
    cmap = colors.ListedColormap(colors_list)
    
    sample_count = 0
    
    print(f"Generating inference visualizations...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1)
            
            # Process each sample in the batch
            batch_size = data.shape[0]
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                # Get current sample
                img = data[i].cpu()  # Shape: (C, H, W)
                gt_mask = targets[i].cpu().numpy()  # Shape: (H, W)
                pred_mask = predictions[i].cpu().numpy()  # Shape: (H, W)
                
                # Create RGB composite for visualization (use first 3 bands)
                rgb_img = img[:3].permute(1, 2, 0).numpy()  # (H, W, 3)
                rgb_img = np.clip((rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min()), 0, 1)
                
                # Create visualization
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                # RGB composite
                axes[0].imshow(rgb_img)
                axes[0].set_title('RGB Composite\n(Bands 1-3)', fontsize=12)
                axes[0].axis('off')
                
                # Ground truth
                im1 = axes[1].imshow(gt_mask, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES-1)
                axes[1].set_title('Ground Truth', fontsize=12)
                axes[1].axis('off')
                
                # Prediction
                im2 = axes[2].imshow(pred_mask, cmap=cmap, vmin=0, vmax=config.NUM_CLASSES-1)
                axes[2].set_title('Model Prediction', fontsize=12)
                axes[2].axis('off')
                
                # Difference map
                diff_map = np.where(gt_mask == pred_mask, 0, 1)  # 0 for correct, 1 for incorrect
                im3 = axes[3].imshow(diff_map, cmap='RdYlBu_r', vmin=0, vmax=1)
                axes[3].set_title('Prediction Errors\n(Red = Wrong)', fontsize=12)
                axes[3].axis('off')
                
                # Add colorbars
                plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='Class')
                plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label='Class')
                plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04, label='Error')
                
                # Calculate accuracy for this sample
                sample_accuracy = np.mean(gt_mask == pred_mask)
                plt.suptitle(f'Sample {sample_count + 1} - Pixel Accuracy: {sample_accuracy:.3f}', fontsize=14)
                
                plt.tight_layout()
                
                # Save figure
                save_path = os.path.join(save_dir, f'inference_result_{sample_count + 1}.png')
                plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
                plt.close()
                
                print(f"Saved inference visualization: {save_path}")
                
                sample_count += 1
    
    print(f"Generated {sample_count} inference visualizations in {save_dir}")


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray,
                         class_names: List[str] = None,
                         save_path: str = None,
                         normalize: bool = True) -> None:
    """
    Plot confusion matrix for model predictions.
    Fixed to handle only classes present in the data.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions  
        class_names: List of class names
        save_path: Path to save the plot
        normalize: Whether to normalize the matrix
    """
    # Get only classes that are actually present in the data
    present_classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Calculate confusion matrix only for present classes
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)
    
    # Create class names for present classes only
    if class_names is None:
        present_class_names = [f'Class_{i}' for i in present_classes]
    else:
        present_class_names = []
        for cls in present_classes:
            if cls < len(class_names):
                present_class_names.append(class_names[cls])
            else:
                present_class_names.append(f'Class_{cls}')
    
    print(f"📊 Confusion Matrix Info:")
    print(f"   Present classes: {len(present_classes)} ({present_classes.tolist()})")
    print(f"   Matrix shape: {cm.shape}")
    
    if normalize:
        # Handle division by zero for classes with no samples
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm[np.isnan(cm_norm)] = 0.0  # Replace NaN with 0
        cm_display = cm_norm
        title = 'Normalized Confusion Matrix'
        fmt = '.2f'
    else:
        cm_display = cm
        title = 'Confusion Matrix (Raw Counts)'
        fmt = 'd'
    
    # Create plot with appropriate size
    fig_size = max(8, min(16, len(present_classes) * 0.8))
    plt.figure(figsize=(fig_size, fig_size))
    
    im = plt.imshow(cm_display, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Set tick marks and labels
    tick_marks = np.arange(len(present_classes))
    plt.xticks(tick_marks, present_class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, present_class_names)
    
    # Add text annotations
    thresh = cm_display.max() / 2. if cm_display.max() > 0 else 0.5
    for i, j in np.ndindex(cm_display.shape):
        plt.text(j, i, format(cm_display[i, j], fmt),
                ha="center", va="center", fontsize=max(8, min(12, 100//len(present_classes))),
                color="white" if cm_display[i, j] > thresh else "black")
    
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"📄 Confusion matrix saved to {save_path}")
    
    plt.close()  # Close to save memory


def plot_training_history(train_losses: List[float],
                         val_losses: List[float],
                         train_accuracies: List[float] = None,
                         val_accuracies: List[float] = None,
                         save_path: str = None) -> None:
    """
    Plot training and validation losses/accuracies over epochs.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accuracies: List of training accuracies (optional)
        val_accuracies: List of validation accuracies (optional)
        save_path: Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Determine number of subplots
    n_plots = 1 + (1 if train_accuracies is not None else 0)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies if provided
    if train_accuracies is not None and n_plots > 1:
        axes[1].plot(epochs[:len(train_accuracies)], train_accuracies, 'b-', 
                    label='Training Accuracy', linewidth=2)
        if val_accuracies is not None:
            axes[1].plot(epochs[:len(val_accuracies)], val_accuracies, 'r-', 
                        label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def calculate_class_weights(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        dataloader: DataLoader to compute weights from
        
    Returns:
        Tensor of class weights
    """
    class_counts = torch.zeros(config.NUM_CLASSES)
    
    print("Calculating class weights...")
    for _, targets in tqdm(dataloader, desc="Computing class distribution"):
        # Flatten targets and count each class
        targets_flat = targets.flatten()
        for class_id in range(config.NUM_CLASSES):
            class_counts[class_id] += (targets_flat == class_id).sum().item()
    
    # Calculate inverse frequency weights
    total_samples = class_counts.sum()
    weights = total_samples / (config.NUM_CLASSES * class_counts)
    
    # Handle zero counts
    weights[class_counts == 0] = 0.0
    
    print(f"Class distribution: {class_counts.numpy()}")
    print(f"Class weights: {weights.numpy()}")
    
    return weights


def save_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         save_path: str,
                         additional_info: Dict = None) -> None:
    """
    Save comprehensive model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        save_path: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'num_bands': config.NUM_BANDS,
            'num_classes': config.NUM_CLASSES,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
        }
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         checkpoint_path: str) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    
    return checkpoint


# Mineral class names (update based on your specific dataset)
MINERAL_CLASSES = [
    "Background",
    "Olivine", 
    "Pyroxene",
    "Feldspar",
    "Quartz",
    "Clay_Minerals",
    "Carbonate",
    "Iron_Oxide",
    "Hydrated_Minerals"
]


def get_class_names() -> List[str]:
    """Return list of mineral class names."""
    return MINERAL_CLASSES[:config.NUM_CLASSES]


if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test colormap creation
    import matplotlib.pyplot as plt
    
    # Create sample data for testing
    dummy_predictions = np.random.randint(0, config.NUM_CLASSES, (100, 100))
    dummy_targets = np.random.randint(0, config.NUM_CLASSES, (100, 100))
    
    # Test confusion matrix
    plot_confusion_matrix(dummy_targets.flatten(), 
                         dummy_predictions.flatten(),
                         class_names=get_class_names()[:config.NUM_CLASSES])
    
    print("Utility functions test completed!")
