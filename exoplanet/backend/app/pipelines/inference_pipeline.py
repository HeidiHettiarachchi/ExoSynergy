#!/usr/bin/env python3
"""
Inference Pipeline for CRISM Hyperspectral Mineral Segmentation.

This script performs comprehensive evaluation of the trained U-Net model:
- Loads the trained model from checkpoint
- Evaluates performance on test dataset
- Generates quantitative metrics and reports
- Creates qualitative visualizations comparing predictions vs ground truth
- Produces confusion matrices and analysis plots

Usage:
    python pipelines/inference_pipeline.py
"""

import sys
import os
from datetime import datetime

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import config
from src.dataset import get_dataloaders
from src.model import create_model
from src.utils import (
    evaluate_model_performance, visualize_inference_results,
    plot_confusion_matrix, get_class_names
)
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.Inference")


def analyze_per_class_performance(predictions, targets, class_names=None):
    """
    Analyze performance metrics for each class individually.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        class_names: List of class names
        
    Returns:
        Dictionary with per-class metrics
    """
    if class_names is None:
        class_names = get_class_names()
    
    unique_classes = np.unique(np.concatenate([predictions, targets]))
    per_class_metrics = {}
    
    print(f"\n=== Per-Class Performance Analysis ===")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 65)
    
    for class_id in unique_classes:
        # Calculate metrics for this class
        true_positive = np.sum((predictions == class_id) & (targets == class_id))
        false_positive = np.sum((predictions == class_id) & (targets != class_id))
        false_negative = np.sum((predictions != class_id) & (targets == class_id))
        
        support = np.sum(targets == class_id)
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        
        per_class_metrics[class_id] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support,
            'name': class_name
        }
        
        print(f"{class_name:<15} {precision:<10.3f} {recall:<10.3f} {f1_score:<10.3f} {support:<10d}")
    
    return per_class_metrics


def generate_detailed_analysis_plots(predictions, targets, save_dir):
    """
    Generate detailed analysis plots for model performance.
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels  
        save_dir: Directory to save plots
    """
    class_names = get_class_names()
    
    # 1. Confusion Matrix (Normalized)
    plt.figure(figsize=(12, 10))
    plot_confusion_matrix(
        targets, predictions, 
        class_names=class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix_normalized.png'),
        normalize=True
    )
    
    # 2. Confusion Matrix (Raw Counts)  
    plt.figure(figsize=(12, 10))
    plot_confusion_matrix(
        targets, predictions,
        class_names=class_names, 
        save_path=os.path.join(save_dir, 'confusion_matrix_counts.png'),
        normalize=False
    )
    
    # 3. Class Distribution Comparison
    plt.figure(figsize=(15, 5))
    
    unique_classes = np.unique(np.concatenate([predictions, targets]))
    target_counts = [np.sum(targets == cls) for cls in unique_classes]
    pred_counts = [np.sum(predictions == cls) for cls in unique_classes]
    
    x = np.arange(len(unique_classes))
    width = 0.35
    
    plt.subplot(1, 3, 1)
    plt.bar(x - width/2, target_counts, width, label='Ground Truth', alpha=0.8)
    plt.bar(x + width/2, pred_counts, width, label='Predictions', alpha=0.8)
    plt.xlabel('Mineral Class')
    plt.ylabel('Number of Pixels')
    plt.title('Class Distribution Comparison')
    plt.xticks(x, [class_names[i] if i < len(class_names) else f'C{i}' for i in unique_classes], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Per-class Accuracy
    plt.subplot(1, 3, 2)
    accuracies = []
    for cls in unique_classes:
        cls_mask = targets == cls
        if np.sum(cls_mask) > 0:
            cls_acc = np.mean(predictions[cls_mask] == targets[cls_mask])
        else:
            cls_acc = 0.0
        accuracies.append(cls_acc)
    
    bars = plt.bar(x, accuracies, alpha=0.8, color='skyblue')
    plt.xlabel('Mineral Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(x, [class_names[i] if i < len(class_names) else f'C{i}' for i in unique_classes], rotation=45)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 5. Prediction Confidence Distribution
    plt.subplot(1, 3, 3)
    # This would require confidence scores from the model
    # For now, show a placeholder
    plt.text(0.5, 0.5, 'Confidence Distribution\n(Requires softmax outputs)', 
             ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    plt.title('Prediction Confidence')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'detailed_analysis.png')
    plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"   - Detailed analysis plots saved: {save_path}")


def test_model_on_individual_samples(model, test_loader, device, num_samples=3):
    """
    Test model on individual samples and show detailed predictions.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on
        num_samples: Number of samples to analyze in detail
    """
    model.eval()
    class_names = get_class_names()
    
    print(f"\n=== Individual Sample Analysis ===")
    
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predictions = torch.argmax(outputs, dim=1)
            
            # Get confidence scores (softmax probabilities)
            probs = torch.softmax(outputs, dim=1)
            
            batch_size = data.shape[0]
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                
                img = data[i].cpu()
                gt_mask = targets[i].cpu().numpy()
                pred_mask = predictions[i].cpu().numpy()
                confidence = probs[i].cpu().numpy()
                
                # Calculate sample statistics
                sample_acc = np.mean(gt_mask == pred_mask)
                unique_gt = np.unique(gt_mask)
                unique_pred = np.unique(pred_mask)
                avg_confidence = np.mean(np.max(confidence, axis=0))
                
                print(f"\nSample {sample_count + 1}:")
                print(f"  Image shape: {img.shape}")
                print(f"  Pixel accuracy: {sample_acc:.3f}")
                print(f"  GT classes: {unique_gt}")
                print(f"  Predicted classes: {unique_pred}")
                print(f"  Average confidence: {avg_confidence:.3f}")
                
                # Per-class breakdown
                print("  Per-class pixel counts:")
                for cls in np.unique(np.concatenate([unique_gt, unique_pred])):
                    gt_count = np.sum(gt_mask == cls)
                    pred_count = np.sum(pred_mask == cls)
                    cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                    print(f"    {cls_name}: GT={gt_count}, Pred={pred_count}")
                
                sample_count += 1


def main():
    """
    Main inference pipeline execution.
    """
    print("=" * 70)
    print("CRISM Hyperspectral Mineral Segmentation - Model Inference")
    print("=" * 70)
    
    try:
        # 1. Check if trained model exists
        print(f"\n1. Checking for trained model...")
        if not os.path.exists(config.MODEL_SAVE_PATH):
            print(f"   ❌ ERROR: No trained model found at {config.MODEL_SAVE_PATH}")
            print(f"   Please run 'python pipelines/model_pipeline.py' first to train the model.")
            sys.exit(1)
        
        print(f"   ✓ Found trained model: {config.MODEL_SAVE_PATH}")
        
        # 2. Load test data
        print(f"\n2. Loading test dataset...")
        _, _, test_loader, label_mapping, original_labels = get_dataloaders()
        
        print(f"   - Label mapping: {label_mapping}")
        print(f"   - Original labels: {original_labels}")
        print(f"   ✓ Test batches: {len(test_loader)}")
        
        # 3. Load trained model
        print(f"\n3. Loading trained model...")
        model = create_model()
        
        checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Print model info from checkpoint
        if 'epoch' in checkpoint:
            print(f"   ✓ Model trained for {checkpoint['epoch'] + 1} epochs")
        if 'val_accuracy' in checkpoint:
            print(f"   ✓ Best validation accuracy: {checkpoint['val_accuracy']:.3f}")
        
        print(f"   ✓ Model loaded successfully")
        
        # 4. Comprehensive model evaluation
        print(f"\n4. Evaluating model performance on test set...")
        
        # Create proper class names based on actual labels in the data
        # Only create names for classes that are present to fix confusion matrix issues
        actual_class_names = [f"Mineral_{original_labels[i]}" for i in range(len(original_labels))]
        
        logger.info(f"📊 Evaluation Setup:")
        logger.info(f"   Test dataset size: {len(test_loader.dataset)} samples")
        logger.info(f"   Total possible classes: {len(original_labels)}")
        logger.info(f"   Class naming: Original labels → Mineral_[ID] format")
        
        metrics = evaluate_model_performance(
            model, test_loader, device=config.DEVICE, 
            class_names=actual_class_names
        )
        
        # 5. Per-class analysis
        print(f"\n5. Analyzing per-class performance...")
        per_class_metrics = analyze_per_class_performance(
            metrics['predictions'], metrics['targets'], 
            class_names=actual_class_names
        )
        
        # 6. Generate detailed analysis plots
        print(f"\n6. Generating detailed analysis plots...")
        generate_detailed_analysis_plots(
            metrics['predictions'], metrics['targets'], config.OUTPUT_DIR
        )
        
        # 7. Generate inference visualizations
        print(f"\n7. Generating inference visualizations...")
        visualize_inference_results(
            model, test_loader, save_dir=config.OUTPUT_DIR, num_samples=8
        )
        
        # 8. Individual sample analysis
        print(f"\n8. Analyzing individual samples...")
        test_model_on_individual_samples(model, test_loader, config.DEVICE)
        
        # 9. Save comprehensive evaluation report
        print(f"\n9. Generating evaluation report...")
        report_path = os.path.join(config.OUTPUT_DIR, 'evaluation_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"CRISM Hyperspectral Mineral Segmentation - Evaluation Report\n")
            f.write(f"=" * 70 + "\n\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model info
            f.write(f"Model Information:\n")
            f.write(f"  - Architecture: U-Net\n")
            f.write(f"  - Input bands: {config.NUM_BANDS}\n")  
            f.write(f"  - Output classes: {config.NUM_CLASSES}\n")
            f.write(f"  - Model path: {config.MODEL_SAVE_PATH}\n")
            if 'epoch' in checkpoint:
                f.write(f"  - Training epochs: {checkpoint['epoch'] + 1}\n")
            f.write(f"  - Device: {config.DEVICE}\n\n")
            
            # Overall performance
            f.write(f"Overall Performance:\n")
            f.write(f"  - Test Loss: {metrics['test_loss']:.4f}\n")
            f.write(f"  - Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
            f.write(f"  - Valid Class Accuracy: {metrics['valid_accuracy']:.4f}\n")
            f.write(f"  - Total test samples: {metrics['total_samples']:,}\n")
            f.write(f"  - Valid samples (non-background): {metrics['valid_samples']:,}\n\n")
            
            # Per-class performance
            f.write(f"Per-Class Performance:\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 65 + "\n")
            
            for class_id, class_metrics in per_class_metrics.items():
                f.write(f"{class_metrics['name']:<15} ")
                f.write(f"{class_metrics['precision']:<10.3f} ")
                f.write(f"{class_metrics['recall']:<10.3f} ")
                f.write(f"{class_metrics['f1_score']:<10.3f} ")
                f.write(f"{class_metrics['support']:<10d}\n")
            
            f.write(f"\nClass Names:\n")
            for i, name in enumerate(get_class_names()):
                f.write(f"  {i}: {name}\n")
            
            f.write(f"\nGenerated Files:\n")
            f.write(f"  - Confusion matrices: confusion_matrix_*.png\n")
            f.write(f"  - Detailed analysis: detailed_analysis.png\n")
            f.write(f"  - Inference visualizations: inference_result_*.png\n")
            f.write(f"  - This report: evaluation_report.txt\n")
        
        print(f"   ✓ Evaluation report saved: {report_path}")
        
        # 10. Summary and recommendations
        print(f"\n10. Evaluation Summary:")
        print(f"   - Overall test accuracy: {metrics['overall_accuracy']:.1%}")
        print(f"   - Valid class accuracy: {metrics['valid_accuracy']:.1%}")
        print(f"   - Test loss: {metrics['test_loss']:.4f}")
        
        # Performance assessment
        if metrics['overall_accuracy'] > 0.9:
            performance_level = "Excellent"
        elif metrics['overall_accuracy'] > 0.8:
            performance_level = "Good"
        elif metrics['overall_accuracy'] > 0.7:
            performance_level = "Fair"
        else:
            performance_level = "Needs Improvement"
        
        print(f"   - Performance level: {performance_level}")
        
        # Find best and worst performing classes
        if per_class_metrics:
            best_class = max(per_class_metrics.items(), key=lambda x: x[1]['f1_score'])
            worst_class = min(per_class_metrics.items(), key=lambda x: x[1]['f1_score'])
            
            print(f"   - Best performing class: {best_class[1]['name']} (F1: {best_class[1]['f1_score']:.3f})")
            print(f"   - Worst performing class: {worst_class[1]['name']} (F1: {worst_class[1]['f1_score']:.3f})")
        
        print(f"\n🎉 Inference pipeline completed successfully!")
        print(f"   - All results saved to: {config.OUTPUT_DIR}")
        print(f"   - Check the evaluation report for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ ERROR in inference pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
