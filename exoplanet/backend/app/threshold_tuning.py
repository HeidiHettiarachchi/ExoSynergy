#!/usr/bin/env python3
"""
Threshold Tuning for Planetary Gatekeeper
==========================================
Loop over thresholds 0.5 to 0.9 in steps of 0.05,
print precision and recall for each using validation_images list

This script helps you find the optimal CLIP threshold for your use case.
It evaluates gatekeeper performance across different threshold values using
a labeled validation set.

Usage:
    python threshold_tuning.py --planetary_dir ./planetary_images --non_planetary_dir ./non_planetary_images
    
    # Or with specific threshold range
    python threshold_tuning.py --planetary_dir ./data/mars --non_planetary_dir ./data/earth --min_threshold 0.4 --max_threshold 0.9 --step 0.05

Requirements:
    - Directory with planetary/geological images (ground truth positive)
    - Directory with non-planetary images (ground truth negative)
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from planetary_gatekeeper import load_gatekeeper, PlanetaryGatekeeper
from src.logger import get_logger

logger = get_logger("ThresholdTuning")


def load_validation_images(planetary_dir: str, non_planetary_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load validation image paths from directories.
    
    Args:
        planetary_dir: Directory containing planetary/geological images
        non_planetary_dir: Directory containing non-planetary images
    
    Returns:
        Tuple of (planetary_paths, non_planetary_paths)
    """
    planetary_paths = []
    non_planetary_paths = []
    
    # Load planetary images
    if os.path.exists(planetary_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            planetary_paths.extend(Path(planetary_dir).glob(ext))
        logger.info(f"Found {len(planetary_paths)} planetary images in {planetary_dir}")
    else:
        logger.warning(f"Planetary directory not found: {planetary_dir}")
    
    # Load non-planetary images
    if os.path.exists(non_planetary_dir):
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            non_planetary_paths.extend(Path(non_planetary_dir).glob(ext))
        logger.info(f"Found {len(non_planetary_paths)} non-planetary images in {non_planetary_dir}")
    else:
        logger.warning(f"Non-planetary directory not found: {non_planetary_dir}")
    
    return [str(p) for p in planetary_paths], [str(p) for p in non_planetary_paths]


def evaluate_threshold(gatekeeper: PlanetaryGatekeeper,
                      planetary_images: List[str],
                      non_planetary_images: List[str],
                      threshold: float) -> Dict:
    """
    Evaluate gatekeeper performance at a specific threshold.
    
    Args:
        gatekeeper: Pre-loaded gatekeeper instance
        planetary_images: List of planetary image paths (ground truth positive)
        non_planetary_images: List of non-planetary image paths (ground truth negative)
        threshold: Threshold to evaluate
    
    Returns:
        Dictionary with precision, recall, f1, accuracy, and confusion matrix values
    """
    # Update gatekeeper threshold
    gatekeeper.threshold = threshold
    
    # Track predictions
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # Evaluate planetary images (should be accepted)
    logger.info(f"Evaluating {len(planetary_images)} planetary images...")
    for img_path in tqdm(planetary_images, desc="Planetary", leave=False):
        try:
            result = gatekeeper.check_image(img_path)
            if result["accepted"]:
                true_positives += 1
            else:
                false_negatives += 1
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            false_negatives += 1
    
    # Evaluate non-planetary images (should be rejected)
    logger.info(f"Evaluating {len(non_planetary_images)} non-planetary images...")
    for img_path in tqdm(non_planetary_images, desc="Non-planetary", leave=False):
        try:
            result = gatekeeper.check_image(img_path)
            if not result["accepted"]:
                true_negatives += 1
            else:
                false_positives += 1
        except Exception as e:
            logger.warning(f"Error processing {img_path}: {e}")
            true_negatives += 1
    
    # Calculate metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total": total,
    }


def tune_threshold(planetary_dir: str,
                  non_planetary_dir: str,
                  min_threshold: float = 0.5,
                  max_threshold: float = 0.9,
                  step: float = 0.05,
                  output_plot: str = None) -> List[Dict]:
    """
    Loop over thresholds 0.5 to 0.9 in steps of 0.05,
    print precision and recall for each using validation_images list.
    
    Args:
        planetary_dir: Directory with planetary images
        non_planetary_dir: Directory with non-planetary images
        min_threshold: Minimum threshold to test
        max_threshold: Maximum threshold to test
        step: Step size for threshold sweep
        output_plot: Path to save results plot (optional)
    
    Returns:
        List of evaluation results for each threshold
    """
    logger.info("🔍 Starting Threshold Tuning")
    logger.info(f"Range: {min_threshold} to {max_threshold}, step={step}")
    
    # Load validation images
    planetary_images, non_planetary_images = load_validation_images(
        planetary_dir, 
        non_planetary_dir
    )
    
    if len(planetary_images) == 0 or len(non_planetary_images) == 0:
        raise ValueError("Need both planetary and non-planetary validation images!")
    
    # Load gatekeeper once
    logger.info("Loading CLIP gatekeeper...")
    gatekeeper = load_gatekeeper(threshold=min_threshold)
    
    # Generate thresholds to test
    thresholds = np.arange(min_threshold, max_threshold + step/2, step)
    
    # Evaluate each threshold
    results = []
    print("\n" + "="*80)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12}")
    print("="*80)
    
    for threshold in thresholds:
        result = evaluate_threshold(
            gatekeeper,
            planetary_images,
            non_planetary_images,
            threshold
        )
        results.append(result)
        
        print(f"{result['threshold']:<12.2f} "
              f"{result['precision']:<12.3f} "
              f"{result['recall']:<12.3f} "
              f"{result['f1']:<12.3f} "
              f"{result['accuracy']:<12.3f}")
    
    print("="*80)
    
    # Find best threshold by F1 score
    best_result = max(results, key=lambda x: x['f1'])
    
    print(f"\n✅ Best Threshold: {best_result['threshold']:.2f}")
    print(f"   Precision: {best_result['precision']:.3f}")
    print(f"   Recall: {best_result['recall']:.3f}")
    print(f"   F1 Score: {best_result['f1']:.3f}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    
    # Plot results
    if output_plot or True:  # Always create plot
        plot_path = output_plot or "threshold_tuning_results.png"
        plot_tuning_results(results, plot_path)
        logger.info(f"📊 Results plot saved to {plot_path}")
    
    return results


def plot_tuning_results(results: List[Dict], output_path: str):
    """
    Plot threshold tuning results.
    
    Args:
        results: List of evaluation results
        output_path: Path to save plot
    """
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1_scores = [r['f1'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, precisions, 'o-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 's-', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, '^-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, accuracies, 'd-', label='Accuracy', linewidth=2)
    
    # Mark best F1 score
    best_idx = np.argmax(f1_scores)
    plt.axvline(thresholds[best_idx], color='red', linestyle='--', alpha=0.5, 
                label=f'Best F1 (threshold={thresholds[best_idx]:.2f})')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Planetary Gatekeeper Threshold Tuning', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Tune CLIP gatekeeper threshold for optimal performance"
    )
    parser.add_argument(
        "--planetary_dir",
        type=str,
        required=True,
        help="Directory containing planetary/geological images"
    )
    parser.add_argument(
        "--non_planetary_dir",
        type=str,
        required=True,
        help="Directory containing non-planetary images"
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=0.5,
        help="Minimum threshold to test (default: 0.5)"
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=0.9,
        help="Maximum threshold to test (default: 0.9)"
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.05,
        help="Step size for threshold sweep (default: 0.05)"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="threshold_tuning_results.png",
        help="Path to save results plot"
    )
    
    args = parser.parse_args()
    
    # Run threshold tuning
    results = tune_threshold(
        planetary_dir=args.planetary_dir,
        non_planetary_dir=args.non_planetary_dir,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
        output_plot=args.output_plot
    )
    
    # Save detailed results to CSV
    import csv
    csv_path = "threshold_tuning_detailed.csv"
    with open(csv_path, 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    logger.info(f"💾 Detailed results saved to {csv_path}")


if __name__ == "__main__":
    main()
