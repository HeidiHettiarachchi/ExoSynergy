#!/usr/bin/env python3
"""
Data Pipeline for CRISM Hyperspectral Mineral Segmentation.

This script handles data verification by visualizing ground truth annotations.
It loads the CRISM data, reconstructs the 2D scenes, and saves ground truth
segmentation masks to verify data integrity before training.

Usage:
    python pipelines/data_pipeline.py
"""

import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.dataset import load_and_reconstruct_data, visualize_ground_truth
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Main data pipeline execution.
    Loads data, performs sanity checks, and generates visualizations.
    """
    print("=" * 60)
    print("CRISM Hyperspectral Data Pipeline - Data Verification")
    print("=" * 60)
    
    try:
        # Load and reconstruct the data
        print("\n1. Loading CRISM data files...")
        images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
        
        print(f"\n🔍 Label Mapping Analysis:")
        print(f"   - Original labels: {original_labels}")
        print(f"   - Continuous mapping: {label_mapping}")
        print(f"   - Number of classes: {len(original_labels)}")
        
        print(f"\n2. Data verification summary:")
        print(f"   - Number of scenes: {len(images)}")
        print(f"   - Image dimensions: {[img.shape[:2] for img in images[:3]]}...")
        print(f"   - Spectral bands: {images[0].shape[2] if len(images) > 0 else 'N/A'}")
        print(f"   - Scene IDs: {scene_ids[:5]}...")
        
        # Basic data integrity checks
        print(f"\n3. Data integrity checks:")
        
        if len(images) == 0:
            print("   ❌ ERROR: No scenes were successfully reconstructed!")
            return
        
        print(f"   ✓ Successfully reconstructed {len(images)} scenes")
        
        # Check spectral band consistency
        expected_bands = config.NUM_BANDS
        actual_bands = images[0].shape[2] if len(images) > 0 else 0
        
        if actual_bands != expected_bands:
            print(f"   ⚠️  WARNING: Expected {expected_bands} spectral bands, got {actual_bands}")
        else:
            print(f"   ✓ Spectral band count matches expectation: {actual_bands}")
        
        # Check for valid data ranges
        # Note: Images have different dimensions, so we can't create a single array
        # Following approach from https://github.com/Banus/crism_ml for CRISM data handling
        all_spectra_values = np.concatenate([img.flatten() for img in images])
        min_val = all_spectra_values.min()
        max_val = all_spectra_values.max()
        mean_val = all_spectra_values.mean()
        
        print(f"   ✓ Spectral value range: [{min_val:.3f}, {max_val:.3f}]")
        print(f"   ✓ Mean spectral value: {mean_val:.3f}")
        
        # Check class distribution
        all_mask_values = np.concatenate([mask.flatten() for mask in masks])
        unique_classes = np.unique(all_mask_values)
        class_counts = {cls: np.sum(all_mask_values == cls) for cls in unique_classes}
        
        print(f"   ✓ Found {len(unique_classes)} unique classes: {list(unique_classes)}")
        print(f"   ✓ Class distribution:")
        for cls, count in class_counts.items():
            percentage = (count / len(all_mask_values)) * 100
            print(f"     Class {cls}: {count:,} pixels ({percentage:.1f}%)")
        
        # Check for potential issues
        print(f"\n4. Quality assessment:")
        
        # Check for empty scenes
        empty_scenes = sum(1 for mask in masks if np.all(mask == 0))
        if empty_scenes > 0:
            print(f"   ⚠️  WARNING: {empty_scenes} scenes appear to be empty (all background)")
        else:
            print(f"   ✓ No empty scenes detected")
        
        # Check scene size distribution
        scene_sizes = [(img.shape[0] * img.shape[1]) for img in images]
        min_size = min(scene_sizes)
        max_size = max(scene_sizes)
        avg_size = np.mean(scene_sizes)
        
        print(f"   ✓ Scene size statistics:")
        print(f"     Min: {min_size:,} pixels")
        print(f"     Max: {max_size:,} pixels") 
        print(f"     Average: {avg_size:,.0f} pixels")
        
        # Generate ground truth visualizations
        print(f"\n5. Generating ground truth visualizations...")
        visualize_ground_truth(save_dir=config.OUTPUT_DIR)
        
        # Generate additional analysis plots
        print(f"\n6. Generating additional analysis plots...")
        generate_analysis_plots(images, masks, scene_ids)
        
        print(f"\n7. Data pipeline completed successfully!")
        print(f"   - Visualizations saved to: {config.OUTPUT_DIR}")
        print(f"   - Ready for model training")
        
    except Exception as e:
        print(f"\n❌ ERROR in data pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_analysis_plots(images, masks, scene_ids):
    """
    Generate additional analysis plots for data understanding.
    
    Args:
        images: List of reconstructed hyperspectral images
        masks: List of ground truth masks
        scene_ids: List of scene identifiers
    """
    
    # 1. Class distribution histogram
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    all_masks = np.concatenate([mask.flatten() for mask in masks])
    unique_classes, counts = np.unique(all_masks, return_counts=True)
    
    bars = plt.bar(unique_classes, counts)
    plt.xlabel('Mineral Class')
    plt.ylabel('Number of Pixels')
    plt.title('Global Class Distribution')
    plt.yscale('log')  # Log scale for better visibility
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count:,}', ha='center', va='bottom', rotation=45, fontsize=8)
    
    # 2. Scene size distribution
    plt.subplot(1, 3, 2)
    scene_sizes = [img.shape[0] * img.shape[1] for img in images]
    plt.hist(scene_sizes, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Scene Size (pixels)')
    plt.ylabel('Number of Scenes')
    plt.title('Scene Size Distribution')
    
    # 3. Spectral band statistics
    plt.subplot(1, 3, 3)
    
    # Calculate mean spectrum across all pixels
    all_spectra = []
    for img in images:
        # Reshape to (pixels, bands) and sample some pixels for efficiency
        pixels_bands = img.reshape(-1, img.shape[2])
        # Sample up to 10000 pixels to avoid memory issues
        if len(pixels_bands) > 10000:
            indices = np.random.choice(len(pixels_bands), 10000, replace=False)
            pixels_bands = pixels_bands[indices]
        all_spectra.append(pixels_bands)
    
    if all_spectra:
        combined_spectra = np.vstack(all_spectra)
        mean_spectrum = np.mean(combined_spectra, axis=0)
        std_spectrum = np.std(combined_spectra, axis=0)
        
        bands = np.arange(len(mean_spectrum))
        plt.plot(bands, mean_spectrum, 'b-', linewidth=2, label='Mean')
        plt.fill_between(bands, mean_spectrum - std_spectrum, 
                        mean_spectrum + std_spectrum, alpha=0.3, label='±1 STD')
        plt.xlabel('Spectral Band')
        plt.ylabel('Reflectance Value')
        plt.title('Average Spectrum')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the analysis plot
    analysis_path = os.path.join(config.OUTPUT_DIR, 'data_analysis.png')
    plt.savefig(analysis_path, dpi=config.DPI, bbox_inches='tight')
    plt.close()
    
    print(f"   - Data analysis plot saved: {analysis_path}")
    
    # 4. Generate per-scene summary
    summary_path = os.path.join(config.OUTPUT_DIR, 'scene_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("CRISM Scene Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for i, (img, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            f.write(f"Scene {i+1}: {scene_id}\n")
            f.write(f"  Dimensions: {img.shape[0]}x{img.shape[1]} pixels\n")
            f.write(f"  Spectral bands: {img.shape[2]}\n")
            
            unique_classes, counts = np.unique(mask, return_counts=True)
            f.write(f"  Classes present: {list(unique_classes)}\n")
            
            for cls, count in zip(unique_classes, counts):
                percentage = (count / mask.size) * 100
                f.write(f"    Class {cls}: {count:,} pixels ({percentage:.1f}%)\n")
            
            f.write("\n")
    
    print(f"   - Scene summary saved: {summary_path}")


if __name__ == "__main__":
    main()
