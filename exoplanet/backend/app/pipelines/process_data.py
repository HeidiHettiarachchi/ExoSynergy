#!/usr/bin/env python3
"""
Data Processing Pipeline for CRISM Hyperspectral Mineral Segmentation.

This script processes all .mat files and converts them to viewable .png images:
- Loads all available CRISM scenes from MATLAB files
- Creates RGB composite images for visual inspection
- Saves ground truth masks for each scene
- Generates spectral analysis plots
- Organizes outputs in data/processed/ directory structure

Usage:
    python pipelines/process_data.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.dataset import load_and_reconstruct_data
from src.utils import get_class_names
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.DataProcessing")


def create_rgb_composite(image: np.ndarray, r_band: int = 50, g_band: int = 150, b_band: int = 250) -> np.ndarray:
    """
    Create RGB composite from hyperspectral image.
    
    Args:
        image: Hyperspectral image array of shape (H, W, C)
        r_band: Red band index
        g_band: Green band index  
        b_band: Blue band index
        
    Returns:
        RGB image array of shape (H, W, 3) normalized to [0, 1]
    """
    # Ensure bands are within range
    max_band = image.shape[2] - 1
    r_band = min(r_band, max_band)
    g_band = min(g_band, max_band)
    b_band = min(b_band, max_band)
    
    # Extract bands
    red = image[:, :, r_band]
    green = image[:, :, g_band]  
    blue = image[:, :, b_band]
    
    # Stack into RGB
    rgb = np.stack([red, green, blue], axis=2)
    
    # Normalize to [0, 1]
    rgb_min = rgb.min()
    rgb_max = rgb.max()
    if rgb_max > rgb_min:
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    else:
        rgb = np.zeros_like(rgb)
    
    return np.clip(rgb, 0, 1)


def save_all_crism_images():
    """
    Process and save all CRISM images from .mat files to .png format.
    """
    logger.info("🚀 Starting comprehensive CRISM data processing...")
    logger.info("=" * 70)
    
    try:
        # Load all available data (remove image limit)
        logger.info("📁 Loading all available CRISM scenes...")
        
        # Temporarily modify the loading function to process all images
        original_load_function = sys.modules['src.dataset'].load_and_reconstruct_data
        
        def load_all_data():
            """Modified loader to process all available images."""
            images, masks, scene_ids, label_mapping, original_labels = original_load_function()
            
            # Override the max_images limit to process all 77 images
            logger.info(f"🔧 Processing ALL available CRISM images (removing size limits)...")
            return images, masks, scene_ids, label_mapping, original_labels
        
        # Load the data
        images, masks, scene_ids, label_mapping, original_labels = load_all_data()
        
        logger.log_dataset_info(images, masks, scene_ids, label_mapping)
        
        # Create color map for ground truth visualization
        num_classes = len(np.unique(np.concatenate([mask.flatten() for mask in masks])))
        colors_list = plt.cm.tab20(np.linspace(0, 1, num_classes))
        cmap = mcolors.ListedColormap(colors_list)
        
        logger.info(f"🎨 Processing {len(images)} scenes into PNG format...")
        
        # Process each scene
        successful_saves = 0
        
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            try:
                logger.info(f"📷 Processing scene {i+1}/{len(images)}: {scene_id}")
                logger.info(f"   Image shape: {image.shape}")
                logger.info(f"   Unique classes: {np.unique(mask)}")
                
                # 1. Save multiple RGB composites with different band combinations
                rgb_combinations = [
                    (50, 150, 250, "standard"),
                    (0, 100, 200, "early_mid_late"),
                    (30, 120, 220, "shifted"),
                    (80, 160, 300, "high_contrast")
                ]
                
                for r, g, b, name in rgb_combinations:
                    rgb_composite = create_rgb_composite(image, r, g, b)
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(rgb_composite)
                    plt.title(f'RGB Composite - {scene_id}\n'
                             f'Bands: R={r}, G={g}, B={b} | Size: {image.shape[0]}×{image.shape[1]}',
                             fontsize=14)
                    plt.axis('off')
                    
                    # Add metadata text
                    plt.figtext(0.02, 0.02, 
                               f'Scene: {scene_id} | Shape: {image.shape} | Bands: {name}',
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                    
                    save_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_{name}_rgb.png')
                    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close()
                
                # 2. Save ground truth mask
                plt.figure(figsize=(12, 8))
                im = plt.imshow(mask, cmap=cmap, interpolation='nearest')
                plt.title(f'Ground Truth Mask - {scene_id}\n'
                         f'Classes: {np.unique(mask)} | Size: {mask.shape[0]}×{mask.shape[1]}',
                         fontsize=14)
                plt.colorbar(im, label='Mineral Class', shrink=0.8)
                plt.axis('off')
                
                # Add class statistics
                unique_classes, counts = np.unique(mask, return_counts=True)
                stats_text = "\n".join([f"Class {cls}: {count:,} pixels" 
                                      for cls, count in zip(unique_classes, counts)[:5]])
                if len(unique_classes) > 5:
                    stats_text += f"\n... and {len(unique_classes)-5} more classes"
                
                plt.figtext(0.02, 0.02, stats_text,
                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                gt_save_path = os.path.join(config.GROUND_TRUTH_DIR, f'{scene_id}_ground_truth.png')
                plt.savefig(gt_save_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close()
                
                # 3. Save spectral analysis for this scene
                if image.size > 0:
                    # Sample pixels for spectral analysis
                    h, w, bands = image.shape
                    sample_size = min(1000, h * w)
                    
                    # Reshape and sample
                    pixels = image.reshape(-1, bands)
                    labels_flat = mask.flatten()
                    
                    if len(pixels) > sample_size:
                        indices = np.random.choice(len(pixels), sample_size, replace=False)
                        pixels = pixels[indices]
                        labels_flat = labels_flat[indices]
                    
                    # Calculate mean spectrum for this scene
                    mean_spectrum = np.mean(pixels, axis=0)
                    std_spectrum = np.std(pixels, axis=0)
                    
                    plt.figure(figsize=(15, 5))
                    
                    # Overall spectrum
                    plt.subplot(1, 3, 1)
                    bands_x = np.arange(bands)
                    plt.plot(bands_x, mean_spectrum, 'b-', linewidth=2, label='Mean')
                    plt.fill_between(bands_x, mean_spectrum - std_spectrum, 
                                   mean_spectrum + std_spectrum, alpha=0.3, label='±1 STD')
                    plt.xlabel('Spectral Band')
                    plt.ylabel('Reflectance')
                    plt.title(f'Mean Spectrum - {scene_id}')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Per-class spectra (if multiple classes present)
                    plt.subplot(1, 3, 2)
                    scene_classes = np.unique(labels_flat)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(scene_classes)))
                    
                    for j, cls in enumerate(scene_classes):
                        if np.sum(labels_flat == cls) > 10:  # At least 10 pixels
                            class_pixels = pixels[labels_flat == cls]
                            class_mean = np.mean(class_pixels, axis=0)
                            plt.plot(bands_x, class_mean, color=colors[j], 
                                   linewidth=2, label=f'Class {cls} (n={len(class_pixels)})')
                    
                    plt.xlabel('Spectral Band')
                    plt.ylabel('Mean Reflectance')
                    plt.title('Per-Class Spectra')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # Band variability
                    plt.subplot(1, 3, 3)
                    band_vars = np.var(pixels, axis=0)
                    plt.plot(bands_x, band_vars, 'r-', linewidth=2)
                    plt.xlabel('Spectral Band')
                    plt.ylabel('Variance')
                    plt.title('Spectral Variability')
                    plt.grid(True, alpha=0.3)
                    
                    # Mark most variable bands
                    top_var_bands = np.argsort(band_vars)[-10:]
                    plt.scatter(top_var_bands, band_vars[top_var_bands], 
                              c='red', s=50, zorder=5, label='Top 10 Variable')
                    plt.legend()
                    
                    plt.suptitle(f'Spectral Analysis - {scene_id}', fontsize=16)
                    plt.tight_layout()
                    
                    spectral_save_path = os.path.join(config.SPECTRAL_ANALYSIS_DIR, f'{scene_id}_spectral_analysis.png')
                    plt.savefig(spectral_save_path, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close()
                
                successful_saves += 1
                logger.info(f"   ✅ Saved scene {scene_id} in all formats")
                
            except Exception as e:
                logger.error(f"   ❌ Failed to process scene {scene_id}: {str(e)}")
                continue
        
        # Generate summary statistics
        logger.info(f"\n📊 Processing Summary:")
        logger.info(f"   Total scenes available: {len(images)}")
        logger.info(f"   Successfully processed: {successful_saves}")
        logger.info(f"   RGB composites per scene: 4 variations")
        logger.info(f"   Total files generated: {successful_saves * 6} (4 RGB + 1 GT + 1 spectral)")
        
        # Create index file
        create_processing_index(images, masks, scene_ids, label_mapping, original_labels)
        
        logger.info(f"\n🎉 Data processing completed!")
        logger.info(f"   📁 RGB composites: {config.RGB_COMPOSITES_DIR}")
        logger.info(f"   🏷️  Ground truth: {config.GROUND_TRUTH_DIR}")
        logger.info(f"   📈 Spectral analysis: {config.SPECTRAL_ANALYSIS_DIR}")
        
    except Exception as e:
        logger.error(f"❌ ERROR in data processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_processing_index(images, masks, scene_ids, label_mapping, original_labels):
    """
    Create an HTML index file to easily browse all processed images.
    """
    index_path = os.path.join(config.PROCESSED_DATA_DIR, 'index.html')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Hyperspectral Data - Processed Images</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; }}
        .scene {{ background-color: white; margin: 20px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .scene-title {{ font-size: 1.5em; color: #2c3e50; margin-bottom: 10px; }}
        .scene-info {{ color: #666; margin-bottom: 15px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; }}
        .image-item {{ text-align: center; }}
        .image-item img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .image-caption {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        .stats {{ background-color: #ecf0f1; padding: 10px; border-radius: 4px; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ CRISM Hyperspectral Mineral Data</h1>
        <p>Processed Martian Surface Images from Mars Reconnaissance Orbiter</p>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <h3>📊 Dataset Statistics</h3>
        <ul>
            <li><strong>Total Scenes:</strong> {len(images)}</li>
            <li><strong>Spectral Bands:</strong> {images[0].shape[2] if len(images) > 0 else 'N/A'}</li>
            <li><strong>Mineral Classes:</strong> {len(original_labels)} unique types</li>
            <li><strong>Original Labels:</strong> {original_labels[:10]}{'...' if len(original_labels) > 10 else ''}</li>
            <li><strong>Image Dimensions:</strong> Variable (smallest to largest)</li>
        </ul>
    </div>
"""
    
    # Add each scene
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        unique_classes = np.unique(mask)
        h, w, c = image.shape
        
        html_content += f"""
    <div class="scene">
        <div class="scene-title">Scene {i+1}: {scene_id}</div>
        <div class="scene-info">
            Dimensions: {h}×{w} pixels | Spectral Bands: {c} | 
            Classes: {len(unique_classes)} ({unique_classes.tolist()})
        </div>
        
        <div class="image-grid">
            <div class="image-item">
                <img src="rgb_composites/{scene_id}_standard_rgb.png" alt="Standard RGB">
                <div class="image-caption">Standard RGB (50,150,250)</div>
            </div>
            <div class="image-item">
                <img src="rgb_composites/{scene_id}_high_contrast_rgb.png" alt="High Contrast RGB">
                <div class="image-caption">High Contrast RGB (80,160,300)</div>
            </div>
            <div class="image-item">
                <img src="ground_truth/{scene_id}_ground_truth.png" alt="Ground Truth">
                <div class="image-caption">Ground Truth Mask</div>
            </div>
            <div class="image-item">
                <img src="spectral_analysis/{scene_id}_spectral_analysis.png" alt="Spectral Analysis">
                <div class="image-caption">Spectral Analysis</div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
    <div class="header">
        <p>🔬 CRISM Machine Learning Project</p>
        <p>Hyperspectral Mineral Segmentation using U-Net</p>
    </div>
</body>
</html>
"""
    
    # Save index file
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 Created browsable index: {index_path}")
    logger.info(f"   Open in browser to view all processed images")


def main():
    """
    Main data processing execution.
    """
    logger.info("=" * 70)
    logger.info("CRISM Data Processing - MAT to PNG Conversion")
    logger.info("=" * 70)
    
    logger.info(f"📂 Directory Structure:")
    logger.info(f"   Raw data: {config.RAW_DATA_DIR}")
    logger.info(f"   Processed output: {config.PROCESSED_DATA_DIR}")
    logger.info(f"   RGB composites: {config.RGB_COMPOSITES_DIR}")
    logger.info(f"   Ground truth: {config.GROUND_TRUTH_DIR}")
    logger.info(f"   Spectral analysis: {config.SPECTRAL_ANALYSIS_DIR}")
    
    # Check if raw data exists
    if not os.path.exists(config.LABELED_DATA_PATH):
        logger.error(f"❌ Labeled data file not found: {config.LABELED_DATA_PATH}")
        logger.error("   Please ensure CRISM data files are in data/raw/ directory")
        sys.exit(1)
    
    # Process all images
    save_all_crism_images()


if __name__ == "__main__":
    from datetime import datetime
    main()
