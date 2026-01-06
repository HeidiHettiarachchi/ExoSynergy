#!/usr/bin/env python3
"""
Simple Data Processing for CRISM - Convert MAT to PNG.
Converts all CRISM .mat images to viewable .png format.

Usage:
    python pipelines/simple_process_data.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.dataset import load_and_reconstruct_data
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.SimpleProcessing")


def create_rgb_composite(image: np.ndarray, r_band: int = 50, g_band: int = 150, b_band: int = 250) -> np.ndarray:
    """Create RGB composite from hyperspectral image."""
    max_band = image.shape[2] - 1
    r_band = min(r_band, max_band)
    g_band = min(g_band, max_band)
    b_band = min(b_band, max_band)
    
    rgb = np.stack([image[:, :, r_band], image[:, :, g_band], image[:, :, b_band]], axis=2)
    
    # Normalize to [0, 1]
    if rgb.max() > rgb.min():
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    
    return np.clip(rgb, 0, 1)


def main():
    """Process all CRISM images and save as PNG files."""
    logger.info("=" * 60)
    logger.info("CRISM Simple Data Processing - MAT to PNG")
    logger.info("=" * 60)
    
    try:
        # Load the data
        logger.info("📁 Loading CRISM data...")
        images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
        
        logger.log_dataset_info(images, masks, scene_ids, label_mapping)
        
        # Create color map
        all_classes = np.unique(np.concatenate([mask.flatten() for mask in masks]))
        colors_list = plt.cm.tab20(np.linspace(0, 1, len(all_classes)))
        cmap = mcolors.ListedColormap(colors_list)
        
        logger.info(f"🖼️ Converting {len(images)} scenes to PNG format...")
        
        # Process each scene
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            logger.info(f"📸 Processing scene {i+1}/{len(images)}: {scene_id}")
            
            # 1. Standard RGB composite
            rgb_composite = create_rgb_composite(image, 50, 150, 250)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(rgb_composite)
            plt.title(f'RGB Composite - {scene_id}\n'
                     f'Size: {image.shape[0]}×{image.shape[1]} | Bands: 50,150,250',
                     fontsize=14)
            plt.axis('off')
            
            rgb_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_rgb.png')
            plt.savefig(rgb_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. High contrast RGB
            rgb_hc = create_rgb_composite(image, 80, 160, 320)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(rgb_hc)
            plt.title(f'High Contrast RGB - {scene_id}\n'
                     f'Size: {image.shape[0]}×{image.shape[1]} | Bands: 80,160,320',
                     fontsize=14)
            plt.axis('off')
            
            rgb_hc_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_rgb_hc.png')
            plt.savefig(rgb_hc_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 3. Ground truth mask
            plt.figure(figsize=(12, 8))
            im = plt.imshow(mask, cmap=cmap, interpolation='nearest')
            
            unique_classes, counts = np.unique(mask, return_counts=True)
            plt.title(f'Ground Truth - {scene_id}\n'
                     f'Size: {mask.shape[0]}×{mask.shape[1]} | '
                     f'Classes: {len(unique_classes)} | '
                     f'Labeled: {np.sum(mask > 0):,}/{mask.size:,} pixels',
                     fontsize=14)
            
            plt.colorbar(im, label='Mineral Class', shrink=0.8)
            plt.axis('off')
            
            gt_path = os.path.join(config.GROUND_TRUTH_DIR, f'{scene_id}_ground_truth.png')
            plt.savefig(gt_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"   ✅ Saved: RGB composite, high-contrast RGB, and ground truth")
        
        # Create simple HTML index
        create_simple_html_index(images, masks, scene_ids)
        
        logger.info(f"\n🎉 Processing completed!")
        logger.info(f"   📊 Processed: {len(images)} scenes")
        logger.info(f"   📁 Files created: {len(images) * 3}")
        logger.info(f"   🌐 Browse: data/processed/index.html")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_simple_html_index(images, masks, scene_ids):
    """Create simple HTML index for browsing images."""
    from datetime import datetime
    
    index_path = os.path.join(config.PROCESSED_DATA_DIR, 'index.html')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Image Gallery</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f8ff; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; margin-bottom: 20px; }}
        .scene {{ background: white; margin: 15px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .scene-title {{ font-size: 1.4em; color: #2c3e50; margin-bottom: 10px; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}
        .image-item {{ text-align: center; background: #f8f9fa; padding: 10px; border-radius: 5px; }}
        .image-item img {{ max-width: 100%; height: auto; border-radius: 5px; }}
        .caption {{ font-weight: bold; margin-top: 8px; color: #444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ CRISM Hyperspectral Image Gallery</h1>
        <p>Mars Reconnaissance Orbiter - {len(images)} Scenes</p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
"""
    
    # Add each scene
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        h, w, c = image.shape
        classes = len(np.unique(mask))
        labeled = np.sum(mask > 0)
        
        html_content += f"""
    <div class="scene">
        <div class="scene-title">Scene {i+1}: {scene_id}</div>
        <p>Dimensions: {h}×{w} | Classes: {classes} | Labeled: {labeled:,} pixels | Bands: {c}</p>
        
        <div class="image-grid">
            <div class="image-item">
                <img src="rgb_composites/{scene_id}_rgb.png" alt="RGB Composite">
                <div class="caption">RGB Composite</div>
            </div>
            <div class="image-item">
                <img src="rgb_composites/{scene_id}_rgb_hc.png" alt="High Contrast">
                <div class="caption">High Contrast RGB</div>
            </div>
            <div class="image-item">
                <img src="ground_truth/{scene_id}_ground_truth.png" alt="Ground Truth">
                <div class="caption">Ground Truth</div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
    <div class="header">
        <p>🔬 CRISM Hyperspectral Mineral Segmentation Project</p>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"🌐 Created image gallery: {index_path}")


if __name__ == "__main__":
    main()
