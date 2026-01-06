#!/usr/bin/env python3
"""
Comprehensive Data Processing for CRISM - Process ALL Available Images.

This script processes ALL 77 available CRISM images from the MATLAB files,
removing the limits used in training to create a complete image gallery.
This is specifically for visualization and exploration purposes.

Usage:
    python pipelines/process_all_data.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import scipy.io as sio

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import config
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.FullDataProcessing")


def load_all_crism_data():
    """
    Load ALL available CRISM data without any size or count limitations.
    This version processes all 77 images for complete visualization.
    """
    logger.info("🔄 Loading ALL CRISM hyperspectral data (no limits)...")
    
    try:
        # Load the labeled data (main dataset with mineral classifications)
        logger.info(f"📁 Loading labeled data: {config.LABELED_DATA_PATH}")
        labeled_data = sio.loadmat(config.LABELED_DATA_PATH)
        
        # Extract data
        spectra = labeled_data['pixspec']
        labels = labeled_data['pixlabs'].flatten()
        image_ids = labeled_data['pixims'].flatten()
        coordinates = labeled_data['pixcrds']
        image_names = labeled_data['im_names']
        
        logger.info(f"📊 Full dataset extracted:")
        logger.info(f"   Total pixels: {len(spectra):,}")
        logger.info(f"   Spectral bands: {spectra.shape[1]}")
        logger.info(f"   Unique labels: {len(np.unique(labels))}")
        logger.info(f"   Images available: {len(np.unique(image_ids))}")
        
        # Create label mapping
        unique_labels_in_data = np.unique(labels)
        label_mapping = {label: i for i, label in enumerate(sorted(unique_labels_in_data))}
        mapped_labels = np.array([label_mapping[label] for label in labels])
        
        # Get unique image IDs
        unique_image_ids = np.unique(image_ids)
        logger.info(f"🖼️  Processing all {len(unique_image_ids)} images...")
        
        images = []
        masks = []
        scene_list = []
        skipped_count = 0
        
        # Process ALL images (no limits)
        for idx, img_id in enumerate(unique_image_ids):
            logger.info(f"🔍 Processing image {idx+1}/{len(unique_image_ids)}: ID {img_id}")
            
            # Get pixels for this image
            img_mask = image_ids == img_id
            img_spectra = spectra[img_mask]
            img_coords = coordinates[img_mask]
            img_labels = mapped_labels[img_mask]
            
            if len(img_spectra) == 0:
                logger.warning(f"   ⚠️ Skipping image {img_id}: No pixels found")
                continue
            
            # Determine spatial dimensions from coordinates
            x_coords = img_coords[:, 0].astype(int)
            y_coords = img_coords[:, 1].astype(int)
            
            # Get bounding box
            min_x, max_x = x_coords.min(), x_coords.max()
            min_y, max_y = y_coords.min(), y_coords.max()
            
            height = max_y - min_y + 1
            width = max_x - min_x + 1
            
            # Only skip extremely small images
            if height < 5 or width < 5:
                logger.warning(f"   ⚠️ Skipping tiny image {img_id}: {height}×{width}")
                skipped_count += 1
                continue
            
            # Allow larger images for complete visualization
            if height * width > 500000:  # 500K pixels limit for memory
                logger.warning(f"   ⚠️ Skipping huge image {img_id}: {height}×{width} pixels")
                skipped_count += 1
                continue
                
            # Initialize arrays
            actual_bands = spectra.shape[1]
            img_array = np.zeros((height, width, actual_bands), dtype=np.float32)
            mask_array = np.zeros((height, width), dtype=np.int64)
            
            # Place pixels at correct locations
            for spec, coord, label in zip(img_spectra, img_coords, img_labels):
                x = int(coord[0]) - min_x
                y = int(coord[1]) - min_y
                
                if 0 <= y < height and 0 <= x < width:
                    img_array[y, x] = spec
                    mask_array[y, x] = int(label)
            
            # Calculate labeled pixel ratio
            labeled_pixels = np.sum(mask_array > 0)
            total_pixels = height * width
            label_ratio = labeled_pixels / total_pixels
            
            # Keep all images with any labeled pixels for visualization
            if label_ratio > 0.001:  # At least 0.1% labeled pixels
                images.append(img_array)
                masks.append(mask_array)
                
                # Create scene name
                try:
                    if hasattr(image_names, '__len__') and len(image_names) > 0:
                        if img_id <= len(image_names):
                            if hasattr(image_names[int(img_id-1)], 'item'):
                                scene_name = str(image_names[int(img_id-1)].item())
                            else:
                                scene_name = str(image_names[int(img_id-1)])
                        else:
                            scene_name = f"image_{img_id}"
                    else:
                        scene_name = f"image_{img_id}"
                except:
                    scene_name = f"image_{img_id}"
                    
                scene_list.append(scene_name)
                
                logger.info(f"   ✅ Added: {height}×{width}, {labeled_pixels}/{total_pixels} labeled ({label_ratio:.2%})")
            else:
                logger.warning(f"   ⚠️ Skipping low-label image {img_id}: {label_ratio:.3%} labeled")
                skipped_count += 1
        
        logger.info(f"\n✅ Data loading complete:")
        logger.info(f"   Successfully loaded: {len(images)} images")
        logger.info(f"   Skipped: {skipped_count} images")
        logger.info(f"   Total processed: {len(unique_image_ids)} images")
        
        return images, masks, scene_list, label_mapping, unique_labels_in_data
        
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise


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


def save_scene_as_images(image, mask, scene_id, scene_idx, total_scenes):
    """
    Save a single scene in all formats (RGB composites, ground truth, spectral analysis).
    """
    logger.info(f"🎨 Processing scene {scene_idx+1}/{total_scenes}: {scene_id}")
    logger.info(f"   Dimensions: {image.shape}")
    logger.info(f"   Classes present: {np.unique(mask)}")
    
    try:
        # Create color map for this scene
        scene_classes = np.unique(mask)
        colors_list = plt.cm.tab20(np.linspace(0, 1, len(scene_classes)))
        cmap = mcolors.ListedColormap(colors_list)
        
        # 1. Save multiple RGB composites
        rgb_combinations = [
            (20, 100, 200, "early_mid_late", "Early-Mid-Late Spectrum"),
            (50, 150, 250, "standard", "Standard RGB Composite"), 
            (80, 160, 300, "high_bands", "High Frequency Bands"),
            (10, 80, 160, "low_mid", "Low-Mid Spectrum"),
            (100, 200, 300, "mid_high", "Mid-High Spectrum")
        ]
        
        for r, g, b, suffix, title in rgb_combinations:
            rgb_composite = create_rgb_composite(image, r, g, b)
            
            plt.figure(figsize=(14, 10))
            plt.imshow(rgb_composite)
            plt.title(f'{title} - {scene_id}\n'
                     f'Shape: {image.shape[0]}×{image.shape[1]} | '
                     f'Bands: R={r}, G={g}, B={b}',
                     fontsize=16, pad=20)
            plt.axis('off')
            
            # Add metadata
            plt.figtext(0.02, 0.02, 
                       f'Scene: {scene_id} | Size: {image.shape[0]}×{image.shape[1]} pixels | '
                       f'Spectral bands: {image.shape[2]} | Band combination: {suffix}',
                       fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
            
            save_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_{suffix}_composite.png')
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
        
        # 2. Save ground truth mask with detailed information
        plt.figure(figsize=(14, 10))
        im = plt.imshow(mask, cmap=cmap, interpolation='nearest')
        
        # Calculate class statistics
        unique_classes, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        
        plt.title(f'Ground Truth Mineral Map - {scene_id}\n'
                 f'Size: {mask.shape[0]}×{mask.shape[1]} | '
                 f'Classes: {len(unique_classes)} | '
                 f'Labeled pixels: {np.sum(mask > 0):,}/{total_pixels:,}',
                 fontsize=16, pad=20)
        
        # Colorbar with class information
        cbar = plt.colorbar(im, label='Mineral Class ID', shrink=0.8, pad=0.02)
        cbar.set_ticks(range(len(scene_classes)))
        cbar.set_ticklabels([f'C{cls}' for cls in scene_classes])
        
        plt.axis('off')
        
        # Add detailed class statistics
        stats_text = "Class Distribution:\n"
        for cls, count in zip(unique_classes, counts):
            percentage = (count / total_pixels) * 100
            stats_text += f"Class {cls}: {count:,} pixels ({percentage:.1f}%)\n"
        
        # Limit text length for readability
        if len(stats_text) > 200:
            lines = stats_text.split('\n')
            stats_text = '\n'.join(lines[:8]) + '\n... and more classes'
        
        plt.figtext(0.02, 0.02, stats_text,
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9))
        
        gt_save_path = os.path.join(config.GROUND_TRUTH_DIR, f'{scene_id}_ground_truth.png')
        plt.savefig(gt_save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        # 3. Save detailed spectral analysis
        if image.size > 0:
            h, w, bands = image.shape
            sample_size = min(2000, h * w // 4)  # Sample up to 2000 pixels
            
            # Reshape and sample
            pixels = image.reshape(-1, bands)
            labels_flat = mask.flatten()
            
            if len(pixels) > sample_size:
                indices = np.random.choice(len(pixels), sample_size, replace=False)
                pixels = pixels[indices]
                labels_flat = labels_flat[indices]
            
            # Create comprehensive spectral analysis
            plt.figure(figsize=(20, 12))
            
            # Overall spectrum
            plt.subplot(2, 3, 1)
            mean_spectrum = np.mean(pixels, axis=0)
            std_spectrum = np.std(pixels, axis=0)
            bands_x = np.arange(bands)
            
            plt.plot(bands_x, mean_spectrum, 'b-', linewidth=2, label='Mean')
            plt.fill_between(bands_x, mean_spectrum - std_spectrum, 
                           mean_spectrum + std_spectrum, alpha=0.3, label='±1 STD')
            plt.xlabel('Spectral Band')
            plt.ylabel('Reflectance')
            plt.title(f'Overall Mean Spectrum\n{scene_id}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Per-class spectra
            plt.subplot(2, 3, 2)
            unique_scene_classes = np.unique(labels_flat)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_scene_classes)))
            
            for j, cls in enumerate(unique_scene_classes):
                class_pixels = pixels[labels_flat == cls]
                if len(class_pixels) > 5:  # At least 5 pixels
                    class_mean = np.mean(class_pixels, axis=0)
                    plt.plot(bands_x, class_mean, color=colors[j], 
                           linewidth=2, label=f'Class {cls} (n={len(class_pixels)})')
            
            plt.xlabel('Spectral Band')
            plt.ylabel('Mean Reflectance')
            plt.title('Per-Class Mean Spectra')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Band variability
            plt.subplot(2, 3, 3)
            band_vars = np.var(pixels, axis=0)
            plt.plot(bands_x, band_vars, 'r-', linewidth=2)
            top_var_bands = np.argsort(band_vars)[-15:]
            plt.scatter(top_var_bands, band_vars[top_var_bands], 
                      c='red', s=30, zorder=5, label='Top 15 Variable Bands')
            plt.xlabel('Spectral Band')
            plt.ylabel('Variance')
            plt.title('Spectral Band Variability')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Reflectance distribution
            plt.subplot(2, 3, 4)
            plt.hist(pixels.flatten(), bins=100, alpha=0.7, edgecolor='black')
            plt.xlabel('Reflectance Value')
            plt.ylabel('Frequency')
            plt.title('Reflectance Distribution')
            plt.yscale('log')
            
            # Class spatial distribution
            plt.subplot(2, 3, 5)
            class_counts = [np.sum(labels_flat == cls) for cls in unique_scene_classes]
            plt.bar(range(len(unique_scene_classes)), class_counts, 
                   color=colors[:len(unique_scene_classes)], alpha=0.8)
            plt.xlabel('Class ID')
            plt.ylabel('Pixel Count')
            plt.title('Class Distribution')
            plt.xticks(range(len(unique_scene_classes)), 
                      [f'C{cls}' for cls in unique_scene_classes])
            
            # Spectral correlation heatmap (sample of bands)
            plt.subplot(2, 3, 6)
            sample_bands = np.linspace(0, bands-1, 20).astype(int)  # Sample 20 bands
            correlation_matrix = np.corrcoef(pixels[:, sample_bands].T)
            
            im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title('Band Correlation Matrix\n(Sample of 20 bands)')
            plt.xlabel('Band Index (sampled)')
            plt.ylabel('Band Index (sampled)')
            plt.colorbar(im, label='Correlation')
            
            plt.suptitle(f'Comprehensive Spectral Analysis - {scene_id}', fontsize=18)
            plt.tight_layout()
            
            spectral_save_path = os.path.join(config.SPECTRAL_ANALYSIS_DIR, f'{scene_id}_full_analysis.png')
            plt.savefig(spectral_save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        
        logger.info(f"✅ Scene processing completed")
        logger.info(f"   Generated all image formats for scene {scene_id}")
        
        return images, masks, scene_list
        
    except Exception as e:
        logger.error(f"❌ Failed to load all data: {e}")
        raise


def main():
    """
    Main processing execution - convert all .mat images to .png format.
    """
    logger.info("=" * 70)
    logger.info("CRISM Complete Data Processing - MAT to PNG Conversion")
    logger.info("=" * 70)
    
    logger.info(f"📂 Output Structure:")
    logger.info(f"   RGB Composites: {config.RGB_COMPOSITES_DIR}")
    logger.info(f"   Ground Truth: {config.GROUND_TRUTH_DIR}")
    logger.info(f"   Spectral Analysis: {config.SPECTRAL_ANALYSIS_DIR}")
    
    try:
        # Load and process all data
        images, masks, scene_ids, label_mapping, original_labels = load_all_crism_data()
        
        # Process each scene
        total_files_created = 0
        
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            save_scene_as_images(image, mask, scene_id, i, len(images))
            total_files_created += 6  # 5 RGB + 1 GT + 1 spectral
        
        # Create browsable HTML index
        create_html_index(images, masks, scene_ids, original_labels)
        
        logger.info(f"\n🎉 Processing completed successfully!")
        logger.info(f"   📊 Scenes processed: {len(images)}")
        logger.info(f"   📁 Files created: {total_files_created}")
        logger.info(f"   🌐 Browse results: data/processed/index.html")
        
        # Create summary report
        create_summary_report(images, masks, scene_ids)
        
    except Exception as e:
        logger.error(f"❌ ERROR in processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_html_index(images, masks, scene_ids, original_labels):
    """Create comprehensive HTML index for browsing all processed images."""
    from datetime import datetime
    
    index_path = os.path.join(config.PROCESSED_DATA_DIR, 'index.html')
    
    # Calculate overall statistics
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
    all_classes = np.unique(np.concatenate([mask.flatten() for mask in masks]))
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Hyperspectral Data Gallery</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.95); padding: 30px; text-align: center; border-radius: 15px; margin-bottom: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); }}
        .stats {{ background: rgba(255,255,255,0.9); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .scene {{ background: rgba(255,255,255,0.95); margin: 20px 0; padding: 20px; border-radius: 12px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
        .scene-title {{ font-size: 1.8em; color: #2c3e50; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .scene-info {{ background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 0.95em; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }}
        .image-item {{ text-align: center; background: #f8f9fa; padding: 15px; border-radius: 8px; }}
        .image-item img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 3px 10px rgba(0,0,0,0.2); transition: transform 0.3s; }}
        .image-item img:hover {{ transform: scale(1.05); }}
        .image-caption {{ font-weight: bold; color: #2c3e50; margin-top: 10px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
        .stat-box {{ background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%); color: white; padding: 15px; border-radius: 8px; text-align: center; }}
        .nav {{ position: sticky; top: 0; background: rgba(44, 62, 80, 0.95); padding: 10px; text-align: center; z-index: 100; }}
        .nav a {{ color: white; text-decoration: none; margin: 0 15px; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#overview">Overview</a>
        <a href="#gallery">Image Gallery</a>
        <a href="#statistics">Statistics</a>
    </div>
    
    <div class="container">
        <div class="header" id="overview">
            <h1>🛰️ CRISM Hyperspectral Mineral Data Gallery</h1>
            <p style="font-size: 1.2em; color: #555;">Mars Reconnaissance Orbiter • Compact Reconnaissance Imaging Spectrometer</p>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="stats" id="statistics">
            <h2>📊 Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <h3>{len(images)}</h3>
                    <p>Processed Scenes</p>
                </div>
                <div class="stat-box">
                    <h3>{total_pixels:,}</h3>
                    <p>Total Pixels</p>
                </div>
                <div class="stat-box">
                    <h3>{images[0].shape[2] if len(images) > 0 else 'N/A'}</h3>
                    <p>Spectral Bands</p>
                </div>
                <div class="stat-box">
                    <h3>{len(all_classes)}</h3>
                    <p>Mineral Classes</p>
                </div>
            </div>
            
            <h3>🏷️ Mineral Classes Found</h3>
            <p style="background: #ecf0f1; padding: 15px; border-radius: 5px; font-family: monospace;">
                {', '.join([f'Class {cls}' for cls in all_classes[:15]])}
                {' ... and more' if len(all_classes) > 15 else ''}
            </p>
        </div>
        
        <div id="gallery">
            <h2 style="text-align: center; color: white; font-size: 2em; margin: 30px 0;">🖼️ Image Gallery</h2>
"""
    
    # Add each scene
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        unique_classes = np.unique(mask)
        labeled_pixels = np.sum(mask > 0)
        total_scene_pixels = mask.size
        h, w, c = image.shape
        
        html_content += f"""
            <div class="scene">
                <div class="scene-title">🗺️ Scene {i+1}: {scene_id}</div>
                <div class="scene-info">
                    <strong>Dimensions:</strong> {h:,} × {w:,} pixels ({h*w:,} total) | 
                    <strong>Spectral Bands:</strong> {c} | 
                    <strong>Mineral Classes:</strong> {len(unique_classes)} types | 
                    <strong>Labeled Coverage:</strong> {labeled_pixels:,}/{total_scene_pixels:,} pixels ({labeled_pixels/total_scene_pixels*100:.1f}%)
                    <br>
                    <strong>Classes Present:</strong> {', '.join([f'Class {cls}' for cls in unique_classes[:10]])}
                    {' ...' if len(unique_classes) > 10 else ''}
                </div>
                
                <div class="image-grid">
                    <div class="image-item">
                        <img src="rgb_composites/{scene_id}_standard_composite.png" alt="Standard RGB">
                        <div class="image-caption">📸 Standard RGB Composite</div>
                    </div>
                    <div class="image-item">
                        <img src="rgb_composites/{scene_id}_high_bands_composite.png" alt="High Frequency">
                        <div class="image-caption">🎨 High Frequency Bands</div>
                    </div>
                    <div class="image-item">
                        <img src="ground_truth/{scene_id}_ground_truth.png" alt="Ground Truth">
                        <div class="image-caption">🏷️ Ground Truth Mask</div>
                    </div>
                    <div class="image-item">
                        <img src="spectral_analysis/{scene_id}_full_analysis.png" alt="Spectral Analysis">
                        <div class="image-caption">📈 Spectral Analysis</div>
                    </div>
                </div>
            </div>
"""
    
    html_content += """
        </div>
        
        <div class="header">
            <h3>🔬 CRISM Hyperspectral Mineral Segmentation Project</h3>
            <p>Deep Learning Pipeline for Martian Surface Analysis</p>
            <p style="font-size: 0.9em; color: #666;">
                This gallery shows all processed CRISM hyperspectral images with RGB composites, 
                ground truth mineral maps, and spectral analysis plots.
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"🌐 Created interactive HTML gallery: {index_path}")


def create_summary_report(images, masks, scene_ids):
    """Create detailed text summary of all processed images."""
    summary_path = os.path.join(config.PROCESSED_DATA_DIR, 'processing_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("CRISM Hyperspectral Data Processing Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Scenes Processed: {len(images)}\n\n")
        
        # Overall statistics
        total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
        all_classes = np.unique(np.concatenate([mask.flatten() for mask in masks]))
        
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total pixels: {total_pixels:,}\n")
        f.write(f"  Spectral bands: {images[0].shape[2] if len(images) > 0 else 'N/A'}\n")
        f.write(f"  Unique mineral classes: {len(all_classes)}\n")
        f.write(f"  Class IDs: {all_classes.tolist()}\n\n")
        
        # Per-scene details
        f.write(f"Per-Scene Details:\n")
        f.write(f"{'Scene':<15} {'ID':<10} {'Dimensions':<15} {'Classes':<10} {'Labeled %':<10}\n")
        f.write("-" * 70 + "\n")
        
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            h, w = mask.shape
            classes = len(np.unique(mask))
            labeled = np.sum(mask > 0)
            total = mask.size
            percentage = (labeled / total) * 100
            
            f.write(f"{i+1:<15} {scene_id:<10} {h}×{w:<12} {classes:<10} {percentage:<10.1f}\n")
        
        f.write(f"\nGenerated Files:\n")
        f.write(f"  RGB composites: {len(images) * 5} files\n")
        f.write(f"  Ground truth maps: {len(images)} files\n") 
        f.write(f"  Spectral analysis: {len(images)} files\n")
        f.write(f"  Total files: {len(images) * 7} files\n")
        f.write(f"  HTML index: index.html\n")
    
    logger.info(f"📄 Created detailed summary: {summary_path}")


if __name__ == "__main__":
    from datetime import datetime
    main()
