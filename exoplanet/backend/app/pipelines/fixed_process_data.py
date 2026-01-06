#!/usr/bin/env python3
"""
Fixed Data Processing for CRISM - Proper Hyperspectral Visualization.
Based on analysis of CRISM ratioed spectral data characteristics.

Usage:
    python pipelines/fixed_process_data.py
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
logger = get_logger("CRISM.FixedProcessing")


def create_enhanced_rgb_composite(image: np.ndarray, r_band: int = 50, g_band: int = 150, b_band: int = 250, 
                                 gamma: float = 2.0, contrast_stretch: bool = True) -> np.ndarray:
    """
    Create enhanced RGB composite from CRISM hyperspectral data.
    Handles outliers, zero values, and applies proper contrast enhancement.
    
    Args:
        image: Hyperspectral image array of shape (H, W, C)
        r_band, g_band, b_band: Band indices for RGB
        gamma: Gamma correction factor
        contrast_stretch: Whether to apply percentile-based contrast stretching
        
    Returns:
        Enhanced RGB image array of shape (H, W, 3) in range [0, 1]
    """
    max_band = image.shape[2] - 1
    r_band = min(r_band, max_band)
    g_band = min(g_band, max_band)
    b_band = min(b_band, max_band)
    
    # Extract bands
    red = image[:, :, r_band].copy()
    green = image[:, :, g_band].copy()
    blue = image[:, :, b_band].copy()
    
    # Stack into RGB
    rgb = np.stack([red, green, blue], axis=2)
    
    # Handle CRISM ratioed data properly
    if contrast_stretch:
        # Use percentile-based normalization to handle outliers
        for i in range(3):
            band = rgb[:, :, i]
            
            # Only consider non-zero pixels for normalization
            non_zero_mask = band > 0
            if np.sum(non_zero_mask) > 0:
                non_zero_values = band[non_zero_mask]
                
                # Use 2nd and 98th percentiles to avoid outliers
                p2 = np.percentile(non_zero_values, 2)
                p98 = np.percentile(non_zero_values, 98)
                
                logger.debug(f"Band {[r_band, g_band, b_band][i]}: p2={p2:.4f}, p98={p98:.4f}")
                
                # Clip and normalize
                band_clipped = np.clip(band, p2, p98)
                if p98 > p2:
                    rgb[:, :, i] = (band_clipped - p2) / (p98 - p2)
                else:
                    rgb[:, :, i] = np.zeros_like(band)
            else:
                rgb[:, :, i] = np.zeros_like(band)
    else:
        # Simple min-max normalization
        rgb_min = rgb.min()
        rgb_max = rgb.max()
        if rgb_max > rgb_min:
            rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    # Apply gamma correction for better visual contrast
    if gamma != 1.0:
        rgb = np.power(rgb, 1.0/gamma)
    
    return np.clip(rgb, 0, 1)


def create_false_color_composite(image: np.ndarray, enhancement: str = "mineral") -> np.ndarray:
    """
    Create false-color composite optimized for mineral detection.
    Based on CRISM spectral band characteristics.
    """
    h, w, bands = image.shape
    
    if enhancement == "mineral":
        # Band combinations known to highlight mineral features in CRISM
        r_band = min(230, bands - 1)  # ~2300nm - olivine, pyroxene
        g_band = min(150, bands - 1)  # ~1500nm - hydrated minerals
        b_band = min(80, bands - 1)   # ~800nm - iron oxides
    elif enhancement == "hydration":
        # Emphasize hydrated minerals
        r_band = min(200, bands - 1)  # ~2000nm
        g_band = min(180, bands - 1)  # ~1800nm  
        b_band = min(160, bands - 1)  # ~1600nm
    elif enhancement == "mafic":
        # Emphasize mafic minerals (olivine, pyroxene)
        r_band = min(250, bands - 1)  # ~2500nm
        g_band = min(200, bands - 1)  # ~2000nm
        b_band = min(100, bands - 1)  # ~1000nm
    else:  # standard
        r_band = min(50, bands - 1)
        g_band = min(150, bands - 1)
        b_band = min(250, bands - 1)
    
    return create_enhanced_rgb_composite(image, r_band, g_band, b_band, 
                                        gamma=1.8, contrast_stretch=True)


def create_enhanced_ground_truth(mask: np.ndarray, scene_id: str) -> tuple:
    """
    Create enhanced ground truth visualization with proper color mapping.
    
    Returns:
        Tuple of (colored_mask, colormap, class_info)
    """
    unique_classes = np.unique(mask)
    
    # Create custom colormap for better mineral visualization
    # Use distinguishable colors
    colors = []
    color_names = []
    
    for i, cls in enumerate(unique_classes):
        if cls == 0:  # Background
            colors.append([0.1, 0.1, 0.1])  # Dark gray
            color_names.append("Background")
        else:
            # Use HSV color space for distinct mineral colors
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            sat = 0.8
            val = 0.9
            
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue/360, sat, val)
            colors.append(list(rgb))
            color_names.append(f"Mineral_{cls}")
    
    cmap = mcolors.ListedColormap(colors)
    
    # Calculate class statistics
    class_stats = {}
    total_pixels = mask.size
    
    for cls in unique_classes:
        count = np.sum(mask == cls)
        percentage = (count / total_pixels) * 100
        class_stats[cls] = {"count": count, "percentage": percentage}
    
    return mask, cmap, class_stats, color_names


def main():
    """Fixed data processing with proper CRISM hyperspectral visualization."""
    logger.info("=" * 70)
    logger.info("CRISM Fixed Data Processing - Enhanced Hyperspectral Visualization")
    logger.info("=" * 70)
    
    try:
        # Load the data
        logger.info("📁 Loading CRISM data with enhanced processing...")
        images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
        
        logger.log_dataset_info(images, masks, scene_ids, label_mapping)
        
        logger.info(f"🎨 Creating enhanced visualizations for {len(images)} scenes...")
        
        # Process each scene with enhanced techniques
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            logger.info(f"🖼️ Processing scene {i+1}/{len(images)}: {scene_id}")
            logger.info(f"   Image shape: {image.shape}")
            logger.info(f"   Value range: [{image.min():.4f}, {image.max():.4f}]")
            logger.info(f"   Non-zero pixels: {np.sum(image != 0):,}/{image.size:,}")
            
            # 1. Standard enhanced RGB
            rgb_standard = create_false_color_composite(image, "standard")
            
            plt.figure(figsize=(15, 10))
            plt.imshow(rgb_standard)
            plt.title(f'Enhanced RGB Composite - {scene_id}\n'
                     f'Size: {image.shape[0]}×{image.shape[1]} | '
                     f'Bands: Standard (50,150,250) with outlier handling',
                     fontsize=14)
            plt.axis('off')
            
            # Add enhancement info
            plt.figtext(0.02, 0.02, 
                       f'Enhancement: Percentile normalization (2-98th) + Gamma correction\n'
                       f'Scene: {scene_id} | Non-zero: {np.sum(image != 0):,} pixels',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
            
            rgb_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_enhanced_rgb.png')
            plt.savefig(rgb_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. Mineral detection false-color
            rgb_mineral = create_false_color_composite(image, "mineral")
            
            plt.figure(figsize=(15, 10))
            plt.imshow(rgb_mineral)
            plt.title(f'Mineral Detection False-Color - {scene_id}\n'
                     f'Optimized for olivine, pyroxene, and iron oxides',
                     fontsize=14)
            plt.axis('off')
            
            mineral_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_mineral_false_color.png')
            plt.savefig(mineral_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 3. Hydration detection false-color
            rgb_hydration = create_false_color_composite(image, "hydration")
            
            plt.figure(figsize=(15, 10))
            plt.imshow(rgb_hydration)
            plt.title(f'Hydrated Mineral Detection - {scene_id}\n'
                     f'Optimized for clay minerals and hydrated phases',
                     fontsize=14)
            plt.axis('off')
            
            hydration_path = os.path.join(config.RGB_COMPOSITES_DIR, f'{scene_id}_hydration_detection.png')
            plt.savefig(hydration_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 4. Enhanced ground truth with detailed mineral information
            mask_enhanced, cmap, class_stats, color_names = create_enhanced_ground_truth(mask, scene_id)
            
            plt.figure(figsize=(15, 10))
            
            # Main plot
            im = plt.imshow(mask_enhanced, cmap=cmap, interpolation='nearest')
            
            plt.title(f'Enhanced Ground Truth - {scene_id}\n'
                     f'Size: {mask.shape[0]}×{mask.shape[1]} | '
                     f'Classes: {len(np.unique(mask))} | '
                     f'Mineral coverage: {np.sum(mask > 0):,}/{mask.size:,} pixels '
                     f'({np.sum(mask > 0)/mask.size*100:.1f}%)',
                     fontsize=14)
            
            # Enhanced colorbar
            cbar = plt.colorbar(im, shrink=0.8, pad=0.02)
            cbar.set_label('Mineral Class', fontsize=12)
            
            # Set colorbar ticks and labels
            unique_classes = np.unique(mask)
            cbar.set_ticks(unique_classes)
            cbar.set_ticklabels([f'C{cls}' for cls in unique_classes])
            
            plt.axis('off')
            
            # Add detailed statistics
            stats_text = "Mineral Distribution:\n"
            for cls in unique_classes:
                if cls in class_stats:
                    count = class_stats[cls]["count"]
                    pct = class_stats[cls]["percentage"]
                    if cls == 0:
                        stats_text += f"Background: {count:,} px ({pct:.1f}%)\n"
                    else:
                        stats_text += f"Class {cls}: {count:,} px ({pct:.1f}%)\n"
            
            plt.figtext(0.02, 0.02, stats_text,
                       fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.95))
            
            gt_path = os.path.join(config.GROUND_TRUTH_DIR, f'{scene_id}_enhanced_ground_truth.png')
            plt.savefig(gt_path, dpi=200, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"   ✅ Generated: Enhanced RGB, mineral detection, hydration, and ground truth")
        
        # Create comprehensive HTML index
        create_enhanced_html_index(images, masks, scene_ids, original_labels)
        
        logger.info(f"\n🎉 Enhanced processing completed!")
        logger.info(f"   📊 Processed: {len(images)} scenes")
        logger.info(f"   🖼️ Total images: {len(images) * 4} (3 RGB + 1 GT per scene)")
        logger.info(f"   🌐 Browse: data/processed/index.html")
        
        # Clean up debug file
        if os.path.exists('debug_rgb_test.png'):
            os.remove('debug_rgb_test.png')
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_enhanced_html_index(images, masks, scene_ids, original_labels):
    """Create enhanced HTML index with proper CRISM information."""
    from datetime import datetime
    
    index_path = os.path.join(config.PROCESSED_DATA_DIR, 'index.html')
    
    # Calculate comprehensive statistics
    total_pixels = sum(img.shape[0] * img.shape[1] for img in images)
    total_labeled = sum(np.sum(mask > 0) for mask in masks)
    all_classes = np.unique(np.concatenate([mask.flatten() for mask in masks]))
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Hyperspectral Mineral Gallery - Enhanced</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; 
               background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: #333; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: rgba(255,255,255,0.96); padding: 30px; text-align: center; 
                  border-radius: 15px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }}
        .stats-overview {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 20px; }}
        .stat-box {{ background: rgba(255,255,255,0.15); padding: 20px; border-radius: 10px; text-align: center; backdrop-filter: blur(10px); }}
        .scene {{ background: rgba(255,255,255,0.97); margin: 25px 0; padding: 25px; 
                 border-radius: 15px; box-shadow: 0 8px 25px rgba(0,0,0,0.15); }}
        .scene-title {{ font-size: 1.8em; color: #2c3e50; margin-bottom: 15px; 
                       border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .scene-info {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .enhancement-note {{ background: #e8f5e8; border-left: 4px solid #27ae60; padding: 15px; margin: 15px 0; }}
        .image-gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
        .image-item {{ background: #f8f9fa; padding: 15px; border-radius: 12px; text-align: center; 
                      transition: transform 0.3s, box-shadow 0.3s; }}
        .image-item:hover {{ transform: translateY(-5px); box-shadow: 0 10px 25px rgba(0,0,0,0.15); }}
        .image-item img {{ max-width: 100%; height: auto; border-radius: 8px; 
                          box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .image-caption {{ font-weight: bold; color: #2c3e50; margin: 12px 0 8px 0; }}
        .image-description {{ font-size: 0.9em; color: #666; line-height: 1.4; }}
        .nav {{ position: sticky; top: 0; background: rgba(30, 60, 114, 0.95); padding: 15px; 
               text-align: center; z-index: 100; backdrop-filter: blur(10px); }}
        .nav a {{ color: white; text-decoration: none; margin: 0 20px; font-weight: bold; 
                 padding: 8px 16px; border-radius: 20px; transition: background 0.3s; }}
        .nav a:hover {{ background: rgba(255,255,255,0.2); }}
    </style>
</head>
<body>
    <div class="nav">
        <a href="#overview">🏠 Overview</a>
        <a href="#enhancements">🔬 Enhancements</a>
        <a href="#gallery">🖼️ Gallery</a>
        <a href="#statistics">📊 Statistics</a>
    </div>
    
    <div class="container">
        <div class="header" id="overview">
            <h1>🛰️ CRISM Hyperspectral Mineral Data Gallery</h1>
            <h2 style="color: #555; font-weight: normal;">Enhanced Visualization with Proper Spectral Processing</h2>
            <p style="font-size: 1.2em; color: #666;">Mars Reconnaissance Orbiter • Compact Reconnaissance Imaging Spectrometer for Mars</p>
            <p style="font-size: 1em; color: #888;">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="stats-overview" id="statistics">
            <h2>📊 Enhanced Dataset Overview</h2>
            <div class="stats-grid">
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{len(images)}</h3>
                    <p style="margin: 5px 0;">CRISM Scenes</p>
                </div>
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{total_pixels:,}</h3>
                    <p style="margin: 5px 0;">Total Pixels</p>
                </div>
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{total_labeled:,}</h3>
                    <p style="margin: 5px 0;">Labeled Pixels</p>
                </div>
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{images[0].shape[2] if len(images) > 0 else 'N/A'}</h3>
                    <p style="margin: 5px 0;">Spectral Bands</p>
                </div>
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{len(all_classes)}</h3>
                    <p style="margin: 5px 0;">Mineral Classes</p>
                </div>
                <div class="stat-box">
                    <h3 style="margin: 0; font-size: 2.2em;">{total_labeled/total_pixels*100:.1f}%</h3>
                    <p style="margin: 5px 0;">Mineral Coverage</p>
                </div>
            </div>
        </div>
        
        <div class="enhancement-note" id="enhancements">
            <h3>🔬 Enhanced Processing Techniques</h3>
            <ul>
                <li><strong>Outlier Handling:</strong> Percentile-based normalization (2-98th percentile) to avoid extreme values</li>
                <li><strong>Contrast Enhancement:</strong> Gamma correction for better visual dynamic range</li>
                <li><strong>CRISM-Optimized Bands:</strong> Mineral-specific band combinations for geological interpretation</li>
                <li><strong>Background Handling:</strong> Proper treatment of unlabeled/zero pixels</li>
                <li><strong>False-Color Composites:</strong> Multiple enhancement modes for different mineral types</li>
            </ul>
        </div>
        
        <div id="gallery">
            <h2 style="text-align: center; color: white; font-size: 2.5em; margin: 40px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                🗺️ Enhanced CRISM Image Gallery
            </h2>
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
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                        <div><strong>📐 Dimensions:</strong> {h:,} × {w:,} pixels</div>
                        <div><strong>🌈 Spectral Bands:</strong> {c} channels</div>
                        <div><strong>🏷️ Mineral Classes:</strong> {len(unique_classes)} types</div>
                        <div><strong>📊 Coverage:</strong> {labeled_pixels/total_scene_pixels*100:.1f}% labeled</div>
                    </div>
                    <div style="margin-top: 15px;">
                        <strong>🧪 Minerals Present:</strong> 
                        {', '.join([f'Class {cls}' for cls in unique_classes if cls != 0][:8])}
                        {' ...' if len([cls for cls in unique_classes if cls != 0]) > 8 else ''}
                    </div>
                </div>
                
                <div class="image-gallery">
                    <div class="image-item">
                        <img src="rgb_composites/{scene_id}_enhanced_rgb.png" alt="Enhanced RGB">
                        <div class="image-caption">📸 Enhanced RGB Composite</div>
                        <div class="image-description">Percentile-normalized with outlier handling and gamma correction for optimal visual contrast</div>
                    </div>
                    <div class="image-item">
                        <img src="rgb_composites/{scene_id}_mineral_false_color.png" alt="Mineral Detection">
                        <div class="image-caption">💎 Mineral Detection False-Color</div>
                        <div class="image-description">Optimized for mafic minerals: olivine, pyroxene, and iron oxides using diagnostic CRISM bands</div>
                    </div>
                    <div class="image-item">
                        <img src="rgb_composites/{scene_id}_hydration_detection.png" alt="Hydration Detection">
                        <div class="image-caption">💧 Hydrated Mineral Detection</div>
                        <div class="image-description">Enhanced for clay minerals and hydrated phases using water absorption bands</div>
                    </div>
                    <div class="image-item">
                        <img src="ground_truth/{scene_id}_enhanced_ground_truth.png" alt="Enhanced Ground Truth">
                        <div class="image-caption">🎯 Enhanced Ground Truth</div>
                        <div class="image-description">High-contrast mineral classification map with distinguishable colors for each mineral type</div>
                    </div>
                </div>
            </div>
"""
    
    html_content += f"""
        </div>
        
        <div class="header">
            <h3>🔬 CRISM Hyperspectral Mineral Segmentation Project</h3>
            <p style="font-size: 1.1em; color: #555;">
                Enhanced Deep Learning Pipeline for Martian Surface Mineral Analysis
            </p>
            <p style="font-size: 0.95em; color: #777; line-height: 1.6;">
                This gallery demonstrates proper CRISM hyperspectral data visualization techniques including 
                outlier handling, contrast enhancement, and mineral-specific false-color composites. 
                Each scene shows multiple enhancement modes optimized for different mineral detection tasks.
            </p>
            <div style="margin-top: 20px; font-size: 0.9em; color: #888;">
                <strong>Processing Stats:</strong> {len(images)} scenes • {total_pixels:,} pixels • 
                {len(all_classes)} mineral classes • {total_labeled:,} labeled pixels
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"🌐 Created enhanced HTML gallery: {index_path}")


if __name__ == "__main__":
    main()
