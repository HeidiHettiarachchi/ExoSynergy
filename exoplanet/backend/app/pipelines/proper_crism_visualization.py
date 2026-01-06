#!/usr/bin/env python3
"""
Proper CRISM RGB Visualization - Enhanced for Ratioed Reflectance Data.

This script creates proper RGB images from CRISM ratioed spectral data by:
- Enhancing subtle spectral variations around 1.0
- Using decorrelation stretch for better color separation
- Applying proper CRISM band combinations for Mars geology
- Creating natural-looking satellite imagery

Usage:
    python pipelines/proper_crism_visualization.py
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
logger = get_logger("CRISM.ProperVisualization")


def decorrelation_stretch(rgb: np.ndarray) -> np.ndarray:
    """
    Apply decorrelation stretch to enhance color separation.
    This technique enhances subtle spectral differences in ratioed data.
    """
    h, w, c = rgb.shape
    
    # Reshape to (pixels, bands)
    pixels = rgb.reshape(-1, c)
    
    # Remove zero/background pixels
    non_zero_mask = np.any(pixels > 0, axis=1)
    if np.sum(non_zero_mask) == 0:
        return rgb
    
    valid_pixels = pixels[non_zero_mask]
    
    # Calculate covariance matrix
    mean_vals = np.mean(valid_pixels, axis=0)
    centered = valid_pixels - mean_vals
    
    try:
        cov_matrix = np.cov(centered.T)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        # Avoid negative eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-10)
        
        # Create decorrelation matrix
        decorr_matrix = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
        
        # Apply decorrelation
        decorrelated = (centered @ decorr_matrix.T) + mean_vals
        
        # Normalize to [0, 1]
        for i in range(c):
            band = decorrelated[:, i]
            p5 = np.percentile(band, 5)
            p95 = np.percentile(band, 95)
            if p95 > p5:
                decorrelated[:, i] = np.clip((band - p5) / (p95 - p5), 0, 1)
        
        # Reconstruct image
        result = np.zeros_like(pixels)
        result[non_zero_mask] = decorrelated
        
        return result.reshape(h, w, c)
        
    except np.linalg.LinAlgError:
        # Fallback to simple contrast enhancement
        return enhance_ratioed_contrast(rgb)


def enhance_ratioed_contrast(rgb: np.ndarray, amplification: float = 3.0) -> np.ndarray:
    """
    Enhance contrast in ratioed CRISM data by amplifying deviations from 1.0.
    
    Args:
        rgb: RGB array with values around 1.0
        amplification: How much to amplify deviations from 1.0
        
    Returns:
        Enhanced RGB with better color contrast
    """
    h, w, c = rgb.shape
    
    for i in range(c):
        band = rgb[:, :, i]
        
        # Only process non-zero pixels
        non_zero_mask = band > 0
        if np.sum(non_zero_mask) > 0:
            # Calculate deviations from 1.0 (baseline for ratioed data)
            deviations = band - 1.0
            
            # Amplify deviations
            enhanced_deviations = deviations * amplification
            
            # Add back to baseline
            enhanced_band = 1.0 + enhanced_deviations
            
            # Apply to only non-zero pixels
            result_band = band.copy()
            result_band[non_zero_mask] = enhanced_band[non_zero_mask]
            
            # Normalize to [0, 1] with percentile clipping
            non_zero_values = result_band[non_zero_mask]
            if len(non_zero_values) > 0:
                p1 = np.percentile(non_zero_values, 1)
                p99 = np.percentile(non_zero_values, 99)
                
                if p99 > p1:
                    result_band[non_zero_mask] = np.clip((non_zero_values - p1) / (p99 - p1), 0, 1)
                else:
                    result_band[non_zero_mask] = 0.5  # Mid-gray for constant values
            
            rgb[:, :, i] = result_band
    
    return rgb


def create_natural_mars_rgb(image: np.ndarray) -> np.ndarray:
    """
    Create natural-looking Mars RGB using enhanced CRISM techniques.
    Uses bands that provide good visual contrast on Mars.
    """
    # Use bands that work well for Mars surface imaging
    # These are empirically good for Mars geology visualization
    total_bands = image.shape[2]
    
    # Near-visible and near-IR bands that show good Mars surface contrast
    red_band = min(45, total_bands - 1)    # ~800nm (near-IR)
    green_band = min(25, total_bands - 1)  # ~600nm (visible red-orange)
    blue_band = min(15, total_bands - 1)   # ~500nm (blue-green)
    
    logger.info(f"   Natural Mars RGB bands: R={red_band}, G={green_band}, B={blue_band}")
    
    # Extract bands
    red = image[:, :, red_band].copy()
    green = image[:, :, green_band].copy() 
    blue = image[:, :, blue_band].copy()
    
    rgb = np.stack([red, green, blue], axis=2)
    
    # Enhanced processing for ratioed data
    enhanced_rgb = enhance_ratioed_contrast(rgb, amplification=2.5)
    
    # Apply decorrelation stretch for better color separation
    final_rgb = decorrelation_stretch(enhanced_rgb)
    
    return final_rgb


def create_high_contrast_mars_rgb(image: np.ndarray) -> np.ndarray:
    """
    Create high-contrast Mars RGB emphasizing geological features.
    """
    total_bands = image.shape[2]
    
    # High-contrast band combination
    red_band = min(80, total_bands - 1)    # ~1200nm
    green_band = min(50, total_bands - 1)  # ~850nm
    blue_band = min(20, total_bands - 1)   # ~550nm
    
    logger.info(f"   High-contrast bands: R={red_band}, G={green_band}, B={blue_band}")
    
    rgb = np.stack([image[:, :, red_band], image[:, :, green_band], image[:, :, blue_band]], axis=2)
    
    # Strong enhancement for geological features
    enhanced_rgb = enhance_ratioed_contrast(rgb, amplification=4.0)
    
    # Apply gamma correction for Mars-like appearance
    enhanced_rgb = np.power(enhanced_rgb, 0.7)
    
    return enhanced_rgb


def main():
    """Generate proper RGB satellite images from CRISM ratioed data."""
    logger.info("=" * 70)
    logger.info("CRISM Proper RGB Visualization - Enhanced for Ratioed Data")
    logger.info("=" * 70)
    
    # Clear previous satellite images
    satellite_dir = os.path.join(config.PROCESSED_DATA_DIR, "proper_satellite_images")
    os.makedirs(satellite_dir, exist_ok=True)
    
    try:
        # Load the data
        logger.info("📡 Loading CRISM data for proper RGB visualization...")
        images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
        
        logger.log_dataset_info(images, masks, scene_ids, label_mapping)
        
        logger.info(f"🎨 Creating proper RGB satellite images for {len(images)} scenes...")
        logger.info("💡 Using enhanced techniques for CRISM ratioed reflectance data")
        
        # Process each scene
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            logger.info(f"🛰️ Processing scene {i+1}/{len(images)}: {scene_id}")
            logger.info(f"   Original dimensions: {image.shape[0]}×{image.shape[1]} pixels")
            logger.info(f"   Data range: [{image.min():.4f}, {image.max():.4f}]")
            
            # Get data statistics for this scene
            non_zero_pixels = np.sum(image != 0)
            total_pixels = image.size
            
            logger.info(f"   Non-zero pixels: {non_zero_pixels:,}/{total_pixels:,} "
                       f"({non_zero_pixels/total_pixels*100:.1f}%)")
            
            # 1. Natural Mars RGB (enhanced for ratioed data)
            natural_rgb = create_natural_mars_rgb(image)
            
            plt.figure(figsize=(18, 12))
            plt.imshow(natural_rgb)
            plt.title(f'Natural Mars Surface Image - {scene_id}\n'
                     f'CRISM Original Satellite Data | Enhanced for Ratioed Reflectance\n'
                     f'Size: {image.shape[0]}×{image.shape[1]} pixels | '
                     f'Processing: Contrast enhancement + Decorrelation stretch',
                     fontsize=16, pad=25)
            plt.axis('off')
            
            # Add technical information
            plt.figtext(0.02, 0.02, 
                       f'CRISM Scene: {scene_id}\n'
                       f'Mars Reconnaissance Orbiter\n'
                       f'Data Type: Ratioed Reflectance (values centered at 1.0)\n'
                       f'Enhancement: Deviation amplification + Decorrelation stretch\n'
                       f'Non-zero coverage: {non_zero_pixels/total_pixels*100:.1f}%\n'
                       f'Spectral bands: 350 channels (362-3920nm)',
                       fontsize=11, bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.95))
            
            natural_path = os.path.join(satellite_dir, f'{scene_id}_natural_satellite.png')
            plt.savefig(natural_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. High-contrast Mars RGB
            contrast_rgb = create_high_contrast_mars_rgb(image)
            
            plt.figure(figsize=(18, 12))
            plt.imshow(contrast_rgb)
            plt.title(f'High-Contrast Mars Surface - {scene_id}\n'
                     f'Enhanced Geological Features | Amplified Spectral Variations\n'
                     f'Optimized for Terrain and Mineral Visibility',
                     fontsize=16, pad=25)
            plt.axis('off')
            
            contrast_path = os.path.join(satellite_dir, f'{scene_id}_high_contrast.png')
            plt.savefig(contrast_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 3. Scientific RGB (showing the actual data values)
            create_scientific_rgb_view(image, scene_id, satellite_dir)
            
            logger.info(f"   ✅ Generated: Natural Mars RGB, high-contrast, and scientific view")
        
        # Create proper satellite gallery
        create_proper_satellite_gallery(images, masks, scene_ids, satellite_dir)
        
        logger.info(f"\n🎉 Proper RGB satellite images completed!")
        logger.info(f"   📊 Enhanced: {len(images)} scenes with proper CRISM visualization")
        logger.info(f"   🖼️ Images per scene: 3 (natural, high-contrast, scientific)")
        logger.info(f"   🌐 Gallery: data/processed/proper_satellite_images/index.html")
        
    except Exception as e:
        logger.error(f"❌ RGB processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_scientific_rgb_view(image: np.ndarray, scene_id: str, save_dir: str):
    """
    Create scientific RGB view showing the actual ratioed data characteristics.
    """
    # Show how the data actually looks with minimal processing
    total_bands = image.shape[2]
    
    # Use well-separated bands
    r_band = min(100, total_bands - 1)
    g_band = min(60, total_bands - 1)
    b_band = min(30, total_bands - 1)
    
    red = image[:, :, r_band]
    green = image[:, :, g_band] 
    blue = image[:, :, b_band]
    
    # Show actual data distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Raw RGB (minimal enhancement)
    rgb_minimal = np.stack([red, green, blue], axis=2)
    
    # Only enhance non-zero pixels slightly
    for i in range(3):
        band = rgb_minimal[:, :, i]
        non_zero_mask = band > 0
        if np.sum(non_zero_mask) > 0:
            non_zero_values = band[non_zero_mask]
            # Very gentle enhancement around 1.0
            enhanced = 0.5 + (non_zero_values - 1.0) * 2.0  # Amplify deviations from 1.0
            enhanced = np.clip(enhanced, 0, 1)
            band[non_zero_mask] = enhanced
    
    ax1.imshow(rgb_minimal)
    ax1.set_title('Scientific RGB\n(Minimal Enhancement)', fontsize=12)
    ax1.axis('off')
    
    # Individual bands
    ax2.imshow(red, cmap='Reds', vmin=0, vmax=red.max())
    ax2.set_title(f'Red Band ({r_band})\nRange: [{red.min():.3f}, {red.max():.3f}]', fontsize=12)
    ax2.axis('off')
    
    ax3.imshow(green, cmap='Greens', vmin=0, vmax=green.max())
    ax3.set_title(f'Green Band ({g_band})\nRange: [{green.min():.3f}, {green.max():.3f}]', fontsize=12)
    ax3.axis('off')
    
    ax4.imshow(blue, cmap='Blues', vmin=0, vmax=blue.max())
    ax4.set_title(f'Blue Band ({b_band})\nRange: [{blue.min():.3f}, {blue.max():.3f}]', fontsize=12)
    ax4.axis('off')
    
    plt.suptitle(f'Scientific Data View - {scene_id}\n'
                 f'Showing actual CRISM ratioed reflectance values', fontsize=16)
    plt.tight_layout()
    
    scientific_path = os.path.join(save_dir, f'{scene_id}_scientific_view.png')
    plt.savefig(scientific_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_proper_satellite_gallery(images, masks, scene_ids, satellite_dir):
    """Create gallery specifically for proper RGB satellite images."""
    from datetime import datetime
    
    index_path = os.path.join(satellite_dir, 'index.html')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Proper RGB Satellite Images</title>
    <style>
        body {{ font-family: 'Arial', sans-serif; margin: 0; background: #0a0a0a; color: #ffffff; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(45deg, #d32f2f 0%, #ff6b35 100%); 
                  padding: 40px; text-align: center; border-radius: 20px; margin-bottom: 40px; }}
        .technical-note {{ background: rgba(255,193,7,0.1); border-left: 5px solid #ffc107; 
                          padding: 20px; margin: 30px 0; border-radius: 8px; }}
        .scene {{ background: rgba(255,255,255,0.05); margin: 30px 0; padding: 25px; 
                 border-radius: 15px; border: 2px solid #ff6b35; }}
        .scene-title {{ font-size: 2em; color: #ff6b35; margin-bottom: 20px; text-align: center; }}
        .image-showcase {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 25px; }}
        .image-panel {{ background: rgba(255,255,255,0.08); padding: 20px; border-radius: 12px; text-align: center; }}
        .image-panel img {{ max-width: 100%; height: auto; border-radius: 10px; 
                           box-shadow: 0 8px 25px rgba(255,107,53,0.3); }}
        .image-title {{ font-size: 1.3em; color: #ffc107; margin: 15px 0 10px 0; font-weight: bold; }}
        .image-details {{ font-size: 0.95em; color: #ccc; line-height: 1.5; }}
        .data-quality {{ background: rgba(76,175,80,0.2); padding: 15px; border-radius: 8px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🪐 CRISM Proper RGB Satellite Images</h1>
            <h2>Mars as Seen from Orbit - Enhanced for Human Vision</h2>
            <p style="font-size: 1.2em;">Mars Reconnaissance Orbiter • Hyperspectral Enhanced RGB</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="technical-note">
            <h3>🔬 Why CRISM Images Look Different</h3>
            <p><strong>CRISM uses ratioed reflectance data</strong> where values are centered around 1.0:</p>
            <ul>
                <li><strong>Value = 1.0:</strong> Normal/average reflectance for that wavelength</li>
                <li><strong>Value > 1.0:</strong> Higher than average reflectance (bright features)</li>
                <li><strong>Value < 1.0:</strong> Lower than average reflectance (dark features)</li>
            </ul>
            <p><strong>Enhancement Applied:</strong> Deviations from 1.0 are amplified and decorrelation stretch is applied to create visible color differences from subtle spectral variations.</p>
        </div>
"""
    
    # Add each scene
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        h, w, c = image.shape
        non_zero_coverage = np.sum(image != 0) / image.size * 100
        value_range = f"[{image.min():.3f}, {image.max():.3f}]"
        
        html_content += f"""
        <div class="scene">
            <div class="scene-title">🛰️ CRISM Scene: {scene_id}</div>
            
            <div class="data-quality">
                <strong>📊 Scene Characteristics:</strong><br>
                Dimensions: {h:,} × {w:,} pixels | 
                Spectral Channels: {c} bands | 
                Data Coverage: {non_zero_coverage:.1f}% | 
                Value Range: {value_range}
            </div>
            
            <div class="image-showcase">
                <div class="image-panel">
                    <img src="{scene_id}_natural_satellite.png" alt="Natural Mars RGB">
                    <div class="image-title">🌍 Natural Mars RGB</div>
                    <div class="image-details">
                        Enhanced natural-looking Mars surface using near-visible bands.
                        Contrast amplified around 1.0 baseline with decorrelation stretch
                        for realistic Martian terrain appearance.
                    </div>
                </div>
                <div class="image-panel">
                    <img src="{scene_id}_high_contrast.png" alt="High-Contrast Mars">
                    <div class="image-title">⚡ High-Contrast Enhancement</div>
                    <div class="image-details">
                        Maximum contrast enhancement emphasizing geological features.
                        Strong amplification of spectral variations with gamma correction
                        for detailed surface structure visibility.
                    </div>
                </div>
                <div class="image-panel">
                    <img src="{scene_id}_scientific_view.png" alt="Scientific View">
                    <div class="image-title">🔬 Scientific Data View</div>
                    <div class="image-details">
                        Minimal processing showing actual CRISM ratioed data values.
                        Individual bands displayed separately to show true data
                        characteristics and spectral response patterns.
                    </div>
                </div>
            </div>
        </div>
"""
    
    html_content += """
        <div class="technical-note">
            <h3>📡 About CRISM Satellite Data</h3>
            <p>
                These images represent the actual Mars surface as captured by the CRISM hyperspectral instrument.
                Unlike regular cameras, CRISM measures 350 different wavelengths of light for each pixel,
                allowing scientists to identify mineral compositions. The "ratioed" data is normalized
                to remove atmospheric and illumination effects, resulting in values centered around 1.0.
            </p>
        </div>
        
        <div class="header">
            <h3>🚀 Mars Reconnaissance Orbiter Mission</h3>
            <p>Original hyperspectral satellite imagery processed for human visualization</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 Created proper RGB gallery: {index_path}")


if __name__ == "__main__":
    main()
