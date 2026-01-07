#!/usr/bin/env python3
"""
Generate Original Satellite Images from CRISM Hyperspectral Data.

This script creates true satellite imagery as seen from Mars orbit:
- Original scene reconstructions showing Mars surface as satellite captured it
- True-color approximations (near-human vision)
- Raw reflectance images
- Original spatial resolution and geometry

Usage:
    python pipelines/generate_satellite_images.py
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
logger = get_logger("CRISM.SatelliteImages")


def create_true_color_satellite_image(image: np.ndarray) -> np.ndarray:
    """
    Create true-color satellite image approximating human vision.
    Uses CRISM bands closest to visible light spectrum.
    
    Args:
        image: Hyperspectral image (H, W, 350)
        
    Returns:
        RGB image approximating human vision
    """
    # CRISM wavelength range is approximately 362-3920 nm
    # Visible spectrum: Red ~700nm, Green ~546nm, Blue ~436nm
    # Map to approximate CRISM band indices
    
    total_bands = image.shape[2]
    
    # Approximate band mapping for CRISM 350 bands covering 362-3920nm range
    # Band 0 ≈ 362nm, Band 349 ≈ 3920nm
    wavelength_per_band = (3920 - 362) / 350
    
    # Find bands closest to visible spectrum
    red_target_nm = 700    # Red
    green_target_nm = 546  # Green  
    blue_target_nm = 436   # Blue
    
    red_band = int((red_target_nm - 362) / wavelength_per_band)
    green_band = int((green_target_nm - 362) / wavelength_per_band) 
    blue_band = int((blue_target_nm - 362) / wavelength_per_band)
    
    # Ensure bands are within valid range
    red_band = min(max(red_band, 0), total_bands - 1)
    green_band = min(max(green_band, 0), total_bands - 1)
    blue_band = min(max(blue_band, 0), total_bands - 1)
    
    logger.info(f"   True-color bands: R={red_band}(~{red_target_nm}nm), "
               f"G={green_band}(~{green_target_nm}nm), B={blue_band}(~{blue_target_nm}nm)")
    
    # Extract visible spectrum bands
    red = image[:, :, red_band]
    green = image[:, :, green_band]
    blue = image[:, :, blue_band]
    
    # Handle ratioed CRISM data (values around 1.0)
    # For ratioed data, enhance contrast around 1.0
    rgb = np.stack([red, green, blue], axis=2)
    
    # Enhanced processing for ratioed reflectance
    for i in range(3):
        band = rgb[:, :, i]
        
        # Find non-zero pixels
        non_zero_mask = band > 0
        if np.sum(non_zero_mask) > 0:
            non_zero_values = band[non_zero_mask]
            
            # For ratioed data, use mean-centered stretching
            mean_val = np.mean(non_zero_values)
            std_val = np.std(non_zero_values)
            
            # Use 3-sigma range for normalization
            lower = max(0, mean_val - 2*std_val)
            upper = mean_val + 2*std_val
            
            # Normalize to [0,1]
            band_norm = np.clip((band - lower) / (upper - lower), 0, 1)
            rgb[:, :, i] = band_norm
        else:
            rgb[:, :, i] = np.zeros_like(band)
    
    return rgb


def create_raw_reflectance_image(image: np.ndarray, band_idx: int = 100) -> np.ndarray:
    """
    Create grayscale image from single spectral band showing raw reflectance.
    
    Args:
        image: Hyperspectral image
        band_idx: Band index to visualize
        
    Returns:
        Normalized grayscale image
    """
    band_idx = min(band_idx, image.shape[2] - 1)
    band = image[:, :, band_idx]
    
    # Non-zero pixel normalization
    non_zero_mask = band > 0
    if np.sum(non_zero_mask) > 0:
        non_zero_values = band[non_zero_mask]
        p5 = np.percentile(non_zero_values, 5)
        p95 = np.percentile(non_zero_values, 95)
        
        normalized = np.clip((band - p5) / (p95 - p5), 0, 1)
    else:
        normalized = np.zeros_like(band)
    
    return normalized


def create_mars_surface_composite(image: np.ndarray) -> np.ndarray:
    """
    Create Mars-like surface composite showing terrain features.
    Uses near-infrared bands that penetrate Mars atmosphere well.
    
    Args:
        image: Hyperspectral image
        
    Returns:
        Mars surface composite
    """
    # Use near-IR bands that show good surface contrast on Mars
    # These bands are less affected by atmospheric scattering
    total_bands = image.shape[2]
    
    # Near-IR bands for Mars surface imaging
    r_band = min(120, total_bands - 1)  # ~1200nm - good surface contrast
    g_band = min(90, total_bands - 1)   # ~900nm - iron absorption edge
    b_band = min(60, total_bands - 1)   # ~600nm - visible red edge
    
    logger.info(f"   Mars surface bands: R={r_band}, G={g_band}, B={b_band}")
    
    rgb = np.stack([image[:, :, r_band], image[:, :, g_band], image[:, :, b_band]], axis=2)
    
    # Enhanced contrast for Mars surface features
    for i in range(3):
        band = rgb[:, :, i]
        non_zero_mask = band > 0
        
        if np.sum(non_zero_mask) > 0:
            non_zero_values = band[non_zero_mask]
            p10 = np.percentile(non_zero_values, 10)
            p90 = np.percentile(non_zero_values, 90)
            
            if p90 > p10:
                band_norm = np.clip((band - p10) / (p90 - p10), 0, 1)
                # Apply slight gamma for Mars-like appearance
                band_norm = np.power(band_norm, 0.8)  
                rgb[:, :, i] = band_norm
            else:
                rgb[:, :, i] = np.zeros_like(band)
        else:
            rgb[:, :, i] = np.zeros_like(band)
    
    return rgb


def main():
    """Generate original satellite images from CRISM hyperspectral data."""
    logger.info("=" * 70)
    logger.info("CRISM Original Satellite Image Generation")
    logger.info("=" * 70)
    
    # Create satellite images directory
    satellite_dir = os.path.join(config.PROCESSED_DATA_DIR, "satellite_images")
    os.makedirs(satellite_dir, exist_ok=True)
    
    try:
        # Load the data
        logger.info("📡 Loading CRISM hyperspectral data for satellite image reconstruction...")
        images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
        
        logger.log_dataset_info(images, masks, scene_ids, label_mapping)
        
        logger.info(f"🛰️ Generating original satellite images for {len(images)} CRISM scenes...")
        
        # Create overview mosaic first
        create_overview_mosaic(images, masks, scene_ids, satellite_dir)
        
        # Process each scene
        for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
            logger.info(f"🗺️ Generating satellite imagery for scene {i+1}/{len(images)}: {scene_id}")
            logger.info(f"   Original size: {image.shape[0]}×{image.shape[1]} pixels")
            logger.info(f"   Spectral range: [{image.min():.4f}, {image.max():.4f}]")
            
            # 1. Original Satellite Image (True Color Approximation)
            true_color = create_true_color_satellite_image(image)
            
            plt.figure(figsize=(16, 12))
            plt.imshow(true_color, extent=[0, image.shape[1], image.shape[0], 0])
            plt.title(f'Original CRISM Satellite Image - {scene_id}\n'
                     f'Mars Reconnaissance Orbiter | Size: {image.shape[0]}×{image.shape[1]} pixels | '
                     f'True-color approximation using visible spectrum bands',
                     fontsize=16, pad=20)
            
            # Add coordinate grid
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlabel('Pixel X Coordinate', fontsize=12)
            plt.ylabel('Pixel Y Coordinate', fontsize=12)
            
            # Add metadata
            plt.figtext(0.02, 0.02, 
                       f'CRISM Scene: {scene_id}\n'
                       f'Instrument: Compact Reconnaissance Imaging Spectrometer for Mars\n'
                       f'Platform: Mars Reconnaissance Orbiter\n'
                       f'Processing: Enhanced true-color composite from 350-band hyperspectral data\n'
                       f'Spatial Resolution: Original CRISM pixel scale\n'
                       f'Spectral Coverage: 362-3920 nm (350 channels)',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9))
            
            satellite_path = os.path.join(satellite_dir, f'{scene_id}_original_satellite.png')
            plt.savefig(satellite_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 2. Mars Surface Composite (Near-IR optimized)
            mars_surface = create_mars_surface_composite(image)
            
            plt.figure(figsize=(16, 12))
            plt.imshow(mars_surface, extent=[0, image.shape[1], image.shape[0], 0])
            plt.title(f'Mars Surface Composite - {scene_id}\n'
                     f'Near-Infrared Enhanced | Optimized for Martian Surface Features',
                     fontsize=16, pad=20)
            
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlabel('Pixel X Coordinate', fontsize=12)
            plt.ylabel('Pixel Y Coordinate', fontsize=12)
            
            mars_path = os.path.join(satellite_dir, f'{scene_id}_mars_surface.png')
            plt.savefig(mars_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 3. Raw Reflectance Map (Single Band)
            raw_reflectance = create_raw_reflectance_image(image, band_idx=100)
            
            plt.figure(figsize=(16, 12))
            plt.imshow(raw_reflectance, cmap='gray', extent=[0, image.shape[1], image.shape[0], 0])
            plt.title(f'Raw Reflectance - {scene_id}\n'
                     f'Single Band (Band 100) | Grayscale Reflectance Map',
                     fontsize=16, pad=20)
            
            plt.colorbar(label='Normalized Reflectance', shrink=0.8)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.xlabel('Pixel X Coordinate', fontsize=12)
            plt.ylabel('Pixel Y Coordinate', fontsize=12)
            
            raw_path = os.path.join(satellite_dir, f'{scene_id}_raw_reflectance.png')
            plt.savefig(raw_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            # 4. Multi-band Overview (showing spectral richness)
            create_multiband_overview(image, scene_id, satellite_dir)
            
            logger.info(f"   ✅ Generated: Original satellite, Mars surface, raw reflectance, and multi-band")
        
        # Create satellite image index
        create_satellite_html_index(images, masks, scene_ids, satellite_dir)
        
        logger.info(f"\n🛰️ Original satellite image generation completed!")
        logger.info(f"   📊 Generated: {len(images)} complete satellite image sets")
        logger.info(f"   📁 Images per scene: 4 (original, Mars surface, raw reflectance, multi-band)")
        logger.info(f"   🌐 Browse satellite images: data/processed/satellite_images/index.html")
        
    except Exception as e:
        logger.error(f"❌ Satellite image generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_multiband_overview(image: np.ndarray, scene_id: str, save_dir: str):
    """
    Create multi-band overview showing the spectral richness of CRISM data.
    """
    h, w, bands = image.shape
    
    # Select representative bands across the spectrum
    band_indices = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    band_indices = [min(b, bands-1) for b in band_indices]
    
    # Create 3x4 grid of single-band images
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, band_idx in enumerate(band_indices):
        # Get wavelength estimate
        wavelength = 362 + band_idx * (3920 - 362) / 350
        
        # Normalize single band
        band_data = image[:, :, band_idx]
        non_zero_mask = band_data > 0
        
        if np.sum(non_zero_mask) > 0:
            non_zero_values = band_data[non_zero_mask]
            p5 = np.percentile(non_zero_values, 5)
            p95 = np.percentile(non_zero_values, 95)
            
            if p95 > p5:
                band_norm = np.clip((band_data - p5) / (p95 - p5), 0, 1)
            else:
                band_norm = np.zeros_like(band_data)
        else:
            band_norm = np.zeros_like(band_data)
        
        # Display band
        axes[i].imshow(band_norm, cmap='viridis')
        axes[i].set_title(f'Band {band_idx}\n~{wavelength:.0f} nm', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle(f'Multi-Band Spectral Overview - {scene_id}\n'
                 f'Representative bands across CRISM 350-channel hyperspectral range',
                 fontsize=18)
    plt.tight_layout()
    
    multiband_path = os.path.join(save_dir, f'{scene_id}_multiband_overview.png')
    plt.savefig(multiband_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def create_overview_mosaic(images, masks, scene_ids, save_dir):
    """
    Create overview mosaic showing all satellite images in one view.
    """
    logger.info("🗺️ Creating overview mosaic of all satellite images...")
    
    if len(images) == 0:
        return
    
    # Calculate grid dimensions
    n_scenes = len(images)
    cols = min(4, n_scenes)
    rows = (n_scenes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        # Create true color for overview
        true_color = create_true_color_satellite_image(image)
        
        axes[i].imshow(true_color)
        axes[i].set_title(f'{scene_id}\n{image.shape[0]}×{image.shape[1]}', fontsize=12)
        axes[i].axis('off')
        
        # Add mineral info
        unique_classes = np.unique(mask)
        mineral_count = len(unique_classes) - 1  # Exclude background
        axes[i].text(0.02, 0.98, f'{mineral_count} minerals', 
                    transform=axes[i].transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    verticalalignment='top')
    
    # Hide empty subplots
    for i in range(n_scenes, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'CRISM Satellite Image Overview\n'
                 f'{n_scenes} Mars Surface Scenes from Mars Reconnaissance Orbiter',
                 fontsize=20)
    plt.tight_layout()
    
    overview_path = os.path.join(save_dir, 'crism_overview_mosaic.png')
    plt.savefig(overview_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"🌐 Created overview mosaic: {overview_path}")


def create_satellite_html_index(images, masks, scene_ids, satellite_dir):
    """Create HTML index specifically for satellite images."""
    from datetime import datetime
    
    index_path = os.path.join(satellite_dir, 'index.html')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM Original Satellite Images</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #000814; color: white; }}
        .header {{ background: linear-gradient(135deg, #001d3d 0%, #003566 100%); 
                  padding: 30px; text-align: center; border-radius: 15px; margin-bottom: 30px; }}
        .mission-info {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .scene {{ background: rgba(255,255,255,0.05); margin: 25px 0; padding: 20px; 
                 border-radius: 12px; border: 1px solid #ffd60a; }}
        .scene-title {{ font-size: 1.6em; color: #ffd60a; margin-bottom: 15px; }}
        .satellite-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 15px; }}
        .sat-image {{ text-align: center; background: rgba(255,255,255,0.08); padding: 15px; border-radius: 8px; }}
        .sat-image img {{ max-width: 100%; height: auto; border: 2px solid #003566; border-radius: 8px; }}
        .sat-caption {{ font-weight: bold; color: #ffd60a; margin-top: 10px; }}
        .sat-description {{ font-size: 0.9em; color: #ccc; line-height: 1.4; margin-top: 5px; }}
        .overview {{ text-align: center; background: rgba(255,255,255,0.08); padding: 20px; border-radius: 10px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ CRISM Original Satellite Images</h1>
        <h2>Mars Reconnaissance Orbiter • Original Scene Reconstructions</h2>
        <p>Authentic Mars surface imagery from hyperspectral orbital data</p>
        <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
    </div>
    
    <div class="mission-info">
        <h3>🚀 Mission Information</h3>
        <ul>
            <li><strong>Spacecraft:</strong> Mars Reconnaissance Orbiter (MRO)</li>
            <li><strong>Instrument:</strong> Compact Reconnaissance Imaging Spectrometer for Mars (CRISM)</li>
            <li><strong>Spectral Range:</strong> 362-3920 nanometers (350 channels)</li>
            <li><strong>Spatial Resolution:</strong> Variable (15-40 meters per pixel typical)</li>
            <li><strong>Mission:</strong> Mars surface mineral composition and geology mapping</li>
        </ul>
    </div>
    
    <div class="overview">
        <h3>🗺️ Complete Mission Overview</h3>
        <img src="crism_overview_mosaic.png" alt="CRISM Overview Mosaic" style="max-width: 100%; border: 2px solid #ffd60a; border-radius: 10px;">
        <p style="margin-top: 15px; font-style: italic;">
            Overview mosaic showing all {len(images)} CRISM scenes as they appear from Mars orbit
        </p>
    </div>
"""
    
    # Add each satellite scene
    for i, (image, mask, scene_id) in enumerate(zip(images, masks, scene_ids)):
        h, w, c = image.shape
        unique_classes = np.unique(mask)
        mineral_classes = len(unique_classes) - 1  # Exclude background
        
        html_content += f"""
    <div class="scene">
        <div class="scene-title">📡 CRISM Scene {i+1}: {scene_id}</div>
        <div style="margin-bottom: 20px; background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
            <strong>📐 Scene Characteristics:</strong><br>
            Spatial Dimensions: {h:,} × {w:,} pixels | 
            Spectral Channels: {c} bands | 
            Mineral Types Detected: {mineral_classes} | 
            Total Scene Coverage: {h*w:,} pixels
        </div>
        
        <div class="satellite-grid">
            <div class="sat-image">
                <img src="{scene_id}_original_satellite.png" alt="Original Satellite">
                <div class="sat-caption">🛰️ Original Satellite Image</div>
                <div class="sat-description">
                    True-color approximation using visible spectrum bands from CRISM hyperspectral data.
                    This is how the Martian surface appears from orbit.
                </div>
            </div>
            <div class="sat-image">
                <img src="{scene_id}_mars_surface.png" alt="Mars Surface">
                <div class="sat-caption">🔴 Mars Surface Composite</div>
                <div class="sat-description">
                    Near-infrared enhanced composite optimized for Martian surface features.
                    Shows terrain details with enhanced contrast.
                </div>
            </div>
            <div class="sat-image">
                <img src="{scene_id}_raw_reflectance.png" alt="Raw Reflectance">
                <div class="sat-caption">📊 Raw Reflectance Map</div>
                <div class="sat-description">
                    Single-band grayscale showing raw surface reflectance values.
                    Direct measurement from CRISM spectrometer.
                </div>
            </div>
            <div class="sat-image">
                <img src="{scene_id}_multiband_overview.png" alt="Multi-band">
                <div class="sat-caption">🌈 Multi-band Spectral Overview</div>
                <div class="sat-description">
                    12 representative bands showing the hyperspectral data richness
                    across the complete 350-channel CRISM spectrum.
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
    <div class="header">
        <h3>🔬 About CRISM Hyperspectral Data</h3>
        <p style="line-height: 1.6; font-size: 1em;">
            These are the original satellite images reconstructed from CRISM hyperspectral data. 
            Each pixel contains 350 spectral measurements allowing scientists to identify 
            mineral compositions on the Martian surface. The images show Mars as seen from 
            the Mars Reconnaissance Orbiter spacecraft using various spectral enhancements.
        </p>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 Created satellite image gallery: {index_path}")


if __name__ == "__main__":
    main()
