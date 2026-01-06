"""
CRISM ML Integration Module
Integrates proven CRISM ML functions with our pipeline for proper image handling.

Based on: Plebani et al. (2022) "A machine learning toolkit for CRISM image analysis", Icarus
Repository: https://github.com/Banus/crism_ml.git
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import bisect

# Add CRISM ML to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from . import config
from .logger import get_logger

# Initialize logger
logger = get_logger("CRISM.ML_Integration")


def norm_minmax(data, vmin=None, vmax=None):
    """Min-max normalization (from CRISM ML preprocessing.py)."""
    if vmin is None:
        vmin = np.min(data, axis=0, keepdims=True)
    if vmax is None:
        vmax = np.max(data, axis=0, keepdims=True)
    
    # Avoid division by zero
    denom = vmax - vmin
    return np.where(denom > 0, (data - vmin) / denom, np.zeros_like(data))


def imadjust(src, tol=5, vin=(0, 255), vout=(0, 255)):
    """
    Adjust image histogram (from CRISM ML plot.py).
    MATLAB-style histogram equalization.
    """
    tol = max(0, min(100, tol))
    if tol > 0:
        # Ensure src is in proper range for histogram
        src_adj = np.clip(src, 0, 255)
        hist = np.histogram(src_adj, bins=list(range(256)), range=(0, 255))[0]
        cum = np.cumsum(hist)

        size = src.shape[0] * src.shape[1]
        if size > 0:
            lb_, ub_ = size * tol / 100, size * (100 - tol) / 100
            vin = (bisect.bisect_left(cum, lb_), bisect.bisect_left(cum, ub_))

    if vin[1] > vin[0]:  # Avoid division by zero
        scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
        vs_ = src - vin[0]
        vs_[src < vin[0]] = 0
        vd_ = vs_*scale + 0.5 + vout[0]
        vd_[vd_ > vout[1]] = vout[1]
        return vd_
    else:
        return src


def get_crism_false_colors(pixspec, badpix, channels=(233, 103, 20)):
    """
    Create false color image using exact CRISM ML methodology.
    
    From CRISM ML plot.py get_false_colors() function.
    This is the proven method for CRISM visualization.
    
    Args:
        pixspec: Spectra shaped as (h, w, n_channels) array
        badpix: Boolean mask of bad pixels to exclude
        channels: Tuple of three indices for R, G, B bands
        
    Returns:
        RGB image with proper CRISM visualization [0,1] range
    """
    shape = badpix.shape
    badpix_flat, pixspec_flat = badpix.ravel(), pixspec.reshape((-1, pixspec.shape[-1]))

    # Apply median filter around selected bands (key CRISM ML technique)
    lsize, rsize = 8, 9  # median filter size: 17 total
    
    # Ensure channels are within bounds
    n_channels = pixspec.shape[-1]
    channels = tuple(min(ch, n_channels - 1) for ch in channels)
    
    img = np.stack([
        np.median(pixspec_flat[:, max(i - lsize, 0):i + rsize], axis=1)
        for i in channels
    ]).T

    # L2 normalization on good pixels only
    goodpx = img[~badpix_flat, :]
    if len(goodpx) > 0:
        # Calculate L2 norms
        l2_norms = np.sqrt(np.einsum('ij,ij->i', goodpx, goodpx))
        mean_l2_per_channel = np.mean(l2_norms.reshape(-1, 1) * goodpx / (l2_norms.reshape(-1, 1) + 1e-8), axis=0, keepdims=True)
        
        # Avoid division by zero
        mean_l2_per_channel = np.where(mean_l2_per_channel == 0, 1.0, mean_l2_per_channel)
        img /= mean_l2_per_channel
        
        # Min-max normalization on good pixels
        vmin = np.min(img[~badpix_flat, :], axis=0, keepdims=True)
        vmax = np.max(img[~badpix_flat, :], axis=0, keepdims=True)
        
        img = 255 * norm_minmax(img, vmin=vmin, vmax=vmax).reshape(shape + (-1,))
    else:
        img = np.zeros(shape + (3,))

    # Histogram equalization per channel (crucial step)
    img_result = []
    for i in range(3):
        channel = img[:, :, i]
        if np.max(channel) > np.min(channel):  # Only adjust if there's variation
            adjusted = imadjust(channel)
            img_result.append(adjusted)
        else:
            img_result.append(channel)
    
    img = np.stack(img_result, axis=2)
    return np.clip(img / 255.0, 0, 1)


def create_bad_pixel_mask(image, threshold_factor=1000):
    """
    Create bad pixel mask using CRISM ML criteria.
    
    Args:
        image: Hyperspectral image (H, W, C)
        threshold_factor: Threshold for detecting bad pixels
        
    Returns:
        Boolean mask of bad pixels
    """
    h, w, c = image.shape
    badpix = np.zeros((h, w), dtype=bool)
    
    # Check each band for invalid values (CRISM ML approach)
    for band_idx in range(c):
        band = image[:, :, band_idx]
        
        # CRISM ML criteria for bad pixels
        invalid_mask = (
            np.isnan(band) | 
            np.isinf(band) | 
            (band > threshold_factor) |  # Very high values (CRISM uses 65535, but be conservative)
            (band == 0) |                # Zero values (background/invalid)
            (band < 0)                   # Negative values (shouldn't occur in reflectance)
        )
        badpix |= invalid_mask
    
    return badpix


def load_and_process_individual_images():
    """
    Load and process individual CRISM images using CRISM ML methodology.
    
    Returns:
        List of processed individual images with metadata
    """
    logger.info("📡 Loading individual CRISM images with CRISM ML approach...")
    
    # Load raw data directly (following CRISM ML pattern)
    try:
        from crism_ml.io import loadmat
        mat_data = loadmat(config.LABELED_DATA_PATH)
    except ImportError:
        # Fallback to scipy if CRISM ML not available
        import scipy.io as sio
        mat_data = sio.loadmat(config.LABELED_DATA_PATH)
    
    # Extract data following CRISM ML structure
    pixspec = mat_data['pixspec']  # (n_pixels, 350)
    pixcrds = mat_data['pixcrds']  # (n_pixels, 2) 
    pixims = mat_data['pixims'].flatten()    # (n_pixels,) image IDs
    pixlabs = mat_data['pixlabs'].flatten()  # (n_pixels,) labels
    
    logger.info(f"📊 CRISM ML Data Structure:")
    logger.info(f"   Pixels: {len(pixspec):,}")
    logger.info(f"   Spectral channels: {pixspec.shape[1]}")
    logger.info(f"   Unique images: {len(np.unique(pixims))}")
    logger.info(f"   Label range: {pixlabs.min()}-{pixlabs.max()}")
    
    # Get unique image IDs
    unique_images = np.unique(pixims)
    logger.info(f"   Processing {len(unique_images)} individual images...")
    
    processed_images = []
    
    # Process each image individually (key improvement)
    for idx, img_id in enumerate(unique_images):
        if idx >= 20:  # Limit for demonstration
            logger.info(f"   Limiting to first 20 images for processing efficiency")
            break
            
        logger.info(f"🖼️ Individual image {idx+1}/{min(20, len(unique_images))}: ID {img_id}")
        
        # Get pixels for this specific image
        img_mask = pixims == img_id
        img_spectra = pixspec[img_mask]     # (n_img_pixels, 350)
        img_coords = pixcrds[img_mask]      # (n_img_pixels, 2)
        img_labels = pixlabs[img_mask]      # (n_img_pixels,)
        
        if len(img_spectra) == 0:
            logger.warning(f"   No pixels found for image {img_id}")
            continue
        
        # Calculate image dimensions from coordinates
        x_coords = img_coords[:, 0].astype(int)
        y_coords = img_coords[:, 1].astype(int)
        
        height = y_coords.max() - y_coords.min() + 1
        width = x_coords.max() - x_coords.min() + 1
        
        logger.info(f"   Dimensions: {height}×{width} pixels")
        logger.info(f"   Valid pixels: {len(img_spectra):,}/{height*width:,} ({len(img_spectra)/(height*width)*100:.1f}%)")
        
        # Skip images that are too large or too sparse
        if height * width > 300000:  # Memory constraint
            logger.info(f"   Skipping large image: {height}×{width}")
            continue
            
        if len(img_spectra) / (height * width) < 0.01:  # Less than 1% coverage
            logger.info(f"   Skipping sparse image: {len(img_spectra)/(height*width)*100:.2f}% coverage")
            continue
        
        # Reconstruct image cube
        img_cube = np.zeros((height, width, pixspec.shape[1]), dtype=np.float64)
        label_map = np.zeros((height, width), dtype=np.int32)
        
        # Place pixels at correct locations
        for spec, coord, label in zip(img_spectra, img_coords, img_labels):
            y = int(coord[1]) - y_coords.min()
            x = int(coord[0]) - x_coords.min()
            
            if 0 <= y < height and 0 <= x < width:
                img_cube[y, x] = spec
                label_map[y, x] = label
        
        # Create bad pixel mask using CRISM ML criteria
        badpix = create_bad_pixel_mask(img_cube)
        
        logger.info(f"   Bad pixels: {np.sum(badpix):,}/{badpix.size:,} ({np.sum(badpix)/badpix.size*100:.1f}%)")
        
        # Store processed image information
        image_info = {
            'id': img_id,
            'cube': img_cube,
            'labels': label_map,
            'badpix': badpix,
            'dimensions': (height, width),
            'n_valid_pixels': len(img_spectra),
            'scene_id': f'{int(img_id):05d}' if isinstance(img_id, (int, float)) else str(img_id)
        }
        
        processed_images.append(image_info)
        logger.info(f"   ✅ Individual image {img_id} processed and stored")
    
    logger.info(f"📊 Individual image processing complete:")
    logger.info(f"   Successfully processed: {len(processed_images)} images")
    logger.info(f"   Ready for CRISM ML RGB generation")
    
    return processed_images


def generate_crism_ml_rgb_for_individual_images(processed_images):
    """
    Generate RGB images for individual CRISM scenes using CRISM ML methodology.
    
    Args:
        processed_images: List of processed image dictionaries
    """
    logger.info("🎨 Generating RGB images using CRISM ML methodology...")
    
    # Create output directory
    individual_rgb_dir = os.path.join(config.PROCESSED_DATA_DIR, "individual_crism_ml")
    os.makedirs(individual_rgb_dir, exist_ok=True)
    
    # CRISM ML proven band combinations
    band_combinations = [
        ((233, 103, 20), "standard", "Standard CRISM ML (Proven)"),
        ((200, 80, 40), "mineral", "Mineral Detection Enhanced"),
        ((180, 120, 60), "hydration", "Hydration Features"),
        ((150, 100, 50), "geology", "General Geology"),
        ((100, 60, 30), "visible", "Near-Visible Spectrum")
    ]
    
    successful_count = 0
    
    for img_info in processed_images:
        img_id = img_info['id']
        scene_id = img_info['scene_id']
        img_cube = img_info['cube']
        badpix = img_info['badpix']
        h, w = img_info['dimensions']
        
        logger.info(f"🖼️ Generating RGB for individual image {scene_id} (ID: {img_id})")
        logger.info(f"   Size: {h}×{w} | Valid: {img_info['n_valid_pixels']:,} pixels")
        
        try:
            # Generate RGB with each band combination
            for channels, suffix, description in band_combinations:
                # Adjust channels to available bands
                max_band = img_cube.shape[2] - 1
                adj_channels = tuple(min(ch, max_band) for ch in channels)
                
                logger.info(f"   Creating {description} RGB (bands: {adj_channels})")
                
                # Use CRISM ML get_false_colors method
                rgb_image = get_crism_false_colors(img_cube, badpix, channels=adj_channels)
                
                # Create high-quality visualization
                plt.figure(figsize=(18, 12))
                plt.imshow(rgb_image, interpolation='none')
                
                plt.title(f'{description} - CRISM Scene {scene_id}\n'
                         f'Individual Image Processing | Bands: {adj_channels[0]}, {adj_channels[1]}, {adj_channels[2]}\n'
                         f'Size: {h}×{w} pixels | Method: CRISM ML get_false_colors()',
                         fontsize=16, pad=25)
                plt.axis('off')
                
                # Add detailed metadata
                plt.figtext(0.02, 0.02, 
                           f'CRISM Image ID: {img_id} | Scene: {scene_id}\n'
                           f'Individual Processing: Pixel-by-pixel reconstruction\n'
                           f'Method: CRISM ML toolkit (Plebani et al., 2022)\n'
                           f'Processing: Median filter (17px) + L2 norm + Min-max + Histogram eq.\n'
                           f'Valid pixels: {img_info["n_valid_pixels"]:,} ({img_info["n_valid_pixels"]/(h*w)*100:.1f}%)\n'
                           f'Band combination: {description}',
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.6", facecolor="lightcyan", alpha=0.95))
                
                save_path = os.path.join(individual_rgb_dir, f'individual_{scene_id}_{suffix}_crism_ml.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
            logger.info(f"   ✅ Generated {len(band_combinations)} RGB variations for scene {scene_id}")
            successful_count += 1
            
        except Exception as e:
            logger.error(f"   ❌ Failed to generate RGB for image {scene_id}: {str(e)}")
            continue
    
    # Create comprehensive gallery
    create_individual_crism_gallery(processed_images, individual_rgb_dir, successful_count)
    
    logger.info(f"\n🎉 Individual CRISM ML RGB generation completed!")
    logger.info(f"   📊 Successfully processed: {successful_count} individual images")
    logger.info(f"   🖼️ Total RGB images: {successful_count * len(band_combinations)}")
    logger.info(f"   🌐 Gallery: data/processed/individual_crism_ml/index.html")
    
    return successful_count


def create_individual_crism_gallery(processed_images, output_dir, successful_count):
    """Create comprehensive gallery for individual CRISM ML images."""
    from datetime import datetime
    
    index_path = os.path.join(output_dir, 'index.html')
    
    # Get list of generated files
    generated_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    generated_files.sort()
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Individual CRISM Images - CRISM ML Processing</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; 
               background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #fff; }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 25px; }}
        .header {{ background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%); 
                  color: #2c3e50; padding: 40px; text-align: center; border-radius: 20px; 
                  margin-bottom: 40px; box-shadow: 0 15px 35px rgba(255,107,107,0.3); }}
        .methodology {{ background: rgba(254,202,87,0.15); border-left: 5px solid #feca57; 
                       padding: 25px; margin: 30px 0; border-radius: 12px; backdrop-filter: blur(10px); }}
        .stats-overview {{ background: rgba(255,107,107,0.15); padding: 25px; border-radius: 12px; 
                          margin: 25px 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .stat-box {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; text-align: center; }}
        .image-section {{ background: rgba(255,255,255,0.05); margin: 30px 0; padding: 25px; 
                         border-radius: 15px; border: 1px solid #ff6b6b; }}
        .scene-title {{ font-size: 2.2em; color: #feca57; margin-bottom: 20px; text-align: center; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 20px; }}
        .image-card {{ background: rgba(255,255,255,0.08); padding: 20px; border-radius: 15px; 
                      text-align: center; transition: all 0.3s; }}
        .image-card:hover {{ transform: translateY(-5px); box-shadow: 0 15px 30px rgba(255,107,107,0.3); }}
        .image-card img {{ max-width: 100%; height: auto; border-radius: 12px; 
                          box-shadow: 0 8px 25px rgba(0,0,0,0.4); }}
        .image-title {{ font-size: 1.3em; color: #ff6b6b; margin: 15px 0 8px 0; font-weight: bold; }}
        .image-description {{ font-size: 0.95em; color: #ddd; line-height: 1.5; }}
        .processing-badge {{ background: #ff6b6b; color: white; padding: 5px 10px; 
                            border-radius: 15px; font-size: 0.8em; margin: 5px 0; display: inline-block; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ Individual CRISM Image Processing</h1>
            <h2>Enhanced with CRISM ML Proven Methodology</h2>
            <p style="font-size: 1.2em;">Pixel-by-pixel reconstruction of individual Mars surface scenes</p>
            <p>Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
        </div>
        
        <div class="stats-overview">
            <div class="stat-box">
                <h3 style="margin: 0; font-size: 2.5em; color: #feca57;">{successful_count}</h3>
                <p>Individual Images</p>
            </div>
            <div class="stat-box">
                <h3 style="margin: 0; font-size: 2.5em; color: #feca57;">{successful_count * 5}</h3>
                <p>Total RGB Images</p>
            </div>
            <div class="stat-box">
                <h3 style="margin: 0; font-size: 2.5em; color: #feca57;">5</h3>
                <p>Band Combinations</p>
            </div>
            <div class="stat-box">
                <h3 style="margin: 0; font-size: 2.5em; color: #feca57;">350</h3>
                <p>Spectral Channels</p>
            </div>
        </div>
        
        <div class="methodology">
            <h3>🔬 Individual Image Processing Methodology</h3>
            <p><strong>Enhanced individual image handling using CRISM ML toolkit:</strong></p>
            <ul>
                <li><strong>Individual Reconstruction:</strong> Each CRISM scene processed separately with its own geometry</li>
                <li><strong>Spatial Accuracy:</strong> Pixel coordinates preserved from original orbital data</li>
                <li><strong>CRISM ML RGB:</strong> Proven get_false_colors() function from published toolkit</li>
                <li><strong>Adaptive Processing:</strong> Bad pixel detection and handling per individual image</li>
                <li><strong>Multiple Enhancements:</strong> 5 different band combinations for geological analysis</li>
            </ul>
        </div>
"""
    
    # Group images by scene
    scenes = {}
    for filename in generated_files:
        if 'individual_' in filename:
            parts = filename.split('_')
            scene_id = parts[1]
            enhancement = parts[2]
            if scene_id not in scenes:
                scenes[scene_id] = []
            scenes[scene_id].append((filename, enhancement))
    
    # Add each individual scene
    for scene_id, images in scenes.items():
        # Find corresponding processed image info
        img_info = None
        for proc_img in processed_images:
            if proc_img['scene_id'] == scene_id:
                img_info = proc_img
                break
        
        if img_info is None:
            continue
            
        h, w = img_info['dimensions']
        n_pixels = img_info['n_valid_pixels']
        
        html_content += f"""
        <div class="image-section">
            <div class="scene-title">🗺️ Individual CRISM Scene: {scene_id}</div>
            
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin-bottom: 25px;">
                <div class="processing-badge">Individual Processing</div>
                <div class="processing-badge">CRISM ML Enhanced</div>
                <br><br>
                <strong>📊 Scene Characteristics:</strong><br>
                Original Dimensions: {h:,} × {w:,} pixels | 
                Valid Data Points: {n_pixels:,} | 
                Coverage: {n_pixels/(h*w)*100:.1f}% | 
                Image ID: {img_info['id']}
            </div>
            
            <div class="image-grid">
"""
        
        # Add each enhancement for this scene
        enhancement_descriptions = {
            "standard": ("🎯 Standard CRISM ML", "Proven band combination (233,103,20) from published research"),
            "mineral": ("💎 Mineral Detection", "Enhanced for mafic and felsic mineral identification"), 
            "hydration": ("💧 Hydration Features", "Optimized for clay minerals and hydrated phases"),
            "geology": ("🏔️ General Geology", "Balanced enhancement for geological features"),
            "visible": ("👁️ Near-Visible", "Approximates human vision for natural appearance")
        }
        
        for filename, enhancement in images:
            if enhancement in enhancement_descriptions:
                title, description = enhancement_descriptions[enhancement]
                html_content += f"""
                <div class="image-card">
                    <img src="{filename}" alt="{title}">
                    <div class="image-title">{title}</div>
                    <div class="image-description">{description}</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>
"""
    
    html_content += f"""
        <div class="methodology">
            <h3>✅ Individual Processing Advantages</h3>
            <p><strong>Why individual image processing produces better results:</strong></p>
            <ul>
                <li><strong>Preserved Geometry:</strong> Each CRISM scene maintains its original spatial structure</li>
                <li><strong>Scene-Specific Optimization:</strong> Processing parameters adapted per individual image</li>
                <li><strong>Reduced Artifacts:</strong> No inter-scene contamination or averaging effects</li>
                <li><strong>Scientific Accuracy:</strong> Maintains orbital data integrity for geological analysis</li>
                <li><strong>Enhanced Detail:</strong> Fine-scale features preserved within each scene</li>
            </ul>
        </div>
        
        <div class="header">
            <h3>🔬 CRISM ML Integration Success</h3>
            <p>Successfully integrated proven CRISM ML methodology for individual image processing!</p>
            <p><strong>Reference:</strong> <a href="https://github.com/Banus/crism_ml" style="color: #feca57;">Banus/crism_ml Repository</a></p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"🌐 Created individual CRISM ML gallery: {index_path}")


def main():
    """Main function for improved CRISM processing with individual image handling."""
    logger.info("=" * 80)
    logger.info("CRISM ML Integration - Enhanced Individual Image Processing")
    logger.info("Based on proven methodology from https://github.com/Banus/crism_ml")
    logger.info("=" * 80)
    
    try:
        # Step 1: Load and process individual images
        processed_images = load_and_process_individual_images()
        
        # Step 2: Generate RGB using CRISM ML methodology 
        successful_count = generate_crism_ml_rgb_for_individual_images(processed_images)
        
        logger.info(f"\n🎉 Complete CRISM ML integration successful!")
        logger.info(f"   📊 Individual images processed: {successful_count}")
        logger.info(f"   🎨 Enhanced with proven CRISM ML RGB methodology")
        logger.info(f"   🌐 Browse results: data/processed/individual_crism_ml/index.html")
        
    except Exception as e:
        logger.error(f"❌ CRISM ML integration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
