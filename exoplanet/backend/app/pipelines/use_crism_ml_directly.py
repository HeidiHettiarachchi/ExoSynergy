#!/usr/bin/env python3
"""
Use CRISM ML Toolkit Directly for Proper RGB Visualization.
This script directly uses the proven Banus/crism_ml functions.

Usage:
    python pipelines/use_crism_ml_directly.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add external repository to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'crism_ml_reference'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import CRISM ML functions directly
from crism_ml.plot import get_false_colors
from crism_ml.io import loadmat

from src import config
from src.logger import get_logger

# Initialize logger
logger = get_logger("CRISM.DirectML")


def main():
    """Use CRISM ML toolkit directly for proper visualization."""
    logger.info("=" * 70)
    logger.info("CRISM Direct ML Toolkit Usage")
    logger.info("Repository: https://github.com/Banus/crism_ml.git")
    logger.info("=" * 70)
    
    # Create output directory
    ml_direct_dir = os.path.join(config.PROCESSED_DATA_DIR, "crism_ml_direct")
    os.makedirs(ml_direct_dir, exist_ok=True)
    
    try:
        logger.info("📡 Loading CRISM data with CRISM ML toolkit...")
        
        # Load data using CRISM ML's proven method
        mat_data = loadmat(config.LABELED_DATA_PATH)
        
        logger.info(f"📊 CRISM ML Data Loaded:")
        logger.info(f"   Available keys: {[k for k in mat_data.keys() if not k.startswith('__')]}")
        
        # Extract data using CRISM ML structure
        pixspec = mat_data['pixspec']  # (592413, 350) spectra
        pixcrds = mat_data['pixcrds']  # (592413, 2) coordinates
        pixims = mat_data['pixims']    # (592413,) image IDs
        pixlabs = mat_data['pixlabs']  # (592413,) labels
        
        logger.info(f"   Spectra: {pixspec.shape}")
        logger.info(f"   Coordinates: {pixcrds.shape}")
        logger.info(f"   Image IDs: {pixims.shape} (range: {pixims.min()}-{pixims.max()})")
        logger.info(f"   Labels: {pixlabs.shape} (unique: {len(np.unique(pixlabs))})")
        
        # Get unique images
        unique_imgs = np.unique(pixims.flatten())
        logger.info(f"   Total images available: {len(unique_imgs)}")
        
        # Process first 5 images for demonstration
        max_images = 5
        successful_count = 0
        
        for img_id in unique_imgs[:max_images]:
            logger.info(f"🖼️ Processing image {img_id}...")
            
            # Get pixels for this image
            img_mask = pixims.flatten() == img_id
            img_spectra = pixspec[img_mask]  # (n_pixels, 350)
            img_coords = pixcrds[img_mask]   # (n_pixels, 2)
            
            if len(img_spectra) == 0:
                logger.warning(f"   No pixels found for image {img_id}")
                continue
            
            # Get image dimensions from coordinates
            x_coords = img_coords[:, 0].astype(int)
            y_coords = img_coords[:, 1].astype(int)
            
            height = y_coords.max() - y_coords.min() + 1
            width = x_coords.max() - x_coords.min() + 1
            
            logger.info(f"   Dimensions: {height}×{width} pixels")
            logger.info(f"   Data pixels: {len(img_spectra):,}")
            
            # Skip very large images for memory
            if height * width > 200000:
                logger.info(f"   Skipping large image: {height}×{width}")
                continue
            
            # Reconstruct image using CRISM ML approach
            # Create full image array
            img_cube = np.zeros((height, width, pixspec.shape[1]), dtype=np.float64)
            
            # Place pixels at correct locations
            for spec, coord in zip(img_spectra, img_coords):
                y = int(coord[1]) - y_coords.min()
                x = int(coord[0]) - x_coords.min()
                
                if 0 <= y < height and 0 <= x < width:
                    img_cube[y, x] = spec
            
            # Create bad pixel mask (pixels with all zeros)
            badpix = np.all(img_cube == 0, axis=2)
            logger.info(f"   Bad pixels: {np.sum(badpix):,}/{badpix.size:,} ({np.sum(badpix)/badpix.size*100:.1f}%)")
            
            # Skip images with too many bad pixels
            if np.sum(badpix) / badpix.size > 0.95:
                logger.info(f"   Skipping sparse image (>95% bad pixels)")
                continue
            
            try:
                # Use CRISM ML's proven false color method
                logger.info(f"   Applying CRISM ML false color method...")
                
                # Standard CRISM ML channels (233, 103, 20)
                rgb_image = get_false_colors(img_cube, badpix, channels=(233, 103, 20))
                
                # Create visualization
                plt.figure(figsize=(16, 12))
                plt.imshow(rgb_image, interpolation='none')
                plt.title(f'CRISM ML Proper RGB - Image {img_id}\n'
                         f'Using Proven CRISM ML Methodology | Channels: 233, 103, 20\n'
                         f'Size: {height}×{width} | Method: Median filter + L2 norm + Histogram eq.',
                         fontsize=16, pad=25)
                plt.axis('off')
                
                # Add technical details
                plt.figtext(0.02, 0.02, 
                           f'CRISM Image ID: {img_id}\n'
                           f'Repository: github.com/Banus/crism_ml\n'
                           f'Method: get_false_colors() function\n'
                           f'Processing: Median filter (17px) + L2 norm + Min-max + Hist. eq.\n'
                           f'Channels: R=233, G=103, B=20 (proven combination)\n'
                           f'Data coverage: {(1 - np.sum(badpix)/badpix.size)*100:.1f}%',
                           fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.95))
                
                save_path = os.path.join(ml_direct_dir, f'image_{img_id}_crism_ml_rgb.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                logger.info(f"   ✅ Successfully created RGB using CRISM ML method")
                successful_count += 1
                
                # Also try alternative channels for comparison
                alternative_channels = [
                    (200, 80, 40),
                    (180, 120, 60), 
                    (150, 100, 50)
                ]
                
                for i, channels in enumerate(alternative_channels):
                    try:
                        alt_rgb = get_false_colors(img_cube, badpix, channels=channels)
                        
                        plt.figure(figsize=(16, 12))
                        plt.imshow(alt_rgb, interpolation='none')
                        plt.title(f'CRISM ML Alternative {i+1} - Image {img_id}\n'
                                 f'Channels: {channels[0]}, {channels[1]}, {channels[2]}',
                                 fontsize=16)
                        plt.axis('off')
                        
                        alt_path = os.path.join(ml_direct_dir, f'image_{img_id}_alt_{i+1}.png')
                        plt.savefig(alt_path, dpi=200, bbox_inches='tight', facecolor='white')
                        plt.close()
                        
                    except Exception as e:
                        logger.warning(f"   Alternative {i+1} failed: {str(e)}")
                
            except Exception as e:
                logger.error(f"   RGB generation failed: {str(e)}")
                continue
        
        # Create simple index
        create_direct_ml_index(ml_direct_dir, successful_count)
        
        logger.info(f"\n🎉 CRISM ML Direct RGB completed!")
        logger.info(f"   📊 Successfully processed: {successful_count} images")
        logger.info(f"   🌐 Gallery: data/processed/crism_ml_direct/index.html")
        logger.info(f"   📚 Using proven methodology from Banus/crism_ml")
        
    except Exception as e:
        logger.error(f"❌ CRISM ML direct processing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_direct_ml_index(output_dir, image_count):
    """Create simple index for CRISM ML direct results."""
    from datetime import datetime
    
    # List all generated images
    generated_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    generated_files.sort()
    
    index_path = os.path.join(output_dir, 'index.html')
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>CRISM ML Direct Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f8ff; }}
        .header {{ background: #2c3e50; color: white; padding: 30px; text-align: center; border-radius: 10px; margin-bottom: 30px; }}
        .success {{ background: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .image-item {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .image-item img {{ max-width: 100%; height: auto; border-radius: 5px; }}
        .caption {{ font-weight: bold; margin: 10px 0; color: #2c3e50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🛰️ CRISM ML Direct RGB Results</h1>
        <p>Using Original CRISM ML Toolkit Functions</p>
        <p>Repository: <a href="https://github.com/Banus/crism_ml.git" style="color: #ffd700;">Banus/crism_ml</a></p>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="success">
        <h3>✅ Success!</h3>
        <p><strong>Successfully processed {image_count} CRISM images using the proven CRISM ML toolkit!</strong></p>
        <p>These RGB images are generated using the exact <code>get_false_colors()</code> function from the 
           published CRISM ML repository (Plebani et al., 2022, Icarus).</p>
    </div>
    
    <div class="image-grid">
"""
    
    # Add each generated image
    for filename in generated_files:
        if 'crism_ml_rgb' in filename:
            img_id = filename.split('_')[1]
            html_content += f"""
        <div class="image-item">
            <img src="{filename}" alt="CRISM ML RGB">
            <div class="caption">🎯 CRISM Image {img_id}</div>
            <p>Proper RGB using CRISM ML get_false_colors() function</p>
        </div>
"""
        elif 'alt_' in filename:
            img_id = filename.split('_')[1]
            alt_num = filename.split('_')[-1].replace('.png', '')
            html_content += f"""
        <div class="image-item">
            <img src="{filename}" alt="Alternative">
            <div class="caption">🔄 Image {img_id} - Alternative {alt_num}</div>
            <p>Alternative band combination for comparison</p>
        </div>
"""
    
    html_content += """
    </div>
    
    <div class="header">
        <h3>🔬 About CRISM ML Method</h3>
        <p>These images use the exact visualization methodology from the published CRISM ML toolkit.</p>
        <p>The proven approach handles CRISM ratioed reflectance data correctly.</p>
    </div>
</body>
</html>
"""
    
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 Created CRISM ML direct gallery: {index_path}")


if __name__ == "__main__":
    main()
