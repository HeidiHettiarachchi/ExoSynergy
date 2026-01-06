#!/usr/bin/env python3
"""
CRISM Overlay Visualizations Pipeline
====================================
Creates combined visualizations showing mineral classifications overlaid on CRISM satellite imagery.
Similar to the tutorial example with mineral patches highlighted on the original RGB images.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
# Try importing cv2, fall back to scipy if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    from scipy.ndimage import zoom
    CV2_AVAILABLE = False

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Add the cloned crism_ml to path
crism_ml_path = project_root / "crism_ml"
sys.path.append(str(crism_ml_path))

# Import from our project modules
from src import config
from src.dataset import load_and_reconstruct_data
from src.logger import get_logger

# Import from CRISM ML for overlay functionality
import crism_ml.plot as crism_plot
import crism_ml.lab as crism_lab

# Initialize logger
logger = get_logger("CRISM.Overlay")

# Define mineral colors (based on scientific standards)
MINERAL_COLORS = {
    0: None,                    # Background/Bland - no overlay
    1: (0.8, 0.2, 0.2),        # Red for primary minerals
    2: (0.2, 0.8, 0.2),        # Green for secondary minerals  
    3: (0.2, 0.2, 0.8),        # Blue for tertiary minerals
    4: (0.8, 0.8, 0.2),        # Yellow for hydrated minerals
    5: (0.8, 0.2, 0.8),        # Magenta for sulfates
    6: (0.2, 0.8, 0.8),        # Cyan for carbonates
    7: (1.0, 0.6, 0.2),        # Orange for oxides
    8: (0.6, 0.4, 0.8),        # Purple for phyllosilicates
    9: (0.4, 0.8, 0.4),        # Light green for pyroxenes
    10: (0.8, 0.4, 0.4),       # Light red for olivines
    11: (0.4, 0.4, 0.8),       # Light blue for plagioclase
    12: (0.6, 0.6, 0.6),       # Gray for unclassified
    13: (1.0, 0.8, 0.4),       # Light yellow for other minerals
}

def get_overlay_advanced(rgb_img, mask, mineral_id, color, alpha=0.7, dilate=True):
    """
    Create mineral overlay on RGB image using CRISM ML methodology.
    
    Parameters:
    -----------
    rgb_img : ndarray
        RGB satellite image as background (H, W, 3)
    mask : ndarray  
        Boolean mask where mineral pixels are True (H, W)
    mineral_id : int
        ID of the mineral class
    color : tuple
        RGB color tuple for the mineral (0-1 range)
    alpha : float
        Transparency of overlay (0=transparent, 1=opaque)
    dilate : bool
        Apply morphological dilation to make minerals more visible
        
    Returns:
    --------
    overlay_img : ndarray
        RGB image with mineral overlay
    """
    if color is None:
        return rgb_img
    
    # Create a copy of the background image
    overlay_img = rgb_img.copy()
    
    # Apply morphological dilation if requested (makes small regions more visible)
    if dilate and np.any(mask):
        if CV2_AVAILABLE:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        else:
            # Fallback: simple dilation using scipy
            from scipy.ndimage import binary_dilation
            kernel = np.ones((3, 3))
            mask = binary_dilation(mask, structure=kernel).astype(bool)
    
    # Create colored overlay
    if np.any(mask):
        # Blend the mineral color with the original image
        for i in range(3):  # RGB channels
            overlay_img[mask, i] = (
                alpha * color[i] + (1 - alpha) * overlay_img[mask, i]
            )
    
    return overlay_img

# Legend creation removed - save only clean images

def process_individual_overlay(image_data, mask_data, scene_id, mineral_names, output_dir):
    """
    Process a single CRISM scene to create overlay visualization.
    """
    logger.info(f"📸 Processing overlay for scene {scene_id}")
    
    # Get image dimensions
    height, width = image_data.shape[:2]
    logger.info(f"    Dimensions: {width}×{height}")
    
    # Start with the RGB image as background
    # Normalize to 0-1 range if needed
    if image_data.max() > 1.0:
        background = image_data / 255.0
    else:
        background = image_data.copy()
    
    # Get unique mineral classes in this scene
    unique_minerals = np.unique(mask_data)
    logger.info(f"    Minerals found: {len(unique_minerals)} classes")
    
    # Start with background image
    overlay_result = background.copy()
    mineral_stats = []
    
    # Apply each mineral overlay
    for mineral_id in unique_minerals:
        if mineral_id == 0:  # Skip background
            continue
            
        # Get mask for this mineral
        mineral_mask = (mask_data == mineral_id)
        pixel_count = np.sum(mineral_mask)
        
        if pixel_count < 5:  # Skip very small regions
            continue
        
        # Get color for this mineral
        color = MINERAL_COLORS.get(mineral_id, (0.5, 0.5, 0.5))  # Default gray
        
        # Apply overlay for this mineral
        overlay_result = get_overlay_advanced(
            overlay_result, mineral_mask, mineral_id, color, 
            alpha=0.6, dilate=True
        )
        
        # Track statistics
        mineral_name = mineral_names.get(mineral_id, f"Class_{mineral_id}")
        mineral_stats.append({
            'id': mineral_id,
            'name': mineral_name, 
            'color': color,
            'pixels': pixel_count,
            'percentage': (pixel_count / (height * width)) * 100
        })
        
        logger.info(f"    ✅ {mineral_name}: {pixel_count:,} pixels ({(pixel_count/(height*width)*100):.2f}%)")
    
    # Save only the clean overlay result - no legends, no text
    output_path = os.path.join(output_dir, f"scene_{scene_id}_overlay.png")
    plt.imsave(output_path, np.clip(overlay_result, 0, 1))
    
    return {
        'scene_id': scene_id,
        'overlay_path': output_path,
        'mineral_stats': mineral_stats,
        'dimensions': f"{width}×{height}"
    }

def create_overlay_gallery(results, output_dir):
    """Create HTML gallery for overlay visualizations."""
    html_path = os.path.join(output_dir, 'index.html')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>CRISM Mineral Overlay Visualizations</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #fff; }}
            .container {{ max-width: 1400px; margin: 0 auto; padding: 30px; }}
            .header {{ background: rgba(255,255,255,0.95); color: #2c3e50; padding: 30px; 
                      text-align: center; border-radius: 15px; margin-bottom: 30px; }}
            .gallery {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                       gap: 25px; }}
            .scene-card {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; 
                          backdrop-filter: blur(10px); }}
            .scene-card img {{ max-width: 100%; border-radius: 10px; margin: 10px 0; }}
            .scene-title {{ font-size: 1.3em; font-weight: bold; margin-bottom: 15px; 
                           color: #feca57; }}
            .mineral-stats {{ background: rgba(46,204,113,0.2); padding: 15px; 
                             border-radius: 10px; margin: 10px 0; }}
            .stat-item {{ margin: 5px 0; font-size: 0.95em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🛰️ CRISM Clean Overlay Visualizations</h1>
                <h2>Pure Mineral Classifications on Satellite Imagery</h2>
                <p>Clean colored mineral overlays - No text, no legends, just pure overlay images</p>
            </div>
            
            <div class="gallery">
    """
    
    for result in results:
        stats_html = ""
        if result['mineral_stats']:
            stats_html = "<div class='mineral-stats'><strong>Detected Minerals:</strong><br>"
            for stat in result['mineral_stats']:
                stats_html += f"<div class='stat-item'>• {stat['name']}: {stat['pixels']:,} pixels ({stat['percentage']:.2f}%)</div>"
            stats_html += "</div>"
        
        html_content += f"""
            <div class="scene-card">
                <div class="scene-title">Scene {result['scene_id']}</div>
                <img src="{os.path.basename(result['overlay_path'])}" alt="Scene {result['scene_id']} Clean Overlay">
                {stats_html}
                <p><strong>Dimensions:</strong> {result['dimensions']}</p>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"📄 Created overlay gallery: {html_path}")

def main():
    """Main pipeline for creating CRISM overlay visualizations."""
    logger.info("================================================================")
    logger.info("CRISM Mineral Overlay Visualizations Pipeline")  
    logger.info("Creates mineral overlays on satellite imagery")
    logger.info("================================================================")
    
    # Create output directory
    overlay_output_dir = os.path.join(config.PROCESSED_DATA_DIR, "mineral_overlays")
    os.makedirs(overlay_output_dir, exist_ok=True)
    
    # Load the CRISM data
    logger.info("📡 Loading CRISM data...")
    images, masks, label_mapping, original_labels, all_coords = load_and_reconstruct_data()
    
    logger.info(f"✅ Loaded {len(images)} individual CRISM scenes")
    
    # Load existing RGB images from individual_crism_ml directory
    rgb_dir = os.path.join(config.PROCESSED_DATA_DIR, "individual_crism_ml")
    if not os.path.exists(rgb_dir):
        logger.error("❌ RGB images not found. Run 'make satellite' first to generate background images.")
        return
    
    # Process each scene
    results = []
    
    # Create mineral name mapping
    mineral_names = {
        0: "Background",
        1: "Kaolinite", 
        2: "Polyhydrated Sulfate",
        3: "Alunite",
        4: "Jarosite",
        5: "Montmorillonite",
        6: "Nontronite", 
        7: "Pyroxene",
        8: "Olivine",
        9: "Hematite",
        10: "Goethite"
    }
    # Add any missing minerals from label_mapping
    for orig_label in original_labels:
        if orig_label not in mineral_names:
            mineral_names[orig_label] = f"Mineral_{orig_label}"
    
    for i in range(min(len(images), 10)):  # Process first 10 scenes
        image_data = images[i] 
        mask_data = masks[i]
        scene_id = i + 1
        
        # Look for corresponding RGB background image
        rgb_files = [f for f in os.listdir(rgb_dir) 
                    if f.startswith(f"individual_{scene_id}_standard") and f.endswith('.png')]
        
        if not rgb_files:
            logger.warning(f"⚠️ No RGB background found for scene {scene_id}")
            continue
            
        # Load the RGB background
        rgb_path = os.path.join(rgb_dir, rgb_files[0])
        background_rgb = plt.imread(rgb_path)
        
        # Resize mask to match RGB if needed
        if mask_data.shape != background_rgb.shape[:2]:
            if CV2_AVAILABLE:
                mask_resized = cv2.resize(
                    mask_data.astype(np.uint8), 
                    (background_rgb.shape[1], background_rgb.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                # Fallback: use zoom from scipy
                h_scale = background_rgb.shape[0] / mask_data.shape[0]
                w_scale = background_rgb.shape[1] / mask_data.shape[1]
                mask_resized = zoom(mask_data.astype(float), (h_scale, w_scale), order=0).astype(np.uint8)
        else:
            mask_resized = mask_data
        
        # Process this scene
        result = process_individual_overlay(
            background_rgb, mask_resized, scene_id, mineral_names, overlay_output_dir
        )
        results.append(result)
    
    # Create gallery
    create_overlay_gallery(results, overlay_output_dir)
    
    logger.info(f"\n🎉 Overlay visualization pipeline complete!")
    logger.info(f"   📸 Processed {len(results)} overlay visualizations")
    logger.info(f"   🌐 Gallery: {overlay_output_dir}/index.html")
    logger.info(f"   📁 Individual overlays: {overlay_output_dir}/")
    
if __name__ == '__main__':
    main()
