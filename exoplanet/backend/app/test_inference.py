#!/usr/bin/env python3
"""
Test script for the inference pipeline.
Creates test PNG images and runs inference on them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src import config
from src.logger import get_logger

logger = get_logger("CRISM.TestInference")

def create_test_images():
    """Use existing CRISM processed images for testing."""
    logger.info("🧪 Selecting test images from processed CRISM data...")
    
    # Create test directory
    test_dir = os.path.join(config.PROCESSED_DATA_DIR, "test_inference")
    os.makedirs(test_dir, exist_ok=True)
    
    test_images = []
    
    # Try to find existing CRISM RGB images
    rgb_dir = os.path.join(config.PROCESSED_DATA_DIR, "individual_crism_ml")
    
    if os.path.exists(rgb_dir):
        # Find RGB composite images
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png') and 'standard' in f.lower()]
        
        if rgb_files:
            # Use first available image (lightweight - just one test)
            src_file = rgb_files[0]
            src_path = os.path.join(rgb_dir, src_file)
            test_path = os.path.join(test_dir, f"crism_test_image.png")
            
            # Copy the real CRISM image
            import shutil
            shutil.copy2(src_path, test_path)
            test_images.append(test_path)
            logger.info(f"   Using CRISM image: {src_file}")
            logger.info(f"   Test image saved to: {test_path}")
            return test_images
    
    # Fallback: check rgb_composites directory
    rgb_comp_dir = os.path.join(config.PROCESSED_DATA_DIR, "rgb_composites")
    if os.path.exists(rgb_comp_dir):
        rgb_files = [f for f in os.listdir(rgb_comp_dir) if f.endswith('.png')]
        
        if rgb_files:
            src_file = rgb_files[0]
            src_path = os.path.join(rgb_comp_dir, src_file)
            test_path = os.path.join(test_dir, f"crism_test_image.png")
            
            import shutil
            shutil.copy2(src_path, test_path)
            test_images.append(test_path)
            logger.info(f"   Using CRISM RGB composite: {src_file}")
            logger.info(f"   Test image saved to: {test_path}")
            return test_images
    
    logger.warning("⚠️ No processed CRISM images found. Please run data processing first.")
    return test_images

def get_test_ground_truth():
    """Get ground truth if available from processed data."""
    logger.info("🏷️ Checking for ground truth data...")
    
    # Check if we have ground truth in processed data
    gt_dir = os.path.join(config.PROCESSED_DATA_DIR, "ground_truth")
    
    if os.path.exists(gt_dir):
        gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.png')]
        if gt_files:
            gt_path = os.path.join(gt_dir, gt_files[0])
            logger.info(f"   Found ground truth: {gt_files[0]}")
            return gt_path
    
    logger.info("   No ground truth found - will run inference without IoU calculation")
    return None

def run_test_inference():
    """Run lightweight inference tests."""
    logger.info("🚀 Running lightweight inference tests...")
    
    # Get test images from processed CRISM data
    test_images = create_test_images()
    gt_path = get_test_ground_truth()
    
    if not test_images:
        logger.warning("⚠️ No test images available. Please run data processing first.")
        logger.info("   Run: python pipelines/process_data.py")
        return
    
    # Test inference on each image
    from inference_script import main as inference_main
    import sys
    
    # Only test the first image for speed
    test_image = test_images[0]
    logger.info(f"🧪 Testing inference on: {os.path.basename(test_image)}")
    
    # Prepare arguments for inference script
    original_argv = sys.argv.copy()
    
    try:
        # Test with ground truth if available
        sys.argv = ['inference_script.py', '--input', test_image]
        
        if gt_path:
            sys.argv.extend(['--ground_truth', gt_path])
        
        # Run inference
        inference_main()
        logger.info(f"   ✅ Inference test completed successfully")
        
    except Exception as e:
        logger.error(f"   ❌ Inference test failed: {e}")
        raise
    
    finally:
        # Restore original arguments
        sys.argv = original_argv
    
    logger.info("💡 Lightweight test complete. For full testing, run inference on all images manually.")

def main():
    """Main test function."""
    logger.info("================================================================")
    logger.info("CRISM Inference Pipeline Test")
    logger.info("================================================================")
    
    try:
        # Check if model exists
        if not os.path.exists(config.MODEL_SAVE_PATH):
            logger.warning(f"⚠️ Model not found at {config.MODEL_SAVE_PATH}")
            logger.info("   Please train the model first using: make train")
            return
        
        # Run tests
        run_test_inference()
        
        logger.info("🎉 Inference tests completed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == '__main__':
    main()
