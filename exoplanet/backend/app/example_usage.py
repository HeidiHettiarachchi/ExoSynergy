#!/usr/bin/env python3
"""
Quick Start Example: Two-Stage Mineral Identification Pipeline
===============================================================
Demonstrates basic usage of the CLIP gatekeeper + CNN mineral classifier

Run this after installing dependencies:
    pip install open-clip-torch

Usage:
    python example_usage.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

print("=" * 80)
print("Two-Stage Mineral Identification Pipeline - Quick Start Example")
print("=" * 80)

# =============================================================================
# Example 1: Basic Gatekeeper Usage
# =============================================================================
print("\n📌 Example 1: Using the Planetary Gatekeeper")
print("-" * 80)

try:
    from planetary_gatekeeper import load_gatekeeper
    
    print("\nLoading CLIP gatekeeper (this may take a moment on first run)...")
    gatekeeper = load_gatekeeper(threshold=0.45)
    print("✅ Gatekeeper loaded successfully!\n")
    
    # Example: Check if an image is planetary
    # Note: You'll need an actual image file to test this
    example_image = "data/processed/rgb_composites"  # Update with actual path
    
    if os.path.exists(example_image):
        result = gatekeeper.check_image(example_image, return_scores=True)
        
        print(f"Image: {example_image}")
        print(f"Accepted: {result['accepted']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Reason: {result['reason']}")
        
        if 'scores' in result:
            print(f"\nDetailed Scores:")
            print(f"  Planetary similarity: {result['scores']['planetary_avg']:.3f}")
            print(f"  Non-planetary similarity: {result['scores']['non_planetary_avg']:.3f}")
    else:
        print(f"⚠️ Example image directory not found: {example_image}")
        print("   Update the path to test with your own images.")

except Exception as e:
    print(f"❌ Error in Example 1: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Example 2: Full Two-Stage Pipeline
# =============================================================================
print("\n📌 Example 2: Full Two-Stage Pipeline")
print("-" * 80)

try:
    from mineral_identification_pipeline import create_pipeline
    from inference_script import load_trained_model
    from src import config
    
    # Check if model exists
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"\nLoading trained mineral model from {config.MODEL_SAVE_PATH}...")
        
        # Create the full pipeline with permissive threshold
        pipeline = create_pipeline(threshold=0.45)
        print("✅ Pipeline created successfully!\n")
        
        # Example: Run prediction on an image
        example_image = "data/processed/rgb_composites"  # Update with actual path
        
        if os.path.exists(example_image):
            print(f"Running two-stage prediction on: {example_image}")
            
            result = pipeline.predict(example_image, return_gatekeeper_scores=False)
            
            print(f"\n📊 Results:")
            print(f"  Stage: {result['stage']}")
            print(f"  Gatekeeper passed: {result['gatekeeper_passed']}")
            print(f"  Gatekeeper confidence: {result['gatekeeper_confidence']:.3f}")
            
            if result['stage'] == 'classified':
                print(f"  Dominant mineral: {result['dominant_mineral']}")
                print(f"  Classification confidence: {result['confidence']:.3f}")
                print(f"  Processing time: {result['processing_time']:.2f}s")
            else:
                print(f"  Rejection reason: {result['gatekeeper_reason']}")
        else:
            print(f"⚠️ Example image not found: {example_image}")
            print("   Update the path to test with your own images.")
    else:
        print(f"⚠️ Trained model not found at: {config.MODEL_SAVE_PATH}")
        print("   Train the model first before using the full pipeline.")
        print("   However, you can still test the gatekeeper independently (Example 1).")

except Exception as e:
    print(f"❌ Error in Example 2: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Example 3: Batch Processing
# =============================================================================
print("\n📌 Example 3: Batch Processing Multiple Images")
print("-" * 80)

try:
    # Collect example images
    data_dir = Path("data/processed/rgb_composites")
    
    if data_dir.exists():
        image_files = list(data_dir.glob("*.png"))[:3]  # First 3 images
        
        if len(image_files) > 0 and 'pipeline' in locals():
            print(f"\nProcessing {len(image_files)} images in batch...")
            
            results = pipeline.batch_predict(image_files, skip_gatekeeper=False)
            
            print(f"\n📊 Batch Results Summary:")
            for i, result in enumerate(results):
                print(f"\nImage {i+1}: {image_files[i].name}")
                print(f"  Stage: {result['stage']}")
                if result['stage'] == 'classified':
                    print(f"  Dominant mineral: {result['dominant_mineral']}")
                else:
                    print(f"  Rejected: {result['gatekeeper_reason']}")
        else:
            print("⚠️ No images found in batch directory or pipeline not loaded.")
    else:
        print(f"⚠️ Image directory not found: {data_dir}")

except Exception as e:
    print(f"❌ Error in Example 3: {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# Example 4: Threshold Exploration
# =============================================================================
print("\n📌 Example 4: Testing Different Thresholds")
print("-" * 80)

try:
    if 'gatekeeper' in locals():
        print("\nTesting the same image with different thresholds:")
        
        test_thresholds = [0.50, 0.65, 0.80]
        
        # You would use an actual image here
        print("(Conceptual example - update with actual image path)")
        
        for threshold in test_thresholds:
            gatekeeper.threshold = threshold
            # result = gatekeeper.check_image("your_image.png")
            print(f"  Threshold {threshold}: [Would check acceptance status]")

except Exception as e:
    print(f"❌ Error in Example 4: {e}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("📚 Summary and Next Steps")
print("=" * 80)

print("""
✅ Quick Start Examples Completed!

To use this pipeline in your code:

1. Basic Gatekeeper Check:
   ```python
   from planetary_gatekeeper import load_gatekeeper
   gatekeeper = load_gatekeeper(threshold=0.65)
   result = gatekeeper.check_image("image.png")
   ```

2. Full Two-Stage Pipeline:
   ```python
   from mineral_identification_pipeline import create_pipeline
   pipeline = create_pipeline(threshold=0.65)
   result = pipeline.predict("image.png")
   ```

3. Threshold Tuning:
   ```bash
   python threshold_tuning.py \\
       --planetary_dir ./mars_images \\
       --non_planetary_dir ./earth_images
   ```

📖 For more details, see TWO_STAGE_PIPELINE_README.md

🔧 Configuration:
   - Adjust threshold (0.5-0.9) based on your precision/recall needs
   - Use threshold_tuning.py to find optimal value for your data
   - Skip gatekeeper with skip_gatekeeper=True if needed

⚡ Performance Tips:
   - Load gatekeeper/pipeline once at startup
   - Use batch_predict() for multiple images
   - Ensure GPU is available for faster inference
""")

print("=" * 80)
