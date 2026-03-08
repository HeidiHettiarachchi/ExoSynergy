#!/usr/bin/env python3
"""
Quick Test Script for Planetary Gatekeeper
===========================================
Tests if the gatekeeper correctly rejects non-planetary images like car photos.

Usage:
    python test_gatekeeper.py <image_path>
    
Example:
    python test_gatekeeper.py car_photo.jpg
    python test_gatekeeper.py mars_image.png
"""

import sys
from pathlib import Path
from planetary_gatekeeper import load_gatekeeper

def test_image(image_path: str, threshold: float = 0.45):
    """Test a single image with the gatekeeper."""
    
    print("=" * 80)
    print("PLANETARY GATEKEEPER TEST")
    print("=" * 80)
    print(f"\nImage: {image_path}")
    print(f"Threshold: {threshold}")
    print(f"\nLoading gatekeeper (this may take a moment)...")
    
    # Load gatekeeper with strict threshold
    gatekeeper = load_gatekeeper(threshold=threshold)
    print("✅ Gatekeeper loaded\n")
    
    # Check the image with detailed scores
    print("🔍 Running planetary image check...\n")
    result = gatekeeper.check_image(image_path, return_scores=True)
    
    # Display results
    print("-" * 80)
    if result["accepted"]:
        print("✅ RESULT: ACCEPTED (Planetary Image)")
    else:
        print("❌ RESULT: REJECTED (Non-Planetary Image)")
    print("-" * 80)
    
    print(f"\nConfidence Score: {result['confidence']:.3f}")
    print(f"Reason: {result['reason']}")
    
    # Detailed scores
    if "scores" in result:
        scores = result["scores"]
        print(f"\n📊 Detailed Analysis:")
        print(f"  Planetary similarity (avg):     {scores['planetary_avg']:.3f}")
        print(f"  Planetary similarity (max):     {scores['planetary_max']:.3f}")
        print(f"  Non-planetary similarity (avg): {scores['non_planetary_avg']:.3f}")
        print(f"  Non-planetary similarity (max): {scores['non_planetary_max']:.3f}")
        
        print(f"\n  Interpretation:")
        if scores['planetary_avg'] > scores['non_planetary_avg']:
            diff = scores['planetary_avg'] - scores['non_planetary_avg']
            print(f"  ✓ Image is {diff:.3f} more similar to planetary images")
        else:
            diff = scores['non_planetary_avg'] - scores['planetary_avg']
            print(f"  ✗ Image is {diff:.3f} more similar to non-planetary images")
    
    # Guidance
    print(f"\n💡 Guidance:")
    if result["accepted"]:
        print("  This image passed and will proceed to mineral classification.")
    else:
        print("  This image was rejected and won't waste compute on mineral classification.")
        print("  Expected input: Mars CRISM satellite imagery or similar planetary geological images.")
        print("  Not accepted: Photos of cars, people, buildings, Earth landscapes, etc.")
    
    print("\n" + "=" * 80)
    
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_gatekeeper.py <image_path> [threshold]")
        print("\nExamples:")
        print("  python test_gatekeeper.py car_photo.jpg")
        print("  python test_gatekeeper.py mars_image.png")
        print("  python test_gatekeeper.py test.jpg 0.70  # Custom threshold")
        print("\nExpected behavior:")
        print("  ✅ Planetary/Mars images should be ACCEPTED")
        print("  ❌ Car photos, portraits, etc. should be REJECTED")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.45
    
    # Check if file exists
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    # Run test
    try:
        result = test_image(image_path, threshold)
        
        # Exit code: 0 if accepted, 1 if rejected
        sys.exit(0 if result["accepted"] else 1)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
