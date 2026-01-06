#!/usr/bin/env python3
"""
Simple API test script - tests the running API server
"""

import requests
import numpy as np
from PIL import Image
import io

def create_test_image():
    """Create a simple test image."""
    # Create a 256x256 RGB test image
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    return Image.fromarray(img)

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing /health endpoint...")
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Device: {data['device']}")
            print(f"   Num classes: {data['num_classes']}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_predict():
    """Test the predict endpoint."""
    print("\n🔮 Testing /predict endpoint...")
    try:
        # Create test image
        img = create_test_image()
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Send request
        files = {'image': ('test.png', img_bytes, 'image/png')}
        data = {'min_area': 50, 'return_image': True}
        
        print("   Sending request...")
        response = requests.post(
            "http://localhost:8000/predict",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful")
            print(f"   Total minerals detected: {result['statistics']['total_minerals_detected']}")
            print(f"   Total regions: {result['statistics']['total_regions']}")
            print(f"   Confidence - Mean: {result['statistics']['confidence_stats']['mean']:.3f}")
            print(f"   Confidence - Min: {result['statistics']['confidence_stats']['min']:.3f}")
            print(f"   Confidence - Max: {result['statistics']['confidence_stats']['max']:.3f}")
            
            # Show some detections
            if result['detections']:
                print(f"\n   First 3 detections:")
                for i, det in enumerate(result['detections'][:3]):
                    print(f"      {i+1}. {det['mineral_name']} - Area: {det['area']} px")
            
            return True
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("\n🌐 Testing / endpoint...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Root endpoint working")
            print(f"   Name: {data['name']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("CRISM API Server Test Suite")
    print("=" * 70)
    print("Make sure the server is running at http://localhost:8000\n")
    
    results = []
    
    # Test endpoints
    results.append(("Root", test_root()))
    results.append(("Health", test_health()))
    results.append(("Predict", test_predict()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    print("-" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

if __name__ == '__main__':
    main()
