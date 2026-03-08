#!/usr/bin/env python3
"""
Download trained model from Google Drive
=========================================
This script downloads the pre-trained model file from Google Drive
and places it in the saved_models directory.

Usage:
    python download_model.py
    
This script is automatically run during deployment on Render.
"""

import os
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Installing gdown...")
    os.system(f"{sys.executable} -m pip install gdown")
    import gdown

# Google Drive file ID from the shareable link
GOOGLE_DRIVE_FILE_ID = "1C3mtxAMuUW4JUA8uDy-hHr1_dAaS-crz"

# Model download configuration
MODEL_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = SCRIPT_DIR / "saved_models"
MODEL_FILE_PATH = SAVED_MODELS_DIR / "best_unet_model.pth"


def download_model():
    """Download the model file from Google Drive if it doesn't exist."""
    
    # Create saved_models directory if it doesn't exist
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if MODEL_FILE_PATH.exists():
        file_size = MODEL_FILE_PATH.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"✓ Model already exists at {MODEL_FILE_PATH}")
        print(f"  Size: {file_size:.2f} MB")
        return True
    
    print(f"Downloading model from Google Drive...")
    print(f"  Source: {MODEL_URL}")
    print(f"  Destination: {MODEL_FILE_PATH}")
    
    try:
        # Download the file
        gdown.download(MODEL_URL, str(MODEL_FILE_PATH), quiet=False)
        
        # Verify download
        if MODEL_FILE_PATH.exists():
            file_size = MODEL_FILE_PATH.stat().st_size / (1024 * 1024)  # Size in MB
            print(f"✓ Model downloaded successfully!")
            print(f"  Size: {file_size:.2f} MB")
            return True
        else:
            print(f"✗ Download failed - file not found at {MODEL_FILE_PATH}")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure the Google Drive link has 'Anyone with the link' viewing permissions")
        print("2. Verify the file ID is correct")
        print(f"3. Try downloading manually from: https://drive.google.com/file/d/{GOOGLE_DRIVE_FILE_ID}/view")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
