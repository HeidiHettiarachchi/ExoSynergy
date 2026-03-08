#!/usr/bin/env bash
# Build script for Render deployment
# This script runs during the build phase before the application starts

set -o errexit  # Exit on error

echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "📥 Downloading pre-trained model from Google Drive..."
python download_model.py

echo "✅ Build completed successfully!"
