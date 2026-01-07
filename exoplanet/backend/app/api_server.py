#!/usr/bin/env python3
"""
CRISM Mineral Classification API Server
========================================
FastAPI REST API for mineral segmentation inference.

Endpoints:
    POST /predict - Upload image and get segmentation with bounding boxes
    GET /health - Check API health status
    GET /docs - Interactive API documentation

Usage:
    python api_server.py
    
Then send requests to: http://localhost:8000/predict
API Docs: http://localhost:8000/docs
"""

import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from typing import Optional, List
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import cv2

# Add project root to path
import sys
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src import config
from src.logger import get_logger
from inference_script import load_trained_model, preprocess_image, run_inference

# Mineral class to name mapping (Mars CRISM minerals)
MINERAL_NAMES = {
    0: "Background/Unclassified",
    1: "Olivine",
    2: "Pyroxene (Low-Ca)",
    3: "Pyroxene (High-Ca)",
    4: "Plagioclase",
    5: "Hydrated Silica",
    6: "Fe/Mg Phyllosilicates",
    7: "Al Phyllosilicates",
    8: "Fe/Mg Smectite",
    9: "Chlorite",
    10: "Serpentine",
    11: "Prehnite",
    12: "Zeolite",
    13: "Carbonate",
    14: "Hydrated Sulfate",
    15: "Gypsum",
    16: "Kieserite",
    17: "Polyhydrated Sulfate",
    18: "Fe Oxide/Hydroxide",
    19: "Hematite",
    20: "Goethite",
    21: "Jarosite",
    22: "Fe/Mg Carbonate",
    23: "Mg Carbonate",
    24: "Ca/Fe Carbonate",
    25: "Chloride",
    26: "Perchlorate",
    27: "Fe Sulfate",
    28: "Al Sulfate",
    29: "Mg Sulfate",
    30: "Ca Sulfate",
    31: "Opaline Silica",
    32: "Amorphous Silica",
    33: "Crystalline Silica",
    34: "Dust/Ice",
    35: "Ferric Oxide",
    36: "Ferrous Minerals",
    37: "Mixed Composition"
}

def get_mineral_name(class_id):
    """Get mineral name from class ID."""
    return MINERAL_NAMES.get(class_id, f"Unknown Mineral Class {class_id}")

# Initialize FastAPI app
app = FastAPI(
    title="CRISM Mineral Classification API",
    description="AI-powered mineral segmentation for Mars CRISM hyperspectral imagery",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger("CRISM.API")

# Load model once at startup
MODEL = None

# Response Models
class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int

class Center(BaseModel):
    x: int
    y: int

class Detection(BaseModel):
    mineral_class: int
    mineral_name: str
    bbox: BoundingBox
    area: int
    center: Center

class ClassDistribution(BaseModel):
    mineral_class: int
    mineral_name: str
    pixel_count: int
    percentage: float

class ConfidenceStats(BaseModel):
    mean: float
    min: float
    max: float

class ImageSize(BaseModel):
    width: int
    height: int

class Statistics(BaseModel):
    total_minerals_detected: int
    total_regions: int
    image_size: ImageSize
    class_distribution: List[ClassDistribution]
    confidence_stats: ConfidenceStats

class PredictionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    statistics: Statistics
    annotated_image: Optional[str] = None
    segmentation_map: Optional[str] = None


def initialize_model():
    """Load model once at server startup."""
    global MODEL
    try:
        MODEL = load_trained_model()
        logger.info("✅ Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


def get_mineral_bounding_boxes(prediction, min_area=100):
    """Extract bounding boxes for each mineral class region."""
    boxes = []
    unique_classes = np.unique(prediction)
    
    for mineral_class in unique_classes:
        mask = (prediction == mineral_class).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append({
                    'mineral_class': int(mineral_class),
                    'bbox': {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)},
                    'area': int(area),
                    'center': {'x': int(x + w/2), 'y': int(y + h/2)}
                })
    
    return boxes


def create_annotated_image(original_image, prediction, confidence, boxes):
    """Create annotated image with bounding boxes and labels."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original with boxes
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image with Detections', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    for box in boxes[:50]:
        mineral_class = box['mineral_class']
        mineral_name = get_mineral_name(mineral_class)
        bbox = box['bbox']
        rect = mpatches.Rectangle(
            (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
            fill=False, edgecolor=colors[mineral_class % 20], linewidth=2
        )
        axes[0].add_patch(rect)
        
        # Use abbreviated mineral name for labels
        short_name = mineral_name[:12] + "..." if len(mineral_name) > 12 else mineral_name
        axes[0].text(
            bbox['x'], bbox['y'] - 5, short_name,
            color=colors[mineral_class % 20], fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )
    
    # 2. Segmentation
    axes[1].imshow(prediction, cmap='tab20', interpolation='nearest')
    axes[1].set_title('Mineral Segmentation Map', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # 3. Confidence
    axes[2].imshow(confidence, cmap='hot', interpolation='bilinear')
    axes[2].set_title('Prediction Confidence', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    annotated_img = Image.open(buf)
    plt.close(fig)
    
    return annotated_img


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup."""
    logger.info("🚀 Starting up API server...")
    initialize_model()


@app.get("/")
async def root():
    """API information."""
    return {
        "name": "CRISM Mineral Classification API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(config.DEVICE),
        'num_classes': config.NUM_CLASSES
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Image file to classify"),
    min_area: int = Form(50, description="Minimum region area in pixels"),
    return_image: bool = Form(True, description="Return annotated images")
):
    """
    Predict mineral segmentation from uploaded image.
    
    Upload an image and receive:
    - List of detected mineral regions with bounding boxes
    - Classification statistics
    - Annotated image with bounding boxes (optional)
    - Segmentation map (optional)
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        logger.info(f"📸 Received image: {image.filename}")
        
        # Load image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        original_np = np.array(pil_image)
        original_size = original_np.shape[:2]
        
        # Save temporarily
        temp_path = f'temp_upload_{image.filename}'
        pil_image.save(temp_path)
        
        try:
            # Preprocess
            image_tensor, _ = preprocess_image(temp_path, target_size=(256, 256))
            
            # Run inference
            prediction, confidence = run_inference(MODEL, image_tensor)
            
            # Resize back
            if prediction.shape != original_size:
                prediction = cv2.resize(
                    prediction.astype(np.uint8),
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                confidence = cv2.resize(
                    confidence,
                    (original_size[1], original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
            
            # Get bounding boxes
            boxes = get_mineral_bounding_boxes(prediction, min_area=min_area)
            
            # Statistics
            unique_classes, counts = np.unique(prediction, return_counts=True)
            total_pixels = prediction.size
            
            statistics = Statistics(
                total_minerals_detected=len(unique_classes),
                total_regions=len(boxes),
                image_size=ImageSize(width=original_size[1], height=original_size[0]),
                class_distribution=[
                    ClassDistribution(
                        mineral_class=int(cls),
                        mineral_name=get_mineral_name(int(cls)),
                        pixel_count=int(count),
                        percentage=float(count / total_pixels * 100)
                    )
                    for cls, count in zip(unique_classes, counts)
                ],
                confidence_stats=ConfidenceStats(
                    mean=float(confidence.mean()),
                    min=float(confidence.min()),
                    max=float(confidence.max())
                )
            )
            
            # Convert boxes to Detection objects with mineral names
            detections = [
                Detection(
                    mineral_class=box['mineral_class'],
                    mineral_name=get_mineral_name(box['mineral_class']),
                    bbox=BoundingBox(**box['bbox']),
                    area=box['area'],
                    center=Center(**box['center'])
                )
                for box in boxes
            ]
            
            response_data = {
                'success': True,
                'detections': detections,
                'statistics': statistics
            }
            
            # Add images if requested
            if return_image:
                annotated_img = create_annotated_image(original_np, prediction, confidence, boxes)
                response_data['annotated_image'] = image_to_base64(annotated_img)
                
                seg_img = Image.fromarray((plt.cm.tab20(prediction / prediction.max())[:, :, :3] * 255).astype(np.uint8))
                response_data['segmentation_map'] = image_to_base64(seg_img)
            
            logger.info(f"✅ Inference completed: {len(boxes)} regions, {len(unique_classes)} minerals")
            
            return response_data
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        import traceback
        raise HTTPException(
            status_code=500,
            detail={
                'error': str(e),
                'traceback': traceback.format_exc()
            }
        )


def main():
    """Start the API server."""
    logger.info("=" * 70)
    logger.info("CRISM Mineral Classification API Server (FastAPI)")
    logger.info("=" * 70)
    logger.info("🌐 Starting server...")
    logger.info("📍 API URL: http://localhost:8000")
    logger.info("📖 Interactive Docs: http://localhost:8000/docs")
    logger.info("📖 ReDoc: http://localhost:8000/redoc")
    logger.info("Press Ctrl+C to stop the server")
    logger.info("=" * 70)
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == '__main__':
    main()
