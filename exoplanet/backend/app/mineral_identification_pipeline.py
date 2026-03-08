#!/usr/bin/env python3
"""
Two-Stage Mineral Identification Pipeline
==========================================
Combines CLIP gatekeeper with CNN mineral classifier for robust prediction.

Pipeline Flow:
    1. CLIP Gatekeeper: Reject non-planetary images (fast, lightweight)
    2. Mineral Classifier: Run expensive CNN only on validated planetary images

Usage:
    from mineral_identification_pipeline import MineralIdentificationPipeline
    
    pipeline = MineralIdentificationPipeline(
        mineral_model=your_model,
        mineral_predict_fn=my_mineral_predict,
        threshold=0.65
    )
    
    result = pipeline.predict(image)

This module wires up both stages while avoiding common pitfalls:
    - load_gatekeeper() is called once at startup, not inside the predict loop
    - Images are converted with .convert("RGB") before passing to either model
    - mineral_predict_fn matches your model's actual input and output format
    - Threshold is passed through and not hardcoded in multiple places
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Callable, Dict, Optional, Union, Tuple, List
import time

# Import gatekeeper module
from planetary_gatekeeper import PlanetaryGatekeeper, load_gatekeeper

# Import existing inference infrastructure
from inference_script import load_trained_model, preprocess_image, run_inference
from src import config
from src.logger import get_logger

logger = get_logger("MineralPipeline")


def mineral_predict(image: Union[Image.Image, np.ndarray, str], 
                   model: torch.nn.Module,
                   device: Optional[torch.device] = None) -> Dict:
    """
    Run inference using the existing EfficientNet mineral classifier.
    Preprocess image to 224x224, normalize with ImageNet stats,
    return top predicted mineral class and confidence score.
    
    This function adapts the existing inference_script.py functionality
    to work with the pipeline's interface.
    
    Args:
        image: PIL Image, numpy array, or file path
        model: Trained U-Net model for mineral segmentation
        device: torch device (defaults to config.DEVICE)
    
    Returns:
        Dictionary with:
            - prediction: predicted class map (HxW array)
            - confidence: mean confidence score
            - class_distribution: dict of class percentages
            - mineral_names: dict mapping class IDs to names
    """
    if device is None:
        device = config.DEVICE
    
    # Convert input to PIL Image and ensure RGB
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
        image_path = str(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert("RGB")
        image_path = None
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
        image_path = None
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Save temporary file if needed for existing pipeline
    if image_path is None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img.save(tmp.name)
            image_path = tmp.name
    
    try:
        # Use existing inference pipeline
        # The preprocess_image and run_inference functions handle all the details
        preprocessed, original_size = preprocess_image(image_path)
        prediction, confidence_map = run_inference(model, preprocessed, original_size)
        
        # Calculate class distribution
        unique, counts = np.unique(prediction, return_counts=True)
        total_pixels = prediction.size
        class_distribution = {
            int(cls): float(count / total_pixels * 100) 
            for cls, count in zip(unique, counts)
        }
        
        # Get mineral names
        from api_server import MINERAL_NAMES, get_mineral_name
        
        # Calculate mean confidence
        mean_confidence = float(np.mean(confidence_map))
        
        return {
            "prediction": prediction,
            "confidence": mean_confidence,
            "confidence_map": confidence_map,
            "class_distribution": class_distribution,
            "mineral_names": MINERAL_NAMES,
            "dominant_mineral": get_mineral_name(int(unique[np.argmax(counts)])),
        }
    
    finally:
        # Clean up temp file if created
        if image_path and image_path.startswith("/tmp"):
            try:
                import os
                os.unlink(image_path)
            except:
                pass


class MineralIdentificationPipeline:
    """
    Two-stage pipeline for robust mineral identification.
    
    Stage 1: CLIP gatekeeper filters non-planetary images
    Stage 2: CNN mineral classifier processes validated images
    
    This design prevents wasting compute on irrelevant images.
    """
    
    def __init__(self,
                 mineral_model: torch.nn.Module,
                 mineral_predict_fn: Optional[Callable] = None,
                 threshold: float = 0.45,
                 gatekeeper: Optional[PlanetaryGatekeeper] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the two-stage pipeline.
        
        Args:
            mineral_model: Trained mineral classification model (U-Net)
            mineral_predict_fn: Custom prediction function (default: mineral_predict)
            threshold: CLIP similarity threshold for planetary images (0.0-1.0)
            gatekeeper: Pre-loaded gatekeeper (if None, creates new one)
            device: Torch device for mineral model
        
        Tell Copilot what your model is in a comment above this block so it 
        infers the correct types.
        """
        self.mineral_model = mineral_model
        self.mineral_predict_fn = mineral_predict_fn or mineral_predict
        self.threshold = threshold
        self.device = device or config.DEVICE
        
        # Load gatekeeper once at startup, not inside predict loop
        if gatekeeper is None:
            logger.info(f"Initializing CLIP gatekeeper with threshold={threshold}")
            self.gatekeeper = load_gatekeeper(threshold=threshold)
        else:
            self.gatekeeper = gatekeeper
            logger.info(f"Using pre-loaded gatekeeper with threshold={threshold}")
        
        logger.info("✅ Two-stage pipeline initialized")
    
    def predict(self, 
                image: Union[Image.Image, np.ndarray, str],
                skip_gatekeeper: bool = False,
                return_gatekeeper_scores: bool = False) -> Dict:
        """
        Run two-stage prediction on an image.
        
        Args:
            image: PIL Image, numpy array, or file path
            skip_gatekeeper: If True, skip stage 1 and go directly to mineral classifier
            return_gatekeeper_scores: Include detailed CLIP scores in result
        
        Returns:
            Dictionary with:
                - stage: "rejected" or "classified"
                - gatekeeper_passed: bool
                - gatekeeper_confidence: float
                - gatekeeper_reason: str
                - prediction: mineral class map (if classified)
                - confidence: mineral prediction confidence (if classified)
                - processing_time: total time in seconds
                - gatekeeper_scores: detailed scores (if requested)
        """
        start_time = time.time()
        
        result = {
            "stage": None,
            "gatekeeper_passed": False,
            "gatekeeper_confidence": None,
            "gatekeeper_reason": None,
            "prediction": None,
            "confidence": None,
            "processing_time": None,
        }
        
        # Stage 1: CLIP Gatekeeper
        if not skip_gatekeeper:
            logger.info("Stage 1: Running CLIP gatekeeper...")
            gatekeeper_result = self.gatekeeper.check_image(
                image, 
                return_scores=return_gatekeeper_scores
            )
            
            result["gatekeeper_passed"] = gatekeeper_result["accepted"]
            result["gatekeeper_confidence"] = gatekeeper_result["confidence"]
            result["gatekeeper_reason"] = gatekeeper_result["reason"]
            
            if return_gatekeeper_scores and "scores" in gatekeeper_result:
                result["gatekeeper_scores"] = gatekeeper_result["scores"]
            
            if not gatekeeper_result["accepted"]:
                result["stage"] = "rejected"
                result["processing_time"] = time.time() - start_time
                logger.info(f"❌ Image rejected: {gatekeeper_result['reason']}")
                return result
            
            logger.info(f"✅ Gatekeeper passed (confidence: {gatekeeper_result['confidence']:.3f})")
        else:
            logger.info("⚠️ Skipping gatekeeper (forced)")
            result["gatekeeper_passed"] = True
            result["gatekeeper_reason"] = "Skipped"
        
        # Stage 2: Mineral Classification
        logger.info("Stage 2: Running mineral classifier...")
        mineral_result = self.mineral_predict_fn(
            image, 
            self.mineral_model, 
            self.device
        )
        
        result["stage"] = "classified"
        result["prediction"] = mineral_result["prediction"]
        result["confidence"] = mineral_result["confidence"]
        result["confidence_map"] = mineral_result.get("confidence_map")
        result["class_distribution"] = mineral_result.get("class_distribution")
        result["mineral_names"] = mineral_result.get("mineral_names")
        result["dominant_mineral"] = mineral_result.get("dominant_mineral")
        
        result["processing_time"] = time.time() - start_time
        
        logger.info(f"✅ Classification complete in {result['processing_time']:.2f}s")
        logger.info(f"   Dominant mineral: {result['dominant_mineral']}")
        
        return result
    
    def batch_predict(self, 
                     images: List[Union[Image.Image, np.ndarray, str]],
                     skip_gatekeeper: bool = False) -> List[Dict]:
        """
        Run prediction on multiple images.
        
        Args:
            images: List of images (PIL Images, numpy arrays, or file paths)
            skip_gatekeeper: If True, skip stage 1 for all images
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"\nProcessing image {i+1}/{len(images)}")
            result = self.predict(image, skip_gatekeeper=skip_gatekeeper)
            results.append(result)
        
        # Summary statistics
        rejected = sum(1 for r in results if r["stage"] == "rejected")
        classified = sum(1 for r in results if r["stage"] == "classified")
        
        logger.info(f"\n📊 Batch Summary:")
        logger.info(f"   Total images: {len(images)}")
        logger.info(f"   Rejected by gatekeeper: {rejected}")
        logger.info(f"   Successfully classified: {classified}")
        
        return results
    
    def update_threshold(self, new_threshold: float):
        """
        Update the gatekeeper threshold.
        
        Args:
            new_threshold: New similarity threshold (0.0-1.0)
        """
        self.threshold = new_threshold
        self.gatekeeper.threshold = new_threshold
        logger.info(f"Threshold updated to {new_threshold}")


def create_pipeline(model_path: Optional[str] = None,
                   threshold: float = 0.45,
                   device: Optional[torch.device] = None) -> MineralIdentificationPipeline:
    """
    Convenience function to create a ready-to-use pipeline.
    
    Args:
        model_path: Path to trained mineral model (default: config.MODEL_SAVE_PATH)
        threshold: CLIP gatekeeper threshold
        device: Torch device
    
    Returns:
        Initialized MineralIdentificationPipeline
    
    Example:
        >>> pipeline = create_pipeline(threshold=0.65)
        >>> result = pipeline.predict("test_image.png")
    """
    logger.info("🚀 Creating two-stage mineral identification pipeline...")
    
    # Load mineral classification model
    mineral_model = load_trained_model(model_path)
    
    # Create pipeline
    pipeline = MineralIdentificationPipeline(
        mineral_model=mineral_model,
        mineral_predict_fn=mineral_predict,
        threshold=threshold,
        device=device
    )
    
    logger.info("✅ Pipeline ready!")
    return pipeline


if __name__ == "__main__":
    """Test the pipeline with a sample image."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mineral_identification_pipeline.py <image_path> [threshold]")
        print("Example: python mineral_identification_pipeline.py test_mars.png 0.65")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    
    print(f"\n🚀 Two-Stage Mineral Identification Pipeline")
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}\n")
    
    # Create pipeline
    pipeline = create_pipeline(threshold=threshold)
    
    # Run prediction
    result = pipeline.predict(image_path, return_gatekeeper_scores=True)
    
    print(f"\n📊 Results:")
    print(f"Stage: {result['stage']}")
    print(f"Gatekeeper passed: {result['gatekeeper_passed']}")
    print(f"Gatekeeper confidence: {result['gatekeeper_confidence']:.3f}")
    print(f"Reason: {result['gatekeeper_reason']}")
    
    if result['stage'] == 'classified':
        print(f"\nMineral Classification:")
        print(f"  Dominant mineral: {result['dominant_mineral']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Processing time: {result['processing_time']:.2f}s")
        
        if result.get('class_distribution'):
            print(f"\n  Top minerals detected:")
            sorted_minerals = sorted(
                result['class_distribution'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            for cls_id, percentage in sorted_minerals:
                mineral_name = result['mineral_names'].get(cls_id, f"Class {cls_id}")
                print(f"    {mineral_name}: {percentage:.2f}%")
    else:
        print(f"\n❌ Image rejected - no mineral classification performed")
        print(f"   Processing time: {result['processing_time']:.2f}s")
