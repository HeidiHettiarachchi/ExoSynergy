#!/usr/bin/env python3
"""
Planetary Image Gatekeeper using CLIP
======================================
Two-stage mineral identification pipeline - Stage 1: CLIP gatekeeper to reject non-planetary images
Stage 2: Existing CNN mineral classifier

This module uses OpenAI's CLIP model to filter out non-planetary images before
running expensive mineral classification. Only images that pass the planetary 
similarity threshold proceed to the mineral CNN.

Key Functions:
    - load_gatekeeper(): Load CLIP model once at startup
    - is_planetary_image(): Check if image is planetary/geological
    - PlanetaryGatekeeper: Class-based interface for integration

Common pitfalls avoided:
    - Uses open_clip (not legacy "clip" package)
    - Always converts images with .convert("RGB")
    - Returns rejection "reason" for debugging
    - Loads model once at startup, not in prediction loop
"""

import torch
import open_clip
from PIL import Image
from typing import Tuple, Dict, Optional
import numpy as np
from pathlib import Path

# Global module logger
try:
    from src.logger import get_logger
    logger = get_logger("PlanetaryGatekeeper")
except:
    import logging
    logger = logging.getLogger("PlanetaryGatekeeper")


class PlanetaryGatekeeper:
    """
    CLIP-based gatekeeper to filter planetary/geological images.
    Rejects images that don't match planetary imagery characteristics.
    """
    
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", threshold: float = 0.45):
        """
        Initialize the planetary image gatekeeper.
        
        Args:
            model_name: CLIP model architecture (default: ViT-B-32)
            pretrained: Pretrained weights to use (default: openai)
            threshold: Minimum similarity score to accept image (0.0-1.0, default: 0.45)
        
        The threshold is passed through and not hardcoded in multiple places.
        """
        self.model_name = model_name
        self.pretrained = pretrained
        self.threshold = threshold
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        # Planetary image descriptors - tuned for Mars CRISM data
        # More specific to distinguish from Earth landscapes
        self.planetary_prompts = [
            "a satellite image of Mars surface showing rocks and minerals",
            "a Mars orbital hyperspectral image",
            "planetary geological survey from space",
            "CRISM spectral data of Martian terrain",
            "mineral composition map from Mars orbit",
            "remote sensing image of an alien planet",
            "extraterrestrial geological surface from satellite",
            "barren rocky landscape from space",
            "scientific planetary survey image",
        ]
        
        # Non-planetary prompts for contrast - more comprehensive
        self.non_planetary_prompts = [
            "a photograph of a car or vehicle",
            "a photograph of everyday objects",
            "a portrait photo of a person or people",
            "a screenshot of text or website",
            "a photo of an animal or pet",
            "a drawing or artwork or painting",
            "random noise or blank white image",
            "a photograph taken on Earth with buildings or trees",
            "urban or city landscape",
            "indoor photograph",
        ]
        
        logger.info(f"Initializing PlanetaryGatekeeper with threshold={threshold}")
    
    def load_model(self):
        """
        Load CLIP model and preprocessing pipeline.
        Call this once at startup, not inside the predict loop.
        """
        if self.model is not None:
            logger.info("Model already loaded, skipping reload")
            return
        
        logger.info(f"Loading CLIP model: {self.model_name} ({self.pretrained})")
        
        # Load the model - uses open_clip (not old OpenAI clip repo)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        self.model.eval()
        
        logger.info(f"✅ CLIP model loaded successfully on {device}")
    
    def check_image(self, image_input, return_scores: bool = False) -> Dict:
        """
        Check if an image appears to be planetary/geological imagery.
        
        Args:
            image_input: PIL Image, numpy array, or file path
            return_scores: If True, include similarity scores in result
        
        Returns:
            Dictionary with keys:
                - accepted: bool, whether image passes threshold
                - confidence: float, planetary similarity score (0.0-1.0)
                - reason: str, explanation for rejection (if rejected)
                - scores: dict, detailed scores (if return_scores=True)
        
        Always converts images with .convert("RGB") before processing.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert input to PIL Image and ensure RGB format
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
        
        # Preprocess image
        device = next(self.model.parameters()).device
        image_tensor = self.preprocess(image).unsqueeze(0).to(device)
        
        # Tokenize text prompts
        planetary_tokens = self.tokenizer(self.planetary_prompts).to(device)
        non_planetary_tokens = self.tokenizer(self.non_planetary_prompts).to(device)
        
        # Get CLIP embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            planetary_text_features = self.model.encode_text(planetary_tokens)
            non_planetary_text_features = self.model.encode_text(non_planetary_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            planetary_text_features = planetary_text_features / planetary_text_features.norm(dim=-1, keepdim=True)
            non_planetary_text_features = non_planetary_text_features / non_planetary_text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarities
            planetary_similarities = (image_features @ planetary_text_features.T).squeeze(0)
            non_planetary_similarities = (image_features @ non_planetary_text_features.T).squeeze(0)
            
            # Aggregate scores
            planetary_score = planetary_similarities.mean().item()
            non_planetary_score = non_planetary_similarities.mean().item()
            max_planetary = planetary_similarities.max().item()
            max_non_planetary = non_planetary_similarities.max().item()
        
        # Calculate confidence (normalized planetary vs non-planetary)
        # Use relative scoring: how much more planetary than non-planetary
        total_score = planetary_score + non_planetary_score
        if total_score > 0:
            confidence = planetary_score / total_score
        else:
            confidence = 0.0
        
        # Decision logic: accept if confidence meets threshold
        accepted = confidence >= self.threshold
        
        result = {
            "accepted": accepted,
            "confidence": confidence,
        }
        
        # Add reason for rejection for debugging
        if not accepted:
            result["reason"] = f"Planetary confidence {confidence:.3f} below threshold {self.threshold:.3f}"
        else:
            result["reason"] = f"Passed planetary check (confidence: {confidence:.3f})"
        
        # Include detailed scores if requested
        if return_scores:
            result["scores"] = {
                "planetary_avg": planetary_score,
                "planetary_max": max_planetary,
                "non_planetary_avg": non_planetary_score,
                "non_planetary_max": max_non_planetary,
                "individual_planetary": planetary_similarities.cpu().numpy().tolist(),
                "individual_non_planetary": non_planetary_similarities.cpu().numpy().tolist(),
            }
        
        return result
    
    def __call__(self, image_input) -> Dict:
        """Convenience method to check image."""
        return self.check_image(image_input)


# Module-level singleton for easy integration
_gatekeeper_instance = None


def load_gatekeeper(threshold: float = 0.45, force_reload: bool = False) -> PlanetaryGatekeeper:
    """
    Load the planetary gatekeeper model.
    Call this once at startup, not inside the predict loop.
    
    Args:
        threshold: Minimum similarity for accepting images (0.0-1.0)
        force_reload: Force reload even if already loaded
    
    Returns:
        Initialized PlanetaryGatekeeper instance
    """
    global _gatekeeper_instance
    
    if _gatekeeper_instance is None or force_reload:
        _gatekeeper_instance = PlanetaryGatekeeper(threshold=threshold)
        _gatekeeper_instance.load_model()
    
    return _gatekeeper_instance


def is_planetary_image(image_input, threshold: float = 0.45, 
                       gatekeeper: Optional[PlanetaryGatekeeper] = None) -> Tuple[bool, float, str]:
    """
    Quick check if an image is planetary/geological.
    
    Args:
        image_input: PIL Image, numpy array, or file path
        threshold: Minimum similarity score (if creating new gatekeeper)
        gatekeeper: Pre-loaded gatekeeper instance (recommended for efficiency)
    
    Returns:
        Tuple of (accepted, confidence, reason)
    
    Example:
        >>> gatekeeper = load_gatekeeper(threshold=0.65)
        >>> accepted, confidence, reason = is_planetary_image(img, gatekeeper=gatekeeper)
        >>> if accepted:
        >>>     # Proceed to mineral classification
    """
    if gatekeeper is None:
        gatekeeper = load_gatekeeper(threshold=threshold)
    
    result = gatekeeper.check_image(image_input)
    return result["accepted"], result["confidence"], result["reason"]


if __name__ == "__main__":
    """Test the gatekeeper with sample images."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python planetary_gatekeeper.py <image_path> [threshold]")
        print("Example: python planetary_gatekeeper.py test_image.png 0.65")
        sys.exit(1)
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.65
    
    print(f"\n🔍 Testing Planetary Gatekeeper")
    print(f"Image: {image_path}")
    print(f"Threshold: {threshold}\n")
    
    # Load gatekeeper
    gatekeeper = load_gatekeeper(threshold=threshold)
    
    # Check image with detailed scores
    result = gatekeeper.check_image(image_path, return_scores=True)
    
    print(f"✅ Result: {'ACCEPTED' if result['accepted'] else '❌ REJECTED'}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Reason: {result['reason']}")
    
    if "scores" in result:
        print(f"\n📊 Detailed Scores:")
        print(f"  Planetary avg: {result['scores']['planetary_avg']:.3f}")
        print(f"  Planetary max: {result['scores']['planetary_max']:.3f}")
        print(f"  Non-planetary avg: {result['scores']['non_planetary_avg']:.3f}")
        print(f"  Non-planetary max: {result['scores']['non_planetary_max']:.3f}")
