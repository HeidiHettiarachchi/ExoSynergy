#!/usr/bin/env python3
"""
CRISM Mineral Segmentation Inference Script
===========================================
Standalone script for mineral segmentation inference on PNG images.

Usage:
    python inference_script.py --input image.png [--output output.png] [--ground_truth gt.png]

Features:
- Takes PNG image as input
- Outputs segmentation map
- Calculates IoU if ground truth provided
- Supports both RGB and hyperspectral inputs
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
# Try importing cv2, fall back to scipy if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available, using scipy fallback for image operations")

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import project modules
from src import config
from src.model import create_model
from src.logger import get_logger
# Note: calculate_iou is implemented directly in this script

# Initialize logger
logger = get_logger("CRISM.Inference")

def load_trained_model(model_path=None):
    """Load the trained U-Net model."""
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    # Load checkpoint first to get model configuration
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    
    # Get number of classes from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Infer from output layer
        n_classes = state_dict['outc.conv.weight'].shape[0]
        logger.info(f"✅ Detected {n_classes} classes from checkpoint model_state_dict")
    else:
        state_dict = checkpoint
        n_classes = state_dict['outc.conv.weight'].shape[0]
        logger.info(f"✅ Detected {n_classes} classes from direct state dict")
    
    # Create model with the correct number of classes
    from src.model import UNet
    model = UNet(n_channels=config.NUM_BANDS, n_classes=n_classes)
    
    # Load trained weights  
    model.load_state_dict(state_dict)
    logger.info(f"✅ Loaded model from checkpoint")
    
    model.eval()
    model.to(config.DEVICE)
    
    logger.info(f"✅ Model loaded from {model_path}")
    logger.info(f"   Classes: {config.NUM_CLASSES}")
    logger.info(f"   Device: {config.DEVICE}")
    
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocess input image for inference.
    
    Args:
        image_path: Path to input PNG image
        target_size: Target size for model input
        
    Returns:
        preprocessed_tensor: Torch tensor ready for model inference
        original_size: Original image dimensions
    """
    logger.info(f"📸 Preprocessing image: {image_path}")
    
    # Load image
    if image_path.endswith('.png') or image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
        # Standard RGB image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        original_size = image_np.shape[:2]
        logger.info(f"   RGB image loaded: {image_np.shape}")
        
        # OPTIMIZATION: Resize BEFORE hyperspectral conversion to avoid processing huge arrays
        if target_size != original_size:
            # Resize RGB image (much faster than resizing 350-band image)
            image_np = np.array(Image.fromarray(image_np).resize(target_size, Image.Resampling.BILINEAR))
            logger.info(f"   Resized RGB to {target_size} (before hyperspectral conversion)")
        
        # Normalize to [0, 1]
        if image_np.max() > 1.0:
            image_np = image_np / 255.0
        
        # Now convert to hyperspectral-like format (on smaller, resized image)
        if image_np.shape[2] == 3:  # RGB
            # Simulate hyperspectral using FAST vectorized linear interpolation
            # R at band 0, G at middle, B at end, with linear interpolation between
            
            h, w = image_np.shape[:2]
            hyperspectral = np.zeros((h, w, config.NUM_BANDS), dtype=np.float32)
            
            # Key band positions for R, G, B
            mid_band = config.NUM_BANDS // 2
            end_band = config.NUM_BANDS - 1
            
            # Vectorized interpolation using broadcasting
            for band_idx in range(config.NUM_BANDS):
                if band_idx <= mid_band:
                    # Interpolate between Red (0) and Green (mid)
                    alpha = band_idx / mid_band if mid_band > 0 else 0
                    hyperspectral[:, :, band_idx] = (1 - alpha) * image_np[:, :, 0] + alpha * image_np[:, :, 1]
                else:
                    # Interpolate between Green (mid) and Blue (end)
                    alpha = (band_idx - mid_band) / (end_band - mid_band) if end_band > mid_band else 0
                    hyperspectral[:, :, band_idx] = (1 - alpha) * image_np[:, :, 1] + alpha * image_np[:, :, 2]
            
            image_np = hyperspectral
            logger.info(f"   Converted to hyperspectral: {image_np.shape}")
        
    else:
        raise ValueError(f"Unsupported image format: {image_path}")
    
    # Convert to PyTorch tensor (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
    image_tensor = image_tensor.to(config.DEVICE)
    
    logger.info(f"   Preprocessed tensor shape: {image_tensor.shape}")
    
    return image_tensor, original_size

def run_inference(model, image_tensor):
    """
    Run model inference on preprocessed image.
    
    Args:
        model: Trained PyTorch model
        image_tensor: Preprocessed input tensor
        
    Returns:
        prediction: Segmentation prediction (H, W)
        confidence: Prediction confidence map (H, W)
    """
    logger.info("🔮 Running inference...")
    
    with torch.no_grad():
        # Forward pass
        outputs = model(image_tensor)
        
        # Get class predictions and confidence
        if len(outputs.shape) == 4:  # (B, C, H, W)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            # Move to CPU and convert to numpy
            prediction = prediction.cpu().numpy()[0]  # Remove batch dimension
            confidence = confidence.cpu().numpy()[0]
            
        else:
            raise ValueError(f"Unexpected model output shape: {outputs.shape}")
    
    logger.info(f"   Prediction shape: {prediction.shape}")
    logger.info(f"   Unique classes found: {np.unique(prediction)}")
    logger.info(f"   Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
    
    return prediction, confidence

def calculate_iou_score(prediction, ground_truth, num_classes):
    """Calculate IoU score between prediction and ground truth."""
    logger.info("📊 Calculating IoU score...")
    
    # Ensure same dimensions
    if prediction.shape != ground_truth.shape:
        logger.warning(f"Shape mismatch: pred {prediction.shape}, gt {ground_truth.shape}")
        # Resize ground truth to match prediction
        if CV2_AVAILABLE:
            ground_truth = cv2.resize(ground_truth.astype(np.uint8), 
                                    (prediction.shape[1], prediction.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        else:
            from scipy.ndimage import zoom
            h_scale = prediction.shape[0] / ground_truth.shape[0]
            w_scale = prediction.shape[1] / ground_truth.shape[1]
            ground_truth = zoom(ground_truth.astype(float), (h_scale, w_scale), order=0).astype(np.uint8)
    
    # Calculate IoU for each class
    iou_scores = []
    for class_id in range(num_classes):
        pred_mask = (prediction == class_id)
        gt_mask = (ground_truth == class_id)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0  # Perfect agreement if both are empty
        
        iou_scores.append(iou)
        logger.info(f"   Class {class_id}: IoU = {iou:.4f}")
    
    mean_iou = np.mean(iou_scores)
    logger.info(f"   Mean IoU: {mean_iou:.4f}")
    
    return mean_iou, iou_scores

def load_ground_truth(gt_path, target_size=None):
    """Load and preprocess ground truth image."""
    logger.info(f"🏷️ Loading ground truth: {gt_path}")
    
    if gt_path.endswith('.png') or gt_path.endswith('.jpg'):
        # Load as grayscale for segmentation mask
        gt_image = Image.open(gt_path).convert('L')
        gt_np = np.array(gt_image)
    else:
        raise ValueError(f"Unsupported ground truth format: {gt_path}")
    
    # Resize if needed
    if target_size and gt_np.shape != target_size:
        if CV2_AVAILABLE:
            gt_np = cv2.resize(gt_np, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            from scipy.ndimage import zoom
            h_scale = target_size[1] / gt_np.shape[0]
            w_scale = target_size[0] / gt_np.shape[1]
            gt_np = zoom(gt_np.astype(float), (h_scale, w_scale), order=0).astype(np.uint8)
        logger.info(f"   Resized ground truth to {target_size}")
    
    logger.info(f"   Ground truth shape: {gt_np.shape}")
    logger.info(f"   Unique classes: {np.unique(gt_np)}")
    
    return gt_np

def save_segmentation_map(prediction, confidence, output_path, colormap='tab10'):
    """Save segmentation map as PNG with colors."""
    logger.info(f"💾 Saving segmentation map: {output_path}")
    
    # Create colored segmentation map
    cmap = plt.get_cmap(colormap)
    num_classes = len(np.unique(prediction))
    
    # Normalize predictions to [0, 1] for colormap
    norm_prediction = prediction / max(1, prediction.max())
    colored_seg = cmap(norm_prediction)
    
    # Convert to RGB (remove alpha channel)
    colored_seg_rgb = (colored_seg[:, :, :3] * 255).astype(np.uint8)
    
    # Save using PIL
    seg_image = Image.fromarray(colored_seg_rgb)
    seg_image.save(output_path)
    
    # Also save confidence map
    confidence_path = output_path.replace('.png', '_confidence.png')
    conf_normalized = (confidence * 255).astype(np.uint8)
    conf_image = Image.fromarray(conf_normalized, mode='L')
    conf_image.save(confidence_path)
    
    logger.info(f"   Segmentation saved: {output_path}")
    logger.info(f"   Confidence saved: {confidence_path}")
    
    return output_path, confidence_path

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='CRISM Mineral Segmentation Inference')
    parser.add_argument('--input', '-i', required=True, help='Input PNG image path')
    parser.add_argument('--output', '-o', help='Output segmentation map path (default: input_name_seg.png)')
    parser.add_argument('--ground_truth', '-gt', help='Ground truth PNG for IoU calculation')
    parser.add_argument('--model', '-m', help='Model checkpoint path (default: from config)')
    parser.add_argument('--size', '-s', type=int, nargs=2, default=[256, 256], 
                       help='Target size for inference (default: 256 256)')
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")
    
    # Set output path
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.parent / f"{input_path.stem}_segmentation.png")
    
    logger.info("================================================================")
    logger.info("CRISM Mineral Segmentation Inference")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    if args.ground_truth:
        logger.info(f"Ground Truth: {args.ground_truth}")
    logger.info("================================================================")
    
    try:
        # 1. Load model
        model = load_trained_model(args.model)
        
        # 2. Preprocess input image
        image_tensor, original_size = preprocess_image(args.input, tuple(args.size))
        
        # 3. Run inference
        prediction, confidence = run_inference(model, image_tensor)
        
        # 4. Resize prediction back to original size if needed
        if prediction.shape != original_size:
            if CV2_AVAILABLE:
                prediction = cv2.resize(prediction.astype(np.uint8), 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_NEAREST)
                confidence = cv2.resize(confidence, 
                                      (original_size[1], original_size[0]), 
                                      interpolation=cv2.INTER_LINEAR)
            else:
                from scipy.ndimage import zoom
                h_scale = original_size[0] / prediction.shape[0]
                w_scale = original_size[1] / prediction.shape[1]
                prediction = zoom(prediction.astype(float), (h_scale, w_scale), order=0).astype(np.uint8)
                confidence = zoom(confidence, (h_scale, w_scale), order=1)
            logger.info(f"   Resized output to original size: {original_size}")
        
        # 5. Save segmentation map
        seg_path, conf_path = save_segmentation_map(prediction, confidence, args.output)
        
        # 6. Calculate IoU if ground truth provided
        if args.ground_truth:
            if os.path.exists(args.ground_truth):
                ground_truth = load_ground_truth(args.ground_truth, original_size)
                mean_iou, class_ious = calculate_iou_score(prediction, ground_truth, config.NUM_CLASSES)
                
                logger.info("📊 IoU Results:")
                logger.info(f"   Mean IoU: {mean_iou:.4f}")
                for i, iou in enumerate(class_ious[:10]):  # Show first 10 classes
                    logger.info(f"   Class {i}: {iou:.4f}")
            else:
                logger.warning(f"Ground truth file not found: {args.ground_truth}")
        
        # 7. Print summary
        unique_classes = np.unique(prediction)
        class_counts = [(cls, np.sum(prediction == cls)) for cls in unique_classes]
        
        logger.info("🎉 Inference completed successfully!")
        logger.info(f"   Segmentation saved: {seg_path}")
        logger.info(f"   Confidence map saved: {conf_path}")
        logger.info(f"   Detected classes: {len(unique_classes)}")
        for cls, count in class_counts:
            percentage = (count / prediction.size) * 100
            logger.info(f"     Class {cls}: {count:,} pixels ({percentage:.2f}%)")
        
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise

if __name__ == '__main__':
    main()
