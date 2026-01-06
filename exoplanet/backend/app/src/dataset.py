"""
Dataset module for CRISM hyperspectral mineral segmentation.
Handles loading, reconstruction, and preprocessing of MATLAB data files.
"""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any
import warnings
import os

from . import config
from .logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def load_and_reconstruct_data() -> Tuple[List[np.ndarray], List[np.ndarray], List[str], dict, List]:
    """
    Load and reconstruct 2D hyperspectral images from MATLAB files.
    
    Based on the dataset documentation:
    - CRISM_labeled_pixels_ratioed.mat contains ratioed spectra with labels
    - Structure: pixspec (592,413 × 350), pixlabs (592,413), pixims (592,413), 
                 pixcrds (592,413 × 2), im_names (77)
    
    Returns:
        Tuple containing:
        - images: List of arrays, each of shape (H, W, C) 
        - masks: List of arrays, each of shape (H, W) with pixel-level class labels  
        - scene_ids: List of unique scene identifiers
        - label_mapping: Dictionary mapping original to continuous labels
        - original_labels: List of original label values
    """
    logger.info("🔄 Loading CRISM hyperspectral data files...")
    
    try:
        # Load the labeled data (main dataset with mineral classifications)
        logger.info(f"📁 Loading labeled data: {config.LABELED_DATA_PATH}")
        labeled_data = sio.loadmat(config.LABELED_DATA_PATH)
        available_keys = [k for k in labeled_data.keys() if not k.startswith('__')]
        logger.info(f"📋 Available data keys: {available_keys}")
        
        # Extract the data based on documented structure
        spectra = labeled_data['pixspec']  # Shape: (592413, 350)
        labels = labeled_data['pixlabs'].flatten()   # Shape: (592413,) - flatten in case it's (n,1)
        image_ids = labeled_data['pixims'].flatten() # Shape: (592413,) - flatten in case it's (n,1)
        coordinates = labeled_data['pixcrds'] # Shape: (592413, 2)
        image_names = labeled_data['im_names'] # Shape: (77,) list of image names
        
        logger.info(f"📊 Extracted data shapes:")
        logger.info(f"   Spectra: {spectra.shape} ({spectra.dtype})")
        logger.info(f"   Labels: {labels.shape} (range: {labels.min()}-{labels.max()})")
        logger.info(f"   Image IDs: {image_ids.shape} (range: {image_ids.min()}-{image_ids.max()})")
        logger.info(f"   Coordinates: {coordinates.shape}")
        logger.info(f"   Image names: {len(image_names) if hasattr(image_names, '__len__') else 'scalar'}")
        
    except Exception as e:
        logger.error(f"Failed to load MATLAB files: {e}")
        raise RuntimeError(f"Failed to load MATLAB files: {e}")
    
    # Verify spectral band count
    actual_bands = spectra.shape[1]
    if actual_bands != config.NUM_BANDS:
        print(f"⚠️  INFO: Config expects {config.NUM_BANDS} bands, data has {actual_bands} bands")
        print(f"Using {actual_bands} spectral bands from data")
    
    # Analyze label distribution to create proper mapping
    unique_labels_in_data = np.unique(labels)
    print(f"Raw unique labels in dataset: {unique_labels_in_data}")
    
    # Create label mapping from sparse to continuous indices
    # Following CRISM ML repository approach for proper class handling
    label_mapping = {}
    continuous_labels = []
    
    for i, original_label in enumerate(sorted(unique_labels_in_data)):
        label_mapping[original_label] = i
        continuous_labels.append(original_label)
    
    print(f"Label mapping: {label_mapping}")
    print(f"Mapped to continuous range: 0 to {len(unique_labels_in_data)-1}")
    
    # Apply label mapping to all labels
    mapped_labels = np.array([label_mapping[label] for label in labels])
    
    # Get unique image IDs
    unique_image_ids = np.unique(image_ids)
    print(f"Found {len(unique_image_ids)} unique images")
    print(f"Image ID range: {unique_image_ids.min()} to {unique_image_ids.max()}")
    
    images = []
    masks = []
    scene_list = []
    
    # Process more images for better performance (inspired by CRISM ML)
    processed_count = 0
    max_images = 20  # Process reasonable subset for training (can be increased)
    
    for img_id in unique_image_ids:
        if processed_count >= max_images:
            print(f"Limiting to first {max_images} images for development")
            break
            
        # Get pixels for this image
        img_mask = image_ids == img_id
        img_spectra = spectra[img_mask]  # (n_pixels, n_bands)
        img_coords = coordinates[img_mask]  # (n_pixels, 2)
        img_labels = mapped_labels[img_mask]  # (n_pixels,) - using mapped labels
        
        if len(img_spectra) == 0:
            continue
        
        # Determine spatial dimensions from coordinates
        # Coordinates are (x, y) format
        x_coords = img_coords[:, 0].astype(int)
        y_coords = img_coords[:, 1].astype(int)
        
        # Get bounding box
        min_x, max_x = x_coords.min(), x_coords.max()
        min_y, max_y = y_coords.min(), y_coords.max()
        
        height = max_y - min_y + 1
        width = max_x - min_x + 1
        
        # Skip very small images
        if height < 10 or width < 10:
            continue
        
        # Skip very large images (memory constraints) - but allow more for better dataset
        if height * width > 100000:  # Allow larger images for better training data
            print(f"    Skipping large image {img_id}: {height}x{width} pixels")
            continue
            
        # Initialize arrays for this image
        img_array = np.zeros((height, width, actual_bands), dtype=np.float32)
        mask_array = np.zeros((height, width), dtype=np.int64)
        
        # Place pixels at their correct locations
        for spec, coord, label in zip(img_spectra, img_coords, img_labels):
            # Convert to relative coordinates
            x = int(coord[0]) - min_x
            y = int(coord[1]) - min_y
            
            # Place pixel (ensuring within bounds)
            if 0 <= y < height and 0 <= x < width:
                img_array[y, x] = spec
                mask_array[y, x] = int(label)
        
        # Only keep images with reasonable number of labeled pixels
        labeled_pixels = np.sum(mask_array > 0)
        total_pixels = height * width
        label_ratio = labeled_pixels / total_pixels
        
        if label_ratio > 0.01:  # At least 1% labeled pixels (more lenient for more data)
            images.append(img_array)
            masks.append(mask_array)
            
            # Use image name if available, otherwise use ID
            try:
                if hasattr(image_names, '__len__') and len(image_names) > 0:
                    # Handle different possible formats of image_names
                    if img_id <= len(image_names):
                        if hasattr(image_names[int(img_id-1)], 'item'):
                            scene_name = str(image_names[int(img_id-1)].item())
                        else:
                            scene_name = str(image_names[int(img_id-1)])
                    else:
                        scene_name = f"image_{img_id}"
                else:
                    scene_name = f"image_{img_id}"
            except:
                scene_name = f"image_{img_id}"
                
            scene_list.append(scene_name)
            
            print(f"  Image {img_id}: {height}x{width}, {labeled_pixels}/{total_pixels} labeled ({label_ratio:.1%})")
            processed_count += 1
    
    print(f"\n✅ Successfully reconstructed {len(images)} images")
    if len(images) > 0:
        print(f"Image dimensions range: {[(img.shape[0], img.shape[1]) for img in images[:3]]}...")
        print(f"Spectral bands: {actual_bands}")
        
        # Check unique labels
        all_labels = np.concatenate([mask.flatten() for mask in masks])
        unique_labels = np.unique(all_labels)
        print(f"Unique labels found: {unique_labels}")
        print(f"Label distribution: {[(label, np.sum(all_labels == label)) for label in unique_labels]}")
    
    # Store label mapping information for global access
    # Update config with actual number of classes found
    actual_num_classes = len(unique_labels_in_data)
    if actual_num_classes != config.NUM_CLASSES:
        print(f"📊 Updating NUM_CLASSES from {config.NUM_CLASSES} to {actual_num_classes}")
        config.NUM_CLASSES = actual_num_classes
    
    # Return as lists since images have different dimensions
    return images, masks, scene_list, label_mapping, continuous_labels


def _extract_field(data_array: np.ndarray, possible_names: List[str]) -> np.ndarray:
    """
    Extract a field from structured array, trying multiple possible field names.
    """
    if data_array.dtype.names is None:
        return data_array
        
    for name in possible_names:
        if name in data_array.dtype.names:
            field_data = data_array[name]
            # Handle nested arrays
            if field_data.dtype == 'O':  # Object array
                field_data = field_data[0, 0] if field_data.ndim == 2 else field_data[0]
            return np.array(field_data)
    
    raise ValueError(f"Could not find any of {possible_names} in array fields: {data_array.dtype.names}")


class CRISMDataset(Dataset):
    """
    PyTorch Dataset for CRISM hyperspectral data.
    
    Handles normalization, tensor conversion, and dimension permutation
    to convert from (H, W, C) to PyTorch's expected (C, H, W) format.
    """
    
    def __init__(self, images: List[np.ndarray], masks: List[np.ndarray], 
                 scene_ids: List[str], normalize: bool = True, target_size: tuple = (128, 128)):
        """
        Initialize dataset.
        
        Args:
            images: List of arrays, each of shape (H, W, C) with hyperspectral data
            masks: List of arrays, each of shape (H, W) with pixel-level labels
            scene_ids: List of scene identifiers
            normalize: Whether to apply per-image normalization
            target_size: Target size (H, W) to resize all images for batching
        """
        self.images = images
        self.masks = masks
        self.scene_ids = scene_ids
        self.normalize = normalize
        self.target_size = target_size
        
        print(f"Dataset initialized with {len(self.images)} scenes")
        
        if len(self.images) > 0:
            print(f"Image shape: {self.images[0].shape}, Mask shape: {self.masks[0].shape}")
            # Compute dataset statistics
            self._compute_statistics()
        else:
            print("Warning: Empty dataset created")
    
    def _compute_statistics(self):
        """Compute and print dataset statistics."""
        # Handle list-based data structure inspired by https://github.com/Banus/crism_ml
        all_mask_values = np.concatenate([mask.flatten() for mask in self.masks])
        unique_classes = np.unique(all_mask_values)
        class_counts = {cls: np.sum(all_mask_values == cls) for cls in unique_classes}
        
        print(f"Unique classes: {unique_classes}")
        print(f"Class distribution: {class_counts}")
        print(f"Total pixels: {len(all_mask_values)}")
        
        # Spectral statistics
        all_spectral_values = np.concatenate([img.flatten() for img in self.images])
        print(f"Spectral range: [{all_spectral_values.min():.3f}, {all_spectral_values.max():.3f}]")
        print(f"Mean spectrum: {all_spectral_values.mean():.3f}")
        print(f"Std spectrum: {all_spectral_values.std():.3f}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Tuple of (image_tensor, mask_tensor) where:
            - image_tensor: Shape (C, target_H, target_W) with normalized spectral data
            - mask_tensor: Shape (target_H, target_W) with class labels
        """
        image = self.images[idx].copy()  # Shape: (H, W, C)
        mask = self.masks[idx].copy()    # Shape: (H, W)
        
        # Normalize per-image if requested
        if self.normalize:
            # Min-max normalization per band
            for band in range(image.shape[2]):
                band_data = image[:, :, band]
                if band_data.max() > band_data.min():
                    image[:, :, band] = (band_data - band_data.min()) / (band_data.max() - band_data.min())
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask_tensor = torch.from_numpy(mask).long()  # (H, W)
        
        # Resize to target size for consistent batching (inspired by CRISM ML approaches)
        target_h, target_w = self.target_size
        
        # Resize image using interpolation
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),  # Add batch dim
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)  # Remove batch dim
        
        # Resize mask using nearest neighbor to preserve class labels
        mask_tensor = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0).float(),  # Add batch and channel dims
            size=(target_h, target_w),
            mode='nearest'
        ).squeeze(0).squeeze(0).long()  # Remove extra dims and convert back to long
        
        return image_tensor, mask_tensor


def get_dataloaders(batch_size: int = None, 
                   train_ratio: float = None,
                   val_ratio: float = None,
                   test_ratio: float = None,
                   target_size: tuple = (128, 128)) -> Tuple[DataLoader, DataLoader, DataLoader, dict, List]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        batch_size: Batch size for all dataloaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation  
        test_ratio: Proportion of data for testing
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Use config defaults if not specified
    batch_size = batch_size or config.BATCH_SIZE
    train_ratio = train_ratio or config.TRAIN_RATIO
    val_ratio = val_ratio or config.VAL_RATIO
    test_ratio = test_ratio or config.TEST_RATIO
    
    # Validate ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # Load and reconstruct data with label mapping
    images, masks, scene_ids, label_mapping, original_labels = load_and_reconstruct_data()
    
    # Create indices for splitting
    num_scenes = len(images)
    indices = np.arange(num_scenes)
    
    # Set random seed for reproducible splits
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(indices)
    
    # Ensure minimum dataset sizes for small datasets (following CRISM ML approach)
    if num_scenes < 10:
        print(f"⚠️  Small dataset detected ({num_scenes} scenes). Adjusting splits...")
        if num_scenes >= 3:
            # For very small datasets: at least 1 for each split
            train_end = max(1, num_scenes - 2)
            val_end = max(2, num_scenes - 1)
        else:
            # For tiny datasets: use simple split
            train_end = max(1, num_scenes // 2)
            val_end = max(train_end + 1, num_scenes - 1)
    else:
        # Normal splitting for larger datasets
        train_end = int(train_ratio * num_scenes)
        val_end = train_end + int(val_ratio * num_scenes)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    print(f"Data split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    
    # Create datasets for each split (images/masks are lists, not arrays)
    # Following list-based approach from https://github.com/Banus/crism_ml for variable-size CRISM images
    train_images = [images[i] for i in train_indices]
    train_masks = [masks[i] for i in train_indices]
    train_scenes = [scene_ids[i] for i in train_indices]
    
    val_images = [images[i] for i in val_indices]
    val_masks = [masks[i] for i in val_indices] 
    val_scenes = [scene_ids[i] for i in val_indices]
    
    test_images = [images[i] for i in test_indices]
    test_masks = [masks[i] for i in test_indices]
    test_scenes = [scene_ids[i] for i in test_indices]
    
    train_dataset = CRISMDataset(train_images, train_masks, train_scenes, normalize=config.NORMALIZE_DATA, target_size=target_size)
    val_dataset = CRISMDataset(val_images, val_masks, val_scenes, normalize=config.NORMALIZE_DATA, target_size=target_size)
    test_dataset = CRISMDataset(test_images, test_masks, test_scenes, normalize=config.NORMALIZE_DATA, target_size=target_size)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader, test_loader, label_mapping, original_labels


# Utility function for ground truth visualization
def visualize_ground_truth(save_dir: str = None):
    """
    Load and visualize ground truth masks for data verification.
    
    Args:
        save_dir: Directory to save visualizations (uses config.OUTPUT_DIR if None)
    """
    import matplotlib.pyplot as plt
    
    save_dir = save_dir or config.OUTPUT_DIR
    
    print("Loading data for ground truth visualization...")
    images, masks, scene_ids = load_and_reconstruct_data()
    
    # Visualize first few scenes
    num_to_show = min(5, len(images))
    
    for i in range(num_to_show):
        plt.figure(figsize=config.FIGURE_SIZE)
        plt.imshow(masks[i], cmap=config.COLORMAP)
        plt.title(f'Ground Truth - Scene {scene_ids[i]}')
        plt.colorbar(label='Mineral Class')
        
        save_path = os.path.join(save_dir, f'ground_truth_scene_{i}.png')
        plt.savefig(save_path, dpi=config.DPI, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ground truth visualization: {save_path}")
    
    print(f"Ground truth visualization complete. Files saved to {save_dir}")
