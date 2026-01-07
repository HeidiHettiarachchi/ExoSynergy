"""
Logging module for CRISM hyperspectral mineral segmentation.
Provides structured logging throughout the pipeline with proper formatting.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import config


class CRISMLogger:
    """
    Custom logger for CRISM project with structured output and file logging.
    """
    
    def __init__(self, name: str, log_level: int = logging.INFO, 
                 log_to_file: bool = True, log_to_console: bool = True):
        """
        Initialize CRISM logger.
        
        Args:
            name: Logger name (usually __name__)
            log_level: Logging level
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Console handler with colors
        if log_to_console:
            # Use UTF-8 encoding for console output to support emojis on Windows
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            # Set UTF-8 encoding for the stream
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8')
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            log_file = os.path.join(config.OUTPUT_DIR, f'crism_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info(f"Logging to file: {log_file}")
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_dataset_info(self, images, masks, scene_ids, label_mapping=None):
        """Log comprehensive dataset information."""
        self.info("=" * 60)
        self.info("DATASET INFORMATION")
        self.info("=" * 60)
        
        # Basic statistics
        num_scenes = len(images)
        self.info(f"📊 Dataset Overview:")
        self.info(f"   Number of scenes: {num_scenes}")
        self.info(f"   Scene IDs: {scene_ids[:3]}..." if len(scene_ids) > 3 else f"   Scene IDs: {scene_ids}")
        
        if num_scenes > 0:
            # Image dimensions
            dimensions = [(img.shape[0], img.shape[1]) for img in images]
            sizes = [h * w for h, w in dimensions]
            
            self.info(f"🖼️  Image Statistics:")
            self.info(f"   Dimensions range: {dimensions[:3]}..." if len(dimensions) > 3 else f"   Dimensions: {dimensions}")
            self.info(f"   Smallest: {min(sizes):,} pixels ({min(dimensions)})")
            self.info(f"   Largest: {max(sizes):,} pixels ({max(dimensions)})")
            self.info(f"   Average: {sum(sizes)/len(sizes):,.0f} pixels")
            self.info(f"   Total pixels: {sum(sizes):,}")
            
            # Spectral information
            spectral_bands = images[0].shape[2]
            self.info(f"🌈 Spectral Information:")
            self.info(f"   Bands per pixel: {spectral_bands}")
            self.info(f"   Expected bands: {config.NUM_BANDS}")
            
            # Memory usage estimate
            memory_gb = sum(sizes) * spectral_bands * 4 / (1024**3)  # 4 bytes per float32
            self.info(f"   Estimated memory: {memory_gb:.2f} GB")
            
            # Class distribution
            all_labels = np.concatenate([mask.flatten() for mask in masks])
            unique_labels, counts = np.unique(all_labels, return_counts=True)
            
            self.info(f"🏷️  Class Information:")
            self.info(f"   Unique classes found: {len(unique_labels)}")
            self.info(f"   Class IDs: {unique_labels.tolist()}")
            
            # Show distribution for classes with significant representation
            self.info(f"   Class distribution:")
            for cls, count in zip(unique_labels, counts):
                percentage = (count / len(all_labels)) * 100
                if percentage > 0.1:  # Only show classes with >0.1% representation
                    self.info(f"     Class {cls}: {count:,} pixels ({percentage:.1f}%)")
            
            # Label mapping if provided
            if label_mapping:
                self.info(f"🔄 Label Mapping:")
                self.info(f"   Mapping strategy: Sparse → Continuous")
                self.info(f"   Original → Mapped: {dict(list(label_mapping.items())[:5])}..." if len(label_mapping) > 5 else f"   Mapping: {label_mapping}")
        
        self.info("=" * 60)
    
    def log_model_info(self, model, optimizer=None, criterion=None):
        """Log comprehensive model information."""
        self.info("=" * 60)
        self.info("MODEL INFORMATION")
        self.info("=" * 60)
        
        # Model architecture
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"🧠 Architecture:")
        self.info(f"   Model type: {model.__class__.__name__}")
        self.info(f"   Input channels: {model.n_channels}")
        self.info(f"   Output classes: {model.n_classes}")
        self.info(f"   Total parameters: {total_params:,}")
        self.info(f"   Trainable parameters: {trainable_params:,}")
        self.info(f"   Model size: {total_params * 4 / (1024**2):.1f} MB")
        
        # Device information
        device = next(model.parameters()).device
        self.info(f"💻 Hardware:")
        self.info(f"   Device: {device}")
        self.info(f"   Device type: {device.type}")
        
        # Training configuration
        if optimizer:
            self.info(f"🎛️  Training Configuration:")
            self.info(f"   Optimizer: {optimizer.__class__.__name__}")
            self.info(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
            if hasattr(optimizer, 'weight_decay'):
                self.info(f"   Weight decay: {optimizer.param_groups[0].get('weight_decay', 0)}")
        
        if criterion:
            self.info(f"   Loss function: {criterion.__class__.__name__}")
            if hasattr(criterion, 'weight') and criterion.weight is not None:
                self.info(f"   Class weighting: Enabled ({len(criterion.weight)} classes)")
        
        self.info("=" * 60)
    
    def log_training_epoch(self, epoch, total_epochs, train_loss, train_acc, 
                          val_loss, val_acc, lr, epoch_time, is_best=False):
        """Log training epoch information."""
        status = "🎉 BEST" if is_best else "    "
        self.info(f"Epoch {epoch+1:3d}/{total_epochs} | {status} | "
                 f"Train: L={train_loss:.4f} A={train_acc:.3f} | "
                 f"Val: L={val_loss:.4f} A={val_acc:.3f} | "
                 f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Get the color for this log level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format the record
        formatted = super().format(record)
        
        # Add color only to console output (not file)
        if hasattr(record, 'stream') and record.stream == sys.stdout:
            return f"{color}{formatted}{reset}"
        return formatted


# Global logger instance
_global_logger = None


def get_logger(name: Optional[str] = None) -> CRISMLogger:
    """
    Get global logger instance.
    
    Args:
        name: Logger name (optional, uses caller's module name if None)
        
    Returns:
        CRISMLogger instance
    """
    global _global_logger
    
    if _global_logger is None:
        logger_name = name or 'CRISM'
        _global_logger = CRISMLogger(logger_name)
    
    return _global_logger


# Convenience functions
def log_info(message: str):
    """Log info message using global logger."""
    get_logger().info(message)

def log_warning(message: str):
    """Log warning message using global logger.""" 
    get_logger().warning(message)

def log_error(message: str):
    """Log error message using global logger."""
    get_logger().error(message)

def log_debug(message: str):
    """Log debug message using global logger."""
    get_logger().debug(message)


# Import numpy for dataset logging
import numpy as np
