"""
Configuration module for CRISM hyperspectral mineral segmentation.
Centralizes all hyperparameters and system settings to prevent magic numbers
and facilitate easy tuning.
"""

import os
import torch


# ===== PATH MANAGEMENT =====
# Construct absolute paths for cross-platform compatibility
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Input data files (now in data/raw/)
BLAND_DATA_PATH = os.path.join(RAW_DATA_DIR, "CRISM_bland_unratioed.mat")
LABELED_DATA_PATH = os.path.join(RAW_DATA_DIR, "CRISM_labeled_pixels_ratioed.mat")

# Processed data directories
RGB_COMPOSITES_DIR = os.path.join(PROCESSED_DATA_DIR, "rgb_composites")
GROUND_TRUTH_DIR = os.path.join(PROCESSED_DATA_DIR, "ground_truth")
SPECTRAL_ANALYSIS_DIR = os.path.join(PROCESSED_DATA_DIR, "spectral_analysis")

# Model save path
MODEL_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "best_unet_model.pth")

# ===== DEVICE MANAGEMENT =====
# Automatically detect best available device for acceleration
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using device: {DEVICE} (CUDA)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"Using device: {DEVICE} (Apple Metal Performance Shaders)")
else:
    DEVICE = torch.device("cpu")
    print(f"Using device: {DEVICE} (CPU fallback)")

print(f"GPU acceleration: {'Yes' if DEVICE.type != 'cpu' else 'No'}")

# ===== MODEL HYPERPARAMETERS =====
# Learning configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50

# Early stopping and model selection
PATIENCE = 10  # Number of epochs to wait for improvement
MIN_DELTA = 1e-4  # Minimum change to qualify as improvement

# ===== DATA CONFIGURATION =====
# Dataset structure parameters
NUM_BANDS = 350  # Number of spectral bands in CRISM data (from dataset documentation)
NUM_CLASSES = 10  # Will be dynamically set based on actual data classes

# Data splitting ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Data preprocessing
NORMALIZE_DATA = True
BACKGROUND_CLASS = 0  # Class label for background/unknown pixels

# ===== MODEL ARCHITECTURE =====
# U-Net configuration
INITIAL_FEATURES = 64  # Number of features in first layer
DROPOUT_RATE = 0.1

# ===== TRAINING CONFIGURATION =====
# Optimization settings
WEIGHT_DECAY = 1e-5
BETA1 = 0.9  # Adam optimizer parameter
BETA2 = 0.999  # Adam optimizer parameter

# Loss function weights (for class imbalance if needed)
CLASS_WEIGHTS = None  # Will be computed dynamically if needed

# ===== VISUALIZATION SETTINGS =====
# Color map for visualization
COLORMAP = "tab10"
FIGURE_SIZE = (15, 5)
DPI = 300

# Random seed for reproducibility
RANDOM_SEED = 42

# ===== VALIDATION =====
# Ensure all directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(RGB_COMPOSITES_DIR, exist_ok=True)
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(SPECTRAL_ANALYSIS_DIR, exist_ok=True)

# Validate data files exist (only warnings for API mode)
if not os.path.exists(BLAND_DATA_PATH):
    print(f"⚠️  Warning: Bland data file not found: {BLAND_DATA_PATH}")
if not os.path.exists(LABELED_DATA_PATH):
    print(f"⚠️  Warning: Labeled data file not found: {LABELED_DATA_PATH}")

print(f"Configuration loaded successfully!")
print(f"Project root: {PROJECT_ROOT}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print(f"Model save path: {MODEL_SAVE_PATH}")
