# CRISM Hyperspectral Mineral Segmentation Pipeline
# Makefile for running the complete machine learning pipeline

# Configuration
PYTHON = python
VENV = .venv
VENV_PYTHON = $(VENV)/bin/python
VENV_PIP = $(VENV)/bin/pip
UV = uv

# Pipeline scripts
DATA_PIPELINE = pipelines/data_pipeline.py
MODEL_PIPELINE = pipelines/model_pipeline.py
INFERENCE_PIPELINE = pipelines/inference_pipeline.py

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[0;33m
BLUE = \033[0;34m
PURPLE = \033[0;35m
CYAN = \033[0;36m
NC = \033[0m # No Color

.PHONY: help setup install data train inference all clean test lint format check-gpu status notebooks

# Default target
help: ## Show this help message
	@echo "$(CYAN)CRISM Hyperspectral Mineral Segmentation Pipeline$(NC)"
	@echo "================================================================"
	@echo ""
	@echo "$(GREEN)Setup Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*$$/ {if ($$1 ~ /setup|install|clean/) printf "  $(YELLOW)%-12s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Pipeline Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*$$/ {if ($$1 ~ /process|data|train|inference|all/) printf "  $(YELLOW)%-12s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## .*$$/ {if ($$1 ~ /test|lint|format|check|status|notebooks/) printf "  $(YELLOW)%-12s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(PURPLE)Usage Examples:$(NC)"
	@echo "  make setup             # One-time environment setup"
	@echo "  make process           # Convert .mat files to viewable .png images"
	@echo "  make all-with-processing # Complete pipeline with image processing"
	@echo "  make all               # Standard pipeline (data → train → inference)"
	@echo "  make data              # Run data analysis only"
	@echo "  make train             # Train the U-Net model"
	@echo "  make inference         # Run model evaluation"
	@echo ""

# Setup and Installation
setup: ## Create virtual environment and install dependencies
	@echo "$(CYAN)🚀 Setting up CRISM project environment...$(NC)"
	@if command -v uv >/dev/null 2>&1; then \
		echo "$(GREEN)✓ Using uv for fast environment setup$(NC)"; \
		uv venv $(VENV); \
		. $(VENV)/bin/activate && uv pip install -r requirements.txt; \
	else \
		echo "$(YELLOW)⚠ uv not found, using standard python venv$(NC)"; \
		$(PYTHON) -m venv $(VENV); \
		. $(VENV)/bin/activate && pip install -r requirements.txt; \
	fi
	@echo "$(GREEN)✅ Environment setup complete!$(NC)"
	@echo "$(BLUE)Activate with: source $(VENV)/bin/activate$(NC)"

install: setup ## Alias for setup

check-venv: ## Check if virtual environment exists
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)❌ Virtual environment not found!$(NC)"; \
		echo "$(YELLOW)Run 'make setup' first$(NC)"; \
		exit 1; \
	fi

# Pipeline Commands
data: check-venv ## Run data analysis and verification pipeline  
	@echo "$(CYAN)📊 Running Data Analysis Pipeline...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) $(DATA_PIPELINE)
	@echo "$(GREEN)✅ Data pipeline completed$(NC)"
	@echo "$(BLUE)📁 Check output/ directory for visualizations$(NC)"

process: check-venv ## Process all .mat files to .png images in data/processed/
	@echo "$(CYAN)🖼️  Processing CRISM images with enhanced hyperspectral techniques...$(NC)"
	@echo "$(YELLOW)📂 Converting .mat files to properly enhanced images...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) pipelines/fixed_process_data.py
	@echo "$(GREEN)✅ Enhanced image processing completed$(NC)"
	@echo "$(BLUE)🌐 Open data/processed/index.html to browse enhanced gallery$(NC)"

train: check-venv ## Train the U-Net model
	@echo "$(CYAN)🧠 Running Model Training Pipeline...$(NC)"
	@echo "$(YELLOW)⏱ This may take 2-4 hours on Apple Silicon GPU$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) $(MODEL_PIPELINE)
	@echo "$(GREEN)✅ Training pipeline completed$(NC)"
	@echo "$(BLUE)💾 Model saved to saved_models/$(NC)"

inference: check-venv ## Run model evaluation and inference
	@echo "$(CYAN)🔬 Running Inference Pipeline...$(NC)"
	@if [ ! -f "saved_models/best_unet_model.pth" ]; then \
		echo "$(RED)❌ No trained model found!$(NC)"; \
		echo "$(YELLOW)Run 'make train' first$(NC)"; \
		exit 1; \
	fi
	@. $(VENV)/bin/activate && $(PYTHON) $(INFERENCE_PIPELINE)
	@echo "$(GREEN)✅ Inference pipeline completed$(NC)"
	@echo "$(BLUE)📊 Check output/ directory for evaluation results$(NC)"

all: check-venv ## Run complete pipeline: data → train → inference
	@echo "$(PURPLE)🚀 Running Complete CRISM Pipeline...$(NC)"
	@echo "$(CYAN)================================================$(NC)"
	@$(MAKE) data
	@echo ""
	@$(MAKE) train
	@echo ""
	@$(MAKE) inference
	@echo ""
	@echo "$(GREEN)🎉 Complete pipeline finished successfully!$(NC)"
	@echo "$(BLUE)📁 Results available in output/ and saved_models/$(NC)"

# Development and Utility Commands
status: check-venv ## Show project status and GPU information
	@echo "$(CYAN)CRISM Project Status$(NC)"
	@echo "===================="
	@echo ""
	@echo "$(GREEN)📁 Project Structure:$(NC)"
	@ls -la | grep -E "(data|src|pipelines|notebooks|output|saved_models)" || true
	@echo ""
	@echo "$(GREEN)🔧 Environment:$(NC)"
	@if [ -d "$(VENV)" ]; then \
		echo "  ✅ Virtual environment: $(VENV)"; \
	else \
		echo "  ❌ Virtual environment: Not found"; \
	fi
	@echo ""
	@echo "$(GREEN)💾 Data Files:$(NC)"
	@if [ -f "data/CRISM_labeled_pixels_ratioed.mat" ]; then \
		echo "  ✅ CRISM data: $(shell ls -lh data/CRISM_labeled_pixels_ratioed.mat | awk '{print $$5}')"; \
	else \
		echo "  ❌ CRISM data: Not found"; \
	fi
	@echo ""
	@echo "$(GREEN)🤖 Model Status:$(NC)"
	@if [ -f "saved_models/best_unet_model.pth" ]; then \
		echo "  ✅ Trained model: $(shell ls -lh saved_models/best_unet_model.pth | awk '{print $$5}')"; \
	else \
		echo "  ❌ Trained model: Not found (run 'make train')"; \
	fi
	@echo ""
	@. $(VENV)/bin/activate && $(PYTHON) -c "import torch; print('🖥 Hardware:'); print('  ✅ Apple Silicon GPU (MPS): Available' if torch.backends.mps.is_available() else '  ✅ NVIDIA GPU: Available' if torch.cuda.is_available() else '  ⚠️  CPU Only: GPU not available'); print(f'  📊 PyTorch version: {torch.__version__}')"

check-gpu: check-venv ## Check GPU availability and acceleration
	@echo "$(CYAN)🖥 GPU Acceleration Status$(NC)"
	@echo "=========================="
	@. $(VENV)/bin/activate && $(PYTHON) -c "import torch; print('PyTorch version:', torch.__version__); print(); print('✅ Apple Silicon (MPS): Available\\n   Device: mps\\n   Expected speedup: 3-5x vs CPU') if torch.backends.mps.is_available() else print('✅ NVIDIA CUDA: Available') if torch.cuda.is_available() else print('⚠️ GPU acceleration not available\\n  Training will use CPU (much slower)')"

notebooks: check-venv ## Launch Jupyter Lab with notebooks
	@echo "$(CYAN)📓 Starting Jupyter Lab...$(NC)"
	@echo "$(YELLOW)Available notebooks:$(NC)"
	@echo "  - notebooks/data_analysis.ipynb  (Data exploration)"
	@echo "  - notebooks/inference.ipynb      (Model testing)"
	@echo ""
	@. $(VENV)/bin/activate && jupyter lab --notebook-dir=notebooks

test: check-venv ## Run basic tests
	@echo "$(CYAN)🧪 Running basic tests...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -c "\
print('Testing imports...'); \
from src import config; \
from src.dataset import load_and_reconstruct_data; \
from src.model import create_model; \
from src.utils import get_class_names; \
print('✅ All imports successful'); \
print(f'Device: {config.DEVICE}'); \
print(f'Spectral bands: {config.NUM_BANDS}'); \
print(f'Classes: {config.NUM_CLASSES}'); \
"

lint: check-venv ## Run code linting with flake8
	@echo "$(CYAN)🔍 Running code linting...$(NC)"
	@. $(VENV)/bin/activate && flake8 src/ pipelines/ --max-line-length=100 --ignore=E501,W503 || true

format: check-venv ## Format code with black
	@echo "$(CYAN)🎨 Formatting code with black...$(NC)"
	@. $(VENV)/bin/activate && black src/ pipelines/ --line-length=100

# Cleanup
clean: ## Remove generated files and cache
	@echo "$(CYAN)🧹 Cleaning up...$(NC)"
	@rm -rf __pycache__ src/__pycache__ pipelines/__pycache__
	@rm -rf .pytest_cache .mypy_cache
	@rm -rf build dist *.egg-info
	@find . -name "*.pyc" -delete
	@find . -name "*.pyo" -delete
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-processed: ## Clean up redundant processed images (keep only best results)
	@echo "$(YELLOW)🧹 Cleaning redundant processed images...$(NC)"
	@rm -rf data/processed/rgb_composites data/processed/satellite_images data/processed/proper_satellite_images data/processed/crism_ml_rgb data/processed/crism_ml_direct data/processed/spectral_analysis 2>/dev/null || true
	@echo "$(GREEN)✅ Redundant images cleaned$(NC)"
	@echo "$(BLUE)📁 Kept: individual_crism_ml/ (best quality) and ground_truth/$(NC)"

clean-all: clean ## Remove all generated files including models and outputs
	@echo "$(YELLOW)⚠ This will delete trained models and outputs!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		rm -rf output/* saved_models/*.pth data/processed/*; \
		echo "$(GREEN)✅ All generated files removed$(NC)"; \
	else \
		echo ""; \
		echo "$(BLUE)Cancelled$(NC)"; \
	fi

clean-env: ## Remove virtual environment
	@echo "$(YELLOW)⚠ This will remove the virtual environment!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		rm -rf $(VENV); \
		echo "$(GREEN)✅ Virtual environment removed$(NC)"; \
		echo "$(BLUE)Run 'make setup' to recreate$(NC)"; \
	else \
		echo ""; \
		echo "$(BLUE)Cancelled$(NC)"; \
	fi

# Quick development shortcuts
quick-test: check-venv ## Quick test with small data subset
	@echo "$(CYAN)⚡ Quick development test...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -c "\
from src.dataset import load_and_reconstruct_data; \
images, masks, scene_ids = load_and_reconstruct_data(); \
print(f'✅ Loaded {len(images)} images for testing'); \
[print(f'  Image {i}: {img.shape}') for i, img in enumerate(images)]; \
"

# Help for specific commands
data-help: ## Show data pipeline help
	@echo "$(CYAN)Data Pipeline (make data)$(NC)"
	@echo "========================="
	@echo "Loads CRISM hyperspectral data and performs analysis:"
	@echo "• Validates data integrity"
	@echo "• Generates ground truth visualizations" 
	@echo "• Creates statistical analysis plots"
	@echo "• Saves results to output/ directory"

process-help: ## Show image processing help
	@echo "$(CYAN)Image Processing Pipeline (make process)$(NC)"
	@echo "========================================"
	@echo "Converts .mat files to viewable .png images:"
	@echo "• Processes all CRISM scenes from data/raw/"
	@echo "• Creates enhanced RGB composites with outlier handling"
	@echo "• Generates mineral-specific false-color images"
	@echo "• Produces hydration detection composites"
	@echo "• Creates high-contrast ground truth visualizations"
	@echo "• Saves organized images to data/processed/"
	@echo "• Creates interactive HTML gallery"

satellite: check-venv ## Generate individual CRISM images using integrated CRISM ML methodology  
	@echo "$(CYAN)🛰️  Processing individual CRISM images with CRISM ML integration...$(NC)"
	@echo "$(YELLOW)📚 Enhanced individual image handling using proven methodology...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) -c "from src.crism_ml_integration import main; main()"
	@echo "$(GREEN)✅ Individual CRISM ML processing completed$(NC)"
	@echo "$(BLUE)🌐 Browse: data/processed/individual_crism_ml/index.html$(NC)"

overlays: check-venv satellite ## Create mineral overlay visualizations (RGB + classifications)
	@echo "$(CYAN)🎨 Creating mineral overlay visualizations...$(NC)"
	@echo "$(YELLOW)📸 Combining satellite imagery with mineral classifications...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) pipelines/create_overlay_visualizations.py
	@echo "$(GREEN)✅ Overlay visualizations completed$(NC)"
	@echo "$(BLUE)🌐 Browse: data/processed/mineral_overlays/index.html$(NC)"

inference-test: check-venv train ## Test the inference pipeline with sample images
	@echo "$(CYAN)🧪 Testing inference pipeline...$(NC)"
	@echo "$(YELLOW)📸 Creating test images and running inference...$(NC)"
	@. $(VENV)/bin/activate && $(PYTHON) test_inference.py
	@echo "$(GREEN)✅ Inference test completed$(NC)"
	@echo "$(BLUE)📁 Check: data/processed/test_inference/$(NC)"

train-help: ## Show training pipeline help
	@echo "$(CYAN)Training Pipeline (make train)$(NC)"
	@echo "=============================="
	@echo "Trains U-Net model for mineral segmentation:"
	@echo "• Loads and splits data (70/15/15)"
	@echo "• Initializes U-Net with 31M parameters"
	@echo "• Uses Apple Silicon MPS GPU acceleration"
	@echo "• Implements early stopping and checkpointing"
	@echo "• Saves best model to saved_models/"
	@echo "• Expected time: 2-4 hours on Apple Silicon"

inference-help: ## Show inference pipeline help
	@echo "$(CYAN)Inference Pipeline (make inference)$(NC)"
	@echo "================================="
	@echo "Evaluates trained model performance:"
	@echo "• Loads best model from training"
	@echo "• Runs evaluation on test dataset"
	@echo "• Generates confusion matrices"
	@echo "• Creates prediction visualizations"
	@echo "• Produces detailed evaluation report"
	@echo "• Saves all results to output/"
