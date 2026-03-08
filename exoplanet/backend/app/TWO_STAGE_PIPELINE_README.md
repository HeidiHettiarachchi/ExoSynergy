# Two-Stage Mineral Identification Pipeline

## Overview

This implementation adds a **two-stage mineral identification pipeline** to the CRISM mineral classification system:

1. **Stage 1: CLIP Gatekeeper** - Fast filtering to reject non-planetary images
2. **Stage 2: CNN Mineral Classifier** - Expensive U-Net segmentation only on validated images

This design prevents wasting compute resources on irrelevant images (photos, screenshots, artwork, etc.).

## Installation

Install the required dependencies:

```bash
pip install torch torchvision open-clip-torch Pillow
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from mineral_identification_pipeline import create_pipeline

# Create the pipeline with default threshold (0.65)
pipeline = create_pipeline(threshold=0.65)

# Run prediction on an image
result = pipeline.predict("mars_image.png")

if result['stage'] == 'classified':
    print(f"Dominant mineral: {result['dominant_mineral']}")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print(f"Image rejected: {result['gatekeeper_reason']}")
```

### Running from Command Line

Test the gatekeeper alone:

```bash
python planetary_gatekeeper.py test_image.png 0.65
```

Test the full pipeline:

```bash
python mineral_identification_pipeline.py mars_surface.png 0.65
```

## File Structure

```
backend/app/
├── planetary_gatekeeper.py           # Stage 1: CLIP-based filtering
├── mineral_identification_pipeline.py # Stage 2: Full two-stage pipeline
├── threshold_tuning.py                # Threshold optimization tool
├── inference_script.py                # Existing mineral classifier
└── api_server.py                      # REST API (can be integrated)
```

## Key Components

### 1. Planetary Gatekeeper (`planetary_gatekeeper.py`)

CLIP-based classifier that checks if an image is planetary/geological:

```python
from planetary_gatekeeper import load_gatekeeper

# Load once at startup (not in loop!)
gatekeeper = load_gatekeeper(threshold=0.65)

# Check an image
result = gatekeeper.check_image("image.png")
print(result["accepted"])      # True/False
print(result["confidence"])    # 0.0-1.0
print(result["reason"])        # Explanation
```

**Key Features:**

- Uses `open-clip-torch` (not legacy OpenAI CLIP)
- Always converts images to RGB with `.convert("RGB")`
- Returns rejection reason for debugging
- Loaded once at startup for efficiency

### 2. Two-Stage Pipeline (`mineral_identification_pipeline.py`)

Combines gatekeeper + mineral classifier:

```python
from mineral_identification_pipeline import MineralIdentificationPipeline
from inference_script import load_trained_model

# Load your mineral model
model = load_trained_model()

# Create pipeline
pipeline = MineralIdentificationPipeline(
    mineral_model=model,
    threshold=0.65
)

# Predict
result = pipeline.predict("test.png")
```

**Pipeline Logic:**

1. Image → CLIP Gatekeeper
2. If rejected → Return early (no expensive CNN)
3. If accepted → Run mineral classification
4. Return comprehensive results

### 3. Threshold Tuning (`threshold_tuning.py`)

Find the optimal threshold for your use case:

```bash
python threshold_tuning.py \
    --planetary_dir ./data/mars_images \
    --non_planetary_dir ./data/earth_images \
    --min_threshold 0.5 \
    --max_threshold 0.9 \
    --step 0.05
```

**Output:**

- Precision, Recall, F1, Accuracy for each threshold
- Plot visualization (`threshold_tuning_results.png`)
- Detailed CSV results (`threshold_tuning_detailed.csv`)
- Recommended optimal threshold

## Advanced Usage

### Custom Prediction Function

If you have a custom mineral prediction function:

```python
def my_mineral_predict(image, model):
    """
    Custom prediction logic.
    Must accept (image, model) and return dict with 'prediction' and 'confidence'.
    """
    # Your preprocessing
    preprocessed = your_preprocess(image)

    # Run model
    output = model(preprocessed)

    # Return results
    return {
        "prediction": predicted_class_map,
        "confidence": mean_confidence,
        # ... other fields
    }

# Use custom function
pipeline = MineralIdentificationPipeline(
    mineral_model=model,
    mineral_predict_fn=my_mineral_predict,
    threshold=0.65
)
```

### Batch Processing

Process multiple images efficiently:

```python
images = ["image1.png", "image2.png", "image3.png"]
results = pipeline.batch_predict(images)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['stage']}")
```

### Skip Gatekeeper (Testing)

To test mineral classification without gatekeeper:

```python
result = pipeline.predict("image.png", skip_gatekeeper=True)
```

### Update Threshold Dynamically

```python
pipeline.update_threshold(0.75)
```

## Integration with API Server

To integrate with the FastAPI server (`api_server.py`):

```python
# At the top of api_server.py
from mineral_identification_pipeline import create_pipeline

# In initialize_model()
global MODEL, PIPELINE
MODEL = load_trained_model()
PIPELINE = create_pipeline(threshold=0.65)

# In the /predict endpoint
result = PIPELINE.predict(image_bytes)
if result['stage'] == 'rejected':
    raise HTTPException(
        status_code=400,
        detail=f"Image rejected: {result['gatekeeper_reason']}"
    )
# Continue with result['prediction']
```

## Common Pitfalls Avoided

✅ **Uses `open_clip` (not old OpenAI `clip` package)**

- Correct: `import open_clip`
- Wrong: `import clip`

✅ **Always converts images to RGB**

```python
image = Image.open(path).convert("RGB")  # ✓ Always include this
```

✅ **Loads gatekeeper once at startup**

```python
# ✓ Correct: Load once
gatekeeper = load_gatekeeper()
for image in images:
    gatekeeper.check_image(image)

# ✗ Wrong: Loading in loop
for image in images:
    gatekeeper = load_gatekeeper()  # Wasteful!
    gatekeeper.check_image(image)
```

✅ **Returns rejection reason for debugging**

```python
result = gatekeeper.check_image(img)
if not result["accepted"]:
    print(result["reason"])  # Know why it was rejected
```

✅ **Threshold passed through, not hardcoded**

```python
# ✓ Correct
pipeline = MineralIdentificationPipeline(threshold=0.65)
pipeline.update_threshold(0.75)

# ✗ Wrong: Hardcoding threshold in multiple places
if score > 0.65:  # Don't do this!
```

## Threshold Recommendations

Based on use case:

- **High Precision (0.75-0.85)**: Few false positives, reject more images
- **Balanced (0.65-0.75)**: Good precision/recall tradeoff (recommended)
- **High Recall (0.50-0.65)**: Accept more images, some false positives

Use `threshold_tuning.py` to find optimal value for your data.

## Performance Tips

1. **Load models once at startup**: Don't reload in prediction loops
2. **Batch processing**: Use `batch_predict()` for multiple images
3. **Skip gatekeeper if certain**: Set `skip_gatekeeper=True` for known planetary images
4. **GPU acceleration**: Ensure PyTorch is using CUDA/MPS if available

## Troubleshooting

### "ImportError: No module named 'clip'"

- You're using the old OpenAI CLIP package
- Solution: `pip install open-clip-torch` and use `import open_clip`

### Images always rejected

- Threshold too high
- Run threshold tuning to find optimal value
- Check image format (ensure RGB conversion)

### Slow performance

- Make sure you're loading gatekeeper once, not in loop
- Check if GPU acceleration is enabled
- Consider higher threshold to reject more images early

### Wrong predictions

- Gatekeeper might be accepting non-planetary images
- Increase threshold or retrain gatekeeper prompts
- Run threshold tuning with your specific data

## License

Same as parent project.

## References

- CLIP Paper: https://arxiv.org/abs/2103.00020
- OpenCLIP: https://github.com/mlfoundations/open_clip
- U-Net for Segmentation: https://arxiv.org/abs/1505.04597
