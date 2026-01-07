# CRISM Mineral Classification Backend

Backend API server for AI-powered CRISM hyperspectral mineral analysis using U-Net deep learning model.

## Features

- FastAPI-based REST API
- U-Net model for semantic segmentation
- Identifies 38 different mineral classes
- Returns annotated images with bounding boxes
- Provides detailed statistics and confidence metrics

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

## Installation

1. **Navigate to the backend directory:**

   ```bash
   cd backend/app
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements_api.txt
   ```

## Running the Server

### Start the API Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Verify Server is Running

Open your browser and navigate to:

- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

## API Endpoints

### Health Check

```
GET /health
```

Returns server status and model information.

### Predict

```
POST /predict
```

Upload an image for mineral classification.

**Parameters:**

- `image` (file): Image file to analyze
- `min_area` (int, optional): Minimum area for detection (default: 50)
- `return_image` (bool, optional): Return annotated images (default: true)

**Response:**

```json
{
  "detections": [...],
  "statistics": {
    "total_minerals": 5,
    "total_area": 12345,
    "mineral_distribution": {...}
  },
  "annotated_image": "base64_encoded_image",
  "segmentation_map": "base64_encoded_image"
}
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # Entry point
│   ├── api_server.py        # FastAPI server
│   ├── inference_script.py  # Inference logic
│   ├── requirements.txt     # ML dependencies
│   ├── requirements_api.txt # API dependencies
│   ├── crism_ml/           # ML model package
│   ├── src/                # Source code
│   └── saved_models/       # Model weights (.pth files)
└── README.md
```

## Model Information

- **Architecture:** U-Net
- **Classes:** 38 mineral types
- **Input:** CRISM hyperspectral or mineral sample images
- **Output:** Semantic segmentation with bounding boxes

## Troubleshooting

### Model not found error

Make sure the model file exists at:

```
backend/app/saved_models/best_unet_model.pth
```

### Port already in use

If port 8000 is already in use, modify the port in `api_server.py`:

```python
uvicorn.run(app, host="0.0.0.0", port=8001)
```

### Import errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
pip install -r requirements_api.txt
```

## Development

### Run with auto-reload (for development)

```bash
uvicorn api_server:app --reload --port 8000
```

### Run tests

```bash
python test_inference.py
```

## License

Part of the ExoSynergy AI project.
