import os
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from spectrumAnalysis.services.preprocess import preprocess_spectrum
from datetime import datetime

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# RAW_DIR = os.path.join(BASE_DIR, "data", "raw_data")
# PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")

# os.makedirs(RAW_DIR, exist_ok=True)
# os.makedirs(PROCESSED_DIR, exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw_data")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed_data")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

@app.post("/preprocess")
async def preprocess_file(file: UploadFile = File(...), data_type: str = "direct"):
    try:
        raw_path = os.path.join(RAW_DIR, file.filename)
        content = await file.read()

        with open(raw_path, "wb") as f:
            f.write(content)

        raw_df = pd.read_csv(raw_path)
        raw_row_count = len(raw_df)

        processed_df = preprocess_spectrum(raw_path, data_type=data_type)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"preprocessed_{timestamp}_{file.filename}"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)
        processed_df.to_csv(processed_path, index=False)

        return {
            "data": processed_df.to_dict(orient="records"),
            "processed_path": processed_path,
            "row_count": raw_row_count,   
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")