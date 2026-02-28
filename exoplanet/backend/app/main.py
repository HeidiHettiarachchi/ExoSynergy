import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from spectrumAnalysis.services.preprocess import preprocess_spectrum
app = FastAPI()

# Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Input + Preprocessing
@app.post("/preprocess")
async def preprocess_file(file: UploadFile = File(...)):

    # Save raw file
    temp_path = f"data/raw_data/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(file.file.read())

    # Preprocess spectrum data
    processed_df = preprocess_spectrum(temp_path)

    # Save processed file
    processed_path = f"data/processed_data/preprocessed_{file.filename}"
    processed_df.to_csv(processed_path, index=False)

    return {
        "data": processed_df.to_dict(orient="records"),
        "processed_path": processed_path
    }