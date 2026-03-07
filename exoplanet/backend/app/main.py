import os
import io
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Literal
from fastapi.middleware.cors import CORSMiddleware
from spectrumAnalysis.services.preprocess import preprocess_spectrum
from datetime import datetime
from app.rules.rule_engine import apply_physics_adjustments

from app.ml.inference import load_model, predict_major_gases
from app.ml.biosignature import analyze_planet                  # NEW

app = FastAPI()
MODEL_LOADED = False

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
async def preprocess_file(
    file: UploadFile = File(...),
    data_type: Literal["direct", "eclipse", "transmission"] = Form(...)
):
    try:
        raw_path = os.path.join(RAW_DIR, file.filename)
        content = await file.read()

        with open(raw_path, "wb") as f:
            f.write(content)

        # ensure the uploaded file is CSV
        try:
            raw_df = pd.read_csv(raw_path)
        except Exception as read_err:
            print(f"[preprocess] failed to read CSV {raw_path}: {read_err}")
            raise

        raw_row_count = len(raw_df)

        processed_df = preprocess_spectrum(raw_path, data_type=data_type)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_filename = f"preprocessed_{timestamp}_{file.filename}"
        processed_path = os.path.join(PROCESSED_DIR, processed_filename)

        processed_df.to_csv(processed_path, index=False)

        major = predict_major_gases(processed_df)

        spectral_features = processed_df.iloc[0].to_dict()
        astro_features = raw_df.iloc[0].to_dict()

        gas_profile = build_full_atmosphere_profile(
            major=major,
            spectral_features=spectral_features,
            astro_features=astro_features,
            data_type=data_type
        )

        # -----------------------------
        # Biosignature + Habitability
        # -----------------------------
        try:

            bio_result = analyze_planet(
                gas_predictions=gas_profile
            )

            habitability = {
                "score": bio_result.score,
                "grade": bio_result.grade,
                "category": bio_result.category,
                "summary": bio_result.summary,
                "factor_scores": bio_result.factor_scores,

                "biosignatures": [
                    {
                        "name": b.name,
                        "detected": b.detected,
                        "reason": b.reason,
                        "gases_involved": b.gases_involved,
                    }
                    for b in bio_result.biosignatures
                ],

                "profile": {
                    "planet_type": bio_result.profile.planet_type,
                    "dominant_gas_fingerprint": bio_result.profile.dominant_gas_fingerprint,
                    "greenhouse_intensity": bio_result.profile.greenhouse_intensity_label,

                    # NEW METRICS
                    "greenhouse_heating_index": bio_result.profile.greenhouse_heating_index,
                    "atmospheric_density": bio_result.profile.atmospheric_density,
                    "thermal_stability": bio_result.profile.thermal_stability,
                    "temperature_potential": bio_result.profile.temperature_potential,

                    "toxicity_index": bio_result.profile.toxicity_index,
                    "toxicity_label": bio_result.profile.toxicity_label,

                    "similar_atmospheres": [
                        {
                            "planet": s.planet,
                            "similarity": s.similarity
                        }
                        for s in bio_result.profile.atmosphere_similarity
                    ]
                }
            }

        except Exception as e:

            habitability = {
                "error": f"Atmospheric analysis failed: {str(e)}"
            }

        return {
            "data": processed_df.to_dict(orient="records"),
            "processed_path": processed_path,
            "row_count": raw_row_count,
            "analysis": {
                "gas_profile": gas_profile,
                "habitability": habitability
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")
    except Exception as e:
        # log the traceback for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")


@app.on_event("startup")
def startup():
    global MODEL_LOADED
    input_dim = 208
    load_model(input_dim)
    MODEL_LOADED = True
    print("Gas prediction model loaded")


def infer_planet_type(radius):
    """Return a simple planet class based on radius (in Jupiter radii)."""
    try:
        radius = float(radius)
    except Exception:
        radius = 1.0

    if radius >= 0.5:
        return "gas_giant"
    elif radius > 0.1:
        return "sub_neptune"
    else:
        return "rocky"


def base_atmosphere(planet):
    """Provide a very basic base atmospheric composition for a planet type."""
    if planet == "gas_giant":
        return {"H2": 90.0, "He": 10.0}
    elif planet == "sub_neptune":
        return {"H2O": 50.0, "H2": 30.0, "He": 20.0}
    else:  # rocky
        return {"N2": 78.0, "O2": 21.0, "CO2": 0.04}


def hybrid_merge(base, adjusted):
    """Merge two gas profiles by summing values, favoring adjusted data."""
    merged = base.copy()
    for k, v in (adjusted or {}).items():
        merged[k] = merged.get(k, 0.0) + v
    return merged


def physical_normalize(profile):
    """Normalize a gas profile so that percentages sum to 100%."""
    total = sum(profile.values())
    if total > 0:
        return {k: v / total * 100.0 for k, v in profile.items()}
    return profile


def build_full_atmosphere_profile(major, spectral_features, astro_features, data_type):
    """Combine prediction with a base atmosphere and physical adjustments."""
    try:
        radius = float(
            astro_features.get("PL_RADJ") or astro_features.get("PL_RAD") or astro_features.get("radius") or 1.0
        )
    except Exception:
        radius = 1.0

    planet = infer_planet_type(radius)

    try:
        adjusted_major = apply_physics_adjustments(major.copy(), spectral_features, astro_features, data_type)
    except Exception:
        adjusted_major = major

    base = base_atmosphere(planet)
    merged = hybrid_merge(base, adjusted_major)
    normalized = physical_normalize(merged)
    return normalized