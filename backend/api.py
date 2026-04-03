"""
FastAPI backend — exposes /predict endpoint.

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.models import PropertyInput, PredictionResponse
from src.inference import predict, _load_artifacts, PLANNING_AREAS, REGIONS, FLOOR_LEVELS, SALE_TYPES


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload model artifacts into memory on startup
    print("Loading model artifacts...")
    _load_artifacts()
    print("Model artifacts loaded. Ready to serve.")
    yield


app = FastAPI(
    title="Factory Price Predictor API",
    description="Predicts Singapore factory unit price ($ psf) using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow Streamlit frontend (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/options")
def options():
    """Return valid categorical options for the frontend dropdowns."""
    return {
        "planning_areas": PLANNING_AREAS,
        "regions": REGIONS,
        "floor_levels": FLOOR_LEVELS,
        "sale_types": SALE_TYPES,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_price(data: PropertyInput):
    try:
        result = predict(**data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
