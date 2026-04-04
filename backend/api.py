"""
FastAPI backend — exposes /predict, /health, /options, /constraints endpoints.

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.dependencies import initialize_services, get_orchestrator
from src.models.prediction import PropertyInput, PredictionResponse
from src.services.inference import PLANNING_AREAS, REGIONS, FLOOR_LEVELS, SALE_TYPES
from src.services.orchestrator import InferenceOrchestrator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_services()
    yield


app = FastAPI(
    title="Factory Price Predictor API",
    description="Predicts Singapore factory unit price ($ psf) using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

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


@app.get("/constraints")
def constraints(orchestrator: InferenceOrchestrator = Depends(get_orchestrator)):
    """Return validation constraints for the frontend input bounds."""
    return OmegaConf.to_container(orchestrator.constraints, resolve=True)


@app.post("/predict", response_model=PredictionResponse)
def predict_price(
    data: PropertyInput,
    orchestrator: InferenceOrchestrator = Depends(get_orchestrator),
):
    try:
        return orchestrator.predict(**data.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed. Please check your inputs.")
