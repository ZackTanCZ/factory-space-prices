"""
FastAPI backend — exposes /predict, /health, /options, /constraints endpoints.

Run with:
    uvicorn backend.api:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.dependencies import get_orchestrator, initialize_services
from src.models.prediction import PredictionResponse, PropertyInput
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


@app.get("/version")
def version():
    """Return the git SHA baked into the image at build time."""
    return {"git_sha": os.environ.get("GIT_SHA", "unknown")}


@app.get("/health/live")
def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
def readiness(orchestrator: InferenceOrchestrator = Depends(get_orchestrator)):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/options")
def options(orchestrator: InferenceOrchestrator = Depends(get_orchestrator)):
    """Return valid categorical options for the frontend dropdowns."""
    return orchestrator.get_feature_values()


@app.get("/constraints")
def constraints(orchestrator: InferenceOrchestrator = Depends(get_orchestrator)):
    """Return validation constraints for the frontend input bounds."""
    return orchestrator.get_constraints()


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
        raise HTTPException(status_code=500, detail=f"Prediction failed. Please check your inputs - {str(e)}")
