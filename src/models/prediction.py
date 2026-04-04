"""
Pydantic models — request and response schemas for the API.

Validation of business rules (bounds, cross-field checks) is handled
by InferenceOrchestrator, not here. This file is types only.
"""

from pydantic import BaseModel


class PropertyInput(BaseModel):
    area_sqft: float
    remaining_lease_years: float
    lease_duration: float
    planning_area: str
    floor_level: str
    type_of_sale: str
    region: str
    dist_to_mrt_m: float


class PredictionResponse(BaseModel):
    predicted_psf: float
    total_price: float
    lower_bound: float
    upper_bound: float
    rmse: float
