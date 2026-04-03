"""
Pydantic models — request and response schemas for the API.
"""

from pydantic import BaseModel, field_validator
from src.inference import PLANNING_AREAS, REGIONS, FLOOR_LEVELS, SALE_TYPES


class PropertyInput(BaseModel):
    area_sqft: float
    remaining_lease_years: float
    lease_duration: float
    planning_area: str
    floor_level: str
    type_of_sale: str
    region: str
    dist_to_mrt_m: float

    @field_validator("area_sqft")
    @classmethod
    def area_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("area_sqft must be greater than 0")
        return v

    @field_validator("remaining_lease_years", "lease_duration")
    @classmethod
    def lease_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Lease values must be greater than 0")
        return v

    @field_validator("dist_to_mrt_m")
    @classmethod
    def distance_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("dist_to_mrt_m must be >= 0")
        return v

    @field_validator("planning_area")
    @classmethod
    def valid_planning_area(cls, v):
        if v not in PLANNING_AREAS:
            raise ValueError(f"Unknown planning area: {v}")
        return v

    @field_validator("region")
    @classmethod
    def valid_region(cls, v):
        if v not in REGIONS:
            raise ValueError(f"Unknown region: {v}")
        return v

    @field_validator("floor_level")
    @classmethod
    def valid_floor_level(cls, v):
        if v not in FLOOR_LEVELS:
            raise ValueError(f"Unknown floor level: {v}")
        return v

    @field_validator("type_of_sale")
    @classmethod
    def valid_sale_type(cls, v):
        if v not in SALE_TYPES:
            raise ValueError(f"Unknown type of sale: {v}")
        return v


class PredictionResponse(BaseModel):
    predicted_psf: float
    total_price: float
    lower_bound: float
    upper_bound: float
    rmse: float
