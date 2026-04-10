"""Unit tests for src/services/inference.py — pure functions only."""

import pytest

from src.services.inference import predict_with, preprocess
from tests.conftest import FEATURE_COLS

# ---------------------------------------------------------------------------
# preprocess()
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_input():
    return {
        "area_sqft": 1000.0,
        "remaining_lease_years": 50.0,
        "lease_duration": 99.0,
        "planning_area": "Ang Mo Kio",
        "floor_level": "Non-First Floor",
        "type_of_sale": "Resale",
        "region": "Central Region",
        "dist_to_mrt_m": 500.0,
    }


def test_preprocess_returns_correct_columns(sample_input, dummy_target_encoder, dummy_ohe):
    X = preprocess(**sample_input, target_encoder=dummy_target_encoder,
                   ohe=dummy_ohe, feature_cols=FEATURE_COLS)
    assert list(X.columns) == FEATURE_COLS


def test_preprocess_returns_single_row(sample_input, dummy_target_encoder, dummy_ohe):
    X = preprocess(**sample_input, target_encoder=dummy_target_encoder,
                   ohe=dummy_ohe, feature_cols=FEATURE_COLS)
    assert X.shape == (1, len(FEATURE_COLS))


def test_preprocess_unknown_planning_area_falls_back_to_global_mean(sample_input, dummy_target_encoder, dummy_ohe):
    sample_input["planning_area"] = "Unknown Area"
    X = preprocess(**sample_input, target_encoder=dummy_target_encoder,
                   ohe=dummy_ohe, feature_cols=FEATURE_COLS)
    assert X["Planning_Area_Encoded"].iloc[0] == dummy_target_encoder["global_mean"]


# ---------------------------------------------------------------------------
# predict_with()
# ---------------------------------------------------------------------------

def test_predict_with_returns_all_keys(dummy_model, dummy_target_encoder, dummy_ohe):
    X = preprocess(
        area_sqft=1000.0,
        remaining_lease_years=50.0,
        lease_duration=99.0,
        planning_area="Ang Mo Kio",
        floor_level="Non-First Floor",
        type_of_sale="Resale",
        region="Central Region",
        dist_to_mrt_m=500.0,
        target_encoder=dummy_target_encoder,
        ohe=dummy_ohe,
        feature_cols=FEATURE_COLS
    )
    result = predict_with(dummy_model, X, area_sqft=1000.0, model_rmse=44.70)
    assert set(result.keys()) == {"predicted_psf", "total_price", "lower_bound", "upper_bound", "rmse"}


def test_predict_with_total_price_correct(dummy_model, dummy_target_encoder, dummy_ohe):
    area_sqft = 1000.0
    X = preprocess(
        area_sqft=area_sqft,
        remaining_lease_years=50.0,
        lease_duration=99.0,
        planning_area="Ang Mo Kio",
        floor_level="Non-First Floor",
        type_of_sale="Resale",
        region="Central Region",
        dist_to_mrt_m=500.0,
        target_encoder=dummy_target_encoder,
        ohe=dummy_ohe,
        feature_cols=FEATURE_COLS
    )
    result = predict_with(dummy_model, X, area_sqft=area_sqft, model_rmse=44.70)
    assert abs(result["total_price"] - result["predicted_psf"] * area_sqft) < area_sqft * 0.005


def test_predict_with_bounds_correct(dummy_model, dummy_target_encoder, dummy_ohe):
    model_rmse = 44.70
    X = preprocess(
        area_sqft=1000.0,
        remaining_lease_years=50.0,
        lease_duration=99.0,
        planning_area="Ang Mo Kio",
        floor_level="Non-First Floor",
        type_of_sale="Resale",
        region="Central Region",
        dist_to_mrt_m=500.0,
        target_encoder=dummy_target_encoder,
        ohe=dummy_ohe,
        feature_cols=FEATURE_COLS
    )
    result = predict_with(dummy_model, X, area_sqft=1000.0, model_rmse=model_rmse)
    assert result["lower_bound"] == round(result["predicted_psf"] - model_rmse, 2)
    assert result["upper_bound"] == round(result["predicted_psf"] + model_rmse, 2)
