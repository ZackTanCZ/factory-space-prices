"""
Shared pytest fixtures for unit tests.

Fixtures create real (but minimal) model artifacts so tests exercise
the actual code paths rather than mocked interfaces.
"""

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from src.services.orchestrator import InferenceOrchestrator

# ---------------------------------------------------------------------------
# Minimal data constants — enough to fit encoders and train a model
# ---------------------------------------------------------------------------

PLANNING_AREAS = ["Ang Mo Kio", "Bedok", "Tuas"]
REGIONS = ["Central Region", "East Region", "North Region", "North-East Region", "West Region"]
FLOOR_LEVELS = ["First Floor", "Non-First Floor", "Unknown"]
SALE_TYPES = ["New Sale", "Resale"]
FEATURE_COLS = ["Log_Area", "Remaining_Lease_Years", "Lease_Remaining_Ratio", "dist_to_mrt_m", "Planning_Area_Encoded",
                "Region_East Region", "Region_North Region", "Region_North-East Region", "Region_West Region",
                "Floor Level_Non-First Floor", "Floor Level_Unknown", "Type of Sale_Resale"]


# ---------------------------------------------------------------------------
# Encoder fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_target_encoder() -> dict:
    """Real target encoder dict matching the structure expected by preprocess()."""
    mapping = {area: float(i * 100) for i, area in enumerate(PLANNING_AREAS)}
    global_mean = float(np.mean(list(mapping.values())))
    return {"map": mapping, "global_mean": global_mean}


@pytest.fixture(scope="session")
def dummy_ohe() -> OneHotEncoder:
    """Real OneHotEncoder fitted on minimal categorical data."""
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_df = pd.DataFrame(
        [[r, f, s] for r in REGIONS for f in FLOOR_LEVELS for s in SALE_TYPES],
        columns=["Region", "Floor Level", "Type of Sale"],
    )
    ohe.fit(cat_df)
    return ohe


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_model(dummy_target_encoder, dummy_ohe) -> XGBRegressor:
    """Real XGBRegressor trained on synthetic data with correct feature columns."""
    rng = np.random.default_rng(42)
    n = 30

    rows = []
    for _ in range(n):
        area_sqft = rng.uniform(500, 5000)
        remaining_lease_years = rng.uniform(10, 90)
        lease_duration = 99.0
        planning_area = rng.choice(PLANNING_AREAS)
        region = rng.choice(REGIONS)
        floor_level = rng.choice(FLOOR_LEVELS)
        type_of_sale = rng.choice(SALE_TYPES)
        dist_to_mrt_m = rng.uniform(100, 5000)

        log_area = np.log(area_sqft)
        lease_ratio = remaining_lease_years / lease_duration
        pa_encoded = float(dummy_target_encoder["map"].get(planning_area, dummy_target_encoder["global_mean"]))

        cat_df = pd.DataFrame(
            [[region, floor_level, type_of_sale]],
            columns=["Region", "Floor Level", "Type of Sale"],
        )
        ohe_arr = dummy_ohe.transform(cat_df)
        ohe_cols = dummy_ohe.get_feature_names_out(["Region", "Floor Level", "Type of Sale"])
        ohe_vals = dict(zip(ohe_cols, ohe_arr[0]))

        row = {
            "Log_Area": log_area,
            "Remaining_Lease_Years": remaining_lease_years,
            "Lease_Remaining_Ratio": lease_ratio,
            "dist_to_mrt_m": dist_to_mrt_m,
            "Planning_Area_Encoded": pa_encoded,
            **ohe_vals,
        }
        rows.append(row)

    X = pd.DataFrame(rows)[FEATURE_COLS]
    y = rng.uniform(200, 800, size=n)

    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_cfg():
    """OmegaConf config matching the structure the orchestrator expects."""
    return OmegaConf.create({
        "constraints": {
            "area_sqft": {"min": 100, "max": 25000},
            "remaining_lease_years": {"min": 1, "max": 99},
            "dist_to_mrt_m": {"min": 0, "max": 10000},
            "lease_duration": {"valid_values": [30, 60, 99]},
        },
        "feature_values": {
            "planning_areas": PLANNING_AREAS,
            "regions": REGIONS,
            "floor_levels": FLOOR_LEVELS,
            "sale_types": SALE_TYPES,
        },
        "feature_cols":{
            "feature_cols": FEATURE_COLS
        },
        "settings": {
            "model_path": "models/xgboost_reduced.pkl",
            "target_encoder_path": "models/target_encoder.pkl",
            "onehot_encoder_path": "models/onehot_encoder.pkl",
            "model_rmse": 44.70,
        },
    })


# ---------------------------------------------------------------------------
# Orchestrator fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def dummy_orchestrator(dummy_cfg, dummy_model, dummy_target_encoder, dummy_ohe) -> InferenceOrchestrator:
    """InferenceOrchestrator with real artifacts injected — bypasses joblib.load()."""
    orchestrator = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orchestrator.cfg = dummy_cfg
    orchestrator.constraints = dummy_cfg.constraints
    orchestrator.feature_values = dummy_cfg.feature_values
    orchestrator.api_settings = dummy_cfg.settings
    orchestrator.feature_cols = dummy_cfg.feature_cols.feature_cols
    orchestrator.model = dummy_model
    orchestrator.target_encoder = dummy_target_encoder
    orchestrator.ohe = dummy_ohe
    return orchestrator
