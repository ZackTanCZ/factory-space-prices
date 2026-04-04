"""
Inference pipeline — pure preprocessing and prediction functions.

All state (model artifacts, encoders) is owned by InferenceOrchestrator
and passed explicitly to these functions.

Preprocessing steps:
  1. Log-transform Area (sqft) → Log_Area
  2. Compute Lease_Remaining_Ratio from Remaining_Lease_Years / Lease_Duration
  3. Target-encode Planning Area
  4. One-hot encode Region, Floor Level, Type of Sale
  5. Assemble feature vector matching training column order
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Expected feature columns in the exact order the model was trained on
FEATURE_COLS = [
    "Log_Area",
    "Remaining_Lease_Years",
    "Lease_Remaining_Ratio",
    "dist_to_mrt_m",
    "Planning_Area_Encoded",
    "Region_East Region",
    "Region_North Region",
    "Region_North-East Region",
    "Region_West Region",
    "Floor Level_Non-First Floor",
    "Floor Level_Unknown",
    "Type of Sale_Resale",
]


def preprocess(
    area_sqft: float,
    remaining_lease_years: float,
    lease_duration: float,
    planning_area: str,
    floor_level: str,
    type_of_sale: str,
    region: str,
    dist_to_mrt_m: float,
    target_encoder: dict,
    ohe,
) -> pd.DataFrame:
    # 1. Log-transform area
    log_area = np.log(area_sqft)

    # 2. Lease remaining ratio
    lease_remaining_ratio = remaining_lease_years / lease_duration

    # 3. Target-encode planning area (fallback to global mean for unseen areas)
    if planning_area not in target_encoder["map"]:
        logger.warning("Unknown planning area '%s' — falling back to global mean", planning_area)
    planning_area_encoded = float(
        target_encoder["map"].get(planning_area, target_encoder["global_mean"])
    )

    # 4. One-hot encode categorical features
    cat_df = pd.DataFrame(
        [[region, floor_level, type_of_sale]],
        columns=["Region", "Floor Level", "Type of Sale"],
    )
    ohe_array = ohe.transform(cat_df)
    ohe_df = pd.DataFrame(
        ohe_array,
        columns=ohe.get_feature_names_out(["Region", "Floor Level", "Type of Sale"]),
    )

    # 5. Assemble feature dict
    feature_dict = {
        "Log_Area": log_area,
        "Remaining_Lease_Years": remaining_lease_years,
        "Lease_Remaining_Ratio": lease_remaining_ratio,
        "dist_to_mrt_m": dist_to_mrt_m,
        "Planning_Area_Encoded": planning_area_encoded,
    }
    for col in ohe_df.columns:
        feature_dict[col] = float(ohe_df[col].values[0])

    # 6. Build DataFrame with exact column order matching training
    return pd.DataFrame([feature_dict])[FEATURE_COLS]


def predict_with(model, X: pd.DataFrame, area_sqft: float, model_rmse: float) -> dict:
    predicted_psf = float(model.predict(X)[0])
    total_price = predicted_psf * area_sqft

    return {
        "predicted_psf": round(predicted_psf, 2),
        "total_price": round(total_price, 2),
        "lower_bound": round(predicted_psf - model_rmse, 2),
        "upper_bound": round(predicted_psf + model_rmse, 2),
        "rmse": model_rmse,
    }
