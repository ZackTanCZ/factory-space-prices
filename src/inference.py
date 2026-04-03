"""
Inference pipeline — preprocessing + prediction.

Replicates the preprocessing steps from notebook 05:
  1. Log-transform Area (sqft) → Log_Area
  2. Compute Lease_Remaining_Ratio from Remaining_Lease_Years / Lease_Duration
     (Lease_Duration is an intermediate input — not passed to the model)
  3. Target-encode Planning Area (load from models/target_encoder.pkl)
  4. One-hot encode Region, Floor Level, Type of Sale (load from models/onehot_encoder.pkl)
  5. Assemble feature vector matching training column order
  6. Predict with xgboost_reduced.pkl (macro features excluded)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"

# Expected feature columns in the exact order the model was trained on
# (X_train.drop(macro_cols) from notebook 09a)
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

# Valid categorical values (sourced from training data)
PLANNING_AREAS = sorted([
    "Ang Mo Kio", "Bedok", "Bishan", "Boon Lay", "Bukit Batok",
    "Bukit Merah", "Changi", "Clementi", "Geylang", "Jurong East",
    "Jurong West", "Kallang", "Pasir Ris", "Paya Lebar", "Pioneer",
    "Queenstown", "Sembawang", "Serangoon", "Sungei Kadut", "Tampines",
    "Toa Payoh", "Tuas", "Woodlands", "Yishun",
])

REGIONS = [
    "Central Region",
    "East Region",
    "North Region",
    "North-East Region",
    "West Region",
]

FLOOR_LEVELS = ["First Floor", "Non-First Floor", "Unknown"]
SALE_TYPES = ["New Sale", "Resale"]

# Model test RMSE from notebook 09a — used for prediction interval
MODEL_RMSE = 44.70


_cache: dict = {}


def _load_artifacts():
    if not _cache:
        _cache["model"] = joblib.load(MODELS_DIR / "xgboost_reduced.pkl")
        _cache["target_encoder"] = joblib.load(MODELS_DIR / "target_encoder.pkl")
        _cache["ohe"] = joblib.load(MODELS_DIR / "onehot_encoder.pkl")
    return _cache["model"], _cache["target_encoder"], _cache["ohe"]


def preprocess(
    area_sqft: float,
    remaining_lease_years: float,
    lease_duration: float,
    planning_area: str,
    floor_level: str,
    type_of_sale: str,
    region: str,
    dist_to_mrt_m: float,
) -> pd.DataFrame:
    _, target_encoder, ohe = _load_artifacts()

    # 1. Log-transform area
    log_area = np.log(area_sqft)

    # 2. Lease remaining ratio — Lease_Duration used here then discarded
    lease_remaining_ratio = remaining_lease_years / lease_duration

    # 3. Target-encode planning area (fallback to global mean for unseen areas)
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


def predict(
    area_sqft: float,
    remaining_lease_years: float,
    lease_duration: float,
    planning_area: str,
    floor_level: str,
    type_of_sale: str,
    region: str,
    dist_to_mrt_m: float,
) -> dict:
    model, _, _ = _load_artifacts()
    X = preprocess(
        area_sqft, remaining_lease_years, lease_duration,
        planning_area, floor_level, type_of_sale, region, dist_to_mrt_m,
    )
    predicted_psf = float(model.predict(X)[0])
    total_price = predicted_psf * area_sqft

    return {
        "predicted_psf": round(predicted_psf, 2),
        "total_price": round(total_price, 2),
        "lower_bound": round(predicted_psf - MODEL_RMSE, 2),
        "upper_bound": round(predicted_psf + MODEL_RMSE, 2),
        "rmse": MODEL_RMSE,
    }
