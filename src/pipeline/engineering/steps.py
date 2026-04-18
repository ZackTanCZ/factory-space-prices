"""
Pure feature engineering functions.

All functions are stateless — they accept data as parameters and return
transformed data. No file I/O or side effects.
"""

import logging
import time
from math import radians

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import haversine_distances

logger = logging.getLogger(__name__)

def apply_feature_transforms(df: pd.DataFrame, target_col: str, drop_cols: list[str]) -> pd.DataFrame:
    """
    Apply feature transforms from notebook 04:
      - Log_Area: fixes right-skew (skewness=+3.95) and linearity assumption
      - Lease_Remaining_Ratio: proportional lease life; more informative than raw duration
      - Log_Unit_Price: log-transform target for linear models only (tree models ignore it)
    """
    df = df.copy()

    df["Log_Area"] = np.log(df["Area (sqft)"])
    logger.info(
        "Log_Area computed - skewness: %.3f -> %.3f",
        df["Area (sqft)"].skew(),
        df["Log_Area"].skew(),
    )

    df["Lease_Remaining_Ratio"] = df["Remaining_Lease_Years"] / df["Lease_Duration"]
    logger.info(
        "Lease_Remaining_Ratio computed - r with target: %.3f",
        df["Lease_Remaining_Ratio"].corr(df[target_col]),
    )

    df["Log_Unit_Price"] = np.log(df[target_col])
    logger.info(
        "Log_Unit_Price computed - skewness: %.3f -> %.3f",
        df[target_col].skew(),
        df["Log_Unit_Price"].skew(),
    )

    df = df.drop(columns=drop_cols)
    logger.info("Dropped %d columns. Shape after transforms: %s", len(drop_cols), df.shape)

    return df


def geocode_onemap(search_term: str) -> tuple[float | None, float | None]:
    """Query OneMap API for lat/long of a building or street name."""
    url = "https://www.onemap.gov.sg/api/common/elastic/search"
    params = {"searchVal": search_term, "returnGeom": "Y", "getAddrDetails": "Y"}
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["found"] > 0:
            result = data["results"][0]
            return float(result["LATITUDE"]), float(result["LONGITUDE"])
    except Exception:
        pass
    return None, None


def geocode_new_buildings(
    df: pd.DataFrame,
    cache: pd.DataFrame,
    api_delay: float = 0.2,
) -> pd.DataFrame:
    """
    Geocode buildings not already in the cache.

    Only buildings absent from property_geocoded.csv are sent to the OneMap API.
    Strategy: try Project Name first, fall back to Street Name if no result.
    """
    cached_keys = set(cache["search_key"].dropna()) if "search_key" in cache.columns else set()

    temp = df[["Project Name", "Street Name"]].copy()
    temp["search_key"] = temp["Project Name"].fillna(temp["Street Name"])
    unique_addresses = temp.drop_duplicates("search_key").reset_index(drop=True)
    new_addresses = unique_addresses[~unique_addresses["search_key"].isin(cached_keys)]

    if new_addresses.empty:
        logger.info("All %d buildings already cached - skipping API calls.", len(unique_addresses))
        return cache

    logger.info(
        "Geocoding %d new buildings (%d already cached)...",
        len(new_addresses),
        len(cached_keys),
    )

    new_results = []
    for _, row in new_addresses.iterrows():
        lat, lon = geocode_onemap(row["search_key"])
        new_results.append({
            "search_key": row["search_key"],
            "Project Name": row["Project Name"],
            "latitude": lat,
            "longitude": lon,
        })

        if len(new_results) % 10 == 0:
            logger.info("  Geocoded %d/%d...", len(new_results), len(new_addresses))

        time.sleep(api_delay)

    new_geo_df = pd.DataFrame(new_results)
    failed = new_geo_df[new_geo_df["latitude"].isna()]
    if not failed.empty:
        raise ValueError(
            f"Geocoding failed for {len(failed)} building(s): "
            f"{failed['search_key'].tolist()} — fix and re-run."
        )

    logger.info("Geocoding complete: %d/%d successful", len(new_geo_df), len(new_addresses))

    return pd.concat([cache, new_geo_df], ignore_index=True) if not cache.empty else new_geo_df.reset_index(drop=True)


def compute_mrt_distances(geo_df: pd.DataFrame, mrt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute distance (metres) from each building to its nearest MRT/LRT station
    using the Haversine formula.
    """
    valid = geo_df.dropna(subset=["latitude", "longitude"]).copy()

    min_distances = []
    nearest_stations = []

    for _, row in valid.iterrows():
        prop_rad = [radians(row["latitude"]), radians(row["longitude"])]

        distances = []
        for _, mrt_row in mrt_df.iterrows():
            mrt_rad = [radians(mrt_row["latitude"]), radians(mrt_row["longitude"])]
            dist_matrix = haversine_distances([prop_rad, mrt_rad])
            dist_km = dist_matrix[0, 1] * 6371
            distances.append((dist_km, mrt_row["mrt_station"]))

        min_dist_km, nearest = min(distances, key=lambda x: x[0])
        min_distances.append(round(min_dist_km * 1000, 1))
        nearest_stations.append(nearest)

    valid["dist_to_mrt_m"] = min_distances
    valid["nearest_mrt"] = nearest_stations

    return valid[["search_key", "Project Name", "latitude", "longitude", "dist_to_mrt_m", "nearest_mrt"]]
