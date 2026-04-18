"""Unit tests for src/pipeline/engineering/steps.py — pure functions only."""

import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline.engineering.steps import (
    apply_feature_transforms,
    compute_mrt_distances,
    geocode_new_buildings,
)

TARGET_COL = "Unit Price ($ psf)"
DROP_COLS = ["Area (sqft)", "Lease_Duration", "Postal District", "Postal Sector", "Year", "Month", "Quarter"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Minimal dataframe with all columns required by apply_feature_transforms."""
    return pd.DataFrame({
        "Area (sqft)": [1000.0, 2000.0, 500.0],
        "Remaining_Lease_Years": [50.0, 30.0, 70.0],
        "Lease_Duration": [99.0, 60.0, 99.0],
        TARGET_COL: [300.0, 500.0, 200.0],
        "Postal District": ["D1", "D2", "D3"],
        "Postal Sector": ["S1", "S2", "S3"],
        "Year": [2020, 2021, 2022],
        "Month": [1, 2, 3],
        "Quarter": ["Q1", "Q2", "Q3"],
        "Project Name": ["A", "B", "C"],
        "Street Name": ["Str A", "Str B", "Str C"],
    })


@pytest.fixture
def sample_cache():
    """Geocode cache with one pre-cached building."""
    return pd.DataFrame({
        "search_key": ["A"],
        "Project Name": ["A"],
        "latitude": [1.35],
        "longitude": [103.8],
    })


@pytest.fixture
def sample_mrt_df():
    """Minimal MRT station dataframe."""
    return pd.DataFrame({
        "mrt_station": ["Station Alpha", "Station Beta", "Station Gamma"],
        "latitude": [1.30, 1.35, 1.40],
        "longitude": [103.75, 103.80, 103.85],
    })


# ---------------------------------------------------------------------------
# apply_feature_transforms
# ---------------------------------------------------------------------------

def test_log_area_computed(sample_df):
    result = apply_feature_transforms(sample_df, TARGET_COL, DROP_COLS)
    expected = np.log(sample_df["Area (sqft)"])
    pd.testing.assert_series_equal(result["Log_Area"].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)


def test_lease_remaining_ratio_computed(sample_df):
    result = apply_feature_transforms(sample_df, TARGET_COL, DROP_COLS)
    expected = sample_df["Remaining_Lease_Years"] / sample_df["Lease_Duration"]
    pd.testing.assert_series_equal(
        result["Lease_Remaining_Ratio"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_log_unit_price_computed(sample_df):
    result = apply_feature_transforms(sample_df, TARGET_COL, DROP_COLS)
    expected = np.log(sample_df[TARGET_COL])
    pd.testing.assert_series_equal(
        result["Log_Unit_Price"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )


def test_drop_cols_removed(sample_df):
    result = apply_feature_transforms(sample_df, TARGET_COL, DROP_COLS)
    for col in DROP_COLS:
        assert col not in result.columns


def test_original_df_not_mutated(sample_df):
    original_cols = set(sample_df.columns)
    apply_feature_transforms(sample_df, TARGET_COL, DROP_COLS)
    assert set(sample_df.columns) == original_cols


# ---------------------------------------------------------------------------
# geocode_new_buildings
# ---------------------------------------------------------------------------

def test_cached_buildings_skipped(sample_df, sample_cache):
    """Buildings already in cache should not trigger new API calls."""
    full_cache = pd.DataFrame({
        "search_key": ["A", "B", "C"],
        "Project Name": ["A", "B", "C"],
        "latitude": [1.35, 1.36, 1.37],
        "longitude": [103.80, 103.81, 103.82],
    })
    result = geocode_new_buildings(sample_df, full_cache, api_delay=0)
    assert len(result) == 3


def test_new_buildings_appended(sample_df, sample_cache, monkeypatch):
    """New buildings not in cache should be appended after geocoding."""
    monkeypatch.setattr(
        "src.pipeline.engineering.steps.geocode_onemap",
        lambda term: (1.40, 103.90),
    )
    result = geocode_new_buildings(sample_df, sample_cache, api_delay=0)
    assert len(result) == 3
    assert set(result["Project Name"]) == {"A", "B", "C"}


def test_empty_cache_geocodes_all(sample_df, monkeypatch):
    """Empty cache should attempt to geocode all unique buildings."""
    calls = []
    monkeypatch.setattr(
        "src.pipeline.engineering.steps.geocode_onemap",
        lambda term: calls.append(term) or (1.30, 103.80),
    )
    empty_cache = pd.DataFrame(columns=["Project Name", "latitude", "longitude"])
    result = geocode_new_buildings(sample_df, empty_cache, api_delay=0)
    assert len(result) == 3


def test_geocoding_failure_raises(sample_df, monkeypatch):
    """Geocoding failure should raise ValueError — not silently fill with NaN."""
    monkeypatch.setattr(
        "src.pipeline.engineering.steps.geocode_onemap",
        lambda term: (None, None),
    )
    empty_cache = pd.DataFrame(columns=["Project Name", "latitude", "longitude"])
    with pytest.raises(ValueError, match="Geocoding failed"):
        geocode_new_buildings(sample_df, empty_cache, api_delay=0)


# ---------------------------------------------------------------------------
# compute_mrt_distances
# ---------------------------------------------------------------------------

def test_nearest_mrt_correct(sample_mrt_df):
    """Building closest to Station Beta should be assigned Station Beta."""
    geo_df = pd.DataFrame({
        "search_key": ["Test"],
        "Project Name": ["Test"],
        "latitude": [1.351],
        "longitude": [103.801],
    })
    result = compute_mrt_distances(geo_df, sample_mrt_df)
    assert result.iloc[0]["nearest_mrt"] == "Station Beta"


def test_dist_to_mrt_is_positive(sample_mrt_df):
    geo_df = pd.DataFrame({
        "search_key": ["Test"],
        "Project Name": ["Test"],
        "latitude": [1.30],
        "longitude": [103.80],
    })
    result = compute_mrt_distances(geo_df, sample_mrt_df)
    assert result.iloc[0]["dist_to_mrt_m"] > 0


def test_dist_to_mrt_unit_is_metres(sample_mrt_df):
    """Distance to the exact station location should be ~0m."""
    geo_df = pd.DataFrame({
        "search_key": ["At Station"],
        "Project Name": ["At Station"],
        "latitude": [sample_mrt_df.iloc[0]["latitude"]],
        "longitude": [sample_mrt_df.iloc[0]["longitude"]],
    })
    result = compute_mrt_distances(geo_df, sample_mrt_df)
    assert result.iloc[0]["dist_to_mrt_m"] < 1.0


def test_output_columns(sample_mrt_df):
    geo_df = pd.DataFrame({
        "search_key": ["Test"],
        "Project Name": ["Test"],
        "latitude": [1.30],
        "longitude": [103.80],
    })
    result = compute_mrt_distances(geo_df, sample_mrt_df)
    assert set(result.columns) == {"search_key", "Project Name", "latitude", "longitude", "dist_to_mrt_m", "nearest_mrt"}


def test_rows_with_missing_coords_excluded(sample_mrt_df):
    """Buildings with NaN coordinates should be excluded from output."""
    geo_df = pd.DataFrame({
        "search_key": ["Valid", "No Coords"],
        "Project Name": ["Valid", "No Coords"],
        "latitude": [1.30, None],
        "longitude": [103.80, None],
    })
    result = compute_mrt_distances(geo_df, sample_mrt_df)
    assert len(result) == 1
    assert result.iloc[0]["Project Name"] == "Valid"
