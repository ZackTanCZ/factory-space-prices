"""Unit tests for src/services/orchestrator.py — validation and serialisation."""

import pytest
from omegaconf import DictConfig

# ---------------------------------------------------------------------------
# _validate() — happy path
# ---------------------------------------------------------------------------

def test_validate_passes_for_valid_inputs(dummy_orchestrator):
    dummy_orchestrator._validate(
        area_sqft=1000.0,
        remaining_lease_years=50.0,
        lease_duration=99.0,
        planning_area="Ang Mo Kio",
        floor_level="Non-First Floor",
        type_of_sale="Resale",
        region="Central Region",
        dist_to_mrt_m=500.0,
    )


# ---------------------------------------------------------------------------
# _validate() — constraint violations
# ---------------------------------------------------------------------------

def test_validate_rejects_area_sqft_below_min(dummy_orchestrator):
    with pytest.raises(ValueError, match="area_sqft"):
        dummy_orchestrator._validate(
            area_sqft=50.0,
            remaining_lease_years=50.0,
            lease_duration=99.0,
            planning_area="Ang Mo Kio",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Central Region",
            dist_to_mrt_m=500.0,
        )


def test_validate_rejects_area_sqft_above_max(dummy_orchestrator):
    with pytest.raises(ValueError, match="area_sqft"):
        dummy_orchestrator._validate(
            area_sqft=99999.0,
            remaining_lease_years=50.0,
            lease_duration=99.0,
            planning_area="Ang Mo Kio",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Central Region",
            dist_to_mrt_m=500.0,
        )


def test_validate_rejects_invalid_lease_duration(dummy_orchestrator):
    with pytest.raises(ValueError, match="lease_duration"):
        dummy_orchestrator._validate(
            area_sqft=1000.0,
            remaining_lease_years=50.0,
            lease_duration=50.0,
            planning_area="Ang Mo Kio",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Central Region",
            dist_to_mrt_m=500.0,
        )


def test_validate_rejects_remaining_lease_exceeds_duration(dummy_orchestrator):
    with pytest.raises(ValueError, match="remaining_lease_years cannot exceed lease_duration"):
        dummy_orchestrator._validate(
            area_sqft=1000.0,
            remaining_lease_years=60.0,
            lease_duration=30.0,
            planning_area="Ang Mo Kio",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Central Region",
            dist_to_mrt_m=500.0,
        )


def test_validate_rejects_unknown_planning_area(dummy_orchestrator):
    with pytest.raises(ValueError, match="planning area"):
        dummy_orchestrator._validate(
            area_sqft=1000.0,
            remaining_lease_years=50.0,
            lease_duration=99.0,
            planning_area="Atlantis",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Central Region",
            dist_to_mrt_m=500.0,
        )


def test_validate_rejects_unknown_region(dummy_orchestrator):
    with pytest.raises(ValueError, match="region"):
        dummy_orchestrator._validate(
            area_sqft=1000.0,
            remaining_lease_years=50.0,
            lease_duration=99.0,
            planning_area="Ang Mo Kio",
            floor_level="Non-First Floor",
            type_of_sale="Resale",
            region="Southern Region",
            dist_to_mrt_m=500.0,
        )


# ---------------------------------------------------------------------------
# get_constraints() and get_feature_values() — serialisation
# ---------------------------------------------------------------------------

def test_get_constraints_returns_plain_dict(dummy_orchestrator):
    result = dummy_orchestrator.get_constraints()
    assert isinstance(result, dict)
    assert not isinstance(result, DictConfig)


def test_get_feature_values_returns_plain_dict(dummy_orchestrator):
    result = dummy_orchestrator.get_feature_values()
    assert isinstance(result, dict)
    assert not isinstance(result, DictConfig)


def test_get_feature_values_contains_expected_keys(dummy_orchestrator):
    result = dummy_orchestrator.get_feature_values()
    assert set(result.keys()) == {"planning_areas", "regions", "floor_levels", "sale_types"}


def test_get_constraints_contains_expected_keys(dummy_orchestrator):
    result = dummy_orchestrator.get_constraints()
    assert set(result.keys()) == {"area_sqft", "remaining_lease_years", "dist_to_mrt_m", "lease_duration"}
