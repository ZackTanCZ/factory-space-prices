"""
InferenceOrchestrator — coordinates validation, preprocessing, and prediction.

Owns model artifacts and config-driven business rules. Instantiated once
at API startup and injected into endpoints via FastAPI's dependency system.
"""

import logging
from pathlib import Path

import joblib
from omegaconf import DictConfig, OmegaConf

from src.services.inference import predict_with, preprocess

logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    def __init__(self, cfg: DictConfig, project_root: Path) -> None:
        self.cfg = cfg
        self.constraints = self.cfg.constraints
        self.feature_values = self.cfg.feature_values
        self.api_settings = self.cfg.settings
        self.feature_cols = self.cfg.feature_cols.feature_cols

        self.model = joblib.load(project_root / self.api_settings.model_path)
        self.target_encoder = joblib.load(project_root / self.api_settings.target_encoder_path)
        self.ohe = joblib.load(project_root / self.api_settings.onehot_encoder_path)

        logger.info("InferenceOrchestrator initialised — model artifacts loaded.")

    def _validate(
        self,
        area_sqft: float,
        remaining_lease_years: float,
        lease_duration: float,
        planning_area: str,
        floor_level: str,
        type_of_sale: str,
        region: str,
        dist_to_mrt_m: float,
    ) -> None:
        c = self.constraints

        if not (c.area_sqft.min < area_sqft <= c.area_sqft.max):
            raise ValueError(f"area_sqft must be between {c.area_sqft.min} and {c.area_sqft.max}")

        valid_durations = list(c.lease_duration.valid_values)
        if lease_duration not in valid_durations:
            raise ValueError(f"lease_duration must be one of {valid_durations}")

        if not (c.remaining_lease_years.min <= remaining_lease_years <= c.remaining_lease_years.max):
            raise ValueError(
                f"remaining_lease_years must be between "
                f"{c.remaining_lease_years.min} and {c.remaining_lease_years.max}"
            )

        if remaining_lease_years > lease_duration:
            raise ValueError("remaining_lease_years cannot exceed lease_duration")

        if not (c.dist_to_mrt_m.min <= dist_to_mrt_m <= c.dist_to_mrt_m.max):
            raise ValueError(f"dist_to_mrt_m must be between {c.dist_to_mrt_m.min} and {c.dist_to_mrt_m.max}")

        if planning_area not in self.feature_values.planning_areas:
            raise ValueError(f"Unknown planning area: {planning_area}")
        if region not in self.feature_values.regions:
            raise ValueError(f"Unknown region: {region}")
        if floor_level not in self.feature_values.floor_levels:
            raise ValueError(f"Unknown floor level: {floor_level}")
        if type_of_sale not in self.feature_values.sale_types:
            raise ValueError(f"Unknown type of sale: {type_of_sale}")

    def get_constraints(self) -> dict:
        return OmegaConf.to_container(self.constraints, resolve=True)

    def get_feature_values(self) -> dict:
        return OmegaConf.to_container(self.feature_values, resolve=True)

    def predict(
        self,
        area_sqft: float,
        remaining_lease_years: float,
        lease_duration: float,
        planning_area: str,
        floor_level: str,
        type_of_sale: str,
        region: str,
        dist_to_mrt_m: float,
    ) -> dict:
        self._validate(
            area_sqft, remaining_lease_years, lease_duration,
            planning_area, floor_level, type_of_sale, region, dist_to_mrt_m,
        )

        X = preprocess(
            area_sqft, remaining_lease_years, lease_duration,
            planning_area, floor_level, type_of_sale, region, dist_to_mrt_m,
            target_encoder=self.target_encoder,
            ohe=self.ohe,
            feature_cols=self.feature_cols
        )

        return predict_with(self.model, X, area_sqft, float(self.api_settings.model_rmse))
