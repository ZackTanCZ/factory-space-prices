"""
FeatureEngineeringPipeline — coordinates feature transforms, geocoding, and
MRT distance computation.

All transformation logic lives in steps.py as pure functions.
"""

import logging
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.pipeline.engineering.steps import (
    apply_feature_transforms,
    compute_mrt_distances,
    geocode_new_buildings,
)

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    def __init__(self, cfg: DictConfig, project_root: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root
        self.input_path = project_root / cfg.feat_engr.input_path
        self.mrt_path = project_root / cfg.feat_engr.mrt_path
        self.geocode_cache_path = project_root / cfg.feat_engr.geocode_cache_path
        self.output_path = project_root / cfg.feat_engr.output_path

    def _validate_config(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        df_cols = pd.read_csv(self.input_path, nrows=0).columns.tolist()

        missing_drop = [c for c in self.cfg.feat_engr.drop_cols if c not in df_cols]
        if missing_drop:
            raise ValueError(f"drop_cols not found in dataset: {missing_drop}")

        required = [*self.cfg.feat_engr.required_cols, self.cfg.feat_engr.target_col]
        missing_required = [c for c in required if c not in df_cols]
        if missing_required:
            raise ValueError(f"Required columns not found in dataset: {missing_required}")

        logger.info("Config validation passed.")

    def load_data(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        self.df = pd.read_csv(self.input_path)
        logger.info("Loaded %d rows, %d columns.", *self.df.shape)

    def apply_transforms(self) -> None:
        self.df = apply_feature_transforms(
            self.df,
            target_col=self.cfg.feat_engr.target_col,
            drop_cols=list(self.cfg.feat_engr.drop_cols),
        )

    def add_mrt_distances(self) -> None:
        mrt_df = pd.read_csv(self.mrt_path)
        mrt_df = mrt_df[["Name", "Latitude", "Longitude"]].rename(columns={
            "Name": "mrt_station",
            "Latitude": "latitude",
            "Longitude": "longitude",
        })
        logger.info("Loaded %d MRT/LRT stations.", len(mrt_df))

        if self.geocode_cache_path.exists():
            cache = pd.read_csv(self.geocode_cache_path)
            logger.info("Loaded geocode cache: %d buildings.", len(cache))
        else:
            cache = pd.DataFrame(columns=["search_key", "Project Name", "latitude", "longitude"])
            logger.info("No geocode cache found — will geocode all buildings.")

        geo_df = geocode_new_buildings(self.df, cache)

        cached_keys_with_dist = (
            set(cache[cache["dist_to_mrt_m"].notna()]["search_key"])
            if "dist_to_mrt_m" in cache.columns
            else set()
        )
        needs_distance = geo_df[~geo_df["search_key"].isin(cached_keys_with_dist)]

        if not needs_distance.empty:
            logger.info("Computing MRT distances for %d buildings...", len(needs_distance))
            dist_df = compute_mrt_distances(needs_distance, mrt_df)

            if "dist_to_mrt_m" in geo_df.columns:
                geo_df = geo_df.drop(columns=["dist_to_mrt_m", "nearest_mrt"], errors="ignore")
            geo_df = geo_df.merge(
                dist_df[["search_key", "dist_to_mrt_m", "nearest_mrt"]],
                on="search_key",
                how="left",
            )

            if "dist_to_mrt_m" in cache.columns:
                cached_with_dist = cache[["search_key", "dist_to_mrt_m", "nearest_mrt"]].dropna(
                    subset=["dist_to_mrt_m"]
                )
                geo_df = geo_df.set_index("search_key")
                geo_df.update(cached_with_dist.set_index("search_key"))
                geo_df = geo_df.reset_index()
        else:
            logger.info("MRT distances already cached for all buildings.")

        geo_df.to_csv(self.geocode_cache_path, index=False)
        logger.info("Geocode cache saved: %s", self.geocode_cache_path)

        logger.info(
            "Distance stats (m) — min: %.0f | median: %.0f | max: %.0f",
            geo_df["dist_to_mrt_m"].min(),
            geo_df["dist_to_mrt_m"].median(),
            geo_df["dist_to_mrt_m"].max(),
        )

        # Build search_key on main df to merge on — drop after merge
        self.df["search_key"] = self.df["Project Name"].fillna(self.df["Street Name"])
        self.df = self.df.merge(
            geo_df[["search_key", "dist_to_mrt_m", "nearest_mrt"]], on="search_key", how="left"
        )
        self.df = self.df.drop(columns=["search_key"])

        missing = self.df["dist_to_mrt_m"].isna().sum()
        if missing > 0:
            raise ValueError(
                f"{missing} rows have no dist_to_mrt_m after merge — check geocode cache."
            )

        logger.info(
            "dist_to_mrt_m merged — r with target: %.3f",
            self.df["dist_to_mrt_m"].corr(self.df[self.cfg.feat_engr.target_col]),
        )

    def save_output(self) -> None:
        self.df.to_csv(self.output_path, index=False)
        logger.info("Saved engineered dataset: %s", self.output_path)
        logger.info("Final shape: %s", self.df.shape)
        logger.info("Columns: %s", self.df.columns.tolist())

    def run(self) -> None:
        logger.info("=== Feature Engineering Pipeline ===")
        logger.info("Input:  %s", self.input_path)
        logger.info("Output: %s", self.output_path)

        self._validate_config()
        self.load_data()
        self.apply_transforms()
        self.add_mrt_distances()
        self.save_output()
