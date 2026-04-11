"""
TrainingPipeline — coordinates data loading, encoding, training, evaluation,
and artifact saving.

Owns pipeline state (train/test splits, encoders, model, metrics).
All transformation logic lives in steps.py as pure functions.
"""

import logging
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from omegaconf import DictConfig
from xgboost import XGBRegressor

from src.pipeline.training.steps import (
    apply_ohe,
    apply_target_encoding,
    evaluate,
    fit_ohe,
    fit_target_encoder,
    split_data,
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, cfg: DictConfig, project_root: Path) -> None:
        self.cfg = cfg
        self.project_root = project_root

    def load_data(self) -> None:
        logger.info("Loading data from %s", self.cfg.train_data.input_path)
        df = pd.read_csv(self.project_root / self.cfg.train_data.input_path)
        df = df.drop(columns=self.cfg.train_data.drop_cols)

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            df=df,
            target_col=self.cfg.train_data.target_col,
            test_size=self.cfg.train_data.test_size,
            random_state=self.cfg.train_data.random_state,
        )
        logger.info("Train: %d rows | Test: %d rows", len(self.X_train), len(self.X_test))

    def encode(self) -> None:
        logger.info("Fitting encoders on training data...")

        cat_cols = self.cfg.train_data.cat_cols
        target_encode_col = self.cfg.train_data.target_encode_col

        self.target_encoder = fit_target_encoder(self.X_train, self.y_train, target_encode_col)
        self.X_train = apply_target_encoding(self.X_train, self.target_encoder, target_encode_col)
        self.X_test = apply_target_encoding(self.X_test, self.target_encoder, target_encode_col)

        self.ohe = fit_ohe(self.X_train, cat_cols)
        self.X_train = apply_ohe(self.X_train, self.ohe, cat_cols, target_encode_col)
        self.X_test = apply_ohe(self.X_test, self.ohe, cat_cols, target_encode_col)

        ohe_cols = [c for c in self.X_train.columns if any(c.startswith(cat) for cat in cat_cols)]
        logger.info("Target encoded: %s → %s_Encoded", target_encode_col, target_encode_col.replace(" ", "_"))
        logger.info("One-hot encoded: %s → %s", list(cat_cols), ohe_cols)
        logger.info("Encoding complete. Total features (%d): %s", self.X_train.shape[1], list(self.X_train.columns))

    def train(self) -> None:
        logger.info("Training XGBRegressor...")
        self.model = XGBRegressor(
            n_estimators=self.cfg.train_model.n_estimators,
            max_depth=self.cfg.train_model.max_depth,
            learning_rate=self.cfg.train_model.learning_rate,
            subsample=self.cfg.train_model.subsample,
            colsample_bytree=self.cfg.train_model.colsample_bytree,
            random_state=self.cfg.train_model.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model.fit(self.X_train, self.y_train)
        logger.info("Training complete.")

    def evaluate(self) -> None:
        self.metrics = evaluate(self.model, self.X_test, self.y_test)
        logger.info("Test RMSE: $%.2f /psf", self.metrics["rmse"])
        logger.info("Test MAE:  $%.2f /psf", self.metrics["mae"])
        logger.info("Test R²:   %.4f", self.metrics["r2"])

    def save_artifacts(self) -> None:
        try:
            model_path = self.project_root / self.cfg.settings.model_path
            target_encoder_path = self.project_root / self.cfg.settings.target_encoder_path
            ohe_path = self.project_root / self.cfg.settings.onehot_encoder_path

            joblib.dump(self.model, model_path)
            joblib.dump(self.target_encoder, target_encoder_path)
            joblib.dump(self.ohe, ohe_path)
            logger.info("Artifacts saved to %s", self.project_root / "models/")

            if mlflow.active_run():
                mlflow.log_artifact(str(model_path), artifact_path="models")
                mlflow.log_artifact(str(target_encoder_path), artifact_path="models")
                mlflow.log_artifact(str(ohe_path), artifact_path="models")
                logger.info("Artifacts logged to MLflow.")
        except OSError as e:
            logger.error("Failed to save artifacts — check output path and permissions: %s", e)
            raise

    def _validate_config(self) -> None:
        input_path = self.project_root / self.cfg.train_data.input_path
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df_cols = pd.read_csv(input_path, nrows=0).columns.tolist()

        missing_drop = [c for c in self.cfg.train_data.drop_cols if c not in df_cols]
        if missing_drop:
            raise ValueError(f"drop_cols not found in dataset: {missing_drop}")

        remaining_cols = [c for c in df_cols if c not in self.cfg.train_data.drop_cols]
        missing_cat = [c for c in self.cfg.train_data.cat_cols if c not in remaining_cols]
        if missing_cat:
            raise ValueError(f"cat_cols not found after dropping columns: {missing_cat}")

        if self.cfg.train_data.target_encode_col not in remaining_cols:
            raise ValueError(f"'{self.cfg.train_data.target_encode_col}' not found after dropping columns")

        if self.cfg.train_data.target_col not in remaining_cols:
            raise ValueError(f"target_col '{self.cfg.train_data.target_col}' not found after dropping columns")

        logger.info("Config validation passed.")

    def run(self) -> None:
        self._validate_config()
        self.load_data()
        self.encode()
        self.train()
        self.evaluate()

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("factory-price-prediction")

        with mlflow.start_run():
            mlflow.log_params(
                {
                    "n_estimators": self.cfg.train_model.n_estimators,
                    "max_depth": self.cfg.train_model.max_depth,
                    "learning_rate": self.cfg.train_model.learning_rate,
                    "subsample": self.cfg.train_model.subsample,
                    "colsample_bytree": self.cfg.train_model.colsample_bytree,
                    "random_state": self.cfg.train_model.random_state,
                    "test_size": self.cfg.train_data.test_size,
                }
            )
            mlflow.log_metrics(
                {
                    "rmse": self.metrics["rmse"],
                    "mae": self.metrics["mae"],
                    "r2": self.metrics["r2"],
                }
            )
            logger.info("Metrics logged to MLflow (tracking URI: %s).", tracking_uri)
            self.save_artifacts()


