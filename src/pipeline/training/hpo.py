"""
Hyperparameter optimisation entry point.

Runs an Optuna sweep via Hydra's Optuna Sweeper. Each trial uses k-fold CV
on the training set — the test set is never touched during HPO. Each trial
logs params and cv_rmse to MLflow. After the sweep, copy the best params
from MLflow into config/train/model.yaml and run the training pipeline.

Usage:
    python -m src.pipeline.training.hpo --multirun
"""

import logging
import os
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

from src.core.settings import get_settings
from src.pipeline.training.orchestrator import TrainingPipeline
from src.pipeline.training.steps import cross_val_rmse

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))


@hydra.main(config_path="../../../../config", config_name="hpo_config", version_base=None)
def main(cfg: DictConfig) -> float:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # Load and encode data — reuse pipeline steps, skip train/evaluate
    pipeline = TrainingPipeline(cfg=cfg, project_root=PROJECT_ROOT)
    pipeline._validate_config()
    pipeline.load_data()
    pipeline.encode()

    # K-fold CV on training set only — test set never touched during HPO
    cv_rmse = cross_val_rmse(
        model_cfg=cfg.train_model,
        X_train=pipeline.X_train,
        y_train=pipeline.y_train,
        n_splits=5,
        random_state=cfg.train_data.random_state,
    )

    # Log tunable params only (exclude fixed non-tunable keys)
    params = {
        k: v
        for k, v in OmegaConf.to_container(cfg.train_model, resolve=True).items()
        if k not in ("_target_", "random_state", "n_jobs", "verbosity")
    }

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("cv_rmse", cv_rmse)
        logger.info("Trial complete — cv_rmse: %.4f", cv_rmse)

    return cv_rmse


if __name__ == "__main__":
    main()
