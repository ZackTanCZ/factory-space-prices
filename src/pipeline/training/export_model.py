"""
Export Production model artifacts from MLflow registry to models/.

Connects to the MLflow tracking server, finds the model version tagged
as Production, and downloads its artifacts to the local models/ directory.
Run this after promoting a model version to Production in the MLflow UI.

Usage:
    python -m src.pipeline.training.export_model
"""

import logging
import os
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.core.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))


def main():
    settings = get_settings()
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    logger.info("Fetching Production model from registry: '%s'", settings.MLFLOW_MODEL_NAME)
    versions = client.get_latest_versions(settings.MLFLOW_MODEL_NAME, stages=["Production"])

    if not versions:
        raise RuntimeError(
            f"No Production model found for '{settings.MLFLOW_MODEL_NAME}'. "
            "Promote a model version to Production in the MLflow UI first."
        )

    version = versions[0]
    logger.info("Found: version %s (run_id: %s)", version.version, version.run_id)

    dest_dir = PROJECT_ROOT / "models"
    dest_dir.mkdir(exist_ok=True)

    # Download MLflow-formatted model (logged via mlflow.sklearn.log_model)
    model_dir = mlflow.artifacts.download_artifacts(
        run_id=version.run_id,
        artifact_path="model",
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    model = mlflow.sklearn.load_model(model_dir)
    import joblib
    joblib.dump(model, dest_dir / "model.pkl")
    logger.info("Exported: model.pkl → %s", dest_dir)

    # Download encoder artifacts (target encoder + OHE)
    encoders_dir = mlflow.artifacts.download_artifacts(
        run_id=version.run_id,
        artifact_path="encoders",
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    for src_file in Path(encoders_dir).iterdir():
        dest_file = dest_dir / src_file.name
        shutil.copy2(src_file, dest_file)
        logger.info("Exported: %s → %s", src_file.name, dest_file)

    logger.info("Export complete. Production artifacts saved to %s", dest_dir)


if __name__ == "__main__":
    main()
