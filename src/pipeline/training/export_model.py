"""
Export champion model artifacts from MLflow registry to models/champion_model/.

Connects to the MLflow tracking server, finds the model version tagged with
the 'champion' alias, and downloads its artifacts to models/champion_model/.
Run this after assigning the 'champion' alias to a model version in the MLflow UI.

Usage:
    python -m src.pipeline.training.export_model
"""

import logging
import os
import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.core.settings import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))


def main():
    settings = get_settings()
    # Must set globally — models:/ URIs in download_artifacts use the global tracking
    # URI, not the tracking_uri parameter (which only applies to runs:/ URIs).
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    client = MlflowClient(tracking_uri=settings.MLFLOW_TRACKING_URI)

    logger.info(
        "Fetching '%s' model from registry: '%s'",
        settings.MLFLOW_CHAMPION_ALIAS,
        settings.MLFLOW_MODEL_NAME,
    )
    try:
        version = client.get_model_version_by_alias(settings.MLFLOW_MODEL_NAME, settings.MLFLOW_CHAMPION_ALIAS)
    except Exception as e:
        raise RuntimeError(
            f"No model version with alias '{settings.MLFLOW_CHAMPION_ALIAS}' found for "
            f"'{settings.MLFLOW_MODEL_NAME}'. Assign the alias in the MLflow UI first."
        ) from e

    logger.info("Found: version %s (run_id: %s)", version.version, version.run_id)

    dest_dir = PROJECT_ROOT / "models" / "champion_model"
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
        logger.info("Cleared existing artifacts from %s", dest_dir)
    dest_dir.mkdir(parents=True)

    # Download model using version.source (MLflow 3.x stores models at models:/m-{id},
    # not under the run artifact path — using run_id + artifact_path="model" won't work)
    model_dir = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source,
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    )
    shutil.copy2(Path(model_dir) / "model.pkl", dest_dir / "model.pkl")
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

    logger.info("Export complete. Champion artifacts saved to %s", dest_dir)


if __name__ == "__main__":
    main()
