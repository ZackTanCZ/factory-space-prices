"""
Application settings loaded from environment variables and .env file.

Usage:
    from src.core.settings import get_settings
    settings = get_settings()
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # MLflow tracking server — update when deploying to a remote server
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "FYP-Factory-Price-Prediction"
    MLFLOW_MODEL_NAME: str = "factory-price-prediction"
    # Alias assigned to the champion model version in the MLflow registry.
    # Stages (Production/Staging) are deprecated in MLflow 3.x — use aliases instead.
    MLFLOW_CHAMPION_ALIAS: str = "champion"

    # Backend URL — update when deploying (must match BACKEND_PORT in local dev)
    BACKEND_URL: str = "http://localhost:8000"

    # Port mappings — used by Docker Compose for host-to-container port binding
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    MLFLOW_PORT: int = 5000

    # Temporary directory for encoder serialisation during training.
    # Files are written here before being uploaded to MLflow, then deleted.
    # Change this if models/ is read-only in your environment.
    TRAINING_TMP_DIR: str = "models"


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton — .env is read once on first call."""
    return Settings()
