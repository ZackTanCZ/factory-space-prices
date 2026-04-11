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
    MLFLOW_EXPERIMENT_NAME: str = ""

    # Backend URL — update when deploying (must match BACKEND_PORT in local dev)
    BACKEND_URL: str = "http://localhost:8000"

    # Port mappings — used by Docker Compose for host-to-container port binding
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 8501
    MLFLOW_PORT: int = 5000


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton — .env is read once on first call."""
    return Settings()
