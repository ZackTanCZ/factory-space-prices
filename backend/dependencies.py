"""
Shared dependencies and state for API routes.

Owns infrastructure concerns: config loading, orchestrator instantiation,
and getter functions for FastAPI dependency injection.
"""

import logging
import os
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.services.orchestrator import InferenceOrchestrator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
CONFIG_DIR = str(PROJECT_ROOT / "config")

_orchestrator: InferenceOrchestrator = None


def get_orchestrator() -> InferenceOrchestrator:
    return _orchestrator


def initialize_services() -> None:
    global _orchestrator

    if os.environ.get("SKIP_MODEL_LOAD"):
        logger.warning("SKIP_MODEL_LOAD is set — skipping model artifact loading (CI mode)")
        return

    logger.info("Loading config...")
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="api_config")
    logger.info("Config loaded.")

    logger.info("Loading model artifacts...")
    try:
        _orchestrator = InferenceOrchestrator(cfg=cfg, project_root=PROJECT_ROOT)
    except FileNotFoundError as e:
        logger.error("Model artifact not found — cannot start: %s", e)
        raise
    logger.info("Model artifacts loaded. Ready to serve.")
