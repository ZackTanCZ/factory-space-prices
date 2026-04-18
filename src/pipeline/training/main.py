"""
Entry point for the training pipeline.

Loads Hydra config and instantiates TrainingPipeline.

Usage:
    python -m src.pipeline.training.main
"""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.pipeline.training.orchestrator import TrainingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))

@hydra.main(config_path="../../../../config", config_name="train_config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info("Starting training pipeline...")
    pipeline = TrainingPipeline(cfg=cfg, project_root=PROJECT_ROOT)
    pipeline.run()
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
