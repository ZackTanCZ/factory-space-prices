"""
Entry point for the training pipeline.

Loads Hydra config and instantiates TrainingPipeline.

Usage:
    python src/pipeline/training/main.py
"""

import logging
import os
from pathlib import Path

from hydra import compose, initialize_config_dir

from src.pipeline.training.orchestrator import TrainingPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent.parent))
CONFIG_DIR = str(PROJECT_ROOT / "config")


def main():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="train_config")

    pipeline = TrainingPipeline(cfg=cfg, project_root=PROJECT_ROOT)
    pipeline.run()


if __name__ == "__main__":
    main()
