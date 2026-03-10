from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .config import PipelineConfig
from .utils.logging import configure_logging


def run_check(config: PipelineConfig) -> None:
    logger = configure_logging(config.log_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    touch_path = output_dir / ".write_test"
    touch_path.write_text("ok")
    touch_path.unlink()
    try:
        import rdkit
        import torch_geometric
        import lmdb
        logger.info("Python packages: torch=%s torch_geometric=%s rdkit=%s lmdb=%s", torch.__version__, torch_geometric.__version__, rdkit.__version__, lmdb.__version__)
    except Exception as exc:
        logger.exception("Environment validation failed: %s", exc)
        raise
    logger.info("CUDA available=%s device_count=%s current_device=%s", torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    logger.info("Output directory writable at %s", config.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate environment for molecular GTN pipeline.")
    parser.add_argument("--output-dir", default="/hy-tmp/result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_check(replace(PipelineConfig(), **vars(args)))


if __name__ == "__main__":
    main()
