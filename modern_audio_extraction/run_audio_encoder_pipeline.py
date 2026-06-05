#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging as transformers_logging

from modern_audio_extraction.pipeline import AudioEncoderExtractionPipeline

PIPELINE_STEPS = {
    "all",
    "preflight",
    "build_query_cache",
    "train_stolen_encoder",
    "evaluate_feature_similarity",
    "evaluate_downstream",
}


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    transformers_logging.set_verbosity_error()


@hydra.main(version_base=None, config_path="configs", config_name="hubert_stolenencoder")
def main(cfg: DictConfig) -> None:
    configure_logging()
    config = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    run_cfg = config.get("run", {})
    step = str(run_cfg.get("step", "all"))
    if step not in PIPELINE_STEPS:
        raise ValueError("Unsupported audio encoder pipeline step: {}".format(step))

    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    logger.info("Running audio encoder extraction pipeline")
    logger.debug("Resolved config:\n{}", json.dumps(config, indent=2))
    AudioEncoderExtractionPipeline(config).run(step)


if __name__ == "__main__":
    main()
