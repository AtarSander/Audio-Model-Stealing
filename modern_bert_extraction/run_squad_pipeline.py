#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from transformers.utils import logging as transformers_logging

from modern_bert_extraction.squad_pipeline import SquadExtractionPipeline

PIPELINE_STEPS = {
    "all",
    "preflight",
    "prepare_wikitext",
    "train_victim",
    "generate_queries",
    "query_victim",
    "build_distill_data",
    "train_extracted",
    "evaluate_agreement",
}


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    transformers_logging.set_verbosity_error()


@hydra.main(version_base=None, config_path="configs", config_name="squad11")
def main(cfg: DictConfig) -> None:
    configure_logging()
    config = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    run_cfg = config.get("run", {})
    scheme = str(run_cfg.get("scheme", "random"))
    step = str(run_cfg.get("step", "all"))
    if step not in PIPELINE_STEPS:
        raise ValueError("Unsupported SQuAD pipeline step: {}".format(step))

    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    logger.info("Running modern SQuAD 1.1 extraction pipeline: scheme={}", scheme)
    logger.debug("Resolved config:\n{}", json.dumps(config, indent=2))
    pipeline = SquadExtractionPipeline(config=config, scheme=scheme)
    pipeline.run(step)


if __name__ == "__main__":
    main()
