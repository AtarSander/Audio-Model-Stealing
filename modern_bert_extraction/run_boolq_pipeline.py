#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from loguru import logger
from transformers.utils import logging as transformers_logging

from modern_bert_extraction.boolq_pipeline import BoolQExtractionPipeline
from modern_bert_extraction.pipeline import load_config

PIPELINE_STEPS = [
    "all",
    "preflight",
    "prepare_wikitext",
    "train_victim",
    "generate_queries",
    "query_victim",
    "build_distill_data",
    "train_extracted",
    "evaluate_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "boolq.yaml"),
        help="Path to the BoolQ pipeline YAML config.",
    )
    parser.add_argument("--scheme", choices=["random", "wiki"], default="random")
    parser.add_argument("--step", choices=PIPELINE_STEPS, default="all")
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--augmentations", type=int, default=None)
    parser.add_argument("--thief-paragraph-limit", type=int, default=None)
    parser.add_argument("--victim-model-dir", default=None)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    transformers_logging.set_verbosity_error()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)

    if args.dataset_size is not None:
        config["query_generation"]["dataset_size"] = args.dataset_size
    if args.augmentations is not None:
        config["query_generation"]["augmentations"] = args.augmentations
    if args.thief_paragraph_limit is not None:
        config["query_generation"]["thief_paragraph_limit"] = args.thief_paragraph_limit
    if args.victim_model_dir is not None:
        config["paths"]["victim_model_dir"] = args.victim_model_dir
    if args.output_root is not None:
        config["paths"]["output_root"] = args.output_root

    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    logger.info("Running modern BoolQ extraction pipeline: scheme={}", args.scheme)
    logger.debug("Resolved config:\n{}", json.dumps(config, indent=2))
    pipeline = BoolQExtractionPipeline(config=config, scheme=args.scheme)
    pipeline.run(args.step)


if __name__ == "__main__":
    main()
