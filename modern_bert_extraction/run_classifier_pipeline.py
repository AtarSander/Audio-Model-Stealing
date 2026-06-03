#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from loguru import logger
from modern_bert_extraction.pipeline import ClassifierExtractionPipeline, load_config
from transformers.utils import logging as transformers_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "classifier.yaml"),
        help="Path to the classifier pipeline YAML config.",
    )
    parser.add_argument("--task", choices=["MNLI", "SST-2", "mnli", "sst-2", "sst2"], default="SST-2")
    parser.add_argument("--scheme", choices=["random", "wiki"], default="random")
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "preflight",
            "prepare_wikitext",
            "train_victim",
            "generate_queries",
            "query_victim",
            "build_distill_data",
            "train_extracted",
            "evaluate_agreement",
        ],
        default="all",
    )
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--augmentations", type=int, default=None)
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
    if args.output_root is not None:
        config["paths"]["output_root"] = args.output_root

    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    logger.info("Running modern classifier extraction pipeline: task={} scheme={}", args.task, args.scheme)
    logger.debug("Resolved config:\n{}", json.dumps(config, indent=2))
    pipeline = ClassifierExtractionPipeline(config=config, task=args.task, scheme=args.scheme)
    pipeline.run(args.step)


if __name__ == "__main__":
    main()
