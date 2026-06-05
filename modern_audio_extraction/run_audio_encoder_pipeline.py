#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from loguru import logger
from transformers.utils import logging as transformers_logging

from modern_audio_extraction.pipeline import AudioEncoderExtractionPipeline, load_config

PIPELINE_STEPS = [
    "all",
    "preflight",
    "build_query_cache",
    "train_stolen_encoder",
    "evaluate_feature_similarity",
    "evaluate_downstream",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "hubert_stolenencoder.yaml"),
    )
    parser.add_argument("--step", choices=PIPELINE_STEPS, default="all")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--student-model", default=None)
    parser.add_argument("--student-init-from-pretrained", type=str, default=None)
    parser.add_argument("--query-source-type", default=None)
    parser.add_argument("--query-budget", type=int, default=None)
    parser.add_argument("--query-max-duration-seconds", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--downstream-train-budget", type=int, default=None)
    parser.add_argument("--downstream-eval-budget", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--require-gpu", type=str, default=None)
    parser.add_argument(
        "--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default=None
    )
    parser.add_argument("--query-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--disable-downstream", action="store_true")
    return parser.parse_args()


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    transformers_logging.set_verbosity_error()


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError("Expected boolean string, got {}".format(value))


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.output_root is not None:
        config["paths"]["output_root"] = args.output_root
    if args.target_model is not None:
        config["target"]["model_name_or_path"] = args.target_model
    if args.student_model is not None:
        config["student"]["model_name_or_path"] = args.student_model
    if args.student_init_from_pretrained is not None:
        config["student"]["init_from_pretrained"] = _parse_bool(args.student_init_from_pretrained)
    if args.query_source_type is not None:
        config["query_source"]["type"] = args.query_source_type
    if args.query_budget is not None:
        config["query_source"]["budget"] = args.query_budget
    if args.query_max_duration_seconds is not None:
        config["query_source"]["max_duration_seconds"] = args.query_max_duration_seconds
    if args.num_train_epochs is not None:
        config["training"]["num_train_epochs"] = args.num_train_epochs
    if args.train_batch_size is not None:
        config["training"]["per_device_train_batch_size"] = args.train_batch_size
    if args.downstream_train_budget is not None:
        config["downstream"]["train_budget"] = args.downstream_train_budget
    if args.downstream_eval_budget is not None:
        config["downstream"]["eval_budget"] = args.downstream_eval_budget
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.require_gpu is not None:
        config["runtime"]["require_gpu"] = _parse_bool(args.require_gpu)
    if args.mixed_precision is not None:
        config["runtime"]["mixed_precision"] = args.mixed_precision
    if args.query_batch_size is not None:
        config["runtime"]["query_batch_size"] = args.query_batch_size
    if args.eval_batch_size is not None:
        config["runtime"]["eval_batch_size"] = args.eval_batch_size
    if args.disable_downstream:
        config["downstream"]["enabled"] = False
    return config


def main() -> None:
    configure_logging()
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)

    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices")
    if cuda_visible_devices is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(cuda_visible_devices))

    logger.info("Running audio encoder extraction pipeline")
    logger.debug("Resolved config:\n{}", json.dumps(config, indent=2))
    AudioEncoderExtractionPipeline(config).run(args.step)


if __name__ == "__main__":
    main()
