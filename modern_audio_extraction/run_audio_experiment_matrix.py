#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

from loguru import logger

from modern_audio_extraction.pipeline import AudioEncoderExtractionPipeline, load_config


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "hubert_stolenencoder.yaml"),
    )
    parser.add_argument(
        "--step",
        choices=[
            "all",
            "preflight",
            "build_query_cache",
            "train_stolen_encoder",
            "evaluate_feature_similarity",
            "evaluate_downstream",
        ],
        default="all",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-name", default="matrix_summary")
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _matrix_configs(base_config: dict):
    matrix = base_config.get("matrix", {})
    budgets = matrix.get("query_budgets", [base_config["query_source"]["budget"]])
    students = matrix.get("student_models", [base_config["student"]])
    sources = matrix.get("query_sources", [base_config["query_source"]])
    for budget in budgets:
        for student in students:
            for source in sources:
                config = deepcopy(base_config)
                config["query_source"].update(source)
                config["query_source"]["budget"] = int(budget)
                if isinstance(student, str):
                    config["student"]["model_name_or_path"] = student
                else:
                    config["student"].update(student)
                source_name = source.get("name", config["query_source"]["type"])
                source_slug = _safe_slug(str(source_name))
                student_slug = _safe_slug(str(config["student"]["model_name_or_path"]))
                config["run_name"] = "hubert__{}__{}__q{}".format(
                    student_slug,
                    source_slug,
                    budget,
                )
                yield config


def main() -> None:
    configure_logging()
    args = parse_args()
    base_config = load_config(args.config)
    final_metrics: list[dict] = []
    for config in _matrix_configs(base_config):
        pipeline = AudioEncoderExtractionPipeline(config)
        final_path = pipeline.metrics_dir / "final_metrics.json"
        if args.skip_existing and final_path.exists():
            logger.info("Skipping existing run {}", pipeline.run_name)
        else:
            logger.info("Running matrix experiment {}", pipeline.run_name)
            pipeline.run(args.step)
        if final_path.exists():
            final_metrics.append(json.loads(final_path.read_text(encoding="utf-8")))
        else:
            final_metrics.append(
                {
                    "run_name": pipeline.run_name,
                    "target_model": str(config["target"]["model_name_or_path"]),
                    "student_model": str(config["student"]["model_name_or_path"]),
                    "query_source_type": str(config["query_source"]["type"]),
                    "query_budget": float(config["query_source"]["budget"]),
                    "status": "missing_final_metrics",
                }
            )

    output_root = Path(base_config["paths"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "{}.json".format(args.output_name)
    csv_path = output_root / "{}.csv".format(args.output_name)
    json_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    if final_metrics:
        fieldnames = sorted({key for row in final_metrics for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(final_metrics)
    logger.info("Wrote matrix summaries to {} and {}", json_path, csv_path)


if __name__ == "__main__":
    main()
