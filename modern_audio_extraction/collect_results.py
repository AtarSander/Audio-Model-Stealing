#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-root", default="output/repro/modern_classifier")
    parser.add_argument("--qa-root", default="output/repro/modern_qa")
    parser.add_argument("--audio-root", default="output/repro/modern_audio")
    parser.add_argument("--output-root", default="output/repro/comparison")
    parser.add_argument("--output-name", default="comparison_summary")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _qa_query_count(run_dir: Path) -> int | None:
    predictions_path = run_dir / "data" / "new_train_victim_predictions.json"
    if not predictions_path.exists():
        return None
    return len(_read_json(predictions_path))


def _parse_task_scheme(run_dir: Path, agreement: dict[str, Any]) -> tuple[str, str]:
    if "task" in agreement and "scheme" in agreement:
        return str(agreement["task"]), str(agreement["scheme"])
    name = run_dir.name
    if "_" in name:
        task, scheme = name.rsplit("_", 1)
        return task.upper(), scheme
    return name, "unknown"


def _score_fields(victim: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
    if "accuracy" in victim or "accuracy" in extracted:
        return {
            "score_name": "accuracy",
            "target_score": victim.get("accuracy"),
            "stolen_score": extracted.get("accuracy"),
        }
    if "f1" in victim or "f1" in extracted:
        return {
            "score_name": "f1",
            "target_score": victim.get("f1"),
            "stolen_score": extracted.get("f1"),
            "target_exact_match": victim.get("exact_match"),
            "stolen_exact_match": extracted.get("exact_match"),
        }
    return {}


def _collect_text_runs(root: Path, *, pipeline: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metrics_dir = run_dir / "metrics"
        victim = _read_json(metrics_dir / "victim_dev_metrics.json")
        extracted = _read_json(metrics_dir / "extracted_dev_metrics.json")
        agreement = _read_json(metrics_dir / "agreement_dev.json")
        if not victim and not extracted and not agreement:
            continue
        task, scheme = _parse_task_scheme(run_dir, agreement)
        resolved = _read_json(run_dir / "resolved_config.json")
        model_name = resolved.get("model", {}).get("pretrained_model_name_or_path")
        query_count = _line_count(run_dir / "data" / "new_train_distill_results.tsv")
        if query_count is None:
            query_count = _line_count(run_dir / "data" / "new_train_distill_probs.tsv")
        if query_count is None:
            query_count = _line_count(run_dir / "data" / "new_train.jsonl")
        if query_count is None:
            query_count = _qa_query_count(run_dir)
        row = {
            "modality": "text",
            "pipeline": pipeline,
            "run_name": run_dir.name,
            "task": task,
            "query_source": scheme,
            "target_model": model_name,
            "student_model": model_name,
            "query_budget": query_count,
            "agreement": agreement.get("agreement", agreement.get("agreement_f1")),
            "agreement_exact_match": agreement.get("agreement_exact_match"),
            "eval_examples": agreement.get("num_examples", victim.get("num_examples")),
        }
        row.update(_score_fields(victim, extracted))
        rows.append(row)
    return rows


def _collect_audio_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for metrics_path in sorted(root.glob("*/metrics/final_metrics.json")):
        metrics = _read_json(metrics_path)
        row = {
            "modality": "audio",
            "pipeline": "encoder",
            "run_name": str(metrics.get("run_name", metrics_path.parents[1].name)),
            "task": "audio_encoder",
            "query_source": metrics.get("query_source_type"),
            "target_model": metrics.get("target_model"),
            "student_model": metrics.get("student_model"),
            "query_budget": metrics.get("query_budget", metrics.get("num_queries")),
            "feature_mse": metrics.get("feature_mse"),
            "feature_cosine": metrics.get("feature_cosine"),
            "agreement": metrics.get("feature_cosine"),
            "target_score": metrics.get("target_accuracy"),
            "stolen_score": metrics.get("stolen_accuracy"),
            "score_name": "downstream_accuracy"
            if "stolen_accuracy" in metrics
            else "feature_cosine",
            "downstream_train_examples": metrics.get("downstream_train_examples"),
            "eval_examples": metrics.get("downstream_eval_examples"),
        }
        rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], output_root: Path, output_name: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "{}.json".format(output_name)
    csv_path = output_root / "{}.csv".format(output_name)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    logger.info("Wrote {} rows to {} and {}", len(rows), json_path, csv_path)


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(_collect_text_runs(Path(args.classifier_root), pipeline="classifier"))
    rows.extend(_collect_text_runs(Path(args.qa_root), pipeline="qa"))
    rows.extend(_collect_audio_runs(Path(args.audio_root)))
    write_outputs(rows, Path(args.output_root), args.output_name)


if __name__ == "__main__":
    main()
