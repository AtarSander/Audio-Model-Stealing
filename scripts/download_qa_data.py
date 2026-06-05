#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

from loguru import logger

SQUAD_URLS = {
    "train-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev-v1.1.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )


def download_file(url: str, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        logger.info("Keeping existing {}", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading {} -> {}", url, output_path)
    request = urllib.request.Request(url, headers={"User-Agent": "modern-bert-extraction"})
    with urllib.request.urlopen(request) as response:
        output_path.write_bytes(response.read())


def download_squad(output_dir: Path, *, force: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in SQUAD_URLS.items():
        download_file(url, output_dir / filename, force=force)


def _load_boolq_hf_dataset():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The `datasets` package is required for BoolQ download. Run `make requirements` first."
        ) from error

    candidates = [
        ("google/boolq", None),
        ("super_glue", "boolq"),
    ]
    errors: list[str] = []
    for dataset_name, config_name in candidates:
        try:
            logger.info(
                "Loading BoolQ from Hugging Face dataset {}{}",
                dataset_name,
                " / {}".format(config_name) if config_name else "",
            )
            if config_name is None:
                return load_dataset(dataset_name)
            return load_dataset(dataset_name, config_name)
        except Exception as error:  # noqa: BLE001
            errors.append("{}: {}".format(dataset_name, error))
    raise RuntimeError("Could not load BoolQ from Hugging Face. Errors: {}".format(errors))


def _normalized_boolq_row(row: dict, index: int) -> dict:
    return {
        "idx": int(row.get("idx", index)),
        "title": str(row.get("title", "")),
        "passage": str(row["passage"]),
        "question": str(row["question"]),
        "answer": bool(row["answer"]),
    }


def download_boolq(output_dir: Path, *, force: bool) -> None:
    train_path = output_dir / "train.jsonl"
    dev_path = output_dir / "dev.jsonl"
    if train_path.exists() and dev_path.exists() and not force:
        logger.info("Keeping existing BoolQ files in {}", output_dir)
        return

    dataset = _load_boolq_hf_dataset()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "train": train_path,
        "validation": dev_path,
    }
    for split_name, output_path in split_map.items():
        rows = [_normalized_boolq_row(row, index) for index, row in enumerate(dataset[split_name])]
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        logger.info("Wrote {} BoolQ {} rows to {}", len(rows), split_name, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad-dir", default="data/raw/squad")
    parser.add_argument("--boolq-dir", default="data/raw/boolq")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["all", "squad", "boolq"],
        default=["all"],
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    tasks = set(args.tasks)
    if "all" in tasks or "squad" in tasks:
        download_squad(Path(args.squad_dir), force=args.force)
    if "all" in tasks or "boolq" in tasks:
        download_boolq(Path(args.boolq_dir), force=args.force)


if __name__ == "__main__":
    main()
