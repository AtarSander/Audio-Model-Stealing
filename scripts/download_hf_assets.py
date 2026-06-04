#!/usr/bin/env python3
"""Download WikiText assets needed by the modern classifier pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

WIKITEXT_SPLIT_TO_FILENAME = {
    "train": "wiki.train.raw",
    "validation": "wiki.valid.raw",
    "test": "wiki.test.raw",
}

WIKITEXT_PARQUET_FILES = {
    "train": [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ],
    "validation": [
        "validation-00000-of-00001.parquet",
    ],
    "test": [
        "test-00000-of-00001.parquet",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    wikitext_parser = subparsers.add_parser("wikitext")
    wikitext_parser.add_argument(
        "--output-dir",
        default="data/raw/wikitext103",
        help="Directory where wiki.train.raw / wiki.valid.raw / wiki.test.raw will be written.",
    )
    wikitext_parser.add_argument(
        "--dataset-repo",
        default="Salesforce/wikitext",
        help="Hugging Face dataset repo to use.",
    )
    wikitext_parser.add_argument(
        "--dataset-name",
        default="wikitext-103-raw-v1",
        help="Dataset configuration name to load from the HF dataset repo.",
    )
    return parser.parse_args()


def iter_text_rows_from_parquet(path: Path):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(columns=["text"]):
        column = batch.column(0)
        for index in range(len(column)):
            value = column[index].as_py()
            if value is None:
                continue
            yield value


def export_wikitext(output_dir: Path, dataset_repo: str, dataset_name: str) -> None:
    from huggingface_hub import hf_hub_download

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, filename in WIKITEXT_SPLIT_TO_FILENAME.items():
        destination = output_dir / filename
        logger.info("Writing {} split to {}", split_name, destination)
        with destination.open("w", encoding="utf-8") as handle:
            for parquet_name in WIKITEXT_PARQUET_FILES[split_name]:
                downloaded = Path(
                    hf_hub_download(
                        repo_id=dataset_repo,
                        repo_type="dataset",
                        filename="{}/{}".format(dataset_name, parquet_name),
                    )
                )
                logger.info("Reading {}", downloaded)
                for text in iter_text_rows_from_parquet(downloaded):
                    handle.write(text)
                    if not text.endswith("\n"):
                        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.command != "wikitext":
        raise SystemExit("Unsupported command: {}".format(args.command))
    export_wikitext(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        dataset_repo=args.dataset_repo,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
