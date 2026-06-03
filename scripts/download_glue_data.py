#!/usr/bin/env python3
"""Download and unpack GLUE data into the repo-local layout.

This is a repo-local version of the long-used GLUE downloader script. It keeps
the logic self-contained and adds a few aliases so the output layout matches
what the paper reproduction code expects, e.g. `SST-2/` and `MNLI/`.
"""

from __future__ import annotations

import argparse
from loguru import logger
from pathlib import Path
import tempfile
import urllib.request
import zipfile


TASK_URLS = {
    "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "SST": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "QQP": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "STS": "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
    "MNLI": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "WNLI": "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
    "diagnostic": "https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
}

MRPC_TRAIN_URL = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST_URL = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"
MRPC_DEV_IDS_URL = (
    "https://raw.githubusercontent.com/MegEngine/Models/master/"
    "official/nlp/bert/glue_data/MRPC/dev_ids.tsv"
)

ALIASES = {
    "SST-2": "SST",
    "STS-B": "STS",
    "AX": "diagnostic",
}

DEFAULT_TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="external/repro/glue",
        help="Directory where the GLUE task folders will be written.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["all"],
        help=(
            "Tasks to download. Use `all` or a subset such as `MNLI SST-2`. "
            "Accepted aliases: `SST-2`, `STS-B`, `AX`."
        ),
    )
    parser.add_argument(
        "--mrpc-source-dir",
        default=None,
        help=(
            "Optional directory containing `msr_paraphrase_train.txt` and "
            "`msr_paraphrase_test.txt`. If omitted, the script downloads them."
        ),
    )
    return parser.parse_args()


def normalize_tasks(requested: list[str]) -> list[str]:
    if len(requested) == 1 and requested[0].lower() == "all":
        return list(DEFAULT_TASKS)

    tasks = []
    for task in requested:
        canonical = ALIASES.get(task, task)
        if canonical not in DEFAULT_TASKS:
            raise SystemExit("Unsupported GLUE task: {}".format(task))
        tasks.append(canonical)
    return tasks


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        logger.info("Using existing file {}", destination)
        return
    logger.info("Downloading {} -> {}", url, destination)
    urllib.request.urlretrieve(url, destination)


def extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(destination)


def download_and_extract_task(task: str, data_dir: Path) -> None:
    if task == "diagnostic":
        mnli_dir = data_dir / "MNLI"
        mnli_dir.mkdir(parents=True, exist_ok=True)
        output_path = mnli_dir / "diagnostic.tsv"
        download_file(TASK_URLS[task], output_path)
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "{}.zip".format(task)
        download_file(TASK_URLS[task], zip_path)
        extract_zip(zip_path, data_dir)


def read_mrpc_dev_ids(path: Path) -> set[tuple[str, str]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return {
            (parts[0], parts[1])
            for line in handle
            if (parts := line.rstrip("\n").split("\t")) and len(parts) >= 2
        }


def read_mrpc_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = [line.rstrip("\n") for line in handle]

    if not lines:
        raise SystemExit("Empty MRPC file: {}".format(path))

    header = lines[0].split("\t")
    rows = [line.split("\t") for line in lines[1:] if line]
    malformed = [row for row in rows if len(row) != 5]
    if malformed:
        raise SystemExit(
            "Malformed MRPC rows found in {}. Expected 5 tab-separated columns, got sample: {}".format(
                path, malformed[0]
            )
        )
    return header, rows


def write_mrpc_split(
    source_path: Path,
    destination_path: Path,
    dev_ids: set[tuple[str, str]],
    split_name: str,
) -> None:
    header, rows = read_mrpc_rows(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as dest_handle:
        dest_handle.write("\t".join(header) + "\n")
        for row in rows:
            pair_id = (row[1], row[2])
            is_dev = pair_id in dev_ids
            if split_name == "dev" and is_dev:
                dest_handle.write("\t".join(row) + "\n")
            elif split_name == "train" and not is_dev:
                dest_handle.write("\t".join(row) + "\n")


def write_mrpc_test(source_path: Path, destination_path: Path) -> None:
    _, rows = read_mrpc_rows(source_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8") as dest_handle:
        dest_handle.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for index, row in enumerate(rows):
            dest_handle.write(
                "{}\t{}\t{}\t{}\t{}\n".format(index, row[1], row[2], row[3], row[4])
            )


def prepare_mrpc(data_dir: Path, mrpc_source_dir: str | None) -> None:
    mrpc_dir = data_dir / "MRPC"
    mrpc_dir.mkdir(parents=True, exist_ok=True)

    if mrpc_source_dir is None:
        train_source = mrpc_dir / "msr_paraphrase_train.txt"
        test_source = mrpc_dir / "msr_paraphrase_test.txt"
        dev_ids_path = mrpc_dir / "dev_ids.tsv"
        download_file(MRPC_TRAIN_URL, train_source)
        download_file(MRPC_TEST_URL, test_source)
        download_file(MRPC_DEV_IDS_URL, dev_ids_path)
    else:
        source_dir = Path(mrpc_source_dir).expanduser().resolve()
        train_source = source_dir / "msr_paraphrase_train.txt"
        test_source = source_dir / "msr_paraphrase_test.txt"
        dev_ids_path = source_dir / "dev_ids.tsv"
        if not dev_ids_path.exists():
            download_file(MRPC_DEV_IDS_URL, mrpc_dir / "dev_ids.tsv")
            dev_ids_path = mrpc_dir / "dev_ids.tsv"

    if not train_source.exists():
        raise SystemExit("Missing MRPC train source: {}".format(train_source))
    if not test_source.exists():
        raise SystemExit("Missing MRPC test source: {}".format(test_source))
    if not dev_ids_path.exists():
        raise SystemExit("Missing MRPC dev_ids.tsv: {}".format(dev_ids_path))

    dev_ids = read_mrpc_dev_ids(dev_ids_path)
    write_mrpc_split(train_source, mrpc_dir / "train.tsv", dev_ids, "train")
    write_mrpc_split(train_source, mrpc_dir / "dev.tsv", dev_ids, "dev")
    write_mrpc_test(test_source, mrpc_dir / "test.tsv")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    tasks = normalize_tasks(args.tasks)
    for task in tasks:
        if task == "MRPC":
            logger.info("Preparing MRPC")
            prepare_mrpc(data_dir, args.mrpc_source_dir)
        else:
            logger.info("Downloading and extracting {}", task)
            download_and_extract_task(task, data_dir)

    logger.info("Done. GLUE data available in {}", data_dir)


if __name__ == "__main__":
    main()
