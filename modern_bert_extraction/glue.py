from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, Sequence


def _set_max_csv_field_size() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


_set_max_csv_field_size()


@dataclass
class TaskSpec:
    name: str
    num_labels: int
    label_names: list[str]
    text_fields: tuple[str, ...]
    train_filename: str
    dev_filename: str
    label_field: str

    def label_to_id(self, label: str) -> int:
        return self.label_names.index(label)


TASK_SPECS = {
    "sst-2": TaskSpec(
        name="sst-2",
        num_labels=2,
        label_names=["0", "1"],
        text_fields=("sentence",),
        train_filename="train.tsv",
        dev_filename="dev.tsv",
        label_field="label",
    ),
    "mnli": TaskSpec(
        name="mnli",
        num_labels=3,
        label_names=["contradiction", "entailment", "neutral"],
        text_fields=("sentence1", "sentence2"),
        train_filename="train.tsv",
        dev_filename="dev_matched.tsv",
        label_field="gold_label",
    ),
}


@dataclass
class ClassifierExample:
    guid: str
    text_a: str
    text_b: str | None
    label: int | None = None
    soft_labels: list[float] | None = None


def normalize_task_name(task: str) -> str:
    lowered = task.strip().lower()
    if lowered in {"sst2", "sst-2"}:
        return "sst-2"
    if lowered == "mnli":
        return "mnli"
    raise ValueError("Unsupported task: {}".format(task))


def task_data_dir(glue_dir: str | Path, task: str) -> Path:
    task_name = normalize_task_name(task)
    folder = "SST-2" if task_name == "sst-2" else "MNLI"
    return Path(glue_dir) / folder


def read_tsv_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Missing TSV header in {}".format(path))
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames), rows


def write_tsv_rows(path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_probability_columns(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[Sequence[float]],
    num_labels: int,
) -> tuple[list[str], list[dict[str, str]]]:
    if len(rows) != len(probabilities):
        raise ValueError("Row/probability count mismatch: {} vs {}".format(len(rows), len(probabilities)))

    extra_fields = ["label{}_prob".format(index) for index in range(num_labels)]
    fieldnames = list(rows[0].keys()) + extra_fields if rows else extra_fields
    output_rows: list[dict[str, str]] = []
    for row, probs in zip(rows, probabilities):
        updated = dict(row)
        for index, prob in enumerate(probs):
            updated["label{}_prob".format(index)] = "{:.8f}".format(float(prob))
        output_rows.append(updated)
    return fieldnames, output_rows


def standard_examples_from_rows(task: str, rows: Sequence[dict[str, str]], split: str) -> list[ClassifierExample]:
    task_name = normalize_task_name(task)
    spec = TASK_SPECS[task_name]
    examples: list[ClassifierExample] = []
    for index, row in enumerate(rows):
        if task_name == "mnli" and row.get(spec.label_field, "").strip() == "-":
            continue
        text_a = row[spec.text_fields[0]]
        text_b = row[spec.text_fields[1]] if len(spec.text_fields) > 1 else None
        guid = "{}-{}".format(split, row.get("index", index))
        label = spec.label_to_id(row[spec.label_field]) if split != "test" else None
        examples.append(ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def distilled_examples_from_rows(task: str, rows: Sequence[dict[str, str]], split: str) -> list[ClassifierExample]:
    task_name = normalize_task_name(task)
    spec = TASK_SPECS[task_name]
    examples: list[ClassifierExample] = []
    prob_fields = ["label{}_prob".format(index) for index in range(spec.num_labels)]
    for index, row in enumerate(rows):
        text_a = row[spec.text_fields[0]]
        text_b = row[spec.text_fields[1]] if len(spec.text_fields) > 1 else None
        soft_labels = [float(row[field]) for field in prob_fields]
        label = max(range(len(soft_labels)), key=lambda i: soft_labels[i])
        guid = "{}-{}".format(split, row.get("index", index))
        examples.append(
            ClassifierExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label,
                soft_labels=soft_labels,
            )
        )
    return examples


def probability_lines(probabilities: Sequence[Sequence[float]]) -> list[str]:
    return ["\t".join("{:.8f}".format(float(prob)) for prob in row) for row in probabilities]
