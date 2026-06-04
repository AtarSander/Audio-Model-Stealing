from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from loguru import logger


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
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        header = handle.readline()
        if not header:
            raise ValueError("Missing TSV header in {}".format(path))
        fieldnames = header.rstrip("\r\n").split("\t")
        rows: list[dict[str, str]] = []
        malformed_rows = 0
        for line_number, line in enumerate(handle, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != len(fieldnames):
                malformed_rows += 1
                logger.warning(
                    "Skipping malformed TSV row {} in {}: expected {} fields, got {}",
                    line_number,
                    input_path,
                    len(fieldnames),
                    len(parts),
                )
                continue
            rows.append(dict(zip(fieldnames, parts)))
    return fieldnames, rows


def _sanitize_tsv_cell(value: object) -> str:
    return (
        str(value if value is not None else "")
        .replace("\t", " ")
        .replace("\r", " ")
        .replace("\n", " ")
    )


def write_tsv_rows(
    path: str | Path, fieldnames: Sequence[str], rows: Iterable[dict[str, str]]
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(_sanitize_tsv_cell(field) for field in fieldnames) + "\n")
        for row in rows:
            handle.write(
                "\t".join(_sanitize_tsv_cell(row.get(field, "")) for field in fieldnames) + "\n"
            )


def append_probability_columns(
    rows: Sequence[dict[str, str]],
    probabilities: Sequence[Sequence[float]],
    num_labels: int,
) -> tuple[list[str], list[dict[str, str]]]:
    if len(rows) != len(probabilities):
        raise ValueError(
            "Row/probability count mismatch: {} vs {}".format(len(rows), len(probabilities))
        )

    extra_fields = ["label{}_prob".format(index) for index in range(num_labels)]
    fieldnames = list(rows[0].keys()) + extra_fields if rows else extra_fields
    output_rows: list[dict[str, str]] = []
    for row, probs in zip(rows, probabilities):
        updated = dict(row)
        for index, prob in enumerate(probs):
            updated["label{}_prob".format(index)] = "{:.8f}".format(float(prob))
        output_rows.append(updated)
    return fieldnames, output_rows


def standard_examples_from_rows(
    task: str, rows: Sequence[dict[str, str]], split: str
) -> list[ClassifierExample]:
    task_name = normalize_task_name(task)
    spec = TASK_SPECS[task_name]
    examples: list[ClassifierExample] = []
    skipped_rows = 0
    for index, row in enumerate(rows):
        label_value = (row.get(spec.label_field) or "").strip()
        if split != "test" and label_value not in spec.label_names:
            skipped_rows += 1
            continue
        text_a = row.get(spec.text_fields[0])
        text_b = row.get(spec.text_fields[1]) if len(spec.text_fields) > 1 else None
        if text_a is None or (len(spec.text_fields) > 1 and text_b is None):
            skipped_rows += 1
            continue
        guid = "{}-{}".format(split, row.get("index", index))
        label = spec.label_to_id(label_value) if split != "test" else None
        examples.append(ClassifierExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    if skipped_rows:
        logger.warning(
            "Skipped {} malformed/unlabeled {} rows for split {}",
            skipped_rows,
            task_name,
            split,
        )
    return examples


def distilled_examples_from_rows(
    task: str, rows: Sequence[dict[str, str]], split: str
) -> list[ClassifierExample]:
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
