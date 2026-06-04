from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
import numpy as np
import torch

from modern_bert_extraction.glue import ClassifierExample, probability_lines
from modern_bert_extraction.qa_generation import (
    generate_boolq_queries,
    load_or_prepare_wikitext_paragraphs,
)
from modern_bert_extraction.training import (
    predict_probabilities,
    resolve_device,
    save_json,
    train_model,
)

BOOLQ_NUM_LABELS = 2


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def boolq_examples_from_rows(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    use_soft_labels: bool,
) -> list[ClassifierExample]:
    examples: list[ClassifierExample] = []
    for index, row in enumerate(rows):
        label = 0 if bool(row.get("answer", False)) else 1
        soft_labels = None
        if use_soft_labels and "soft_answer" in row:
            soft_labels = [float(value) for value in row["soft_answer"]]
            label = int(np.argmax(soft_labels))
        examples.append(
            ClassifierExample(
                guid="{}-{}".format(split, row.get("idx", index)),
                text_a=str(row["passage"]),
                text_b=str(row["question"]),
                label=label,
                soft_labels=soft_labels,
            )
        )
    return examples


def boolq_accuracy(rows: Sequence[dict[str, Any]], probabilities: np.ndarray) -> dict[str, float]:
    gold = np.array([0 if bool(row.get("answer", False)) else 1 for row in rows], dtype=np.int64)
    predicted = probabilities.argmax(axis=-1)
    accuracy = float((predicted == gold).mean()) if len(gold) else 0.0
    return {
        "accuracy": accuracy,
        "num_examples": float(len(gold)),
    }


def boolq_agreement(
    *,
    victim_probabilities: np.ndarray,
    extracted_probabilities: np.ndarray,
) -> dict[str, float]:
    if len(victim_probabilities) != len(extracted_probabilities):
        raise ValueError(
            "Victim/extracted probability count mismatch: {} vs {}".format(
                len(victim_probabilities),
                len(extracted_probabilities),
            )
        )
    if len(victim_probabilities) == 0:
        return {"agreement": 0.0, "num_examples": 0.0}

    eps = 1.0e-12
    victim_safe = np.clip(victim_probabilities, eps, 1.0)
    extracted_safe = np.clip(extracted_probabilities, eps, 1.0)
    agreement = float(
        (victim_probabilities.argmax(axis=-1) == extracted_probabilities.argmax(axis=-1)).mean()
    )
    l2 = np.linalg.norm(victim_probabilities - extracted_probabilities, axis=-1)
    kld_victim_to_extracted = np.sum(
        victim_safe * (np.log(victim_safe) - np.log(extracted_safe)),
        axis=-1,
    )
    return {
        "agreement": agreement,
        "l2_mean": float(l2.mean()),
        "l2_max": float(l2.max()),
        "l2_min": float(l2.min()),
        "l2_std": float(l2.std()),
        "kld_victim_to_extracted_mean": float(kld_victim_to_extracted.mean()),
        "num_examples": float(len(victim_probabilities)),
    }


class BoolQExtractionPipeline:
    def __init__(self, config: dict, scheme: str):
        self.config = config
        self.scheme = scheme.lower()
        if self.scheme not in {"random", "wiki"}:
            raise ValueError("Unsupported scheme: {}".format(scheme))

        self.paths = config["paths"]
        self.model_cfg = config["model"]
        self.training_cfg = config["training"]
        self.query_cfg = config["query_generation"]
        self.runtime_cfg = config["runtime"]
        self.seed = int(config.get("seed", 42))

        self.run_name = "boolq_{}".format(self.scheme)
        self.run_root = Path(self.paths["output_root"]) / self.run_name
        configured_checkpoint_root = self.paths.get("checkpoint_root")
        self.checkpoint_run_root = (
            Path(configured_checkpoint_root) / self.run_name
            if configured_checkpoint_root
            else self.run_root
        )
        self.data_dir = self.run_root / "data"
        configured_victim_dir = self.paths.get("victim_model_dir")
        self.reused_victim_dir = Path(configured_victim_dir) if configured_victim_dir else None
        self.victim_dir = self.reused_victim_dir or self.checkpoint_run_root / "victim_model"
        self.extracted_dir = self.checkpoint_run_root / "extracted_model"
        self.metrics_dir = self.run_root / "metrics"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_run_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.train_jsonl = Path(self.paths["boolq_dir"]) / "train.jsonl"
        self.dev_jsonl = Path(self.paths["boolq_dir"]) / "dev.jsonl"
        self.wikitext_paragraphs_path = Path(self.paths["wikitext_paragraphs"])
        self.query_jsonl = self.data_dir / "new_train.jsonl"
        self.query_probs_tsv = self.data_dir / "new_train_distill_probs.tsv"
        self.distilled_train_jsonl = self.data_dir / "train.jsonl"
        self.victim_dev_probs_tsv = self.metrics_dir / "victim_dev_probabilities.tsv"
        self.extracted_dev_probs_tsv = self.metrics_dir / "extracted_dev_probabilities.tsv"

    def preflight(self) -> None:
        device = resolve_device(self.runtime_cfg)
        logger.info("Using device {}", device)
        if device.type == "cuda":
            logger.info("GPU {}", torch.cuda.get_device_name(device))
        if self.reused_victim_dir is not None:
            if not self.reused_victim_dir.exists():
                raise FileNotFoundError(
                    "Configured victim checkpoint does not exist: {}".format(
                        self.reused_victim_dir
                    )
                )
            logger.info("Reusing victim checkpoint from {}", self.reused_victim_dir)
        save_json(self.run_root / "resolved_config.json", self.config)
        logger.info("Saved resolved config to {}", self.run_root / "resolved_config.json")

    def prepare_wikitext(self) -> list[str]:
        return load_or_prepare_wikitext_paragraphs(
            raw_path=self.paths["wikitext_raw"],
            paragraphs_path=self.wikitext_paragraphs_path,
        )

    def _load_train_rows(self) -> list[dict[str, Any]]:
        if not self.train_jsonl.exists():
            raise FileNotFoundError(
                "BoolQ train data not found at {}. Run `make download_qa_data` first.".format(
                    self.train_jsonl
                )
            )
        return read_jsonl(self.train_jsonl)

    def _load_dev_rows(self) -> list[dict[str, Any]]:
        if not self.dev_jsonl.exists():
            raise FileNotFoundError(
                "BoolQ dev data not found at {}. Run `make download_qa_data` first.".format(
                    self.dev_jsonl
                )
            )
        return read_jsonl(self.dev_jsonl)

    def train_victim(self) -> None:
        dev_rows = self._load_dev_rows()
        if self.reused_victim_dir is not None:
            logger.info(
                "Skipping victim training; evaluating reused checkpoint {}", self.victim_dir
            )
            dev_examples = boolq_examples_from_rows(
                dev_rows,
                split="dev",
                use_soft_labels=False,
            )
            probabilities = predict_probabilities(
                model_dir_or_name=self.victim_dir,
                examples=dev_examples,
                num_labels=BOOLQ_NUM_LABELS,
                training_cfg=self.training_cfg,
                model_cfg=self.model_cfg,
                runtime_cfg=self.runtime_cfg,
            )
            self.victim_dev_probs_tsv.write_text(
                "\n".join(probability_lines(probabilities)) + "\n",
                encoding="utf-8",
            )
            metrics = boolq_accuracy(dev_rows, probabilities)
            metrics["source_checkpoint"] = str(self.victim_dir)
            save_json(self.metrics_dir / "victim_dev_metrics.json", metrics)
            return

        train_rows = self._load_train_rows()
        artifacts = train_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.victim_dir,
            train_examples=boolq_examples_from_rows(
                train_rows,
                split="train",
                use_soft_labels=False,
            ),
            eval_examples=boolq_examples_from_rows(
                dev_rows,
                split="dev",
                use_soft_labels=False,
            ),
            num_labels=BOOLQ_NUM_LABELS,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "victim_dev_metrics.json", artifacts.metrics)

    def generate_queries(self) -> None:
        train_rows = self._load_train_rows()
        thief_paragraphs = self.prepare_wikitext()
        thief_paragraph_limit = self.query_cfg.get("thief_paragraph_limit")
        if thief_paragraph_limit is not None:
            thief_paragraphs = thief_paragraphs[: int(thief_paragraph_limit)]
            logger.info("Using first {} WikiText paragraphs", len(thief_paragraphs))
        query_rows = generate_boolq_queries(
            source_rows=train_rows,
            thief_paragraphs=thief_paragraphs,
            scheme=self.scheme,
            question_sampling_scheme=self.query_cfg["question_sampling_scheme"],
            augmentations=int(self.query_cfg.get("augmentations", 1)),
            seed=self.seed,
            dataset_size=self.query_cfg.get("dataset_size"),
        )
        write_jsonl(self.query_jsonl, query_rows)
        logger.info("Wrote {} BoolQ queries to {}", len(query_rows), self.query_jsonl)

    def query_victim(self) -> None:
        query_rows = read_jsonl(self.query_jsonl)
        query_examples = boolq_examples_from_rows(
            query_rows,
            split="query",
            use_soft_labels=False,
        )
        probabilities = predict_probabilities(
            model_dir_or_name=self.victim_dir,
            examples=query_examples,
            num_labels=BOOLQ_NUM_LABELS,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        self.query_probs_tsv.write_text(
            "\n".join(probability_lines(probabilities)) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote victim BoolQ query probabilities to {}", self.query_probs_tsv)

    def build_distill_data(self) -> None:
        query_rows = read_jsonl(self.query_jsonl)
        probabilities = np.array(
            [
                [float(value) for value in line.split("\t")]
                for line in self.query_probs_tsv.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ],
            dtype=np.float64,
        )
        if len(query_rows) != len(probabilities):
            raise ValueError(
                "Query/probability count mismatch: {} vs {}".format(
                    len(query_rows),
                    len(probabilities),
                )
            )

        distilled_rows: list[dict[str, Any]] = []
        for row, probs in zip(query_rows, probabilities):
            updated = dict(row)
            updated["soft_answer"] = [float(probs[0]), float(probs[1])]
            updated["answer"] = bool(int(np.argmax(probs)) == 0)
            distilled_rows.append(updated)
        write_jsonl(self.distilled_train_jsonl, distilled_rows)
        logger.info(
            "Wrote distilled BoolQ training data with {} rows to {}",
            len(distilled_rows),
            self.distilled_train_jsonl,
        )

    def train_extracted(self) -> None:
        train_rows = read_jsonl(self.distilled_train_jsonl)
        dev_rows = self._load_dev_rows()
        artifacts = train_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.extracted_dir,
            train_examples=boolq_examples_from_rows(
                train_rows,
                split="train",
                use_soft_labels=True,
            ),
            eval_examples=boolq_examples_from_rows(
                dev_rows,
                split="dev",
                use_soft_labels=False,
            ),
            num_labels=BOOLQ_NUM_LABELS,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "extracted_dev_metrics.json", artifacts.metrics)

    def evaluate_agreement(self) -> None:
        dev_rows = self._load_dev_rows()
        dev_examples = boolq_examples_from_rows(
            dev_rows,
            split="dev",
            use_soft_labels=False,
        )
        victim_probs = predict_probabilities(
            model_dir_or_name=self.victim_dir,
            examples=dev_examples,
            num_labels=BOOLQ_NUM_LABELS,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        extracted_probs = predict_probabilities(
            model_dir_or_name=self.extracted_dir,
            examples=dev_examples,
            num_labels=BOOLQ_NUM_LABELS,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        self.victim_dev_probs_tsv.write_text(
            "\n".join(probability_lines(victim_probs)) + "\n",
            encoding="utf-8",
        )
        self.extracted_dev_probs_tsv.write_text(
            "\n".join(probability_lines(extracted_probs)) + "\n",
            encoding="utf-8",
        )
        save_json(
            self.metrics_dir / "victim_dev_metrics_from_probs.json",
            boolq_accuracy(dev_rows, victim_probs),
        )
        save_json(
            self.metrics_dir / "extracted_dev_metrics_from_probs.json",
            boolq_accuracy(dev_rows, extracted_probs),
        )
        save_json(
            self.metrics_dir / "agreement_dev.json",
            boolq_agreement(
                victim_probabilities=victim_probs,
                extracted_probabilities=extracted_probs,
            ),
        )
        logger.info("Saved BoolQ agreement metrics to {}", self.metrics_dir / "agreement_dev.json")

    def run(self, step: str) -> None:
        if step == "all":
            ordered_steps = [
                "preflight",
                "prepare_wikitext",
                "train_victim",
                "generate_queries",
                "query_victim",
                "build_distill_data",
                "train_extracted",
                "evaluate_agreement",
            ]
        else:
            ordered_steps = [step]
        for step_name in ordered_steps:
            logger.info("Starting step '{}'", step_name)
            getattr(self, step_name)()
