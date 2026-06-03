from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from loguru import logger
import numpy as np
from omegaconf import OmegaConf
import torch

from modern_bert_extraction.glue import (
    TASK_SPECS,
    append_probability_columns,
    distilled_examples_from_rows,
    normalize_task_name,
    probability_lines,
    read_tsv_rows,
    standard_examples_from_rows,
    task_data_dir,
    write_tsv_rows,
)
from modern_bert_extraction.query_generation import (
    build_top_k_vocab,
    generate_queries,
    load_or_prepare_wikitext_sentences,
)
from modern_bert_extraction.training import (
    predict_probabilities,
    resolve_device,
    save_json,
    train_model,
)


def load_config(path: str | Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def _task_display_name(task_name: str) -> str:
    return "SST-2" if task_name == "sst-2" else "MNLI"


class ClassifierExtractionPipeline:
    def __init__(self, config: dict, task: str, scheme: str):
        self.config = config
        self.task_name = normalize_task_name(task)
        self.scheme = scheme.lower()
        if self.scheme not in {"random", "wiki"}:
            raise ValueError("Unsupported scheme: {}".format(scheme))
        self.spec = TASK_SPECS[self.task_name]
        self.paths = config["paths"]
        self.model_cfg = config["model"]
        self.training_cfg = config["training"]
        self.query_cfg = config["query_generation"]
        self.runtime_cfg = config["runtime"]
        self.seed = int(config.get("seed", 42))

        task_slug = self.task_name.replace("-", "")
        self.run_root = Path(self.paths["output_root"]) / "{}_{}".format(task_slug, self.scheme)
        self.data_dir = self.run_root / "data"
        self.victim_dir = self.run_root / "victim_model"
        self.extracted_dir = self.run_root / "extracted_model"
        self.metrics_dir = self.run_root / "metrics"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.base_task_dir = task_data_dir(self.paths["glue_dir"], self.task_name)
        self.query_tsv = self.data_dir / "new_train_sents.tsv"
        self.query_probs_tsv = self.data_dir / "new_train_distill_results.tsv"
        self.distilled_train_tsv = self.data_dir / "train.tsv"
        self.distilled_dev_tsv = self.data_dir / self.spec.dev_filename
        self.wikitext_sentences_path = Path(self.paths["wikitext_sentences"])
        self.victim_dev_probs_tsv = self.metrics_dir / "victim_dev_probabilities.tsv"
        self.extracted_dev_probs_tsv = self.metrics_dir / "extracted_dev_probabilities.tsv"

    def preflight(self) -> None:
        device = resolve_device(self.runtime_cfg)
        logger.info("Using device {}", device)
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            logger.info("GPU {}", gpu_name)
        save_json(self.run_root / "resolved_config.json", self.config)
        logger.info("Saved resolved config to {}", self.run_root / "resolved_config.json")

    def prepare_wikitext(self) -> list[str]:
        sentences = load_or_prepare_wikitext_sentences(
            raw_path=self.paths["wikitext_raw"],
            sentences_path=self.wikitext_sentences_path,
        )
        logger.info(
            "Prepared {} WikiText sentences at {}",
            len(sentences),
            self.wikitext_sentences_path,
        )
        return sentences

    def train_victim(self) -> None:
        _, train_rows = read_tsv_rows(self.base_task_dir / self.spec.train_filename)
        _, dev_rows = read_tsv_rows(self.base_task_dir / self.spec.dev_filename)
        train_examples = standard_examples_from_rows(self.task_name, train_rows, split="train")
        dev_examples = standard_examples_from_rows(self.task_name, dev_rows, split="dev")
        logger.info(
            "Training victim model on {} train examples and {} dev examples",
            len(train_examples),
            len(dev_examples),
        )
        artifacts = train_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.victim_dir,
            train_examples=train_examples,
            eval_examples=dev_examples,
            num_labels=self.spec.num_labels,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "victim_dev_metrics.json", artifacts.metrics)
        logger.info(
            "Victim metrics saved to {}",
            self.metrics_dir / "victim_dev_metrics.json",
        )

    def generate_queries(self) -> None:
        fieldnames, train_rows = read_tsv_rows(self.base_task_dir / self.spec.train_filename)
        thief_sentences = self.prepare_wikitext()
        vocab = build_top_k_vocab(thief_sentences, top_k=int(self.query_cfg["top_k_vocab"]))
        query_rows = generate_queries(
            task=self.task_name,
            scheme=self.scheme,
            base_rows=train_rows,
            thief_sentences=thief_sentences,
            vocab=vocab,
            max_query_length=int(self.query_cfg["max_query_length"]),
            thief_sentence_threshold=int(self.query_cfg["thief_sentence_threshold"]),
            ed1_changes=int(self.query_cfg["ed1_changes"]),
            dataset_size=self.query_cfg.get("dataset_size"),
            augmentations=int(self.query_cfg.get("augmentations", 1)),
            sanitize_samples=bool(self.query_cfg.get("sanitize_samples", True)),
            seed=self.seed,
        )
        write_tsv_rows(self.query_tsv, fieldnames, query_rows)
        logger.info("Wrote {} generated queries to {}", len(query_rows), self.query_tsv)

    def query_victim(self) -> None:
        _, query_rows = read_tsv_rows(self.query_tsv)
        query_examples = standard_examples_from_rows(self.task_name, query_rows, split="test")
        probabilities = predict_probabilities(
            model_dir_or_name=self.victim_dir,
            examples=query_examples,
            num_labels=self.spec.num_labels,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        self.query_probs_tsv.write_text("\n".join(probability_lines(probabilities)) + "\n", encoding="utf-8")
        logger.info(
            "Wrote victim probability outputs for {} examples to {}",
            len(probabilities),
            self.query_probs_tsv,
        )

    def build_distill_data(self) -> None:
        query_fieldnames, query_rows = read_tsv_rows(self.query_tsv)
        probabilities = [
            [float(value) for value in line.split("\t")]
            for line in self.query_probs_tsv.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_fieldnames, distilled_train_rows = append_probability_columns(
            query_rows, probabilities, self.spec.num_labels
        )
        write_tsv_rows(self.distilled_train_tsv, train_fieldnames, distilled_train_rows)
        logger.info(
            "Wrote distilled training set with {} rows to {}",
            len(distilled_train_rows),
            self.distilled_train_tsv,
        )

        dev_fieldnames, dev_rows = read_tsv_rows(self.base_task_dir / self.spec.dev_filename)
        one_hot_dev_probs = []
        for example in standard_examples_from_rows(self.task_name, dev_rows, split="dev"):
            vector = [0.0] * self.spec.num_labels
            assert example.label is not None
            vector[example.label] = 1.0
            one_hot_dev_probs.append(vector)
        dev_fieldnames, distilled_dev_rows = append_probability_columns(
            dev_rows, one_hot_dev_probs, self.spec.num_labels
        )
        write_tsv_rows(self.distilled_dev_tsv, dev_fieldnames, distilled_dev_rows)
        logger.info(
            "Wrote distilled dev set with {} rows to {}",
            len(distilled_dev_rows),
            self.distilled_dev_tsv,
        )

    def train_extracted(self) -> None:
        _, train_rows = read_tsv_rows(self.distilled_train_tsv)
        _, dev_rows = read_tsv_rows(self.distilled_dev_tsv)
        train_examples = distilled_examples_from_rows(self.task_name, train_rows, split="train")
        dev_examples = distilled_examples_from_rows(self.task_name, dev_rows, split="dev")
        logger.info(
            "Training extracted model on {} train examples and {} dev examples",
            len(train_examples),
            len(dev_examples),
        )
        artifacts = train_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.extracted_dir,
            train_examples=train_examples,
            eval_examples=dev_examples,
            num_labels=self.spec.num_labels,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "extracted_dev_metrics.json", artifacts.metrics)
        logger.info(
            "Extracted-model metrics saved to {}",
            self.metrics_dir / "extracted_dev_metrics.json",
        )

    def evaluate_agreement(self) -> None:
        _, dev_rows = read_tsv_rows(self.base_task_dir / self.spec.dev_filename)
        dev_examples = standard_examples_from_rows(self.task_name, dev_rows, split="dev")
        victim_probs = predict_probabilities(
            model_dir_or_name=self.victim_dir,
            examples=dev_examples,
            num_labels=self.spec.num_labels,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        extracted_probs = predict_probabilities(
            model_dir_or_name=self.extracted_dir,
            examples=dev_examples,
            num_labels=self.spec.num_labels,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
        )

        self.victim_dev_probs_tsv.write_text(
            "\n".join(probability_lines(victim_probs)) + "\n", encoding="utf-8"
        )
        self.extracted_dev_probs_tsv.write_text(
            "\n".join(probability_lines(extracted_probs)) + "\n", encoding="utf-8"
        )

        eps = 1.0e-12
        victim_safe = np.clip(victim_probs, eps, 1.0)
        extracted_safe = np.clip(extracted_probs, eps, 1.0)
        agreement = float(
            (victim_probs.argmax(axis=-1) == extracted_probs.argmax(axis=-1)).mean()
        )
        l2 = np.linalg.norm(victim_probs - extracted_probs, axis=-1)
        kld_victim_to_extracted = np.sum(
            victim_safe * (np.log(victim_safe) - np.log(extracted_safe)), axis=-1
        )
        kld_extracted_to_victim = np.sum(
            extracted_safe * (np.log(extracted_safe) - np.log(victim_safe)), axis=-1
        )
        metrics = {
            "task": _task_display_name(self.task_name),
            "scheme": self.scheme,
            "num_examples": int(len(victim_probs)),
            "agreement": agreement,
            "l2_mean": float(l2.mean()),
            "l2_max": float(l2.max()),
            "l2_min": float(l2.min()),
            "l2_std": float(l2.std()),
            "kld_victim_to_extracted_mean": float(kld_victim_to_extracted.mean()),
            "kld_extracted_to_victim_mean": float(kld_extracted_to_victim.mean()),
        }
        save_json(self.metrics_dir / "agreement_dev.json", metrics)
        logger.info("Agreement metrics saved to {}", self.metrics_dir / "agreement_dev.json")

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
