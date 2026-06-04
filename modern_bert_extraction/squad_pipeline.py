from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import torch

from modern_bert_extraction.qa_generation import (
    combine_squad_predictions_with_queries,
    generate_squad_queries,
    load_or_prepare_wikitext_paragraphs,
)
from modern_bert_extraction.qa_training import (
    evaluate_prediction_agreement,
    evaluate_squad_predictions,
    predict_squad,
    read_squad_json,
    squad_examples_from_json,
    train_qa_model,
    write_squad_json,
)
from modern_bert_extraction.training import resolve_device, save_json


class SquadExtractionPipeline:
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

        self.run_name = "squad11_{}".format(self.scheme)
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

        self.train_json = Path(self.paths["squad_dir"]) / "train-v1.1.json"
        self.dev_json = Path(self.paths["squad_dir"]) / "dev-v1.1.json"
        self.wikitext_paragraphs_path = Path(self.paths["wikitext_paragraphs"])
        self.query_json = self.data_dir / "new_train.json"
        self.query_predictions_json = self.data_dir / "new_train_victim_predictions.json"
        self.distilled_train_json = self.data_dir / "train-v1.1.json"
        self.victim_dev_predictions_json = self.metrics_dir / "victim_dev_predictions.json"
        self.extracted_dev_predictions_json = self.metrics_dir / "extracted_dev_predictions.json"

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

    def _load_train_data(self) -> dict:
        if not self.train_json.exists():
            raise FileNotFoundError(
                "SQuAD train data not found at {}. Run `make download_qa_data` first.".format(
                    self.train_json
                )
            )
        return read_squad_json(self.train_json)

    def _load_dev_data(self) -> dict:
        if not self.dev_json.exists():
            raise FileNotFoundError(
                "SQuAD dev data not found at {}. Run `make download_qa_data` first.".format(
                    self.dev_json
                )
            )
        return read_squad_json(self.dev_json)

    def train_victim(self) -> None:
        dev_examples = squad_examples_from_json(self._load_dev_data())
        if self.reused_victim_dir is not None:
            logger.info(
                "Skipping victim training; evaluating reused checkpoint {}", self.victim_dir
            )
            predictions = predict_squad(
                model_dir_or_name=self.victim_dir,
                examples=dev_examples,
                model_cfg=self.model_cfg,
                training_cfg=self.training_cfg,
                runtime_cfg=self.runtime_cfg,
            )
            metrics = evaluate_squad_predictions(dev_examples, predictions)
            metrics["source_checkpoint"] = str(self.victim_dir)
            save_json(self.metrics_dir / "victim_dev_metrics.json", metrics)
            self.victim_dev_predictions_json.write_text(
                json.dumps(predictions, indent=2),
                encoding="utf-8",
            )
            return

        train_examples = squad_examples_from_json(self._load_train_data())
        artifacts = train_qa_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.victim_dir,
            train_examples=train_examples,
            eval_examples=dev_examples,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "victim_dev_metrics.json", artifacts.metrics)

    def generate_queries(self) -> None:
        train_data = self._load_train_data()
        thief_paragraphs = self.prepare_wikitext()
        thief_paragraph_limit = self.query_cfg.get("thief_paragraph_limit")
        if thief_paragraph_limit is not None:
            thief_paragraphs = thief_paragraphs[: int(thief_paragraph_limit)]
            logger.info("Using first {} WikiText paragraphs", len(thief_paragraphs))
        fraction = self.query_cfg.get("fraction")
        generated = generate_squad_queries(
            source_data=train_data,
            thief_paragraphs=thief_paragraphs,
            scheme=self.scheme,
            question_sampling_scheme=self.query_cfg["question_sampling_scheme"],
            augmentations=int(self.query_cfg.get("augmentations", 1)),
            seed=self.seed,
            fraction=float(fraction) if fraction is not None else None,
            dataset_size=self.query_cfg.get("dataset_size"),
        )
        write_squad_json(self.query_json, generated)
        logger.info("Wrote SQuAD extraction queries to {}", self.query_json)

    def query_victim(self) -> None:
        query_examples = squad_examples_from_json(read_squad_json(self.query_json))
        predictions = predict_squad(
            model_dir_or_name=self.victim_dir,
            examples=query_examples,
            model_cfg=self.model_cfg,
            training_cfg=self.training_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        self.query_predictions_json.write_text(
            json.dumps(predictions, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote victim SQuAD query predictions to {}", self.query_predictions_json)

    def build_distill_data(self) -> None:
        generated_queries = read_squad_json(self.query_json)
        predictions = json.loads(self.query_predictions_json.read_text(encoding="utf-8"))
        distilled = combine_squad_predictions_with_queries(
            generated_queries=generated_queries,
            predictions=predictions,
        )
        write_squad_json(self.distilled_train_json, distilled)
        logger.info("Wrote distilled SQuAD training data to {}", self.distilled_train_json)

    def train_extracted(self) -> None:
        train_examples = squad_examples_from_json(read_squad_json(self.distilled_train_json))
        dev_examples = squad_examples_from_json(self._load_dev_data())
        artifacts = train_qa_model(
            model_name_or_path=self.model_cfg["pretrained_model_name_or_path"],
            output_dir=self.extracted_dir,
            train_examples=train_examples,
            eval_examples=dev_examples,
            training_cfg=self.training_cfg,
            model_cfg=self.model_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "extracted_dev_metrics.json", artifacts.metrics)

    def evaluate_agreement(self) -> None:
        dev_examples = squad_examples_from_json(self._load_dev_data())
        victim_predictions = predict_squad(
            model_dir_or_name=self.victim_dir,
            examples=dev_examples,
            model_cfg=self.model_cfg,
            training_cfg=self.training_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        extracted_predictions = predict_squad(
            model_dir_or_name=self.extracted_dir,
            examples=dev_examples,
            model_cfg=self.model_cfg,
            training_cfg=self.training_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        self.victim_dev_predictions_json.write_text(
            json.dumps(victim_predictions, indent=2),
            encoding="utf-8",
        )
        self.extracted_dev_predictions_json.write_text(
            json.dumps(extracted_predictions, indent=2),
            encoding="utf-8",
        )
        save_json(
            self.metrics_dir / "victim_dev_metrics_from_predictions.json",
            evaluate_squad_predictions(dev_examples, victim_predictions),
        )
        save_json(
            self.metrics_dir / "extracted_dev_metrics_from_predictions.json",
            evaluate_squad_predictions(dev_examples, extracted_predictions),
        )
        save_json(
            self.metrics_dir / "agreement_dev.json",
            evaluate_prediction_agreement(
                victim_predictions=victim_predictions,
                extracted_predictions=extracted_predictions,
            ),
        )
        logger.info("Saved SQuAD agreement metrics to {}", self.metrics_dir / "agreement_dev.json")

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
