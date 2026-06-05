from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf
import torch

from modern_audio_extraction.distillation import (
    build_query_cache,
    evaluate_feature_similarity,
    train_stolen_encoder,
)
from modern_audio_extraction.downstream import evaluate_downstream_classifier
from modern_audio_extraction.models import (
    load_audio_encoder,
    load_audio_processor,
    load_stolen_encoder,
)
from modern_bert_extraction.training import resolve_device, save_json


def load_config(path: str | Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


class AudioEncoderExtractionPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.seed = int(config.get("seed", 42))
        self.paths = config["paths"]
        self.target_cfg = config["target"]
        self.student_cfg = config["student"]
        self.query_cfg = config["query_source"]
        self.training_cfg = config["training"]
        self.downstream_cfg = config["downstream"]
        self.runtime_cfg = config["runtime"]

        self.run_name = str(config.get("run_name") or self._default_run_name())
        self.run_root = Path(self.paths["output_root"]) / self.run_name
        configured_checkpoint_root = self.paths.get("checkpoint_root")
        self.checkpoint_run_root = (
            Path(configured_checkpoint_root) / self.run_name
            if configured_checkpoint_root
            else self.run_root
        )
        self.cache_dir = self.run_root / "cache"
        self.stolen_dir = self.checkpoint_run_root / "stolen_encoder"
        self.metrics_dir = self.run_root / "metrics"
        self.query_cache_path = self.cache_dir / "target_feature_cache.pt"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.checkpoint_run_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def _default_run_name(self) -> str:
        target = Path(str(self.target_cfg["model_name_or_path"])).name.replace("/", "_")
        student = Path(str(self.student_cfg["model_name_or_path"])).name.replace("/", "_")
        modality = str(self.query_cfg["type"])
        budget = int(self.query_cfg["budget"])
        return "{}__student-{}__{}__q{}".format(target, student, modality, budget)

    def preflight(self) -> None:
        device = resolve_device(self.runtime_cfg)
        logger.info("Using device {}", device)
        if device.type == "cuda":
            logger.info("GPU {}", torch.cuda.get_device_name(device))
        save_json(self.run_root / "resolved_config.json", self.config)
        logger.info("Saved resolved config to {}", self.run_root / "resolved_config.json")

    def build_query_cache(self) -> None:
        build_query_cache(
            cache_path=self.query_cache_path,
            target_cfg=self.target_cfg,
            query_cfg=self.query_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
            force=bool(self.query_cfg.get("force_rebuild_cache", False)),
        )

    def train_stolen_encoder(self) -> None:
        artifacts = train_stolen_encoder(
            cache_path=self.query_cache_path,
            output_dir=self.stolen_dir,
            target_cfg=self.target_cfg,
            student_cfg=self.student_cfg,
            training_cfg=self.training_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        save_json(self.metrics_dir / "distillation_metrics.json", artifacts.metrics)

    def evaluate_feature_similarity(self) -> None:
        stolen_encoder, processor, _ = load_stolen_encoder(self.stolen_dir)
        metrics = evaluate_feature_similarity(
            cache_path=self.query_cache_path,
            stolen_encoder=stolen_encoder,
            processor=processor,
            training_cfg=self.training_cfg,
            runtime_cfg=self.runtime_cfg,
        )
        metrics.update(self._experiment_metadata())
        save_json(self.metrics_dir / "feature_similarity.json", metrics)
        if not self.downstream_cfg.get("enabled", True):
            save_json(self.metrics_dir / "final_metrics.json", metrics)

    def evaluate_downstream(self) -> None:
        target_processor = load_audio_processor(self.target_cfg["model_name_or_path"])
        target_model = load_audio_encoder(
            self.target_cfg["model_name_or_path"],
            init_from_pretrained=True,
        )
        stolen_encoder, stolen_processor, _ = load_stolen_encoder(self.stolen_dir)
        metrics = evaluate_downstream_classifier(
            target_model=target_model,
            target_processor=target_processor,
            stolen_encoder=stolen_encoder,
            stolen_processor=stolen_processor,
            downstream_cfg=self.downstream_cfg,
            runtime_cfg=self.runtime_cfg,
            seed=self.seed,
        )
        metrics.update(self._experiment_metadata())
        save_json(self.metrics_dir / "downstream_metrics.json", metrics)

        final_metrics = {}
        feature_path = self.metrics_dir / "feature_similarity.json"
        if feature_path.exists():
            final_metrics.update(json.loads(feature_path.read_text(encoding="utf-8")))
        final_metrics.update(metrics)
        save_json(self.metrics_dir / "final_metrics.json", final_metrics)

    def _experiment_metadata(self) -> dict[str, float | str]:
        return {
            "target_model": str(self.target_cfg["model_name_or_path"]),
            "student_model": str(self.student_cfg["model_name_or_path"]),
            "query_source_type": str(self.query_cfg["type"]),
            "query_budget": float(self.query_cfg["budget"]),
            "run_name": self.run_name,
        }

    def run(self, step: str) -> None:
        if step == "all":
            ordered_steps = [
                "preflight",
                "build_query_cache",
                "train_stolen_encoder",
                "evaluate_feature_similarity",
            ]
            if self.downstream_cfg.get("enabled", True):
                ordered_steps.append("evaluate_downstream")
            else:
                logger.info("Skipping downstream evaluation because downstream.enabled=false")
        else:
            ordered_steps = [step]
        for step_name in ordered_steps:
            logger.info("Starting step '{}'", step_name)
            getattr(self, step_name)()
