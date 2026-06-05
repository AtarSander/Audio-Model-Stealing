from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from modern_audio_extraction.audio_data import load_hf_audio_examples
from modern_audio_extraction.distillation import _model_kwargs, _processor_batch
from modern_audio_extraction.models import StolenAudioEncoder
from modern_bert_extraction.training import (
    _autocast_context,
    auto_mixed_precision_dtype,
    resolve_device,
    set_seed,
)


@dataclass
class ProbeResult:
    accuracy: float
    num_train_examples: int
    num_eval_examples: int


class MlpProbe(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, hidden_sizes: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_dim, int(hidden_size)))
            layers.append(nn.ReLU())
            current_dim = int(hidden_size)
        layers.append(nn.Linear(current_dim, num_labels))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def _mean_pool(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.mean(dim=1)


def _label_count(labels: Sequence[int]) -> int:
    return max(labels) + 1 if labels else 0


def _extract_embeddings_with_model(
    *,
    model,
    processor,
    examples,
    sampling_rate: int,
    batch_size: int,
    runtime_cfg: dict[str, Any],
    is_stolen: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(runtime_cfg.get("mixed_precision", "auto"), device)
    model.to(device)
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(examples), batch_size), desc="embed audio", leave=False):
            batch_examples = examples[start : start + batch_size]
            batch = _processor_batch(
                processor,
                [example.audio for example in batch_examples],
                sampling_rate=sampling_rate,
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast_context(device, dtype):
                if is_stolen:
                    hidden = model(
                        batch["input_values"],
                        batch.get("attention_mask"),
                    )
                else:
                    hidden = model(**_model_kwargs(batch)).last_hidden_state
            embeddings.append(_mean_pool(hidden.float()).cpu())
            labels.extend(int(example.label) for example in batch_examples)
    return torch.cat(embeddings, dim=0), torch.tensor(labels, dtype=torch.long)


def train_probe(
    *,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    eval_embeddings: torch.Tensor,
    eval_labels: torch.Tensor,
    probe_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
) -> ProbeResult:
    set_seed(seed)
    device = resolve_device(runtime_cfg)
    num_labels = int(train_labels.max().item()) + 1
    probe = MlpProbe(
        input_dim=train_embeddings.shape[-1],
        num_labels=num_labels,
        hidden_sizes=probe_cfg.get("hidden_sizes", [512, 256]),
    )
    probe.to(device)
    train_loader = DataLoader(
        TensorDataset(train_embeddings.float(), train_labels),
        batch_size=int(probe_cfg.get("batch_size", 128)),
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=float(probe_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(probe_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(float(probe_cfg.get("num_train_epochs", 20)))
    for epoch_index in range(epochs):
        probe.train()
        losses: list[float] = []
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = probe(features)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
        logger.info(
            "Probe epoch {} loss={:.6f}",
            epoch_index + 1,
            float(np.mean(losses)) if losses else 0.0,
        )

    probe.eval()
    eval_loader = DataLoader(
        TensorDataset(eval_embeddings.float(), eval_labels),
        batch_size=int(probe_cfg.get("batch_size", 128)),
        shuffle=False,
    )
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in eval_loader:
            logits = probe(features.to(device))
            predictions = logits.argmax(dim=-1).cpu()
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    return ProbeResult(
        accuracy=float(correct / total) if total else 0.0,
        num_train_examples=int(train_labels.numel()),
        num_eval_examples=int(eval_labels.numel()),
    )


def evaluate_downstream_classifier(
    *,
    target_model,
    target_processor,
    stolen_encoder: StolenAudioEncoder,
    stolen_processor,
    downstream_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    if not downstream_cfg.get("enabled", True):
        return {}
    sampling_rate = int(downstream_cfg.get("sampling_rate", 16000))
    max_duration_seconds = float(downstream_cfg.get("max_duration_seconds", 4.0))
    train_examples = load_hf_audio_examples(
        dataset_name=downstream_cfg["dataset_name"],
        dataset_config=downstream_cfg.get("dataset_config"),
        split=downstream_cfg["train_split"],
        audio_column=downstream_cfg.get("audio_column", "audio"),
        label_column=downstream_cfg.get("label_column", "label"),
        budget=downstream_cfg.get("train_budget"),
        seed=seed,
        sampling_rate=sampling_rate,
        max_duration_seconds=max_duration_seconds,
    )
    eval_examples = load_hf_audio_examples(
        dataset_name=downstream_cfg["dataset_name"],
        dataset_config=downstream_cfg.get("dataset_config"),
        split=downstream_cfg["eval_split"],
        audio_column=downstream_cfg.get("audio_column", "audio"),
        label_column=downstream_cfg.get("label_column", "label"),
        budget=downstream_cfg.get("eval_budget"),
        seed=seed + 1,
        sampling_rate=sampling_rate,
        max_duration_seconds=max_duration_seconds,
    )
    labels = [example.label for example in train_examples if example.label is not None]
    if _label_count(labels) < 2:
        raise ValueError("Downstream dataset must contain at least two labels.")

    batch_size = int(runtime_cfg.get("eval_batch_size", 4))
    logger.info("Extracting target downstream embeddings")
    target_train_embeddings, train_labels = _extract_embeddings_with_model(
        model=target_model,
        processor=target_processor,
        examples=train_examples,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        runtime_cfg=runtime_cfg,
        is_stolen=False,
    )
    target_eval_embeddings, eval_labels = _extract_embeddings_with_model(
        model=target_model,
        processor=target_processor,
        examples=eval_examples,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        runtime_cfg=runtime_cfg,
        is_stolen=False,
    )
    logger.info("Extracting stolen downstream embeddings")
    stolen_train_embeddings, stolen_train_labels = _extract_embeddings_with_model(
        model=stolen_encoder,
        processor=stolen_processor,
        examples=train_examples,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        runtime_cfg=runtime_cfg,
        is_stolen=True,
    )
    stolen_eval_embeddings, stolen_eval_labels = _extract_embeddings_with_model(
        model=stolen_encoder,
        processor=stolen_processor,
        examples=eval_examples,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        runtime_cfg=runtime_cfg,
        is_stolen=True,
    )
    if not torch.equal(train_labels, stolen_train_labels) or not torch.equal(
        eval_labels,
        stolen_eval_labels,
    ):
        raise RuntimeError("Target/stolen downstream labels are misaligned.")

    probe_cfg = downstream_cfg.get("probe", {})
    logger.info("Training target downstream probe")
    target_result = train_probe(
        train_embeddings=target_train_embeddings,
        train_labels=train_labels,
        eval_embeddings=target_eval_embeddings,
        eval_labels=eval_labels,
        probe_cfg=probe_cfg,
        runtime_cfg=runtime_cfg,
        seed=seed,
    )
    logger.info("Training stolen downstream probe")
    stolen_result = train_probe(
        train_embeddings=stolen_train_embeddings,
        train_labels=stolen_train_labels,
        eval_embeddings=stolen_eval_embeddings,
        eval_labels=stolen_eval_labels,
        probe_cfg=probe_cfg,
        runtime_cfg=runtime_cfg,
        seed=seed,
    )
    ratio = stolen_result.accuracy / target_result.accuracy if target_result.accuracy > 0 else 0.0
    return {
        "target_accuracy": target_result.accuracy,
        "stolen_accuracy": stolen_result.accuracy,
        "stolen_to_target_accuracy_ratio": ratio,
        "downstream_train_examples": float(target_result.num_train_examples),
        "downstream_eval_examples": float(target_result.num_eval_examples),
    }
