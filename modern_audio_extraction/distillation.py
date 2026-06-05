from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from modern_audio_extraction.audio_data import load_query_examples
from modern_audio_extraction.models import (
    StolenAudioEncoder,
    freeze_feature_encoder_if_available,
    load_audio_encoder,
    load_audio_processor,
    save_stolen_encoder,
)
from modern_bert_extraction.training import (
    _autocast_context,
    auto_mixed_precision_dtype,
    resolve_device,
    set_seed,
)


@dataclass
class DistillationArtifacts:
    output_dir: Path
    metrics: dict[str, float]


def _processor_batch(processor, waveforms: Sequence[torch.Tensor], sampling_rate: int):
    arrays = [waveform.detach().cpu().float().numpy() for waveform in waveforms]
    return processor(
        arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )


def _model_kwargs(batch) -> dict[str, torch.Tensor]:
    kwargs = {"input_values": batch["input_values"]}
    if "attention_mask" in batch:
        kwargs["attention_mask"] = batch["attention_mask"]
    return kwargs


def _audio_augment(
    waveform: torch.Tensor,
    *,
    cfg: dict[str, Any],
    generator: torch.Generator,
) -> torch.Tensor:
    output = waveform.clone()
    gain_min = float(cfg.get("gain_min", 0.9))
    gain_max = float(cfg.get("gain_max", 1.1))
    if gain_min != 1.0 or gain_max != 1.0:
        gain = gain_min + (gain_max - gain_min) * torch.rand((), generator=generator).item()
        output = output * gain
    noise_std = float(cfg.get("noise_std", 0.0))
    if noise_std > 0:
        output = output + torch.randn(output.shape, generator=generator) * noise_std
    max_shift_seconds = float(cfg.get("max_time_shift_seconds", 0.0))
    sampling_rate = int(cfg.get("sampling_rate", 16000))
    max_shift = int(max_shift_seconds * sampling_rate)
    if max_shift > 0 and output.numel() > 1:
        shift = int(torch.randint(-max_shift, max_shift + 1, (), generator=generator).item())
        output = torch.roll(output, shifts=shift, dims=0)
    return output.clamp(-1.0, 1.0).contiguous()


def build_query_cache(
    *,
    cache_path: str | Path,
    target_cfg: dict[str, Any],
    query_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
    force: bool = False,
) -> Path:
    output_path = Path(cache_path)
    if output_path.exists() and not force:
        logger.info("Keeping existing query cache at {}", output_path)
        return output_path

    set_seed(seed)
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(runtime_cfg.get("mixed_precision", "auto"), device)
    processor = load_audio_processor(target_cfg["model_name_or_path"])
    target_model = load_audio_encoder(target_cfg["model_name_or_path"], init_from_pretrained=True)
    target_model.to(device)
    target_model.eval()

    examples = load_query_examples(query_cfg, seed=seed)
    sampling_rate = int(query_cfg.get("sampling_rate", 16000))
    batch_size = int(runtime_cfg.get("query_batch_size", runtime_cfg.get("eval_batch_size", 4)))
    records: list[dict[str, Any]] = []

    logger.info("Querying target encoder for {} examples", len(examples))
    with torch.no_grad():
        for start in tqdm(range(0, len(examples), batch_size), desc="query target", leave=False):
            batch_examples = examples[start : start + batch_size]
            batch = _processor_batch(
                processor,
                [example.audio for example in batch_examples],
                sampling_rate=sampling_rate,
            )
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast_context(device, dtype):
                outputs = target_model(**_model_kwargs(batch))
            hidden = outputs.last_hidden_state.float().cpu()
            for example, features in zip(batch_examples, hidden):
                records.append(
                    {
                        "guid": example.guid,
                        "audio": example.audio.cpu().to(torch.float16),
                        "sampling_rate": example.sampling_rate,
                        "target_features": features.to(torch.float16),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_model": target_cfg["model_name_or_path"],
            "query_source": query_cfg,
            "records": records,
        },
        output_path,
    )
    logger.info("Saved {} queried target feature records to {}", len(records), output_path)
    return output_path


class AudioFeatureCacheDataset(Dataset):
    def __init__(self, cache_path: str | Path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.metadata = {key: value for key, value in cache.items() if key != "records"}
        self.records = cache["records"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class DistillationCollator:
    def __init__(
        self,
        *,
        processor,
        sampling_rate: int,
        augmentation_cfg: dict[str, Any],
        seed: int,
    ):
        self.processor = processor
        self.sampling_rate = sampling_rate
        self.augmentation_cfg = dict(augmentation_cfg)
        self.augmentation_cfg["sampling_rate"] = sampling_rate
        self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [record["audio"].float() for record in batch]
        augmented_waveforms = [
            _audio_augment(waveform, cfg=self.augmentation_cfg, generator=self.generator)
            for waveform in waveforms
        ]
        encoded = _processor_batch(self.processor, waveforms, self.sampling_rate)
        augmented_encoded = _processor_batch(
            self.processor,
            augmented_waveforms,
            self.sampling_rate,
        )

        targets = [record["target_features"].float() for record in batch]
        max_time = max(target.shape[0] for target in targets)
        feature_dim = targets[0].shape[-1]
        padded_targets = torch.zeros(len(targets), max_time, feature_dim)
        target_mask = torch.zeros(len(targets), max_time, dtype=torch.bool)
        for index, target in enumerate(targets):
            padded_targets[index, : target.shape[0]] = target
            target_mask[index, : target.shape[0]] = True

        return {
            "input_values": encoded["input_values"],
            "attention_mask": encoded.get("attention_mask"),
            "augmented_input_values": augmented_encoded["input_values"],
            "augmented_attention_mask": augmented_encoded.get("attention_mask"),
            "target_features": padded_targets,
            "target_mask": target_mask,
        }


def _move_optional(tensor: torch.Tensor | None, device: torch.device):
    return None if tensor is None else tensor.to(device)


def _align_time(predicted: torch.Tensor, target_time: int) -> torch.Tensor:
    if predicted.shape[1] == target_time:
        return predicted
    return torch.nn.functional.interpolate(
        predicted.transpose(1, 2),
        size=target_time,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def feature_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    metric: str,
) -> torch.Tensor:
    predicted = _align_time(predicted, target.shape[1])
    if metric == "mse":
        per_step = ((predicted - target) ** 2).mean(dim=-1)
    elif metric == "l2":
        per_step = torch.sqrt(((predicted - target) ** 2).sum(dim=-1) + 1.0e-12)
    elif metric == "cosine":
        per_step = 1.0 - torch.nn.functional.cosine_similarity(predicted, target, dim=-1)
    else:
        raise ValueError("Unsupported feature distance metric: {}".format(metric))
    return (per_step * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def train_stolen_encoder(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
    target_cfg: dict[str, Any],
    student_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    seed: int,
) -> DistillationArtifacts:
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(runtime_cfg.get("mixed_precision", "auto"), device)

    processor = load_audio_processor(student_cfg["model_name_or_path"])
    student_model = load_audio_encoder(
        student_cfg["model_name_or_path"],
        init_from_pretrained=bool(student_cfg.get("init_from_pretrained", True)),
    )
    if student_cfg.get("freeze_feature_encoder", False):
        freeze_feature_encoder_if_available(student_model)
    target_hidden_size = int(
        target_cfg.get("hidden_size") or FeatureCacheInfo(cache_path).hidden_size
    )
    stolen_encoder = StolenAudioEncoder(student_model, target_hidden_size=target_hidden_size)
    stolen_encoder.to(device)

    dataset = AudioFeatureCacheDataset(cache_path)
    collator = DistillationCollator(
        processor=processor,
        sampling_rate=int(training_cfg.get("sampling_rate", 16000)),
        augmentation_cfg=training_cfg.get("augmentation", {}),
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(training_cfg["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(
        stolen_encoder.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(float(training_cfg["num_train_epochs"]))
    grad_accum = int(training_cfg.get("gradient_accumulation_steps", 1))
    update_steps_per_epoch = max(1, (len(dataloader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, update_steps_per_epoch * epochs)
    warmup_steps = int(total_steps * float(training_cfg.get("warmup_proportion", 0.0)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and dtype == torch.float16,
    )
    lambda_aug = float(training_cfg.get("lambda_aug", 20.0))
    metric = str(training_cfg.get("distance_metric", "mse"))

    logger.info(
        "Training stolen audio encoder on {} queried examples for {} epochs",
        len(dataset),
        epochs,
    )
    last_metrics: dict[str, float] = {}
    for epoch_index in range(epochs):
        stolen_encoder.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        loss1_values: list[float] = []
        loss2_values: list[float] = []
        progress = tqdm(
            dataloader, desc="audio steal epoch {}".format(epoch_index + 1), leave=False
        )
        for step_index, batch in enumerate(progress, start=1):
            input_values = batch["input_values"].to(device)
            attention_mask = _move_optional(batch["attention_mask"], device)
            augmented_input_values = batch["augmented_input_values"].to(device)
            augmented_attention_mask = _move_optional(batch["augmented_attention_mask"], device)
            target_features = batch["target_features"].to(device)
            target_mask = batch["target_mask"].to(device)

            with _autocast_context(device, dtype):
                original_features = stolen_encoder(input_values, attention_mask)
                augmented_features = stolen_encoder(
                    augmented_input_values,
                    augmented_attention_mask,
                )
                loss1 = feature_distance(
                    original_features,
                    target_features,
                    target_mask,
                    metric=metric,
                )
                loss2 = feature_distance(
                    augmented_features,
                    target_features,
                    target_mask,
                    metric=metric,
                )
                loss = (loss1 + lambda_aug * loss2) / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_index % grad_accum == 0 or step_index == len(dataloader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    stolen_encoder.parameters(),
                    float(training_cfg.get("max_grad_norm", 1.0)),
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item() * grad_accum))
            loss1_values.append(float(loss1.item()))
            loss2_values.append(float(loss2.item()))

        last_metrics = {
            "epoch": float(epoch_index + 1),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "loss_l1": float(np.mean(loss1_values)) if loss1_values else 0.0,
            "loss_l2": float(np.mean(loss2_values)) if loss2_values else 0.0,
            "num_queries": float(len(dataset)),
        }
        logger.info(
            "Epoch {} loss={:.6f} l1={:.6f} l2={:.6f}",
            epoch_index + 1,
            last_metrics["loss"],
            last_metrics["loss_l1"],
            last_metrics["loss_l2"],
        )

    save_stolen_encoder(
        output_dir=output_path,
        stolen_encoder=stolen_encoder,
        processor=processor,
        metadata={
            "student": student_cfg,
            "target": target_cfg,
            "training": training_cfg,
        },
    )
    (output_path / "metrics.json").write_text(
        json.dumps(last_metrics, indent=2),
        encoding="utf-8",
    )
    return DistillationArtifacts(output_dir=output_path, metrics=last_metrics)


class FeatureCacheInfo:
    def __init__(self, cache_path: str | Path):
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        records = cache["records"]
        if not records:
            raise ValueError("Query cache is empty: {}".format(cache_path))
        self.hidden_size = int(records[0]["target_features"].shape[-1])


def evaluate_feature_similarity(
    *,
    cache_path: str | Path,
    stolen_encoder: StolenAudioEncoder,
    processor,
    training_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
) -> dict[str, float]:
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(runtime_cfg.get("mixed_precision", "auto"), device)
    stolen_encoder.to(device)
    stolen_encoder.eval()
    dataset = AudioFeatureCacheDataset(cache_path)
    collator = DistillationCollator(
        processor=processor,
        sampling_rate=int(training_cfg.get("sampling_rate", 16000)),
        augmentation_cfg={},
        seed=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(runtime_cfg.get("eval_batch_size", 4)),
        shuffle=False,
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        collate_fn=collator,
    )
    mse_values: list[float] = []
    cosine_values: list[float] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="feature eval", leave=False):
            input_values = batch["input_values"].to(device)
            attention_mask = _move_optional(batch["attention_mask"], device)
            target_features = batch["target_features"].to(device)
            target_mask = batch["target_mask"].to(device)
            with _autocast_context(device, dtype):
                predicted = stolen_encoder(input_values, attention_mask)
                predicted = _align_time(predicted, target_features.shape[1])
            mse_values.append(
                float(
                    feature_distance(predicted, target_features, target_mask, metric="mse").item()
                )
            )
            cosine_loss = feature_distance(
                predicted,
                target_features,
                target_mask,
                metric="cosine",
            )
            cosine_values.append(float(1.0 - cosine_loss.item()))
    return {
        "feature_mse": float(np.mean(mse_values)) if mse_values else 0.0,
        "feature_cosine": float(np.mean(cosine_values)) if cosine_values else 0.0,
        "num_queries": float(len(dataset)),
    }
