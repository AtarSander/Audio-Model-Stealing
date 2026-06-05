from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence

from loguru import logger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modern_bert_extraction.glue import ClassifierExample


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(runtime_cfg: dict) -> torch.device:
    requested = str(runtime_cfg.get("device", "auto")).lower()
    require_gpu = bool(runtime_cfg.get("require_gpu", False))
    if requested == "cpu":
        if require_gpu:
            raise RuntimeError("GPU is required by config but device was forced to CPU.")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested explicitly but is not available.")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_gpu:
        raise RuntimeError("GPU is required by config but torch.cuda.is_available() is False.")
    return torch.device("cpu")


def auto_mixed_precision_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    mode = str(mode).lower()
    if device.type != "cuda":
        return None
    if mode == "none":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError("Unsupported mixed_precision mode: {}".format(mode))


class ExampleDataset(Dataset[ClassifierExample]):
    def __init__(self, examples: Sequence[ClassifierExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ClassifierExample:
        return self.examples[index]


class ExampleCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[ClassifierExample]) -> dict[str, torch.Tensor]:
        text_pairs = [example.text_b for example in batch]
        tokenizer_kwargs = {
            "text": [example.text_a for example in batch],
            "truncation": True,
            "padding": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if any(text is not None for text in text_pairs):
            tokenizer_kwargs["text_pair"] = text_pairs
        encoded = self.tokenizer(**tokenizer_kwargs)

        hard_labels = [example.label for example in batch]
        if any(label is None for label in hard_labels):
            hard_labels = [
                0 if example.soft_labels is None else int(np.argmax(example.soft_labels))
                for example in batch
            ]
        encoded["hard_labels"] = torch.tensor(hard_labels, dtype=torch.long)

        if batch[0].soft_labels is not None:
            encoded["soft_labels"] = torch.tensor(
                [example.soft_labels for example in batch], dtype=torch.float32
            )
        return encoded


def soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(soft_labels * log_probs).sum(dim=-1).mean()


@dataclass
class TrainingArtifacts:
    output_dir: Path
    metrics: dict[str, float]


def _autocast_context(device: torch.device, dtype: torch.dtype | None):
    if device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def _is_local_classifier_checkpoint_dir(model_name_or_path: str | Path) -> bool:
    model_path = Path(str(model_name_or_path))
    if not model_path.is_dir():
        return False
    checkpoint_filenames = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((model_path / filename).exists() for filename in checkpoint_filenames):
        return False

    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = config_data.get("architectures") or []
    return any("ForSequenceClassification" in architecture for architecture in architectures)


def _initialize_classifier_from_base(model_name_or_path: str, num_labels: int):
    logger.info(
        "Initializing classifier from base encoder '{}' with {} labels",
        model_name_or_path,
        num_labels,
    )
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_config(config)
    encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
    missing_keys, unexpected_keys = model.base_model.load_state_dict(
        encoder.state_dict(),
        strict=False,
    )
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Base encoder weight transfer failed. Missing keys: {}. Unexpected keys: {}.".format(
                missing_keys,
                unexpected_keys,
            )
        )
    return model


def train_model(
    *,
    model_name_or_path: str,
    output_dir: str | Path,
    train_examples: Sequence[ClassifierExample],
    eval_examples: Sequence[ClassifierExample],
    num_labels: int,
    training_cfg: dict,
    model_cfg: dict,
    runtime_cfg: dict,
    seed: int,
) -> TrainingArtifacts:
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(training_cfg.get("mixed_precision", "auto"), device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if _is_local_classifier_checkpoint_dir(model_name_or_path):
        logger.info("Loading fine-tuned classifier checkpoint from {}", model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_name_or_path),
            num_labels=num_labels,
        )
    else:
        model = _initialize_classifier_from_base(
            model_name_or_path=str(model_name_or_path),
            num_labels=num_labels,
        )
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)

    per_device_train_batch_size = int(training_cfg["per_device_train_batch_size"])
    per_device_eval_batch_size = int(training_cfg["per_device_eval_batch_size"])
    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    num_epochs = float(training_cfg["num_train_epochs"])
    num_workers = int(training_cfg.get("dataloader_num_workers", 0))

    train_loader = DataLoader(
        ExampleDataset(train_examples),
        batch_size=per_device_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ExampleCollator(tokenizer, max_length=int(model_cfg["max_seq_length"])),
    )
    eval_loader = DataLoader(
        ExampleDataset(eval_examples),
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=ExampleCollator(tokenizer, max_length=int(model_cfg["max_seq_length"])),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    num_update_steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_train_steps = int(num_epochs * num_update_steps_per_epoch)
    warmup_steps = int(total_train_steps * float(training_cfg.get("warmup_proportion", 0.1)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and dtype == torch.float16,
    )

    effective_batch_size = per_device_train_batch_size * grad_accum
    logger.info(
        "Training {} examples for {} epochs on {} with per-device batch size {}, gradient accumulation {}, effective batch size {}, total update steps {}",
        len(train_examples),
        int(num_epochs),
        device,
        per_device_train_batch_size,
        grad_accum,
        effective_batch_size,
        total_train_steps,
    )

    best_accuracy = float("-inf")
    best_metrics: dict[str, float] = {}
    for epoch_index in range(int(num_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc="train epoch {}".format(epoch_index + 1), leave=False)
        for step_index, batch in enumerate(progress, start=1):
            hard_labels = batch.pop("hard_labels")
            soft_labels = batch.pop("soft_labels", None)
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            hard_labels = hard_labels.to(device)
            if soft_labels is not None:
                soft_labels = soft_labels.to(device)

            with _autocast_context(device, dtype):
                logits = model(**batch).logits
                if soft_labels is not None:
                    loss = soft_cross_entropy(logits, soft_labels)
                else:
                    loss = nn.functional.cross_entropy(logits, hard_labels)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_index % grad_accum == 0 or step_index == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training_cfg.get("max_grad_norm", 1.0))
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate_model(model=model, dataloader=eval_loader, device=device, dtype=dtype)
        metrics["epoch"] = epoch_index + 1
        logger.info(
            "Epoch {} eval accuracy={:.4f} loss={:.4f}",
            epoch_index + 1,
            metrics["accuracy"],
            metrics["loss"],
        )
        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_metrics = dict(metrics)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(
                "Saved new best checkpoint to {} with accuracy {:.4f}",
                output_path,
                best_accuracy,
            )

    metrics_path = output_path / "metrics.json"
    metrics_path.write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    return TrainingArtifacts(output_dir=output_path, metrics=best_metrics)


def load_model_and_tokenizer(
    model_dir_or_name: str | Path,
    num_labels: int,
    training_cfg: dict,
    runtime_cfg: dict,
):
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(training_cfg.get("mixed_precision", "auto"), device)
    logger.info("Loading classifier for inference from {}", model_dir_or_name)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir_or_name), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir_or_name), num_labels=num_labels
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device, dtype


def predict_probabilities(
    *,
    model_dir_or_name: str | Path,
    examples: Sequence[ClassifierExample],
    num_labels: int,
    training_cfg: dict,
    model_cfg: dict,
    runtime_cfg: dict,
) -> np.ndarray:
    model, tokenizer, device, dtype = load_model_and_tokenizer(
        model_dir_or_name=model_dir_or_name,
        num_labels=num_labels,
        training_cfg=training_cfg,
        runtime_cfg=runtime_cfg,
    )
    dataloader = DataLoader(
        ExampleDataset(examples),
        batch_size=int(training_cfg["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
        collate_fn=ExampleCollator(tokenizer, max_length=int(model_cfg["max_seq_length"])),
    )
    logger.info("Predicting probabilities for {} examples", len(examples))
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="predict", leave=False):
            batch.pop("hard_labels")
            batch.pop("soft_labels", None)
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast_context(device, dtype):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            probabilities.append(probs)
    return np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, num_labels))


def evaluate_model(
    model, dataloader: DataLoader, device: torch.device, dtype: torch.dtype | None
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="eval", leave=False):
            hard_labels = batch.pop("hard_labels").to(device)
            soft_labels = batch.pop("soft_labels", None)
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            if soft_labels is not None:
                soft_labels = soft_labels.to(device)

            with _autocast_context(device, dtype):
                logits = model(**batch).logits
                if soft_labels is not None:
                    loss = soft_cross_entropy(logits, soft_labels)
                else:
                    loss = nn.functional.cross_entropy(logits, hard_labels)
            losses.append(float(loss.item()))
            predictions.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
            labels.append(hard_labels.cpu().numpy())

    probabilities = np.concatenate(predictions, axis=0)
    label_array = np.concatenate(labels, axis=0)
    predicted = probabilities.argmax(axis=-1)
    accuracy = float((predicted == label_array).mean()) if len(label_array) else 0.0
    return {
        "accuracy": accuracy,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "num_examples": int(len(label_array)),
    }


def save_json(path: str | Path, data: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
