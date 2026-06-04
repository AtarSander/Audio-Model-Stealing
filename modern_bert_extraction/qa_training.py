from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import string
from typing import Any, Sequence

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from modern_bert_extraction.training import (
    _autocast_context,
    auto_mixed_precision_dtype,
    resolve_device,
    set_seed,
)


@dataclass(frozen=True)
class SquadExample:
    qid: str
    title: str
    context: str
    question: str
    answers: list[dict[str, Any]]


@dataclass
class QaTrainingArtifacts:
    output_dir: Path
    metrics: dict[str, float]


class QaFeatureDataset(Dataset):
    def __init__(self, features: Sequence[dict[str, Any]], *, include_positions: bool):
        self.features = list(features)
        self.include_positions = include_positions

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        feature = self.features[index]
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in feature:
            item["token_type_ids"] = torch.tensor(feature["token_type_ids"], dtype=torch.long)
        if self.include_positions:
            item["start_positions"] = torch.tensor(feature["start_position"], dtype=torch.long)
            item["end_positions"] = torch.tensor(feature["end_position"], dtype=torch.long)
        return item


def read_squad_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_squad_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def squad_examples_from_json(data: dict[str, Any]) -> list[SquadExample]:
    examples: list[SquadExample] = []
    for article in data["data"]:
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                examples.append(
                    SquadExample(
                        qid=str(qa["id"]),
                        title=title,
                        context=context,
                        question=qa["question"],
                        answers=list(qa.get("answers") or []),
                    )
                )
    return examples


def _is_local_qa_checkpoint_dir(model_name_or_path: str | Path) -> bool:
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
    return any("ForQuestionAnswering" in architecture for architecture in architectures)


def _initialize_qa_from_base(model_name_or_path: str):
    logger.info("Initializing QA model from base encoder '{}'", model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    model = AutoModelForQuestionAnswering.from_config(config)
    encoder = AutoModel.from_pretrained(model_name_or_path, config=config)
    missing_keys, unexpected_keys = model.base_model.load_state_dict(
        encoder.state_dict(),
        strict=False,
    )
    # Extractive QA heads do not use BERT's pooled CLS output.
    unexpected_keys = [key for key in unexpected_keys if not key.startswith("pooler.")]
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Base encoder weight transfer failed. Missing keys: {}. Unexpected keys: {}.".format(
                missing_keys,
                unexpected_keys,
            )
        )
    return model


def _load_qa_model_and_tokenizer(
    model_name_or_path: str | Path,
    *,
    model_cfg: dict,
    training_cfg: dict,
    runtime_cfg: dict,
):
    device = resolve_device(runtime_cfg)
    dtype = auto_mixed_precision_dtype(training_cfg.get("mixed_precision", "auto"), device)
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)
    if _is_local_qa_checkpoint_dir(model_name_or_path):
        logger.info("Loading fine-tuned QA checkpoint from {}", model_name_or_path)
        model = AutoModelForQuestionAnswering.from_pretrained(str(model_name_or_path))
    else:
        model = _initialize_qa_from_base(str(model_name_or_path))
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.to(device)
    return model, tokenizer, device, dtype


def _prepare_train_features(
    *,
    examples: Sequence[SquadExample],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    cls_token_id = tokenizer.cls_token_id
    for example in examples:
        tokenized = tokenizer(
            example.question.strip(),
            example.context,
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index, input_ids in enumerate(tokenized["input_ids"]):
            offsets = tokenized["offset_mapping"][feature_index]
            sequence_ids = tokenized.sequence_ids(feature_index)
            cls_index = input_ids.index(cls_token_id)
            feature = {
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"][feature_index],
                "start_position": cls_index,
                "end_position": cls_index,
            }
            if "token_type_ids" in tokenized:
                feature["token_type_ids"] = tokenized["token_type_ids"][feature_index]

            if not example.answers:
                features.append(feature)
                continue

            answer = example.answers[0]
            answer_start = int(answer["answer_start"])
            answer_text = str(answer["text"])
            answer_end = answer_start + len(answer_text)

            context_start = 0
            while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while context_end >= 0 and sequence_ids[context_end] != 1:
                context_end -= 1

            if (
                context_start >= len(offsets)
                or context_end < 0
                or offsets[context_start][0] > answer_start
                or offsets[context_end][1] < answer_end
            ):
                features.append(feature)
                continue

            token_start = context_start
            while token_start <= context_end and offsets[token_start][0] <= answer_start:
                token_start += 1
            token_end = context_end
            while token_end >= context_start and offsets[token_end][1] >= answer_end:
                token_end -= 1
            feature["start_position"] = token_start - 1
            feature["end_position"] = token_end + 1
            features.append(feature)
    return features


def _prepare_validation_features(
    *,
    examples: Sequence[SquadExample],
    tokenizer,
    max_seq_length: int,
    doc_stride: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for example in examples:
        tokenized = tokenizer(
            example.question.strip(),
            example.context,
            truncation="only_second",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for feature_index, input_ids in enumerate(tokenized["input_ids"]):
            sequence_ids = tokenized.sequence_ids(feature_index)
            offsets = []
            for offset, sequence_id in zip(
                tokenized["offset_mapping"][feature_index], sequence_ids
            ):
                offsets.append(tuple(offset) if sequence_id == 1 else None)
            feature = {
                "input_ids": input_ids,
                "attention_mask": tokenized["attention_mask"][feature_index],
                "example_id": example.qid,
                "offset_mapping": offsets,
            }
            if "token_type_ids" in tokenized:
                feature["token_type_ids"] = tokenized["token_type_ids"][feature_index]
            features.append(feature)
    return features


def _postprocess_predictions(
    *,
    examples: Sequence[SquadExample],
    features: Sequence[dict[str, Any]],
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    n_best_size: int,
    max_answer_length: int,
) -> dict[str, str]:
    features_per_example: dict[str, list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[feature["example_id"]].append(feature_index)

    predictions: dict[str, str] = {}
    for example in examples:
        best_score = float("-inf")
        best_answer = ""
        for feature_index in features_per_example[example.qid]:
            offsets = features[feature_index]["offset_mapping"]
            start_indexes = np.argsort(start_logits[feature_index])[-n_best_size:][::-1]
            end_indexes = np.argsort(end_logits[feature_index])[-n_best_size:][::-1]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if start_index >= len(offsets) or end_index >= len(offsets):
                        continue
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if end_index < start_index:
                        continue
                    answer_length = end_index - start_index + 1
                    if answer_length > max_answer_length:
                        continue
                    score = float(start_logits[feature_index][start_index])
                    score += float(end_logits[feature_index][end_index])
                    if score <= best_score:
                        continue
                    start_char = offsets[start_index][0]
                    end_char = offsets[end_index][1]
                    best_score = score
                    best_answer = example.context[start_char:end_char]
        predictions[example.qid] = best_answer
    return predictions


def _predict_with_model(
    *,
    model,
    tokenizer,
    examples: Sequence[SquadExample],
    model_cfg: dict,
    training_cfg: dict,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, str]:
    features = _prepare_validation_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=int(model_cfg["max_seq_length"]),
        doc_stride=int(model_cfg["doc_stride"]),
    )
    dataloader = DataLoader(
        QaFeatureDataset(features, include_positions=False),
        batch_size=int(training_cfg["per_device_eval_batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
    )
    model.eval()
    start_logits: list[np.ndarray] = []
    end_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="qa predict", leave=False):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast_context(device, dtype):
                outputs = model(**batch)
            start_logits.append(outputs.start_logits.float().cpu().numpy())
            end_logits.append(outputs.end_logits.float().cpu().numpy())

    if not start_logits:
        return {}
    return _postprocess_predictions(
        examples=examples,
        features=features,
        start_logits=np.concatenate(start_logits, axis=0),
        end_logits=np.concatenate(end_logits, axis=0),
        n_best_size=int(model_cfg.get("n_best_size", 20)),
        max_answer_length=int(model_cfg.get("max_answer_length", 30)),
    )


def predict_squad(
    *,
    model_dir_or_name: str | Path,
    examples: Sequence[SquadExample],
    model_cfg: dict,
    training_cfg: dict,
    runtime_cfg: dict,
) -> dict[str, str]:
    model, tokenizer, device, dtype = _load_qa_model_and_tokenizer(
        model_dir_or_name,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        runtime_cfg=runtime_cfg,
    )
    logger.info("Predicting SQuAD answers for {} examples", len(examples))
    return _predict_with_model(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        device=device,
        dtype=dtype,
    )


def train_qa_model(
    *,
    model_name_or_path: str,
    output_dir: str | Path,
    train_examples: Sequence[SquadExample],
    eval_examples: Sequence[SquadExample],
    training_cfg: dict,
    model_cfg: dict,
    runtime_cfg: dict,
    seed: int,
) -> QaTrainingArtifacts:
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device, dtype = _load_qa_model_and_tokenizer(
        model_name_or_path,
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        runtime_cfg=runtime_cfg,
    )

    train_features = _prepare_train_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=int(model_cfg["max_seq_length"]),
        doc_stride=int(model_cfg["doc_stride"]),
    )
    train_loader = DataLoader(
        QaFeatureDataset(train_features, include_positions=True),
        batch_size=int(training_cfg["per_device_train_batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("dataloader_num_workers", 0)),
    )

    grad_accum = int(training_cfg["gradient_accumulation_steps"])
    num_epochs = int(float(training_cfg["num_train_epochs"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    update_steps_per_epoch = max(1, (len(train_loader) + grad_accum - 1) // grad_accum)
    total_train_steps = max(1, num_epochs * update_steps_per_epoch)
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

    logger.info(
        "Training QA model on {} examples / {} features for {} epochs on {}",
        len(train_examples),
        len(train_features),
        num_epochs,
        device,
    )
    best_f1 = float("-inf")
    best_metrics: dict[str, float] = {}
    for epoch_index in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            train_loader, desc="qa train epoch {}".format(epoch_index + 1), leave=False
        )
        for step_index, batch in enumerate(progress, start=1):
            batch = {name: tensor.to(device) for name, tensor in batch.items()}
            with _autocast_context(device, dtype):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step_index % grad_accum == 0 or step_index == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(training_cfg.get("max_grad_norm", 1.0)),
                )
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        predictions = _predict_with_model(
            model=model,
            tokenizer=tokenizer,
            examples=eval_examples,
            model_cfg=model_cfg,
            training_cfg=training_cfg,
            device=device,
            dtype=dtype,
        )
        metrics = evaluate_squad_predictions(eval_examples, predictions)
        metrics["epoch"] = float(epoch_index + 1)
        logger.info(
            "Epoch {} SQuAD eval exact_match={:.4f} f1={:.4f}",
            epoch_index + 1,
            metrics["exact_match"],
            metrics["f1"],
        )
        if metrics["f1"] >= best_f1:
            best_f1 = metrics["f1"]
            best_metrics = dict(metrics)
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info("Saved new best QA checkpoint to {}", output_path)

    (output_path / "metrics.json").write_text(
        json.dumps(best_metrics, indent=2),
        encoding="utf-8",
    )
    return QaTrainingArtifacts(output_dir=output_path, metrics=best_metrics)


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(character for character in value if character not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn,
    prediction: str,
    ground_truths: Sequence[str],
) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def evaluate_squad_predictions(
    examples: Sequence[SquadExample],
    predictions: dict[str, str],
) -> dict[str, float]:
    exact_match = 0.0
    f1 = 0.0
    for example in examples:
        ground_truths = [answer["text"] for answer in example.answers]
        if not ground_truths:
            ground_truths = [""]
        prediction = predictions.get(example.qid, "")
        exact_match += metric_max_over_ground_truths(
            exact_match_score,
            prediction,
            ground_truths,
        )
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    total = len(examples)
    return {
        "exact_match": 100.0 * exact_match / total if total else 0.0,
        "f1": 100.0 * f1 / total if total else 0.0,
        "num_examples": float(total),
    }


def evaluate_prediction_agreement(
    *,
    victim_predictions: dict[str, str],
    extracted_predictions: dict[str, str],
) -> dict[str, float]:
    qids = sorted(set(victim_predictions) & set(extracted_predictions))
    exact = 0.0
    f1 = 0.0
    for qid in qids:
        victim_answer = victim_predictions[qid]
        extracted_answer = extracted_predictions[qid]
        exact += exact_match_score(extracted_answer, victim_answer)
        f1 += f1_score(extracted_answer, victim_answer)
    total = len(qids)
    return {
        "agreement_exact_match": 100.0 * exact / total if total else 0.0,
        "agreement_f1": 100.0 * f1 / total if total else 0.0,
        "num_examples": float(total),
    }
