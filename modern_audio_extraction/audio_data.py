from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Audio, load_dataset
from loguru import logger
import numpy as np
import torch
import torchaudio.functional as AF


@dataclass
class AudioExample:
    guid: str
    audio: torch.Tensor
    sampling_rate: int
    label: int | None = None


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 1:
        return audio
    return audio.mean(dim=0)


def normalize_audio(
    audio: torch.Tensor,
    *,
    source_sampling_rate: int,
    target_sampling_rate: int,
    max_duration_seconds: float,
) -> torch.Tensor:
    audio = _to_mono(audio.float())
    if source_sampling_rate != target_sampling_rate:
        audio = AF.resample(audio, source_sampling_rate, target_sampling_rate)
    max_samples = int(target_sampling_rate * max_duration_seconds)
    if max_samples > 0 and audio.numel() > max_samples:
        audio = audio[:max_samples]
    if audio.numel() == 0:
        audio = torch.zeros(1, dtype=torch.float32)
    peak = audio.abs().max()
    if peak > 1.0:
        audio = audio / peak
    return audio.contiguous()


def _take_budget(dataset, budget: int | None, seed: int):
    if budget is None:
        return dataset
    budget = min(int(budget), len(dataset))
    if budget <= 0:
        return dataset.select([])
    return dataset.shuffle(seed=seed).select(range(budget))


def _load_dataset_with_split_alias(dataset_name: str, dataset_config: str | None, split: str):
    try:
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=split)
        return load_dataset(dataset_name, split=split)
    except ValueError as exc:
        alias = {
            "train.clean.100": "train.100",
            "train.clean.360": "train.360",
            "validation.clean": "validation",
            "test.clean": "test",
        }.get(split)
        if alias is None or "Unknown split" not in str(exc):
            raise
        logger.info(
            "Dataset split '{}' is unavailable for {} {}; retrying with '{}'",
            split,
            dataset_name,
            dataset_config or "",
            alias,
        )
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, split=alias)
        return load_dataset(dataset_name, split=alias)


def load_hf_audio_examples(
    *,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    audio_column: str,
    label_column: str | None,
    budget: int | None,
    seed: int,
    sampling_rate: int,
    max_duration_seconds: float,
) -> list[AudioExample]:
    logger.info(
        "Loading HF audio dataset name={} config={} split={} budget={}",
        dataset_name,
        dataset_config,
        split,
        budget,
    )
    dataset = _load_dataset_with_split_alias(dataset_name, dataset_config, split)
    dataset = dataset.cast_column(audio_column, Audio(sampling_rate=sampling_rate))
    dataset = _take_budget(dataset, budget=budget, seed=seed)

    examples: list[AudioExample] = []
    for index, row in enumerate(dataset):
        audio_data = row[audio_column]
        waveform = torch.tensor(np.asarray(audio_data["array"]), dtype=torch.float32)
        waveform = normalize_audio(
            waveform,
            source_sampling_rate=int(audio_data["sampling_rate"]),
            target_sampling_rate=sampling_rate,
            max_duration_seconds=max_duration_seconds,
        )
        label = int(row[label_column]) if label_column else None
        examples.append(
            AudioExample(
                guid=str(row.get("id", row.get("file", index))),
                audio=waveform,
                sampling_rate=sampling_rate,
                label=label,
            )
        )
    logger.info("Loaded {} HF audio examples", len(examples))
    return examples


def load_synthetic_audio_examples(
    *,
    source_type: str,
    budget: int,
    seed: int,
    sampling_rate: int,
    duration_seconds: float,
) -> list[AudioExample]:
    generator = torch.Generator().manual_seed(seed)
    num_samples = max(1, int(sampling_rate * duration_seconds))
    examples: list[AudioExample] = []
    timeline = torch.arange(num_samples, dtype=torch.float32) / sampling_rate
    for index in range(int(budget)):
        if source_type == "synthetic_noise":
            waveform = torch.randn(num_samples, generator=generator) * 0.2
        elif source_type == "synthetic_sine":
            frequency = 120.0 + 40.0 * (index % 20)
            phase = torch.rand((), generator=generator).item() * 2.0 * torch.pi
            waveform = 0.2 * torch.sin(2.0 * torch.pi * frequency * timeline + phase)
        else:
            raise ValueError("Unsupported synthetic audio source: {}".format(source_type))
        examples.append(
            AudioExample(
                guid="{}-{}".format(source_type, index),
                audio=waveform.contiguous(),
                sampling_rate=sampling_rate,
            )
        )
    logger.info("Generated {} synthetic {} examples", len(examples), source_type)
    return examples


def load_query_examples(query_cfg: dict[str, Any], *, seed: int) -> list[AudioExample]:
    source_type = str(query_cfg["type"])
    sampling_rate = int(query_cfg.get("sampling_rate", 16000))
    max_duration_seconds = float(query_cfg.get("max_duration_seconds", 4.0))
    budget = int(query_cfg["budget"])
    if source_type == "hf_audio":
        return load_hf_audio_examples(
            dataset_name=query_cfg["dataset_name"],
            dataset_config=query_cfg.get("dataset_config"),
            split=query_cfg["split"],
            audio_column=query_cfg.get("audio_column", "audio"),
            label_column=query_cfg.get("label_column"),
            budget=budget,
            seed=seed,
            sampling_rate=sampling_rate,
            max_duration_seconds=max_duration_seconds,
        )
    if source_type in {"synthetic_noise", "synthetic_sine"}:
        return load_synthetic_audio_examples(
            source_type=source_type,
            budget=budget,
            seed=seed,
            sampling_rate=sampling_rate,
            duration_seconds=max_duration_seconds,
        )
    raise ValueError("Unsupported query source type: {}".format(source_type))
