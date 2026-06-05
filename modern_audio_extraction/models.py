from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
import torch
from torch import nn
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoProcessor


def load_audio_processor(model_name_or_path: str):
    try:
        return AutoProcessor.from_pretrained(model_name_or_path)
    except (OSError, TypeError, ValueError) as exc:
        logger.info(
            "AutoProcessor is not usable for '{}'; falling back to AutoFeatureExtractor. Reason: {}",
            model_name_or_path,
            exc,
        )
        return AutoFeatureExtractor.from_pretrained(model_name_or_path)


def load_audio_encoder(model_name_or_path: str, *, init_from_pretrained: bool = True):
    if init_from_pretrained:
        logger.info("Loading pretrained audio encoder {}", model_name_or_path)
        return AutoModel.from_pretrained(model_name_or_path)
    logger.info("Initializing audio encoder from config {}", model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)
    return AutoModel.from_config(config)


def freeze_feature_encoder_if_available(model) -> None:
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        return
    if hasattr(model, "freeze_feature_extractor"):
        model.freeze_feature_extractor()


class StolenAudioEncoder(nn.Module):
    def __init__(self, student_model, *, target_hidden_size: int):
        super().__init__()
        self.student_model = student_model
        student_hidden_size = int(student_model.config.hidden_size)
        self.target_hidden_size = int(target_hidden_size)
        if student_hidden_size == self.target_hidden_size:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(student_hidden_size, self.target_hidden_size)

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None):
        kwargs = {"input_values": input_values}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        outputs = self.student_model(**kwargs)
        return self.projection(outputs.last_hidden_state)

    @property
    def hidden_size(self) -> int:
        return self.target_hidden_size


def save_stolen_encoder(
    *,
    output_dir: str | Path,
    stolen_encoder: StolenAudioEncoder,
    processor,
    metadata: dict[str, Any],
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    student_dir = output_path / "student_model"
    stolen_encoder.student_model.save_pretrained(student_dir)
    processor.save_pretrained(output_path / "processor")
    torch.save(stolen_encoder.projection.state_dict(), output_path / "projection.pt")
    (output_path / "stolen_encoder_config.json").write_text(
        json.dumps(
            {
                "target_hidden_size": stolen_encoder.target_hidden_size,
                "metadata": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_stolen_encoder(output_dir: str | Path):
    output_path = Path(output_dir)
    config = json.loads((output_path / "stolen_encoder_config.json").read_text(encoding="utf-8"))
    processor = load_audio_processor(str(output_path / "processor"))
    student_model = AutoModel.from_pretrained(output_path / "student_model")
    encoder = StolenAudioEncoder(
        student_model,
        target_hidden_size=int(config["target_hidden_size"]),
    )
    projection_state = torch.load(output_path / "projection.pt", map_location="cpu")
    encoder.projection.load_state_dict(projection_state)
    return encoder, processor, config
