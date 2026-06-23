from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AssistantConfig:
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    chunk_size: int = 900
    chunk_overlap: int = 140
    top_k: int = 5
    index_path: str = "storage/faiss"
    faithfulness_judge: bool = False


def load_config(path: str | Path = "configs/default.yaml") -> AssistantConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping.")
    return AssistantConfig(**raw)


def merge_config(config: AssistantConfig, **overrides: Any) -> AssistantConfig:
    data = {**config.__dict__}
    data.update({key: value for key, value in overrides.items() if value is not None})
    return AssistantConfig(**data)
