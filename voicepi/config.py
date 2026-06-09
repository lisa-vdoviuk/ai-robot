from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class Config:
    data: dict[str, Any]
    root: Path

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Config":
        cfg_path = Path(path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(data=data, root=cfg_path.parent)

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, dotted: str, default: str | None = None) -> Path:
        value = self.get(dotted, default)
        if value is None:
            raise ValueError(f"Missing path config: {dotted}")
        p = Path(str(value)).expanduser()
        if not p.is_absolute():
            p = self.root / p
        return p

    def require_files(self) -> list[str]:
        missing: list[str] = []
        required_paths = ["stt.model_path"]

        llm_engine = str(self.get("llm.engine", "groq")).strip().lower()
        if llm_engine in {"llama_cpp", "local", "llama"}:
            required_paths.append("llm.model_path")

        tts_engine = str(self.get("tts.engine", "kokoro")).strip().lower()
        if tts_engine != "kokoro":
            missing.append(f"tts.engine: unsupported value {tts_engine!r}; this build uses 'kokoro'")
        else:
            required_paths.extend(["tts.kokoro.model_path", "tts.kokoro.voices_path"])

        vision_enabled = bool(self.get("vision.object_detection.enabled", True))
        vision_backend = str(self.get("vision.object_detection.backend", "yolo")).strip().lower()
        if vision_enabled:
            if vision_backend != "yolo":
                missing.append(f"vision.object_detection.backend: unsupported value {vision_backend!r}; use 'yolo'")
            else:
                required_paths.append("vision.object_detection.model_path")

        for dotted in required_paths:
            p = self.path(dotted)
            if not p.exists():
                missing.append(f"{dotted}: {p}")

        return missing
