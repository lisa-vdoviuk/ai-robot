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

        llm_engine = str(self.get("llm.engine", "llama_cpp")).strip().lower()
        if llm_engine in {"llama_cpp", "local", "llama"}:
            required_paths.append("llm.model_path")

        tts_engine = str(self.get("tts.engine", "piper")).strip().lower()
        if tts_engine in {"piper", "local"}:
            required_paths.append("tts.model_path")
        elif tts_engine == "kokoro":
            required_paths.append("tts.kokoro.model_path")
            required_paths.append("tts.kokoro.voices_path")

        for dotted in required_paths:
            p = self.path(dotted)
            if not p.exists():
                missing.append(f"{dotted}: {p}")

        if tts_engine in {"piper", "local"}:
            tts_config = self.get("tts.config_path")
            if tts_config:
                p = self.path("tts.config_path")
                if not p.exists():
                    missing.append(f"tts.config_path: {p}")
        # edge-tts: no model files needed

        return missing
