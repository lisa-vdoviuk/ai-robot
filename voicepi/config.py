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
        for dotted in ["stt.model_path", "llm.model_path", "tts.model_path"]:
            p = self.path(dotted)
            if not p.exists():
                missing.append(f"{dotted}: {p}")
        tts_config = self.get("tts.config_path")
        if tts_config:
            p = self.path("tts.config_path")
            if not p.exists():
                missing.append(f"tts.config_path: {p}")
        return missing
