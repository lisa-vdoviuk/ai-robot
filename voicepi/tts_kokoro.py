"""Kokoro TTS via kokoro-onnx.

This engine is the default offline TTS path for Raspberry Pi 5. It keeps one
ONNX Runtime model loaded, synthesizes one chunk at a time, warms up in the
background, and applies a small pronunciation dictionary before inference.
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
from typing import Any

import soundfile as sf

from .text_utils import clean_for_tts

log = logging.getLogger(__name__)

_DEFAULT_PRONUNCIATIONS: dict[str, str] = {
    "YOLO": "you only look once",
    "ONNX": "onix",
    "GPIO": "G P I O",
    "HC-SR04": "H C S R zero four",
    "ESP32": "E S P thirty two",
    "Raspberry Pi": "Raspberry Pie",
    "Pi 5": "Pie Five",
    "Vosk": "vosk",
    "Kokoro": "ko ko ro",
    "Groq": "grock",
}


class KokoroTTS:
    """Kokoro 82M via ONNX Runtime; no PyTorch required."""

    mime_type = "audio/wav"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.normalize_text = bool(cfg.get("tts.normalize_text", True))
        self.voice = str(cfg.get("tts.kokoro.voice", "af_heart"))
        self.speed = float(cfg.get("tts.kokoro.speed", 1.05))
        self.lang = str(cfg.get("tts.kokoro.lang", "en-us"))
        self.sample_rate = int(cfg.get("tts.kokoro.sample_rate", 24000))
        self.timeout_s = float(cfg.get("tts.synth_timeout_s", 25))
        self.max_chars = int(cfg.get("tts.kokoro.max_chars", 420))
        self.warmup_text = str(cfg.get("tts.kokoro.warmup_text", "Ready."))
        self.threads = max(1, int(cfg.get("tts.kokoro.threads", 2)))

        # Keep Pi 5 responsive: avoid ONNX/OpenBLAS oversubscribing all cores.
        os.environ.setdefault("OMP_NUM_THREADS", str(self.threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.threads))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(self.threads))

        self._model_path = str(cfg.path("tts.kokoro.model_path", "models/kokoro/kokoro-v1.0.int8.onnx"))
        self._voices_path = str(cfg.path("tts.kokoro.voices_path", "models/kokoro/voices-v1.0.bin"))
        self._kokoro = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._pronunciation = self._load_pronunciations(cfg)

        if bool(cfg.get("tts.kokoro.warmup", True)):
            threading.Thread(target=self._warmup, name="voicepi-kokoro-warmup", daemon=True).start()
        else:
            self._ready.set()

    def synthesize_wav(self, text: str, cancel_event: threading.Event | None = None) -> bytes:
        text = self._prepare_text(text)
        if not text:
            return b""
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rsplit(" ", 1)[0] or text[: self.max_chars]

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS cancelled")

        started = time.monotonic()
        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("TTS cancelled")
            if self.timeout_s > 0 and time.monotonic() - started > self.timeout_s:
                raise RuntimeError(f"Kokoro TTS timed out after {self.timeout_s:.1f}s")

            kokoro = self._load()
            samples, sample_rate = kokoro.create(
                text,
                voice=self.voice,
                speed=self.speed,
                lang=self.lang,
            )

        out = io.BytesIO()
        sf.write(out, samples, sample_rate, format="WAV")
        return out.getvalue()

    def _prepare_text(self, text: str) -> str:
        text = clean_for_tts(text) if self.normalize_text else (text or "").strip()
        return self._apply_pronunciation(text)

    def _load_pronunciations(self, cfg) -> dict[str, str]:
        custom = cfg.get("tts.pronunciation", {}) or {}
        merged: dict[str, str] = dict(_DEFAULT_PRONUNCIATIONS)
        if isinstance(custom, dict):
            for key, value in custom.items():
                if key and value:
                    merged[str(key)] = str(value)
        return merged

    def _apply_pronunciation(self, text: str) -> str:
        for phrase, spoken in sorted(self._pronunciation.items(), key=lambda item: len(item[0]), reverse=True):
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
            text = pattern.sub(spoken, text)
        return text

    def _load(self):
        if self._kokoro is not None:
            return self._kokoro
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run: pip install -U kokoro-onnx soundfile\n"
                "Then: python scripts/download_models.py --skip-llm --tts-engine kokoro"
            ) from exc
        self._kokoro = Kokoro(self._model_path, self._voices_path)
        return self._kokoro

    def _warmup(self) -> None:
        try:
            with self._lock:
                kokoro = self._load()
                kokoro.create(self.warmup_text, voice=self.voice, speed=self.speed, lang=self.lang)
            log.info("kokoro warmup done")
        except Exception as exc:
            log.warning("kokoro warmup failed (non-fatal): %s", exc)
        finally:
            self._ready.set()
