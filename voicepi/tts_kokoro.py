"""Kokoro TTS via kokoro-onnx (ONNX Runtime) - no PyTorch required.

Install model files once:
  python scripts/download_models.py --tts-engine kokoro

Voices: af_heart (default), af_bella, af_nicole, am_adam, am_michael,
        bf_emma, bf_isabella, bm_george, bm_lewis
"""
from __future__ import annotations

import io
import logging
import threading
import time

import soundfile as sf

from .text_utils import clean_for_tts

log = logging.getLogger(__name__)


class KokoroTTS:
    """Kokoro 82M via ONNX Runtime - ARM-friendly, no PyTorch needed.

    Speed tips (all tunable in config.yaml):
      - tts.stream_during_generation: true   <- biggest win: play first sentence ASAP
      - tts.sentence_min_chars: 25           <- shorter first chunk = faster first audio
      - tts.kokoro.warmup: true              <- synthesize dummy text at startup so
                                                first real call skips ONNX JIT overhead
    """

    mime_type = "audio/wav"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.normalize_text = bool(cfg.get("tts.normalize_text", True))
        self.voice = str(cfg.get("tts.kokoro.voice", "af_heart"))
        self.speed = float(cfg.get("tts.kokoro.speed", 1.0))
        self.lang = str(cfg.get("tts.kokoro.lang", "en-us"))
        self.sample_rate = int(cfg.get("tts.kokoro.sample_rate", 24000))
        self.timeout_s = float(cfg.get("tts.synth_timeout_s", 35))

        self._model_path = str(cfg.path("tts.kokoro.model_path", "models/kokoro/kokoro-v1.0.int8.onnx"))
        self._voices_path = str(cfg.path("tts.kokoro.voices_path", "models/kokoro/voices-v1.0.bin"))

        self._kokoro = None
        self._lock = threading.Lock()
        self._ready = threading.Event()

        # Pre-warm in background so first user request doesn't pay cold-start cost.
        # Synthesizing a short phrase forces ONNX to load and JIT-compile the graph.
        if bool(cfg.get("tts.kokoro.warmup", True)):
            t = threading.Thread(target=self._warmup, name="voicepi-kokoro-warmup", daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def synthesize_wav(self, text: str, cancel_event: threading.Event | None = None) -> bytes:
        text = clean_for_tts(text) if self.normalize_text else (text or "").strip()
        if not text:
            return b""

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

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self):
        """Load Kokoro (call inside self._lock)."""
        if self._kokoro is not None:
            return self._kokoro
        try:
            from kokoro_onnx import Kokoro  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "kokoro-onnx is not installed. Run: pip install kokoro-onnx\n"
                "Then: python scripts/download_models.py --tts-engine kokoro"
            ) from exc
        self._kokoro = Kokoro(self._model_path, self._voices_path)
        return self._kokoro

    def _warmup(self) -> None:
        """Background pre-warm: load model + run one inference so ONNX JIT is done
        before the first real user request arrives."""
        try:
            with self._lock:
                kokoro = self._load()
            # A very short phrase is enough to trigger full graph compilation.
            kokoro.create("Hi.", voice=self.voice, speed=self.speed, lang=self.lang)
            log.info("kokoro warmup done")
        except Exception as exc:
            log.warning("kokoro warmup failed (non-fatal): %s", exc)
        finally:
            self._ready.set()
