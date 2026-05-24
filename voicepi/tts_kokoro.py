from __future__ import annotations

import io
import threading
import time

import soundfile as sf

from .text_utils import clean_for_tts


class KokoroTTS:
    """Free local TTS engine based on Kokoro.

    It exposes the same synthesize_wav() method as PiperTTS, so the rest of
    the project does not need to know which TTS backend is active.
    """

    mime_type = "audio/wav"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.normalize_text = bool(cfg.get("tts.normalize_text", True))
        self.lang_code = str(cfg.get("tts.kokoro.lang_code", "a"))
        self.voice = str(cfg.get("tts.kokoro.voice", "af_heart"))
        self.sample_rate = int(cfg.get("tts.kokoro.sample_rate", 24000))
        self.timeout_s = float(cfg.get("tts.synth_timeout_s", 35))

        self._pipeline = None
        self._lock = threading.Lock()

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from kokoro import KPipeline  # type: ignore

        self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def synthesize_wav(self, text: str, cancel_event: threading.Event | None = None) -> bytes:
        text = clean_for_tts(text) if self.normalize_text else (text or "").strip()

        if not text:
            return b""

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS cancelled")

        started = time.monotonic()
        audio_parts = []

        with self._lock:
            pipeline = self._load_pipeline()

            generator = pipeline(text, voice=self.voice)

            for _graphemes, _phonemes, audio in generator:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("TTS cancelled")

                if self.timeout_s > 0 and time.monotonic() - started > self.timeout_s:
                    raise RuntimeError(f"Kokoro TTS timed out after {self.timeout_s:.1f}s")

                audio_parts.append(audio)

        if not audio_parts:
            return b""

        # Kokoro returns numpy arrays. Concatenate safely.
        try:
            import numpy as np

            audio_all = np.concatenate(audio_parts)
        except Exception:
            audio_all = audio_parts[0]

        out = io.BytesIO()
        sf.write(out, audio_all, self.sample_rate, format="WAV")
        return out.getvalue()