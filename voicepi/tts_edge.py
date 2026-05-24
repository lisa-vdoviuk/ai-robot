"""Edge TTS -- Microsoft Edge neural voices, free, no API key, requires internet.

Excellent neural quality (AriaNeural, GuyNeural, JennyNeural, SoniaNeural...).
Latency: ~0.5-1 s on typical home internet.

Install: pip install edge-tts
No model files needed.
"""
from __future__ import annotations

import asyncio
import subprocess
import threading
import time

from .text_utils import clean_for_tts

# Best voices by use case:
# Female warm:        en-US-AriaNeural
# Male professional:  en-US-GuyNeural
# Female friendly:    en-US-JennyNeural
# Male casual:        en-US-EricNeural
# British female:     en-GB-SoniaNeural
# British male:       en-GB-RyanNeural
# Australian female:  en-AU-NatashaNeural


class EdgeTTS:
    """Text-to-speech using Microsoft Edge TTS (neural quality, free, internet required)."""

    mime_type = "audio/wav"

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.normalize_text = bool(cfg.get("tts.normalize_text", True))
        self.voice = str(cfg.get("tts.edge.voice", "en-US-AriaNeural"))
        self.rate = str(cfg.get("tts.edge.rate", "+0%"))
        self.volume = str(cfg.get("tts.edge.volume", "+0%"))
        self.timeout_s = float(cfg.get("tts.synth_timeout_s", 35))

    def synthesize_wav(self, text: str, cancel_event: threading.Event | None = None) -> bytes:
        try:
            import edge_tts  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is not installed. Run: pip install edge-tts"
            ) from exc

        text = clean_for_tts(text) if self.normalize_text else (text or "").strip()
        if not text:
            return b""

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS cancelled")

        try:
            mp3_data = asyncio.run(self._synth_mp3(text))
        except Exception as exc:
            raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc

        if not mp3_data:
            return b""

        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("TTS cancelled")

        return _mp3_to_wav(mp3_data, timeout=min(self.timeout_s, 20))

    async def _synth_mp3(self, text: str) -> bytes:
        import edge_tts  # type: ignore

        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
        )
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)


def _mp3_to_wav(mp3_data: bytes, timeout: float = 20.0) -> bytes:
    """Convert MP3 bytes -> WAV bytes using ffmpeg (already installed via apt)."""
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "mp3",
            "-i", "pipe:0",
            "-f", "wav",
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "pipe:1",
        ],
        input=mp3_data,
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg MP3->WAV conversion failed: {stderr}")
    return proc.stdout
