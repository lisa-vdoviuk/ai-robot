from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from .text_utils import clean_for_tts


class PiperTTS:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        configured_bin = str(cfg.get("tts.piper_bin", "piper"))
        self.piper_bin = shutil.which(configured_bin) or configured_bin
        self.model_path: Path = cfg.path("tts.model_path")
        self.config_path: Optional[Path] = None
        if cfg.get("tts.config_path"):
            self.config_path = cfg.path("tts.config_path")
        self.normalize_text = bool(cfg.get("tts.normalize_text", True))

    def synthesize_wav(self, text: str, cancel_event: threading.Event | None = None) -> bytes:
        text = clean_for_tts(text) if self.normalize_text else (text or "").strip()
        if not text:
            return b""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper voice model not found: {self.model_path}")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = Path(tmp.name)

        cmd = [
            self.piper_bin,
            "--model",
            str(self.model_path),
            "--output_file",
            str(out_path),
        ]
        if self.config_path and self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])
        speaker_id = self.cfg.get("tts.speaker_id")
        if speaker_id is not None:
            cmd.extend(["--speaker", str(speaker_id)])
        length_scale = self.cfg.get("tts.length_scale")
        if length_scale is not None:
            cmd.extend(["--length-scale", str(length_scale)])
        noise_scale = self.cfg.get("tts.noise_scale")
        if noise_scale is not None:
            cmd.extend(["--noise-scale", str(noise_scale)])
        noise_w = self.cfg.get("tts.noise_w")
        if noise_w is not None:
            cmd.extend(["--noise-w", str(noise_w)])
        sentence_silence = self.cfg.get("tts.sentence_silence")
        if sentence_silence is not None:
            cmd.extend(["--sentence-silence", str(sentence_silence)])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        assert proc.stdin is not None
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()

        while proc.poll() is None:
            if cancel_event is not None and cancel_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=0.8)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError("TTS cancelled")
            time.sleep(0.02)

        stderr = b""
        if proc.stderr is not None:
            try:
                stderr = proc.stderr.read()
            except Exception:
                stderr = b""
        try:
            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed with code {proc.returncode}: {stderr.decode('utf-8', errors='ignore')}")
            return out_path.read_bytes()
        finally:
            try:
                os.unlink(out_path)
            except FileNotFoundError:
                pass
