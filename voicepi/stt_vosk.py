from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal, Optional

_MODEL_CACHE: dict[str, object] = {}
_MODEL_LOCK = threading.Lock()


class VoskStreamSTT:
    """A Vosk recognizer scoped to one browser session.

    Important: Vosk's AcceptWaveform() "final" means it found a phrase segment, not
    necessarily that the human is done speaking. The session commits a user turn only
    when the browser sends speech_end after client-side VAD silence.
    """

    def __init__(self, model_path: Path, sample_rate: int = 16000) -> None:
        try:
            import vosk  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Vosk is not installed. Run scripts/install_pi.sh or pip install vosk.") from exc

        self.vosk = vosk
        self.sample_rate = sample_rate
        key = str(model_path.resolve())
        with _MODEL_LOCK:
            if key not in _MODEL_CACHE:
                if not model_path.exists():
                    raise FileNotFoundError(f"Vosk model not found: {model_path}")
                _MODEL_CACHE[key] = vosk.Model(str(model_path))
            self.model = _MODEL_CACHE[key]
        self.recognizer = self._new_recognizer()
        self.last_partial = ""
        self.segment_texts: list[str] = []

    def _new_recognizer(self):
        recognizer = self.vosk.KaldiRecognizer(self.model, float(self.sample_rate))
        recognizer.SetWords(True)
        return recognizer

    def begin_utterance(self) -> None:
        self.recognizer = self._new_recognizer()
        self.last_partial = ""
        self.segment_texts = []

    def accept_pcm16le(self, pcm: bytes) -> tuple[Literal["partial", "segment", "none"], str, dict]:
        if not pcm:
            return "none", "", {}
        if self.recognizer.AcceptWaveform(pcm):
            raw = json.loads(self.recognizer.Result())
            text = (raw.get("text") or "").strip()
            self.last_partial = ""
            if text:
                self.segment_texts.append(text)
                return "segment", self._combined_text(), raw
            return "none", "", raw
        raw = json.loads(self.recognizer.PartialResult())
        partial = (raw.get("partial") or "").strip()
        if partial and partial != self.last_partial:
            self.last_partial = partial
            combined = " ".join([*self.segment_texts, partial]).strip()
            return "partial", combined, raw
        return "none", "", raw

    def finish_utterance(self) -> tuple[str, dict]:
        raw = json.loads(self.recognizer.FinalResult())
        final_text = (raw.get("text") or "").strip()
        parts = [*self.segment_texts]
        if final_text:
            parts.append(final_text)
        text = " ".join(p for p in parts if p).strip()
        self.begin_utterance()
        return text, raw

    def finish(self) -> Optional[str]:
        text, _raw = self.finish_utterance()
        return text or None

    def _combined_text(self) -> str:
        return " ".join(p for p in self.segment_texts if p).strip()
