"""Optional local text embeddings for semantic memory recall.

The memory layer works without embeddings (FTS5 keyword recall). When an embedder
is configured, ``MemoryStore.recall`` additionally re-ranks candidates by cosine
similarity, so the robot can recall *meaning* ("what do I enjoy?" -> a past
"I love building robots" turn) rather than only matching words.

Design notes for Raspberry Pi 5
-------------------------------
* A separate, tiny GGUF embedding model is used (e.g. nomic-embed-text or bge-small,
  ~30-130 MB). It is loaded lazily on first use so boot stays fast, and guarded by a
  lock because llama.cpp model handles are not safe for concurrent calls.
* Embedding one short sentence is cheap; we only embed when storing an episode and
  once per recall query, never inside the token-generation hot path.
* If llama-cpp-python or the model file is missing, ``build_embedder`` returns None
  and the system silently falls back to keyword recall.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional


class LlamaEmbedder:
    """Wraps a small GGUF embedding model via llama-cpp-python."""

    def __init__(self, model_path: str | Path, n_ctx: int = 512, n_threads: int = 4) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {self.model_path}")
        self.n_ctx = int(n_ctx)
        self.n_threads = int(n_threads)
        self._lock = threading.Lock()
        self._model = None  # lazy

    def _ensure(self):
        if self._model is None:
            from llama_cpp import Llama  # type: ignore

            self._model = Llama(
                model_path=str(self.model_path),
                embedding=True,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,
                verbose=False,
            )
        return self._model

    def __call__(self, text: str) -> list[float]:
        text = (text or "").strip()
        if not text:
            return []
        with self._lock:
            model = self._ensure()
            out = model.embed(text)
        # llama-cpp may return list[float] or list[list[float]] depending on version.
        if out and isinstance(out[0], (list, tuple)):
            return [float(x) for x in out[0]]
        return [float(x) for x in out]


def build_embedder(cfg) -> Optional[LlamaEmbedder]:
    """Construct an embedder from config, or return None to use keyword-only recall."""
    if not bool(cfg.get("memory.embeddings.enabled", False)):
        return None
    try:
        model_path = cfg.path("memory.embeddings.model_path")
    except Exception:
        return None
    try:
        return LlamaEmbedder(
            model_path,
            n_ctx=int(cfg.get("memory.embeddings.n_ctx", 512)),
            n_threads=int(cfg.get("memory.embeddings.n_threads", cfg.get("llm.n_threads", 4))),
        )
    except Exception:
        # Missing model file or llama-cpp not built with embedding support.
        return None