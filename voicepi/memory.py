"""Local long-term memory for the VoicePi robot.

Design goals
------------
* **Fast.** Memory is read on every conversational turn, before generation.
  A network round-trip (e.g. Supabase) would fight directly against the
  latency target, so the primary store is local SQLite. Reads are sub-millisecond.
* **Offline-first.** The robot keeps working with no Wi-Fi.
* **Vector-ready, but not vector-required.** Episodic recall uses SQLite FTS5
  (BM25 keyword search) out of the box. If an embedding function is supplied,
  recall additionally scores candidates by cosine similarity for semantic hits.
  This keeps the Pi light while leaving a clean upgrade path.
* **Safe to add.** Everything degrades gracefully: no FTS5 -> LIKE fallback,
  no embedder -> keyword-only recall, DB error -> empty context (never crashes a turn).

Three kinds of memory are stored:

1. ``user_profile``  - durable key/value facts about the user
   (name, language, preferences, important people/objects).
2. ``episodes``      - episodic memory: one row per conversational turn,
   indexed for hybrid recall.
3. ``summaries``     - rolling natural-language summaries of a session, so the
   assistant has continuity without replaying every raw turn.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

# An embedder turns text into a list[float]. Optional; supply later for semantic recall.
Embedder = Callable[[str], list[float]]


def _now() -> float:
    return time.time()


def _iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts if ts is not None else _now()))


# ---------------------------------------------------------------------------
# Lightweight, no-LLM fact extraction (English + Ukrainian).
# This runs after a turn to capture durable facts cheaply, without spending a
# second model forward pass. A stronger optional LLM extractor can be layered on
# top later; this heuristic alone already personalizes the assistant.
# ---------------------------------------------------------------------------

_NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-zЀ-ӿ][\w'\-]{1,30})", re.IGNORECASE),
    re.compile(r"\bcall me\s+([A-Za-zЀ-ӿ][\w'\-]{1,30})", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Z][\w'\-]{1,30})\b"),  # "I am Stas" (capitalized -> likely a name)
    re.compile(r"\b(?:мене звати|моє ім'я|я)\s+([А-ЯІЇЄҐ][\w'\-]{1,30})"),
]

_LIKE_PATTERNS = [
    re.compile(r"\bi (?:really )?like\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi love\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bmy favou?rite\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bя люблю\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bмені подобається\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bмій улюблений\s+(.{2,60})", re.IGNORECASE),
]

_DISLIKE_PATTERNS = [
    re.compile(r"\bi (?:really )?(?:don't|do not|dont) like\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bi hate\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bя не люблю\s+(.{2,60})", re.IGNORECASE),
    re.compile(r"\bя ненавиджу\s+(.{2,60})", re.IGNORECASE),
]


def _clean_value(text: str) -> str:
    text = text.strip().strip(".,!?;:…").strip()
    # Cut at a sentence boundary / conjunction so we don't store a whole monologue.
    text = re.split(r"\b(?:and|but|because|so|though|бо|але|тому|і)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    return text.strip().strip(".,!?;:").strip()


def extract_facts(user_text: str) -> list[dict[str, Any]]:
    """Return a list of {key, value, confidence} durable facts from one utterance."""
    facts: list[dict[str, Any]] = []
    if not user_text:
        return facts
    text = user_text.strip()

    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            name = _clean_value(m.group(1))
            if name and name.lower() not in {"a", "the", "not", "sorry", "here", "good", "fine"}:
                facts.append({"key": "name", "value": name, "confidence": 0.9})
                break

    seen_likes: set[str] = set()
    for pat in _LIKE_PATTERNS:
        for m in pat.finditer(text):
            val = _clean_value(m.group(1))
            if val and val.lower() not in seen_likes:
                seen_likes.add(val.lower())
                facts.append({"key": "likes", "value": val, "confidence": 0.7})

    for pat in _DISLIKE_PATTERNS:
        for m in pat.finditer(text):
            val = _clean_value(m.group(1))
            if val:
                facts.append({"key": "dislikes", "value": val, "confidence": 0.7})

    return facts


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class MemoryStore:
    """Thread-safe local memory backed by SQLite.

    Parameters
    ----------
    db_path:    Path to the SQLite file. Parent dirs are created.
    embedder:   Optional callable str -> list[float] for semantic recall.
    logger:     Optional object with ``.event(source, level, message, **fields)``.
    """

    def __init__(self, db_path: str | Path, embedder: Optional[Embedder] = None, logger=None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.logger = logger
        # How many recent embedded episodes to scan per semantic recall. Bounded so
        # the cosine scan stays cheap on a small device even as the DB grows.
        self.vector_scan_limit = 300
        self._lock = threading.RLock()
        # check_same_thread=False: turns run on per-session worker threads.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.fts_enabled = self._init_db()

    # -- setup ---------------------------------------------------------------

    def _log(self, level: str, message: str, **fields: Any) -> None:
        if self.logger is not None:
            try:
                self.logger.event("memory", level, message, **fields)
            except Exception:
                pass

    def _init_db(self) -> bool:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profile (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, key, value)
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    ts REAL NOT NULL,
                    iso TEXT NOT NULL,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL DEFAULT '',
                    meta TEXT,
                    embedding TEXT
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    ts REAL NOT NULL,
                    iso TEXT NOT NULL,
                    summary TEXT NOT NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_episodes_user_ts ON episodes(user_id, ts)")

            fts_ok = False
            try:
                cur.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                    USING fts5(user_text, assistant_text, content='episodes', content_rowid='id')
                    """
                )
                # Triggers keep the FTS index in sync with the base table.
                cur.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                        INSERT INTO episodes_fts(rowid, user_text, assistant_text)
                        VALUES (new.id, new.user_text, new.assistant_text);
                    END
                    """
                )
                cur.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                        INSERT INTO episodes_fts(episodes_fts, rowid, user_text, assistant_text)
                        VALUES ('delete', old.id, old.user_text, old.assistant_text);
                    END
                    """
                )
                cur.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE ON episodes BEGIN
                        INSERT INTO episodes_fts(episodes_fts, rowid, user_text, assistant_text)
                        VALUES ('delete', old.id, old.user_text, old.assistant_text);
                        INSERT INTO episodes_fts(rowid, user_text, assistant_text)
                        VALUES (new.id, new.user_text, new.assistant_text);
                    END
                    """
                )
                fts_ok = True
            except sqlite3.OperationalError as exc:
                self._log("warning", f"FTS5 unavailable, falling back to LIKE recall: {exc}")
                fts_ok = False

            self._conn.commit()
            return fts_ok

    # -- profile facts -------------------------------------------------------

    def remember_fact(
        self,
        key: str,
        value: str,
        confidence: float = 0.6,
        source: str = "conversation",
        user_id: str = "default",
    ) -> None:
        key = (key or "").strip()
        value = (value or "").strip()
        if not key or not value:
            return
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO user_profile (user_id, key, value, confidence, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, key, value) DO UPDATE SET
                    confidence = MAX(confidence, excluded.confidence),
                    updated_at = excluded.updated_at,
                    source = excluded.source
                """,
                (user_id, key, value, float(confidence), source, now, now),
            )
            self._conn.commit()

    def remember_facts(self, facts: list[dict[str, Any]], source: str = "conversation", user_id: str = "default") -> int:
        n = 0
        for f in facts or []:
            try:
                self.remember_fact(f["key"], f["value"], float(f.get("confidence", 0.6)), source, user_id)
                n += 1
            except Exception:
                continue
        return n

    def get_facts(self, user_id: str = "default", min_confidence: float = 0.0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, value, confidence, source, updated_at
                FROM user_profile
                WHERE user_id = ? AND confidence >= ?
                ORDER BY confidence DESC, updated_at DESC
                """,
                (user_id, float(min_confidence)),
            ).fetchall()
        return [dict(r) for r in rows]

    def forget_fact(self, key: str, value: str | None = None, user_id: str = "default") -> int:
        with self._lock:
            if value is None:
                cur = self._conn.execute(
                    "DELETE FROM user_profile WHERE user_id = ? AND key = ?", (user_id, key)
                )
            else:
                cur = self._conn.execute(
                    "DELETE FROM user_profile WHERE user_id = ? AND key = ? AND value = ?",
                    (user_id, key, value),
                )
            self._conn.commit()
            return cur.rowcount

    # -- episodic memory -----------------------------------------------------

    def add_episode(
        self,
        user_text: str,
        assistant_text: str = "",
        meta: dict[str, Any] | None = None,
        user_id: str = "default",
    ) -> int:
        ts = _now()
        embedding_json = None
        if self.embedder is not None:
            try:
                vec = self.embedder(f"{user_text}\n{assistant_text}".strip())
                if vec:
                    embedding_json = json.dumps([round(float(x), 6) for x in vec])
            except Exception as exc:
                self._log("warning", f"embedding failed, storing episode without vector: {exc}")
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO episodes (user_id, ts, iso, user_text, assistant_text, meta, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    ts,
                    _iso(ts),
                    user_text or "",
                    assistant_text or "",
                    json.dumps(meta or {}),
                    embedding_json,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent(self, n: int = 5, user_id: str = "default") -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, ts, iso, user_text, assistant_text
                FROM episodes WHERE user_id = ?
                ORDER BY ts DESC LIMIT ?
                """,
                (user_id, int(n)),
            ).fetchall()
        return [dict(r) for r in rows][::-1]  # chronological order

    def recall(self, query: str, k: int = 3, user_id: str = "default") -> list[dict[str, Any]]:
        """Hybrid recall.

        Keyword candidates come from FTS5/LIKE. When an embedder is configured we
        ALSO scan recent embedded episodes by cosine similarity and merge them in,
        because keyword search misses semantic and morphological matches (e.g. the
        query "robot motor" would not match a stored "robots and motors" via FTS,
        since FTS matches exact tokens). The merged pool is then ranked by a blend
        of semantic similarity and keyword score.
        """
        query = (query or "").strip()
        if not query:
            return []
        candidates = self._keyword_candidates(query, k * 4, user_id)
        by_id: dict[Any, dict[str, Any]] = {c["id"]: c for c in candidates}

        if self.embedder is not None:
            try:
                qvec = self.embedder(query)
                # Pull in semantically similar episodes that keyword search missed.
                for c in self._embedded_episodes(user_id, limit=self.vector_scan_limit):
                    by_id.setdefault(c["id"], c)
                for c in by_id.values():
                    emb = c.get("embedding")
                    c["_sim"] = _cosine(qvec, json.loads(emb)) if emb else 0.0
                    # Semantic signal dominates; keyword score breaks ties / helps when no vector.
                    c["_score"] = 0.6 * c["_sim"] + 0.4 * c.get("_kw_score", 0.0)
                ranked = sorted(by_id.values(), key=lambda c: c["_score"], reverse=True)
            except Exception as exc:
                self._log("warning", f"vector recall failed, using keyword order: {exc}")
                ranked = candidates
        else:
            ranked = candidates

        out = []
        for c in ranked[:k]:
            out.append({
                "id": c["id"], "ts": c["ts"], "iso": c["iso"],
                "user_text": c["user_text"], "assistant_text": c["assistant_text"],
            })
        return out

    def _embedded_episodes(self, user_id: str, limit: int) -> list[dict[str, Any]]:
        """Recent episodes that have a stored embedding, for vector scanning."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, ts, iso, user_text, assistant_text, embedding
                FROM episodes
                WHERE user_id = ? AND embedding IS NOT NULL
                ORDER BY ts DESC LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["_kw_score"] = 0.0
            result.append(d)
        return result

    def _keyword_candidates(self, query: str, limit: int, user_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if self.fts_enabled:
                try:
                    match = self._fts_query(query)
                    rows = self._conn.execute(
                        """
                        SELECT e.id, e.ts, e.iso, e.user_text, e.assistant_text, e.embedding,
                               bm25(episodes_fts) AS rank
                        FROM episodes_fts
                        JOIN episodes e ON e.id = episodes_fts.rowid
                        WHERE episodes_fts MATCH ? AND e.user_id = ?
                        ORDER BY rank LIMIT ?
                        """,
                        (match, user_id, int(limit)),
                    ).fetchall()
                    result = []
                    for r in rows:
                        d = dict(r)
                        # bm25 returns lower=better; map to a 0..1-ish keyword score.
                        d["_kw_score"] = 1.0 / (1.0 + abs(float(d.pop("rank", 0.0))))
                        result.append(d)
                    if result:
                        return result
                except sqlite3.OperationalError:
                    pass  # fall through to LIKE

            # LIKE fallback (also used when FTS finds nothing useful).
            like_terms = [t for t in re.findall(r"\w{3,}", query.lower())][:6]
            if not like_terms:
                return []
            clause = " OR ".join(["LOWER(user_text) LIKE ? OR LOWER(assistant_text) LIKE ?"] * len(like_terms))
            params: list[Any] = []
            for t in like_terms:
                params.extend([f"%{t}%", f"%{t}%"])
            rows = self._conn.execute(
                f"""
                SELECT id, ts, iso, user_text, assistant_text, embedding
                FROM episodes WHERE user_id = ? AND ({clause})
                ORDER BY ts DESC LIMIT ?
                """,
                (user_id, *params, int(limit)),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["_kw_score"] = 0.5
                result.append(d)
            return result

    @staticmethod
    def _fts_query(query: str) -> str:
        """Turn free text into a safe FTS5 OR-query of quoted terms."""
        terms = re.findall(r"\w+", query.lower())
        terms = [t for t in terms if len(t) >= 2][:8]
        if not terms:
            return '""'
        return " OR ".join(f'"{t}"' for t in terms)

    # -- summaries -----------------------------------------------------------

    def add_summary(self, summary: str, user_id: str = "default") -> None:
        summary = (summary or "").strip()
        if not summary:
            return
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO summaries (user_id, ts, iso, summary) VALUES (?, ?, ?, ?)",
                (user_id, ts, _iso(ts), summary),
            )
            self._conn.commit()

    def latest_summary(self, user_id: str = "default") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM summaries WHERE user_id = ? ORDER BY ts DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return row["summary"] if row else ""

    # -- prompt context ------------------------------------------------------

    def build_context_block(
        self,
        user_text: str,
        user_id: str = "default",
        max_facts: int = 8,
        k_recall: int = 3,
        include_recent: int = 0,
    ) -> str:
        """Render a compact [USER MEMORY] block to inject into the LLM prompt.

        Returns an empty string when there is nothing worth injecting, so callers
        can simply do ``if block: prompt += block``.
        """
        try:
            lines: list[str] = []

            facts = self.get_facts(user_id=user_id, min_confidence=0.4)[:max_facts]
            if facts:
                grouped: dict[str, list[str]] = {}
                for f in facts:
                    grouped.setdefault(f["key"], []).append(f["value"])
                fact_strs = []
                for key, values in grouped.items():
                    fact_strs.append(f"{key}: {', '.join(values[:4])}")
                lines.append("Known about the user: " + "; ".join(fact_strs) + ".")

            summary = self.latest_summary(user_id=user_id)
            if summary:
                lines.append(f"Previous context: {summary}")

            recalled = self.recall(user_text, k=k_recall, user_id=user_id) if k_recall > 0 else []
            if recalled:
                mem_lines = []
                for r in recalled:
                    u = (r["user_text"] or "").strip().replace("\n", " ")[:120]
                    a = (r["assistant_text"] or "").strip().replace("\n", " ")[:120]
                    if a:
                        mem_lines.append(f"- earlier the user said \"{u}\" and you answered \"{a}\".")
                    else:
                        mem_lines.append(f"- earlier the user said \"{u}\".")
                if mem_lines:
                    lines.append("Relevant past moments:\n" + "\n".join(mem_lines))

            if include_recent > 0:
                rec = self.recent(include_recent, user_id=user_id)
                if rec:
                    lines.append("Most recent exchanges:\n" + "\n".join(
                        f"- {r['user_text'][:80]}" for r in rec
                    ))

            if not lines:
                return ""
            return "[USER MEMORY]\n" + "\n".join(lines)
        except Exception as exc:
            self._log("error", f"build_context_block failed: {exc}")
            return ""

    # -- maintenance ---------------------------------------------------------

    def stats(self, user_id: str = "default") -> dict[str, Any]:
        with self._lock:
            facts = self._conn.execute(
                "SELECT COUNT(*) c FROM user_profile WHERE user_id=?", (user_id,)
            ).fetchone()["c"]
            eps = self._conn.execute(
                "SELECT COUNT(*) c FROM episodes WHERE user_id=?", (user_id,)
            ).fetchone()["c"]
            sums = self._conn.execute(
                "SELECT COUNT(*) c FROM summaries WHERE user_id=?", (user_id,)
            ).fetchone()["c"]
        return {"facts": facts, "episodes": eps, "summaries": sums, "fts_enabled": self.fts_enabled}

    def prune_episodes(self, keep: int = 2000, user_id: str = "default") -> int:
        """Keep the DB bounded on a small device; drops oldest episodes beyond `keep`."""
        with self._lock:
            cur = self._conn.execute(
                """
                DELETE FROM episodes WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM episodes WHERE user_id = ? ORDER BY ts DESC LIMIT ?
                )
                """,
                (user_id, user_id, int(keep)),
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()