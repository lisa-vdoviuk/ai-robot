from __future__ import annotations

import base64
import queue
import threading
import time
import uuid
from typing import Any

from .decision_manager import evaluate_robot_command
from .memory import MemoryStore, extract_facts
from .robot_controller import RobotCommand, RobotController
from .stt_vosk import VoskStreamSTT
from .text_utils import (
    SentenceBuffer,
    StreamingAnswerExtractor,
    clean_for_tts,
    extract_tag,
    extract_tag_progress,
    fallback_visible_rationale,
    is_rejected_transcript,
    strip_llm_markup,
)

_RESERVED_LOG_KEYS = {"source", "level", "message", "sid", "ts", "iso"}


class ConversationSession:
    def __init__(self, sid: str, socketio, cfg, logger, llm_engine, tts_engine, robot_controller: RobotController | None = None, vision_service=None, memory_store: "MemoryStore | None" = None) -> None:
        self.sid = sid
        self.socketio = socketio
        self.cfg = cfg
        self.logger = logger
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.robot_controller = robot_controller or RobotController(cfg)
        self.vision_service = vision_service
        self.memory_store = memory_store
        # A stable id for this user. With a single owner it's "default"; a multi-user
        # build (e.g. after face recognition) can set this per recognized person.
        self.user_id = str(cfg.get("memory.user_id", "default"))
        self._summarizing = False
        self.history: list[dict[str, str]] = []
        self.cancel_event = threading.Event()
        self.turn_thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.stt = VoskStreamSTT(cfg.path("stt.model_path"), int(cfg.get("stt.sample_rate", 16000)))
        self.min_final_chars = int(cfg.get("stt.min_final_chars", 2))
        self.min_final_words = int(cfg.get("stt.min_final_words", 1))
        self.reject_phrases = list(cfg.get("stt.reject_phrases", []) or [])
        self.accepting_audio = False
        self.last_committed_text = ""
        self.last_committed_at = 0.0

    def emit(self, event: str, data: dict[str, Any]) -> None:
        self.socketio.emit(event, data, to=self.sid)

    def log(self, log_source: str, log_level: str, log_message: str, **fields: Any) -> None:
        # Browser telemetry can contain keys named "source", "level", "message", etc.
        # Keep parameter names distinct so Python argument binding cannot collide
        # before this sanitizer runs.
        safe_fields = {f"field_{k}" if k in _RESERVED_LOG_KEYS else k: v for k, v in fields.items()}
        record = self.logger.event(log_source, log_level, log_message, sid=self.sid, **safe_fields)
        self.emit("log", record)

    def handle_audio_chunk(self, data: bytes | bytearray | memoryview) -> None:
        if not data or not self.accepting_audio:
            return
        pcm = bytes(data)
        try:
            kind, text, raw = self.stt.accept_pcm16le(pcm)
        except Exception as exc:
            self.log("stt", "error", f"STT error: {exc}")
            return
        if kind in {"partial", "segment"}:
            self.emit("stt_partial", {"text": text, "kind": kind, "raw": raw})

    def handle_speech_start(self, data: dict[str, Any] | None = None) -> None:
        with self.lock:
            if not self.accepting_audio:
                self.stt.begin_utterance()
                self.accepting_audio = True
                self.emit("listening", {"state": "speech_started"})
                self.log("stt", "info", "speech started", **(data or {}))
            self.interrupt("client_speech_start")

    def handle_speech_end(self) -> None:
        with self.lock:
            if not self.accepting_audio:
                return
            self.accepting_audio = False
            try:
                text, raw = self.stt.finish_utterance()
            except Exception as exc:
                self.log("stt", "error", f"STT finalization error: {exc}")
                return

        text = text.strip()
        self.emit("stt_partial", {"text": ""})
        if is_rejected_transcript(text, self.min_final_chars, self.min_final_words, self.reject_phrases):
            self.log("stt", "info", "rejected short/noise transcript", text=text, raw=raw)
            self.emit("stt_rejected", {"text": text, "reason": "too_short_or_noise"})
            return

        now = time.monotonic()
        if text == self.last_committed_text and now - self.last_committed_at < 2.0:
            self.log("stt", "info", "rejected duplicate transcript", text=text)
            self.emit("stt_rejected", {"text": text, "reason": "duplicate"})
            return
        self.last_committed_text = text
        self.last_committed_at = now

        self.emit("stt_final", {"text": text, "raw": raw})
        self.log("stt", "info", "final utterance transcript", text=text)
        self.start_turn(text)

    def interrupt(self, reason: str = "interrupt") -> None:
        with self.lock:
            if self.turn_thread and self.turn_thread.is_alive():
                self.cancel_event.set()
                self.emit("cancelled", {"reason": reason})
                self.log("session", "info", "generation cancelled", reason=reason)
                # If the user interrupts while the robot is moving, stop the motors too.
                if bool(self.cfg.get("robot.stop_on_barge_in", True)):

                    result = self.robot_controller.execute(RobotCommand(action="stop", reason=reason, confidence=1.0))
                    self.emit("robot_update", result.to_dict())

    def start_turn(self, user_text: str) -> None:
        with self.lock:
            self.interrupt("new_user_turn")
            self.cancel_event = threading.Event()
            turn_id = str(uuid.uuid4())
            self.turn_thread = threading.Thread(
                target=self._run_turn,
                args=(turn_id, user_text, self.cancel_event),
                name=f"voicepi-turn-{turn_id[:8]}",
                daemon=True,
            )
            self.turn_thread.start()

    def close(self) -> None:
        self.cancel_event.set()
        if bool(self.cfg.get("robot.stop_on_disconnect", True)):
            self.robot_controller.execute(RobotCommand(action="stop", reason="client disconnected", confidence=1.0))
        final = self.stt.finish()
        if final:
            self.log("stt", "info", "final transcript on close", text=final)

    def _tts_worker(self, turn_id: str, q: "queue.Queue[str | None]", cancel_event: threading.Event) -> None:
        # Piper sounds choppy when every short sentence becomes a separate WAV.
        # Coalesce nearby short fragments into one speakable chunk while keeping
        # enough streaming responsiveness for voice dialogue.
        coalesce_wait_s = float(self.cfg.get("tts.coalesce_wait_s", 0.18))
        max_chunk_chars = int(self.cfg.get("tts.chunk_max_chars", 320))
        pending: str | None = None
        stop_after_chunk = False

        while not cancel_event.is_set():
            try:
                if pending is None:
                    item = q.get(timeout=0.1)
                else:
                    item = pending
                    pending = None
            except queue.Empty:
                continue

            if item is None:
                return

            first = clean_for_tts(item)
            if not first:
                if stop_after_chunk:
                    return
                continue

            parts = [first]
            deadline = time.monotonic() + max(0.0, coalesce_wait_s)
            while not cancel_event.is_set() and len(" ".join(parts)) < max_chunk_chars:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = q.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is None:
                    stop_after_chunk = True
                    break
                next_text = clean_for_tts(next_item)
                if not next_text:
                    continue
                if len(" ".join(parts)) + 1 + len(next_text) > max_chunk_chars:
                    pending = next_text
                    break
                parts.append(next_text)

            text = clean_for_tts(" ".join(parts))
            if not text:
                if stop_after_chunk:
                    return
                continue

            try:
                started = time.perf_counter()
                wav = self.tts_engine.synthesize_wav(text, cancel_event=cancel_event)
                elapsed = time.perf_counter() - started
                if cancel_event.is_set() or not wav:
                    return
                audio_b64 = base64.b64encode(wav).decode("ascii")
                mime_type = getattr(self.tts_engine, "mime_type", "audio/wav")

                self.emit(
                    "tts_audio",
                    {
                        "turn_id": turn_id,
                        "text": text,
                        "audio_b64": audio_b64,
                        "wav_b64": audio_b64,
                        "mime_type": mime_type,
                    },
                )
                self.log("tts", "info", "audio synthesized", chars=len(text), elapsed_s=round(elapsed, 3))
            except Exception as exc:
                if not cancel_event.is_set():
                    self.log("tts", "error", f"TTS error: {exc}", text=text[:120])

            if stop_after_chunk:
                return

    def _speak_text_once(self, turn_id: str, text: str, cancel_event: threading.Event) -> None:
        """Low-load TTS mode: synthesize only after LLM generation has finished."""
        text = clean_for_tts(text)

        if not text or cancel_event.is_set():
            return

        try:
            started = time.perf_counter()
            wav = self.tts_engine.synthesize_wav(text, cancel_event=cancel_event)
            elapsed = time.perf_counter() - started

            if cancel_event.is_set() or not wav:
                return

            self.emit(
                "tts_audio",
                {
                    "turn_id": turn_id,
                    "text": text,
                    "wav_b64": base64.b64encode(wav).decode("ascii"),
                },
            )

            self.log(
                "tts",
                "info",
                "audio synthesized",
                chars=len(text),
                elapsed_s=round(elapsed, 3),
                mode="nonstreaming",
            )

        except Exception as exc:
            if not cancel_event.is_set():
                self.log("tts", "error", f"TTS error: {exc}", text=text[:120])

    def _run_turn(self, turn_id: str, user_text: str, cancel_event: threading.Event) -> None:
        started = time.perf_counter()
        self.emit("turn_start", {"turn_id": turn_id, "user_text": user_text})
        self.history.append({"role": "user", "content": user_text})

        vision_context = self._capture_vision_context(user_text)
        planner_text = user_text
        if vision_context:
            planner_text = f"{user_text}\n\n[CAMERA OBSERVATION]\n{vision_context}"

        robot_context = self._plan_and_execute_robot(planner_text, cancel_event)
        guarded_text = self._maybe_guard_user_text(user_text)
        memory_context = self._memory_context(user_text)
        if memory_context:
            guarded_text = f"{memory_context}\n\n{guarded_text}"
        if vision_context:
            guarded_text = f"{guarded_text}\n\n[CAMERA OBSERVATION]\n{vision_context}"
        if robot_context:
            guarded_text = f"{guarded_text}\n\n[ROBOT TOOL RESULT]\n{robot_context}"
        messages = self.llm_engine.build_messages(self.history[:-1], guarded_text)
        self.log("llm", "info", "generation started", turn_id=turn_id, user_text=user_text)

        stream_tts = bool(self.cfg.get("tts.stream_during_generation", True))

        sentence_buffer = SentenceBuffer(
            min_chars=int(self.cfg.get("tts.sentence_min_chars", 22)),
            max_chars=int(self.cfg.get("tts.sentence_max_chars", 220)),
        )

        answer_streamer = StreamingAnswerExtractor("answer")
        tts_queue: queue.Queue[str | None] | None = None
        tts_thread: threading.Thread | None = None

        if stream_tts:
            tts_queue = queue.Queue(maxsize=16)
            tts_thread = threading.Thread(
                target=self._tts_worker,
                args=(turn_id, tts_queue, cancel_event),
                daemon=True,
                name=f"voicepi-tts-{turn_id[:8]}",
            )
            tts_thread.start()

        raw_text = ""
        answer_progress_sent = 0
        streamed_answer_chars = 0
        rationale_progress_sent = 0
        visible_rationale = bool(self.cfg.get("assistant.visible_rationale", True))

        def on_token(token: str) -> None:
            nonlocal raw_text, answer_progress_sent, rationale_progress_sent, streamed_answer_chars
            raw_text += token
            self.emit("llm_token", {"turn_id": turn_id, "token": token, "raw": raw_text})

            if visible_rationale:
                rationale_now = extract_tag_progress(raw_text, "rationale")
                if len(rationale_now) > rationale_progress_sent:
                    self.emit(
                        "xai_update",
                        {
                            "turn_id": turn_id,
                            "rationale_partial": rationale_now,
                        },
                    )
                    rationale_progress_sent = len(rationale_now)

                safe_new_answer = answer_streamer.feed(token)
                if safe_new_answer:
                    streamed_answer_chars += len(safe_new_answer)
                    answer_progress_sent = streamed_answer_chars
                    if stream_tts and tts_queue is not None:
                        for sent in sentence_buffer.feed(safe_new_answer):
                            self._enqueue_tts(tts_queue, sent, cancel_event)
            else:
                cleaned = clean_for_tts(token)

                if stream_tts and tts_queue is not None:
                    for sent in sentence_buffer.feed(cleaned):
                        self._enqueue_tts(tts_queue, sent, cancel_event)
                else:
                    sentence_buffer.feed(cleaned)

        try:
            metrics = self.llm_engine.stream_chat(messages, cancel_event=cancel_event, on_token=on_token)
        except Exception as exc:
            self.log("llm", "error", f"LLM error: {exc}")
            self.emit("turn_error", {"turn_id": turn_id, "error": str(exc)})
            if tts_queue is not None:
                tts_queue.put(None)
            return

        raw_text = metrics["raw"]
        if visible_rationale:
            answer = clean_for_tts(extract_tag(raw_text, "answer") or strip_llm_markup(raw_text))
            rationale = extract_tag(raw_text, "rationale") or fallback_visible_rationale(user_text, answer)
        else:
            answer = clean_for_tts(strip_llm_markup(raw_text))
            rationale = ""

        final_tail = clean_for_tts(sentence_buffer.flush())

        if stream_tts and tts_queue is not None:
            if final_tail:
                self._enqueue_tts(tts_queue, final_tail, cancel_event)

            if visible_rationale and answer_progress_sent == 0 and answer and not cancel_event.is_set():
                fallback_buffer = SentenceBuffer(
                    min_chars=int(self.cfg.get("tts.sentence_min_chars", 22)),
                    max_chars=int(self.cfg.get("tts.sentence_max_chars", 220)),
                )

                for sent in fallback_buffer.feed(answer):
                    self._enqueue_tts(tts_queue, sent, cancel_event)

                tail = fallback_buffer.flush()
                if tail:
                    self._enqueue_tts(tts_queue, tail, cancel_event)

            tts_queue.put(None)

            join_timeout = float(self.cfg.get("tts.join_timeout_s", 30))

            if tts_thread is not None:
                tts_thread.join(timeout=join_timeout)

                if tts_thread.is_alive():
                    self.log("tts", "warning", "TTS worker did not finish before timeout", timeout_s=join_timeout)

        else:
            self._speak_text_once(turn_id, answer, cancel_event)

        elapsed = time.perf_counter() - started
        if cancel_event.is_set() or metrics.get("cancelled"):
            self.emit("turn_cancelled", {"turn_id": turn_id})
            return

        self.history.append({"role": "assistant", "content": answer})
        self._remember_turn(user_text, answer)
        self.emit(
            "xai_update",
            {
                "turn_id": turn_id,
                "rationale": rationale,
                "answer": answer,
                "messages": messages,
                "metrics": {**metrics, "total_turn_elapsed_s": elapsed},
            },
        )
        self.emit("turn_done", {"turn_id": turn_id, "answer": answer})
        self.log(
            "llm",
            "info",
            "generation completed",
            turn_id=turn_id,
            tokens=metrics.get("tokens"),
            tokens_per_s=round(float(metrics.get("tokens_per_s", 0.0)), 3),
            elapsed_s=round(elapsed, 3),
        )


    def handle_vision_analyze(self) -> None:
        if not self.vision_service:
            self.emit("vision_update", {"ok": False, "summary": "vision service is not configured", "error": "vision_unavailable"})
            return
        try:
            obs = self.vision_service.analyze_now(reason="manual")
            self.emit("vision_update", obs.to_dict())
        except Exception as exc:
            self.log("vision", "error", f"manual vision analysis failed: {exc}")
            self.emit("vision_update", {"ok": False, "summary": "manual vision analysis failed", "error": str(exc)})

    def _capture_vision_context(self, user_text: str) -> str:
        if not self.vision_service:
            return ""
        try:
            context = self.vision_service.context_for_prompt(user_text)
            latest = self.vision_service.latest()
            if latest:
                self.emit("vision_update", latest.to_dict())
            if context:
                self.log("vision", "info", "camera observation attached to LLM turn", context=context)
            return context
        except Exception as exc:
            self.log("vision", "error", f"vision context failed: {exc}")
            return ""

    def _memory_context(self, user_text: str) -> str:
        """Build the [USER MEMORY] block injected before the user's message."""
        if not self.memory_store or not bool(self.cfg.get("memory.attach_to_prompt", True)):
            return ""
        try:
            block = self.memory_store.build_context_block(
                user_text,
                user_id=self.user_id,
                max_facts=int(self.cfg.get("memory.max_facts", 8)),
                k_recall=int(self.cfg.get("memory.recall_k", 3)),
            )
            if block:
                self.log("memory", "info", "memory context attached", chars=len(block))
            return block
        except Exception as exc:
            self.log("memory", "error", f"memory context failed: {exc}")
            return ""

    def _remember_turn(self, user_text: str, answer: str) -> None:
        """Persist the exchange and any durable facts. Never raises into the turn."""
        if not self.memory_store:
            return
        try:
            self.memory_store.add_episode(user_text, answer, meta={"sid": self.sid}, user_id=self.user_id)
            if bool(self.cfg.get("memory.extract_facts", True)):
                facts = extract_facts(user_text)
                if facts:
                    n = self.memory_store.remember_facts(facts, source="conversation", user_id=self.user_id)
                    self.log("memory", "info", "stored durable facts", count=n, facts=facts)
            keep = int(self.cfg.get("memory.max_episodes", 2000))
            if keep > 0:
                self.memory_store.prune_episodes(keep=keep, user_id=self.user_id)
        except Exception as exc:
            self.log("memory", "error", f"remember turn failed: {exc}")
        self._maybe_summarize()

    def _maybe_summarize(self) -> None:
        """When the conversation grows long, fold old turns into a stored summary.

        Runs on a background thread so it never adds latency to the live reply.
        Only one summarization runs at a time. The summary is what feeds the
        'Previous context' line of the [USER MEMORY] block on later turns.
        """
        if not self.memory_store or not bool(self.cfg.get("memory.summary.enabled", True)):
            return
        if self._summarizing:
            return
        trigger_turns = int(self.cfg.get("memory.summary.trigger_turns", 12))
        keep_recent = int(self.cfg.get("memory.summary.keep_recent_turns", 6))
        # history holds 2 entries per exchange (user + assistant).
        if len(self.history) < trigger_turns * 2:
            return
        self._summarizing = True
        threading.Thread(target=self._summarize_worker, args=(keep_recent,), daemon=True, name="voicepi-summary").start()

    def _summarize_worker(self, keep_recent: int) -> None:
        try:
            with self.lock:
                old = self.history[: -keep_recent * 2] if keep_recent > 0 else list(self.history)
                if not old:
                    return
            previous = self.memory_store.latest_summary(user_id=self.user_id)
            summary = self.llm_engine.summarize(
                old,
                previous_summary=previous,
                max_tokens=int(self.cfg.get("memory.summary.max_tokens", 160)),
            )
            if summary and summary != previous:
                self.memory_store.add_summary(summary, user_id=self.user_id)
                self.log("memory", "info", "rolling summary updated", chars=len(summary))
            # Trim short-term history; the summary preserves the older context.
            with self.lock:
                if keep_recent > 0 and len(self.history) > keep_recent * 2:
                    self.history = self.history[-keep_recent * 2 :]
        except Exception as exc:
            self.log("memory", "error", f"summarization failed: {exc}")
        finally:
            self._summarizing = False

    def _should_run_robot_planner(self, user_text: str) -> bool:
        if not bool(self.cfg.get("robot.fast_intent_gate", True)):
            return True
        text = user_text.casefold()
        keywords = [
            "move", "drive", "go", "forward", "backward", "reverse", "left", "right", "turn",
            "stop", "halt", "freeze", "wait", "motor", "wheel", "robot", "esp32",
            "рух", "їд", "вперед", "назад", "ліворуч", "праворуч", "повер",
            "зупин", "стоп", "мотор", "колес", "робот",
        ]
        return any(k in text for k in keywords)

    def _plan_and_execute_robot(self, user_text: str, cancel_event: threading.Event) -> str:
        if not bool(self.cfg.get("robot.enabled", False)):
            return ""
        if not self._should_run_robot_planner(user_text):
            self.log("robot", "info", "planner skipped by fast intent gate", user_text=user_text[:240])
            return ""
        try:
            self.log("robot", "info", "planner started", user_text=user_text)
            planner_raw = self.llm_engine.plan_robot_action(user_text, self.history[:-1], cancel_event)
            command = RobotCommand.from_planner_dict(planner_raw, self.cfg)
            self.emit("robot_update", {"planner": planner_raw, "command": command.to_dict()})
            self.log("robot", "info", "planner completed", planner=planner_raw, command=command.to_dict())
            latest_vision = self.vision_service.latest() if self.vision_service else None
            decision = evaluate_robot_command(command, latest_vision, self.cfg)

            self.emit("robot_update", {
                "decision": decision,
                "command": command.to_dict(),
            })

            self.log(
                "decision",
                "info" if decision.get("allowed") else "warning",
                "robot action safety decision",
                decision=decision,
                command=command.to_dict(),
            )

            if not decision.get("allowed", False):
                blocked_result = {
                    "ok": True,
                    "skipped": True,
                    "command": command.to_dict(),
                    "response": "blocked by decision manager",
                    "decision": decision,
                }
                self.emit("robot_update", blocked_result)
                return f"Robot command blocked by safety decision manager. Decision: {decision}."
            result = self.robot_controller.execute(command)
            result_dict = result.to_dict()
            self.emit("robot_update", result_dict)
            self.log("robot", "info" if result.ok else "error", "robot command result", **result_dict)
            if command.action == "none" or result.skipped:
                return f"Planner chose no robot movement. Planner: {command.to_dict()}. Result: {result_dict}."
            return f"Robot command: {command.to_dict()}. Result: {result_dict}."
        except Exception as exc:
            self.log("robot", "error", f"robot planning/execution failed: {exc}")
            self.emit("robot_update", {"ok": False, "error": str(exc)})
            return f"Robot tool failed before execution: {exc}"

    def _enqueue_tts(self, q: "queue.Queue[str | None]", text: str, cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            return
        try:
            q.put(text, timeout=0.2)
        except queue.Full:
            self.log("tts", "warning", "TTS queue full; dropping sentence", text=text[:120])

    def _maybe_guard_user_text(self, user_text: str) -> str:
        if not bool(self.cfg.get("quality_guard.enabled", True)):
            return user_text
        words = user_text.strip().split()
        if bool(self.cfg.get("quality_guard.ask_clarification_for_single_word", True)) and len(words) <= 1:
            return (
                f"Speech transcript: {user_text!r}. This is likely incomplete or ambiguous. "
                "Ask one short clarification unless the intent is obvious."
            )
        return user_text