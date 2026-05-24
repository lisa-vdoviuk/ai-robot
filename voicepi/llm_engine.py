from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable
import os
import requests


_VISIBLE_FORMAT = """Output exactly this XML-like format every time:\n<rationale>One short visible rationale for debugging: what intent you inferred and any uncertainty. Do not reveal hidden chain-of-thought.</rationale>\n<answer>The spoken answer in natural English. No markdown. No labels like \"Answer:\". Do not include XML closing tags in the spoken text.</answer>"""

_ROBOT_PLANNER_SYSTEM = """You are a robot motion tool planner for a small Raspberry Pi + ESP32 wheeled robot.
Your only job is to decide whether the latest user message requests physical robot movement.
Return JSON only. No markdown. No prose.

Valid schema:
{
  "action": "none" | "move" | "stop",
  "direction": null | "forward" | "backward" | "left" | "right",
  "speed": 0.2-0.75,
  "duration_ms": 150-1800,
  "confidence": 0.0-1.0,
  "reason": "short visible reason"
}

Rules:
- Use action "move" only for a clear command to move the robot.
- Use action "stop" for stop, halt, freeze, wait, do not move.
- Use action "none" for questions, greetings, code help, explanations, ambiguous speech, or anything not commanding the robot.
- "left" and "right" mean turn in place.
- The latest user message may include a [CAMERA OBSERVATION] block. Treat it as sensor input from the robot camera.
- If the latest camera observation reports a close obstacle and the user asks to move toward it, prefer action "none" or "stop" with a safety reason.
- Never execute motion from vision alone; it must match the user's current instruction or a clear active condition in the latest message.
- Keep durations short for safety. If the user says "a little", use 350-600 ms. If unspecified, use about 700 ms.
- If uncertain, choose "none" with confidence below 0.55.
"""


class LLMEngine:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.engine = str(cfg.get("llm.engine", "llama_cpp")).strip().lower()
        self.lock = threading.Lock()
        self.llm = None
        self.model_path: Path | None = None

        if self.engine in {"groq", "groq_api", "api_groq"}:
            self.engine = "groq"
            self.groq_base_url = str(
                cfg.get("llm.groq.base_url", "https://api.groq.com/openai/v1")
            ).rstrip("/")
            self.groq_model = str(cfg.get("llm.groq.model", "llama-3.1-8b-instant"))
            self.groq_timeout_s = float(cfg.get("llm.groq.timeout_s", 30))

            api_key_env = str(cfg.get("llm.groq.api_key_env", "GROQ_API_KEY"))
            self.groq_api_key = str(
                cfg.get("llm.groq.api_key", "") or os.environ.get(api_key_env, "")
            )

            if not self.groq_api_key:
                raise RuntimeError(
                    f"Groq API key is missing. Set environment variable {api_key_env}, "
                    "or configure llm.groq.api_key for local testing only."
                )

            self.session = requests.Session()
            return

        if self.engine not in {"llama_cpp", "local", "llama"}:
            raise RuntimeError(
                f"Unsupported llm.engine: {self.engine!r}. Use 'llama_cpp' or 'groq'."
            )

        self.engine = "llama_cpp"
        self.model_path = cfg.path("llm.model_path")

        if not self.model_path.exists():
            raise FileNotFoundError(f"LLM GGUF not found: {self.model_path}")

        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run scripts/install_pi.sh; it builds with OpenBLAS on Pi."
            ) from exc

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=int(cfg.get("llm.n_ctx", 1024)),
            n_threads=int(cfg.get("llm.n_threads", 2)),
            n_batch=int(cfg.get("llm.n_batch", 64)),
            n_gpu_layers=int(cfg.get("llm.n_gpu_layers", 0)),
            chat_format=str(cfg.get("llm.chat_format", "chatml")),
            verbose=False,
        )

    def build_messages(self, history: list[dict[str, str]], user_text: str) -> list[dict[str, str]]:
        max_turns = int(self.cfg.get("assistant.max_history_turns", 8))
        system_prompt = str(self.cfg.get("assistant.system_prompt", "You are a helpful assistant."))
        visible = bool(self.cfg.get("assistant.visible_rationale", True))
        if visible and "<rationale>" not in system_prompt:
            system_prompt = f"{system_prompt.rstrip()}\n\n{_VISIBLE_FORMAT}"
        if not visible:
            system_prompt = (
                system_prompt
                .replace(_VISIBLE_FORMAT, "")
                .replace("Output exactly this XML-like format every time:", "")
            )
            system_prompt += "\nRespond as normal speakable English text without XML tags."

        if bool(self.cfg.get("camera.enabled", False)):
            system_prompt += """

You may receive a [CAMERA OBSERVATION] note from the robot camera. Treat it as current but imperfect sensor input.
Use it to answer visual questions about visible objects, people, rough left/center/right zones, motion and possible close obstacles. If confidence is low, say that briefly.
Do not claim to see details that are not present in the observation.
"""

        if bool(self.cfg.get("robot.enabled", False)):
            system_prompt += """

You may receive a [ROBOT TOOL RESULT] note from the server. Treat it as ground truth.
If a robot action was executed, briefly acknowledge it. If it failed, say so briefly.
Do not invent robot actions that are not listed in the tool result.
"""

        trimmed = history[-max_turns * 2 :]
        # A tiny few-shot strongly improves tag compliance on 1.5B/3B models.
        examples: list[dict[str, str]] = []
        if visible:
            examples = [
                {"role": "user", "content": "How are you?"},
                {
                    "role": "assistant",
                    "content": "<rationale>The user is greeting me and asking for a simple status update.</rationale>\n<answer>I’m doing well and ready to help. How are you?</answer>",
                },
            ]
        return [
            {"role": "system", "content": system_prompt},
            *examples,
            *trimmed,
            {"role": "user", "content": user_text},
        ]
    def _common_params(self, *, max_tokens_default: int, stream: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temperature": float(self.cfg.get("llm.temperature", 0.25)),
            "top_p": float(self.cfg.get("llm.top_p", 0.85)),
            "max_tokens": int(self.cfg.get("llm.max_tokens", max_tokens_default)),
            "stream": stream,
        }

        if self.engine == "llama_cpp":
            params["repeat_penalty"] = float(self.cfg.get("llm.repeat_penalty", 1.05))

            stop = self.cfg.get("llm.stop", None)
            if stop:
                params["stop"] = stop

        return params

    def _groq_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

    def _groq_chat_completion(self, payload: dict[str, Any], *, stream: bool):
        url = f"{self.groq_base_url}/chat/completions"

        response = self.session.post(
            url,
            headers=self._groq_headers(),
            json=payload,
            stream=stream,
            timeout=self.groq_timeout_s,
        )

        if response.status_code >= 400:
            raise RuntimeError(f"Groq API error {response.status_code}: {response.text[:1000]}")

        return response

    def _create_chat_completion(self, params: dict[str, Any]):
        if self.engine == "llama_cpp":
            assert self.llm is not None
            return self.llm.create_chat_completion(**params)

        payload = dict(params)
        payload["model"] = self.groq_model
        payload.pop("repeat_penalty", None)

        response = self._groq_chat_completion(payload, stream=False)
        return response.json()
    
    def plan_robot_action(
        self,
        user_text: str,
        history: list[dict[str, str]],
        cancel_event: threading.Event,
    ) -> dict[str, Any]:
        """Ask the same local LLM for a constrained movement tool decision."""
        if not bool(self.cfg.get("robot.enabled", False)):
            return {"action": "none", "confidence": 0.0, "reason": "robot disabled"}

        history_lines: list[str] = []
        for msg in history[-6:]:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:300].replace("\n", " ")
            history_lines.append(f"{role}: {content}")
        user_payload = (
            "Recent conversation:\n"
            + ("\n".join(history_lines) if history_lines else "(none)")
            + "\n\nLatest user message:\n"
            + user_text
        )

        params: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": _ROBOT_PLANNER_SYSTEM},
                {"role": "user", "content": user_payload},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "repeat_penalty": 1.0,
            "max_tokens": int(self.cfg.get("robot.planner_max_tokens", 120)),
            "stream": False,
        }
        stop = self.cfg.get("llm.stop", None)
        if stop:
            params["stop"] = stop

        with self.lock:
            if cancel_event.is_set():
                return {"action": "none", "confidence": 0.0, "reason": "cancelled"}
            result = self._create_chat_completion(params)

        try:
            text = result["choices"][0]["message"].get("content", "")
        except Exception:
            text = str(result)
        return _parse_first_json_object(text)

    def summarize(
        self,
        turns: list[dict[str, str]],
        previous_summary: str = "",
        max_tokens: int = 160,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Condense old conversation turns into a short running memory summary.

        Used by the rolling-summary memory feature. Runs as a single non-streaming
        completion. It shares the model lock with normal generation, so callers
        should invoke it off the live turn (e.g. on a background thread when idle).
        """
        if not turns:
            return previous_summary
        convo_lines: list[str] = []
        for msg in turns:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = str(msg.get("content", "")).replace("\n", " ").strip()
            if content:
                convo_lines.append(f"{role}: {content}")
        if not convo_lines:
            return previous_summary

        system = (
            "You compress a voice assistant's conversation into a compact running memory. "
            "Write 2-4 short sentences capturing durable, useful facts: who the user is, "
            "their stated preferences, ongoing goals, and any commitments. "
            "Do not include greetings, small talk, or one-off pleasantries. Plain text only."
        )
        user_payload = ""
        if previous_summary:
            user_payload += f"Existing memory summary:\n{previous_summary}\n\n"
        user_payload += "New conversation to fold in:\n" + "\n".join(convo_lines)

        params: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        stop = self.cfg.get("llm.stop", None)
        if stop:
            params["stop"] = stop

        with self.lock:
            if cancel_event is not None and cancel_event.is_set():
                return previous_summary
            result = self._create_chat_completion(params)
        try:
            text = result["choices"][0]["message"].get("content", "")
        except Exception:
            text = ""
        text = (text or "").strip()
        return text or previous_summary

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event,
        on_token: Callable[[str], None],
    ) -> dict[str, Any]:
        params = self._common_params(max_tokens_default=80, stream=True)
        params["messages"] = messages

        started = time.perf_counter()
        token_count = 0
        raw_parts: list[str] = []
        cancelled = False

        with self.lock:
            if self.engine == "llama_cpp":
                assert self.llm is not None
                stream = self.llm.create_chat_completion(**params)

                for chunk in stream:
                    if cancel_event.is_set():
                        cancelled = True
                        break

                    token = _extract_stream_delta(chunk)

                    if token:
                        token_count += 1
                        raw_parts.append(token)
                        on_token(token)

            else:
                payload = dict(params)
                payload["model"] = self.groq_model
                payload.pop("repeat_penalty", None)

                response = self._groq_chat_completion(payload, stream=True)

                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if cancel_event.is_set():
                            cancelled = True
                            break

                        if not line or not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        token = _extract_stream_delta(chunk)

                        if token:
                            token_count += 1
                            raw_parts.append(token)
                            on_token(token)

                finally:
                    response.close()

        elapsed = max(time.perf_counter() - started, 1e-6)

        return {
            "raw": "".join(raw_parts),
            "tokens": token_count,
            "elapsed_s": elapsed,
            "tokens_per_s": token_count / elapsed,
            "cancelled": cancelled,
            "engine": self.engine,
            "model": self.groq_model if self.engine == "groq" else str(self.model_path),
        }

def _extract_stream_delta(chunk: dict[str, Any]) -> str:
    try:
        delta = chunk["choices"][0].get("delta", {})
        return delta.get("content") or ""
    except Exception:
        return ""
    
def _parse_first_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {"action": "none", "confidence": 0.0, "reason": "empty planner output"}
    # Remove common markdown fences if a small model ignores the prompt.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return {"action": "none", "confidence": 0.0, "reason": "no JSON object in planner output"}
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    data = json.loads(candidate)
                    if isinstance(data, dict):
                        return data
                except Exception:
                    break
    return {"action": "none", "confidence": 0.0, "reason": "invalid planner JSON"}