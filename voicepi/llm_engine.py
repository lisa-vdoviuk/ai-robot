from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable


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
- If the user gives a gesture-conditioned command and the camera observation confirms that gesture, execute the matching action. Example: if the user says a fist means stop and the observation says gesture=fist, return action "stop". If the user says thumbs-up means go/drive and the observation says gesture=thumbs_up, return a short forward move.
- Never execute motion from vision alone; it must match the user's current instruction or a clear active condition in the latest message.
- Keep durations short for safety. If the user says "a little", use 350-600 ms. If unspecified, use about 700 ms.
- If uncertain, choose "none" with confidence below 0.55.
"""


class LLMEngine:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.model_path: Path = cfg.path("llm.model_path")
        if not self.model_path.exists():
            raise FileNotFoundError(f"LLM GGUF not found: {self.model_path}")
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run scripts/install_pi.sh; it builds with OpenBLAS on Pi."
            ) from exc

        self.lock = threading.Lock()
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=int(cfg.get("llm.n_ctx", 2048)),
            n_threads=int(cfg.get("llm.n_threads", 4)),
            n_batch=int(cfg.get("llm.n_batch", 128)),
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
Use it to answer visual questions, count visible fingers, and explain gestures. If confidence is low, say that briefly.
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
            result = self.llm.create_chat_completion(**params)

        try:
            text = result["choices"][0]["message"].get("content", "")
        except Exception:
            text = str(result)
        return _parse_first_json_object(text)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        cancel_event: threading.Event,
        on_token: Callable[[str], None],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "messages": messages,
            "temperature": float(self.cfg.get("llm.temperature", 0.45)),
            "top_p": float(self.cfg.get("llm.top_p", 0.9)),
            "repeat_penalty": float(self.cfg.get("llm.repeat_penalty", 1.08)),
            "max_tokens": int(self.cfg.get("llm.max_tokens", 180)),
            "stream": True,
        }
        stop = self.cfg.get("llm.stop", None)
        if stop:
            params["stop"] = stop

        started = time.perf_counter()
        token_count = 0
        raw_parts: list[str] = []
        cancelled = False

        # llama.cpp model execution is not thread-safe for concurrent generations.
        with self.lock:
            stream = self.llm.create_chat_completion(**params)
            for chunk in stream:
                if cancel_event.is_set():
                    cancelled = True
                    break
                try:
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content") or ""
                except Exception:
                    token = ""
                if token:
                    token_count += 1
                    raw_parts.append(token)
                    on_token(token)

        elapsed = max(time.perf_counter() - started, 1e-6)
        return {
            "raw": "".join(raw_parts),
            "tokens": token_count,
            "elapsed_s": elapsed,
            "tokens_per_s": token_count / elapsed,
            "cancelled": cancelled,
        }


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
