from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


VALID_ACTIONS = {"none", "move", "stop"}
VALID_DIRECTIONS = {"forward", "backward", "left", "right"}


@dataclass(frozen=True)
class RobotCommand:
    action: str = "none"
    direction: str | None = None
    speed: float = 0.45
    duration_ms: int = 700
    reason: str = ""
    confidence: float = 0.0

    @classmethod
    def from_planner_dict(cls, data: dict[str, Any], cfg) -> "RobotCommand":
        action = str(data.get("action", "none")).strip().lower()
        if action not in VALID_ACTIONS:
            action = "none"

        direction_raw = data.get("direction")
        direction = str(direction_raw).strip().lower() if direction_raw is not None else None
        if direction in {"back", "reverse"}:
            direction = "backward"
        if direction not in VALID_DIRECTIONS:
            direction = None

        default_speed = float(cfg.get("robot.default_speed", 0.45))
        min_speed = float(cfg.get("robot.min_speed", 0.2))
        max_speed = float(cfg.get("robot.max_speed", 0.75))
        try:
            speed = float(data.get("speed", default_speed))
        except Exception:
            speed = default_speed
        speed = max(min_speed, min(max_speed, speed))

        min_ms = int(cfg.get("robot.min_duration_ms", 150))
        max_ms = int(cfg.get("robot.max_duration_ms", 1800))
        try:
            duration_ms = int(data.get("duration_ms", cfg.get("robot.default_duration_ms", 700)))
        except Exception:
            duration_ms = int(cfg.get("robot.default_duration_ms", 700))
        duration_ms = max(min_ms, min(max_ms, duration_ms))

        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        reason = str(data.get("reason", ""))[:240]

        if action == "move" and not direction:
            action = "none"
        if action in {"none", "stop"}:
            direction = None

        return cls(
            action=action,
            direction=direction,
            speed=speed,
            duration_ms=duration_ms,
            reason=reason,
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "direction": self.direction,
            "speed": self.speed,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RobotResult:
    ok: bool
    skipped: bool = False
    command: dict[str, Any] | None = None
    response: dict[str, Any] | str | None = None
    error: str | None = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "skipped": self.skipped,
            "command": self.command,
            "response": self.response,
            "error": self.error,
            "elapsed_s": round(self.elapsed_s, 3),
        }


class RobotController:
    """Safe HTTP client for a small ESP32 motor controller.

    The LLM can only request a tiny command schema. This class clamps all values
    before they reach the ESP32 so a malformed generation cannot run the motors
    indefinitely or at unsafe speed.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("robot.enabled", False))
        self.base_url = str(cfg.get("robot.base_url", "http://192.168.4.1")).rstrip("/")
        self.token = str(cfg.get("robot.token", "change-me"))
        self.timeout_s = float(cfg.get("robot.timeout_s", 2.0))
        self.min_confidence = float(cfg.get("robot.min_confidence", 0.55))

    def execute(self, command: RobotCommand) -> RobotResult:
        started = time.perf_counter()
        if not self.enabled:
            return RobotResult(ok=True, skipped=True, command=command.to_dict(), response="robot disabled")
        if command.action == "none" or command.confidence < self.min_confidence:
            return RobotResult(ok=True, skipped=True, command=command.to_dict(), response="no confident robot action")

        try:
            if command.action == "stop":
                response = self._post_json("/api/stop", {})
            elif command.action == "move":
                response = self._post_json(
                    "/api/move",
                    {
                        "direction": command.direction,
                        "speed": command.speed,
                        "duration_ms": command.duration_ms,
                    },
                )
            else:
                response = "unsupported action"
                return RobotResult(ok=False, command=command.to_dict(), response=response)
            return RobotResult(
                ok=True,
                command=command.to_dict(),
                response=response,
                elapsed_s=time.perf_counter() - started,
            )
        except Exception as exc:
            return RobotResult(
                ok=False,
                command=command.to_dict(),
                error=str(exc),
                elapsed_s=time.perf_counter() - started,
            )

    def status(self) -> RobotResult:
        started = time.perf_counter()
        if not self.enabled:
            return RobotResult(ok=True, skipped=True, response="robot disabled")
        try:
            response = self._get_json("/api/status")
            return RobotResult(ok=True, response=response, elapsed_s=time.perf_counter() - started)
        except Exception as exc:
            return RobotResult(ok=False, error=str(exc), elapsed_s=time.perf_counter() - started)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-VoicePi-Token": self.token,
        }

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any] | str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self._headers(),
            method="POST",
        )
        return self._open_json(req)

    def _get_json(self, path: str) -> dict[str, Any] | str:
        req = urllib.request.Request(
            self.base_url + path,
            headers=self._headers(),
            method="GET",
        )
        return self._open_json(req)

    def _open_json(self, req: urllib.request.Request) -> dict[str, Any] | str:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                if not body:
                    return {"status": resp.status}
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return body
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ESP32 HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ESP32 connection failed: {exc.reason}") from exc
