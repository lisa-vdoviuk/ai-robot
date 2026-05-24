from __future__ import annotations

from typing import Any

from .robot_controller import RobotCommand


def evaluate_robot_command(command: RobotCommand, latest_vision, cfg) -> dict[str, Any]:
    """First safety/decision layer before any motor command reaches ESP32.

    This is not full autonomy yet. It is the first Decision Manager stage:
    the LLM may propose an action, but the robot decides whether the action is
    currently allowed based on world state.
    """

    world_state = build_world_state(latest_vision)
    reasons: list[str] = []
    warnings: list[str] = []

    allowed = True

    if command.action == "move":
        if command.confidence < float(cfg.get("robot.min_confidence", 0.55)):
            allowed = False
            reasons.append("planner confidence is below robot.min_confidence")

        vision = world_state.get("vision", {})
        close_obstacles = vision.get("close_obstacles", [])

        if command.direction == "forward":
            center_obstacles = [
                obj for obj in close_obstacles
                if obj.get("zone") in {"center", "unknown"}
            ]

            if center_obstacles:
                allowed = False
                labels = ", ".join(str(obj.get("label", "object")) for obj in center_obstacles[:3])
                reasons.append(f"forward movement blocked because close obstacle is visible ahead: {labels}")

        if command.direction in {"left", "right"} and close_obstacles:
            warnings.append("movement allowed, but close obstacle exists in camera scene")

    if command.action == "stop":
        allowed = True
        reasons.append("stop is always allowed")

    if command.action == "none":
        allowed = True
        reasons.append("no movement requested")

    return {
        "allowed": allowed,
        "reasons": reasons,
        "warnings": warnings,
        "world_state": world_state,
    }


def build_world_state(latest_vision) -> dict[str, Any]:
    if latest_vision is None:
        return {
            "vision": {
                "available": False,
                "attention": "unknown",
                "objects": [],
                "close_obstacles": [],
            }
        }

    data = latest_vision.to_dict() if hasattr(latest_vision, "to_dict") else dict(latest_vision)
    scene = data.get("scene") or {}

    return {
        "vision": {
            "available": bool(data.get("ok", False)),
            "backend": data.get("backend"),
            "summary": data.get("summary"),
            "confidence": data.get("confidence", 0.0),
            "attention": scene.get("attention", "unknown"),
            "person_count": scene.get("person_count", 0),
            "object_count": scene.get("object_count", len(data.get("objects") or [])),
            "objects": data.get("objects") or [],
            "close_obstacles": scene.get("close_obstacles") or [],
            "motion": data.get("motion") or {},
        }
    }