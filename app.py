from __future__ import annotations

import argparse
import atexit
import json
import os
import threading
import uuid
from typing import Any

from flask import Flask, Response, jsonify, render_template
from flask_sock import Sock

from voicepi.camera import CameraManager
from voicepi.config import Config
from voicepi.llm_engine import LLMEngine
from voicepi.logging_utils import JsonlLogger
from voicepi.robot_controller import RobotController
from voicepi.session import ConversationSession
from voicepi.tts_piper import PiperTTS
from voicepi.vision import VisionObservation, VisionService


class WebSocketTransport:
    """Tiny adapter exposing .emit(event, data, to=...) like Socket.IO."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.lock = threading.Lock()

    def emit(self, event: str, data: dict[str, Any], to: str | None = None) -> None:
        payload = json.dumps({"event": event, "data": data}, ensure_ascii=False)
        with self.lock:
            try:
                self.ws.send(payload)
            except Exception:
                # The client may have closed the tab while a background LLM/TTS thread was emitting.
                pass


def create_app(cfg: Config) -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = cfg.get("server.secret_key", "voicepi-dev")
    sock = Sock(app)

    logger = JsonlLogger(cfg.path("logging.jsonl_path", "logs/voicepi.jsonl"), bool(cfg.get("logging.console", True)))

    missing = cfg.require_files()
    if missing:
        raise RuntimeError(
            "Missing model files:\n  " + "\n  ".join(missing) + "\nRun: python scripts/download_models.py"
        )

    logger.event("boot", "info", "loading LLM")
    llm_engine = LLMEngine(cfg)
    logger.event("boot", "info", "loading TTS")
    tts_engine = PiperTTS(cfg)
    robot_controller = RobotController(cfg)
    if robot_controller.enabled:
        logger.event("boot", "info", "robot controller enabled", base_url=robot_controller.base_url)
    else:
        logger.event("boot", "info", "robot controller disabled")

    camera_manager = CameraManager(cfg, logger=logger)
    camera_manager.start()
    if camera_manager.enabled:
        logger.event("boot", "info", "camera enabled", status=camera_manager.status().to_dict())
    else:
        logger.event("boot", "info", "camera disabled")

    sessions: dict[str, ConversationSession] = {}

    def broadcast(event: str, data: dict[str, Any]) -> None:
        for active_session in list(sessions.values()):
            try:
                active_session.emit(event, data)
            except Exception:
                pass

    vision_service = VisionService(cfg, camera_manager, logger=logger)

    def on_vision_observation(obs: VisionObservation) -> None:
        broadcast("vision_update", obs.to_dict())

    vision_service.add_listener(on_vision_observation)
    vision_service.start()
    atexit.register(vision_service.stop)
    atexit.register(camera_manager.stop)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            assistant_name=cfg.get("assistant.name", "VoicePi"),
            vad_threshold=cfg.get("client.vad_threshold", 0.018),
            vad_hold_ms=cfg.get("client.vad_hold_ms", 650),
            barge_in_threshold=cfg.get("client.barge_in_threshold", 0.045),
            min_speech_ms=cfg.get("client.min_speech_ms", 180),
            preroll_ms=cfg.get("client.preroll_ms", 450),
            audio_chunk_ms=cfg.get("client.audio_chunk_ms", 80),
            sample_rate=cfg.get("stt.sample_rate", 16000),
        )

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "sessions": len(sessions), "robot_enabled": robot_controller.enabled})

    @app.get("/robot/status")
    def robot_status():
        return jsonify(robot_controller.status().to_dict())

    @app.get("/camera/status")
    def camera_status():
        return jsonify({"camera": camera_manager.status().to_dict(), "vision": vision_service.status()})

    @app.get("/camera/snapshot.jpg")
    def camera_snapshot():
        frame = camera_manager.latest_jpeg()
        if not frame:
            return jsonify({"ok": False, "error": "no camera frame available"}), 503
        return Response(frame, mimetype="image/jpeg")

    @app.get("/camera/stream.mjpg")
    def camera_stream():
        status = camera_manager.status()
        if not status.enabled:
            return jsonify({"ok": False, "error": "camera disabled"}), 503
        if not status.running:
            return jsonify({"ok": False, "error": status.error or "camera not running"}), 503
        return Response(
            camera_manager.mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=FRAME",
            headers={
                "Age": "0",
                "Cache-Control": "no-cache, private",
                "Pragma": "no-cache",
            },
        )

    @sock.route("/ws")
    def ws_route(ws):
        sid = str(uuid.uuid4())
        transport = WebSocketTransport(ws)
        sess: ConversationSession | None = None
        try:
            sess = ConversationSession(sid, transport, cfg, logger, llm_engine, tts_engine, robot_controller, vision_service)
            sessions[sid] = sess
            sess.log("session", "info", "client connected")
            sess.emit("server_ready", {"sid": sid, "assistant_name": cfg.get("assistant.name", "VoicePi")})
            sess.emit("camera_status", {"camera": camera_manager.status().to_dict(), "vision": vision_service.status()})
            latest_vision = vision_service.latest()
            if latest_vision:
                sess.emit("vision_update", latest_vision.to_dict())

            while True:
                msg = ws.receive()
                if msg is None:
                    break
                if isinstance(msg, (bytes, bytearray, memoryview)):
                    sess.handle_audio_chunk(msg)
                    continue
                try:
                    envelope = json.loads(str(msg))
                except json.JSONDecodeError:
                    sess.log("ws", "warning", "invalid websocket json")
                    continue
                event = envelope.get("event")
                data = envelope.get("data") or {}
                if event == "speech_start":
                    sess.handle_speech_start(data if isinstance(data, dict) else {})
                elif event == "speech_end":
                    sess.handle_speech_end()
                elif event == "barge_in":
                    reason = str(data.get("reason", "barge_in")) if isinstance(data, dict) else "barge_in"
                    sess.interrupt(reason)
                elif event == "manual_text":
                    text = str(data.get("text", "")).strip() if isinstance(data, dict) else ""
                    if text:
                        sess.emit("stt_final", {"text": text, "manual": True})
                        sess.log("ui", "info", "manual text submitted", text=text)
                        sess.start_turn(text)
                elif event == "vision_analyze":
                    sess.handle_vision_analyze()
                elif event == "ping":
                    sess.emit("pong", {"ok": True})
                else:
                    sess.log("ws", "warning", f"unknown event: {event}")
        except Exception as exc:
            logger.event("session", "error", f"websocket failed: {exc}", sid=sid)
            try:
                transport.emit("fatal_error", {"error": str(exc)})
            except Exception:
                pass
        finally:
            sessions.pop(sid, None)
            if sess:
                try:
                    sess.log("session", "info", "client disconnected")
                    sess.close()
                except Exception:
                    pass

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="VoicePi local voice LLM assistant")
    parser.add_argument("--config", default=os.environ.get("VOICEPI_CONFIG", "config.yaml"))
    args = parser.parse_args()

    cfg = Config.load(args.config)
    app = create_app(cfg)

    ssl_context = None
    if bool(cfg.get("server.https", False)):
        cert = cfg.get("server.certfile")
        key = cfg.get("server.keyfile")
        if cert and key:
            ssl_context = (str(cfg.path("server.certfile")), str(cfg.path("server.keyfile")))
        else:
            # Requires pyOpenSSL. Good enough for LAN testing; install a real cert for production.
            ssl_context = "adhoc"

    app.run(
        host=str(cfg.get("server.host", "0.0.0.0")),
        port=int(cfg.get("server.port", 5443)),
        ssl_context=ssl_context,
        threaded=True,
    )


if __name__ == "__main__":
    main()
