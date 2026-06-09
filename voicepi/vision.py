from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .camera import CameraManager


OBSTACLE_LABELS = {
    "person", "chair", "dining table", "couch", "bottle", "cat", "dog", "car", "bus",
    "bicycle", "motorcycle", "potted plant", "tv", "laptop", "backpack",
}


@dataclass(frozen=True)
class VisionObservation:
    ok: bool
    ts: float
    iso: str
    backend: str
    summary: str
    confidence: float = 0.0
    objects: list[dict[str, Any]] = field(default_factory=list)
    scene: dict[str, Any] = field(default_factory=dict)
    motion: dict[str, Any] = field(default_factory=dict)
    frame: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    reason: str = "poll"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ts": self.ts,
            "iso": self.iso,
            "backend": self.backend,
            "summary": self.summary,
            "confidence": self.confidence,
            "objects": self.objects,
            "scene": self.scene,
            "motion": self.motion,
            "frame": self.frame,
            "error": self.error,
            "reason": self.reason,
        }

    def to_prompt_text(self) -> str:
        parts = [
            f"time={self.iso}",
            f"backend={self.backend}",
            f"summary={self.summary}",
            f"confidence={self.confidence:.2f}",
        ]
        if self.scene:
            compact_scene = {
                "person_count": self.scene.get("person_count", 0),
                "close_obstacles": self.scene.get("close_obstacles", []),
                "object_zones": self.scene.get("object_zones", {}),
                "attention": self.scene.get("attention", "clear"),
            }
            parts.append(f"scene={compact_scene}")
        if self.objects:
            compact_objects = [
                {
                    "label": obj.get("label"),
                    "confidence": obj.get("confidence"),
                    "zone": obj.get("zone"),
                    "area_ratio": obj.get("area_ratio"),
                    "bbox": obj.get("bbox"),
                }
                for obj in self.objects[:6]
            ]
            parts.append(f"objects={compact_objects}")
        if self.motion:
            compact_motion = {
                "detected": self.motion.get("detected", False),
                "zone": self.motion.get("zone"),
                "changed_area_ratio": self.motion.get("changed_area_ratio"),
            }
            parts.append(f"motion={compact_motion}")
        if self.error:
            parts.append(f"error={self.error}")
        return "\n".join(parts)


class VisionService:
    """Converts camera frames into concise local scene observations.

    The local LLM is text-only, so this service turns the Pi camera stream into
    structured text: known objects, rough zones, close-obstacle hints and motion.
    This is a useful computer-vision base for later navigation and interaction
    without the previous hand-control pipeline.
    """

    def __init__(self, cfg, camera: CameraManager, logger=None) -> None:
        self.cfg = cfg
        self.camera = camera
        self.logger = logger
        self.enabled = bool(cfg.get("vision.enabled", False))
        self.poll_enabled = bool(cfg.get("vision.poll_enabled", True))
        self.poll_interval_s = float(cfg.get("vision.poll_interval_s", 2.0))
        self.max_prompt_age_s = float(cfg.get("vision.max_prompt_age_s", 3.0))
        self.snapshot_on_turn = bool(cfg.get("vision.snapshot_on_turn", True))
        self.always_attach = bool(cfg.get("vision.always_attach_to_prompt", True))
        self.log_poll_observations = bool(cfg.get("vision.log_poll_observations", False))
        self.analyzer = BasicVisionAnalyzer(cfg)
        self._latest: VisionObservation | None = None
        self._latest_lock = threading.RLock()
        self._analyze_lock = threading.Lock()
        self._listeners: list[Callable[[VisionObservation], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return

        if not self.poll_enabled:
            self._log("vision", "info", "vision service enabled; background polling disabled")
            return

        if self._thread and self._thread.is_alive():
            return

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="voicepi-vision", daemon=True)
        self._thread.start()
        self._log("vision", "info", "vision service started")

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def add_listener(self, callback: Callable[[VisionObservation], None]) -> None:
        self._listeners.append(callback)

    def latest(self) -> VisionObservation | None:
        with self._latest_lock:
            return self._latest

    def analyze_now(self, reason: str = "manual") -> VisionObservation:
        if not self.enabled:
            obs = self._observation(False, "disabled", "vision disabled", reason=reason)
            self._publish(obs)
            return obs
        if not self.camera.enabled:
            obs = self._observation(False, "disabled", "camera disabled", reason=reason)
            self._publish(obs)
            return obs
        frame = self.camera.latest_jpeg()
        if not frame:
            obs = self._observation(False, "camera", "no camera frame available yet", reason=reason)
            self._publish(obs)
            return obs

        if not self._analyze_lock.acquire(timeout=2.0):
            existing = self.latest()
            if existing:
                return existing
            return self._observation(
                False, "busy", "vision analysis timed out waiting for lock",
                reason=reason
            )
        try:
            try:
                obs = self.analyzer.analyze(frame, reason=reason)
            except Exception as exc:
                obs = self._observation(False, "error", f"vision analysis failed: {exc}", error=str(exc), reason=reason)
            self._publish(obs)
            return obs
        finally:
            try:
                self._analyze_lock.release()
            except RuntimeError:
                pass

    def context_for_prompt(self, user_text: str) -> str:
        if not self.enabled or not self.camera.enabled:
            return ""
        lower = user_text.casefold()
        vision_words = [
            "see", "seeing", "look", "camera", "frame", "image", "photo", "picture",
            "object", "objects", "recognize", "detect", "what is", "what are",
            "person", "face", "obstacle", "motion", "movement", "around", "nearby",
            "бач", "камера", "камер", "зір", "фото", "картин", "об'єкт", "обєкт",
            "предмет", "розпізн", "детект", "що бач", "людин", "облич", "перешкод",
            "рух", "руха", "навколо", "поруч",
        ]
        likely_visual = any(word in lower for word in vision_words)

        obs = self.latest()

        if self.snapshot_on_turn and (self.always_attach or likely_visual):
            obs = self.analyze_now(reason="turn")

        if obs and (self.always_attach or likely_visual):
            fresh = (time.time() - obs.ts) <= self.max_prompt_age_s
            if fresh or likely_visual:
                return obs.to_prompt_text()

        return ""

    def status(self) -> dict[str, Any]:
        latest = self.latest()

        analyzer_backend = "unavailable"
        detector_status: dict[str, Any] = {}

        try:
            analyzer_backend = getattr(self.analyzer, "backend_name", "unknown")
        except Exception as exc:
            analyzer_backend = f"error: {exc}"

        try:
            detector_status = self.analyzer.object_detector.status()
        except Exception as exc:
            detector_status = {
                "enabled": False,
                "backend": "error",
                "error": str(exc),
            }

        return {
            "enabled": self.enabled,
            "poll_enabled": bool(getattr(self, "poll_enabled", True)),
            "backend": analyzer_backend,
            "object_detector": detector_status,
            "latest": latest.to_dict() if latest else None,
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.analyze_now(reason="poll")
            except Exception as exc:
                self._log("vision", "error", f"vision loop failed: {exc}")
            self._stop.wait(self.poll_interval_s)

    def _publish(self, obs: VisionObservation) -> None:
        with self._latest_lock:
            self._latest = obs
        should_log = obs.reason != "poll" or self.log_poll_observations or not obs.ok
        if should_log:
            self._log("vision", "info" if obs.ok else "warning", obs.summary, observation=obs.to_dict())
        for cb in list(self._listeners):
            try:
                cb(obs)
            except Exception:
                pass

    def _observation(
        self,
        ok: bool,
        backend: str,
        summary: str,
        *,
        confidence: float = 0.0,
        objects: list[dict[str, Any]] | None = None,
        scene: dict[str, Any] | None = None,
        motion: dict[str, Any] | None = None,
        frame: dict[str, Any] | None = None,
        error: str | None = None,
        reason: str = "poll",
    ) -> VisionObservation:
        now = time.time()
        return VisionObservation(
            ok=ok,
            ts=now,
            iso=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
            backend=backend,
            summary=summary,
            confidence=confidence,
            objects=objects or [],
            scene=scene or {},
            motion=motion or {},
            frame=frame or {},
            error=error,
            reason=reason,
        )

    def _log(self, source: str, level: str, message: str, **fields: Any) -> None:
        if self.logger is not None:
            try:
                self.logger.event(source, level, message, **fields)
            except Exception:
                pass


class BasicVisionAnalyzer:
    """Object + motion scene analyzer.

    Kept under the old class name so the rest of the app does not need to know
    the implementation is now scene awareness.
    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backend_name = "scene"
        self.object_detector = ObjectDetector(cfg)
        self._prev_gray = None
        self._prev_motion_ts = 0.0

    def analyze(self, jpeg: bytes, reason: str = "poll") -> VisionObservation:
        now = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now))

        cv2, np = self._load_cv()
        if cv2 is None or np is None:
            return VisionObservation(
                ok=True,
                ts=now,
                iso=iso,
                backend="frame-only",
                summary="Camera frame is available. Install python3-opencv for object and motion detection.",
                confidence=0.2,
                frame={"jpeg_bytes": len(jpeg)},
                reason=reason,
            )

        arr = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            return VisionObservation(
                ok=False,
                ts=now,
                iso=iso,
                backend="opencv",
                summary="Camera frame could not be decoded as JPEG.",
                confidence=0.0,
                error="decode_failed",
                reason=reason,
            )

        height, width = image.shape[:2]
        frame = {"width": int(width), "height": int(height), "jpeg_bytes": len(jpeg)}
        objects = self.object_detector.detect(image, cv2, np)
        objects = [self._add_spatial_features(obj, width, height) for obj in objects]
        motion = self._detect_motion(image, cv2, np)
        scene = self._build_scene(objects, motion)
        summary = self._summary(objects, scene, motion)
        confidence = max(
            self._objects_confidence(objects),
            float(motion.get("confidence", 0.0) or 0.0),
            0.35 if objects or motion.get("detected") else 0.25,
        )
        self.backend_name = self._backend_name()
        return VisionObservation(
            ok=True,
            ts=now,
            iso=iso,
            backend=self.backend_name,
            summary=summary,
            confidence=round(min(1.0, confidence), 3),
            objects=objects,
            scene=scene,
            motion=motion,
            frame=frame,
            reason=reason,
        )

    def _load_cv(self):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            return cv2, np
        except Exception:
            return None, None

    def _backend_name(self) -> str:
        detector_backend = self.object_detector.backend_name or "no-objects"
        return f"scene:{detector_backend}+motion"

    def _add_spatial_features(self, obj: dict[str, Any], width: int, height: int) -> dict[str, Any]:
        bbox = obj.get("bbox") or []
        enriched = dict(obj)
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if cx < width / 3:
                zone = "left"
            elif cx > (width * 2) / 3:
                zone = "right"
            else:
                zone = "center"
            area_ratio = (box_w * box_h) / max(1, width * height)
            enriched.update({
                "zone": zone,
                "center": [round(cx / max(width, 1), 3), round(cy / max(height, 1), 3)],
                "area_ratio": round(float(area_ratio), 4),
            })
        except Exception:
            enriched.setdefault("zone", "unknown")
        return enriched

    def _detect_motion(self, image, cv2, np) -> dict[str, Any]:
        if not bool(self.cfg.get("vision.motion.enabled", True)):
            return {"enabled": False, "detected": False}
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self._prev_gray is None:
                self._prev_gray = gray
                self._prev_motion_ts = time.time()
                return {"enabled": True, "detected": False, "reason": "priming"}

            frame_delta = cv2.absdiff(self._prev_gray, gray)
            self._prev_gray = gray
            self._prev_motion_ts = time.time()

            threshold = int(self.cfg.get("vision.motion.threshold", 24))
            _, thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = image.shape[:2]
            frame_area = max(1, width * height)
            min_area_ratio = float(self.cfg.get("vision.motion.min_area_ratio", 0.01))
            min_area = frame_area * min_area_ratio
            moving = []
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w / 2.0
                zone = "left" if cx < width / 3 else ("right" if cx > width * 2 / 3 else "center")
                moving.append({"area": area, "bbox": [int(x), int(y), int(x + w), int(y + h)], "zone": zone})
            if not moving:
                return {"enabled": True, "detected": False, "changed_area_ratio": 0.0, "contours": 0}
            total_area_ratio = sum(item["area"] for item in moving) / frame_area
            largest = max(moving, key=lambda item: item["area"])
            confidence = min(1.0, 0.35 + total_area_ratio * 4.0)
            return {
                "enabled": True,
                "detected": True,
                "zone": largest["zone"],
                "bbox": largest["bbox"],
                "changed_area_ratio": round(float(total_area_ratio), 4),
                "contours": len(moving),
                "confidence": round(float(confidence), 3),
            }
        except Exception as exc:
            return {"enabled": True, "detected": False, "error": str(exc)}

    def _build_scene(self, objects: list[dict[str, Any]], motion: dict[str, Any]) -> dict[str, Any]:
        object_zones = {"left": [], "center": [], "right": [], "unknown": []}
        close_threshold = float(self.cfg.get("vision.obstacle_area_ratio", 0.10))
        close_obstacles = []
        person_count = 0
        face_count = 0
        for obj in objects:
            label = str(obj.get("label", "object"))
            zone = str(obj.get("zone", "unknown"))
            if zone not in object_zones:
                zone = "unknown"
            object_zones[zone].append(label)
            if label == "person":
                person_count += 1
            if "face" in label:
                face_count += 1
            area_ratio = float(obj.get("area_ratio", 0.0) or 0.0)
            if label in OBSTACLE_LABELS and area_ratio >= close_threshold:
                close_obstacles.append({
                    "label": label,
                    "zone": zone,
                    "area_ratio": round(area_ratio, 4),
                    "confidence": obj.get("confidence"),
                })
        object_zones = {k: v[:5] for k, v in object_zones.items() if v}
        attention = "clear"
        if close_obstacles:
            attention = "obstacle_close"
        elif person_count or face_count:
            attention = "person_visible"
        elif motion.get("detected"):
            attention = "motion_visible"
        return {
            "person_count": person_count,
            "face_count": face_count,
            "object_count": len(objects),
            "object_zones": object_zones,
            "close_obstacles": close_obstacles[:4],
            "attention": attention,
        }

    def _summary(self, objects: list[dict[str, Any]], scene: dict[str, Any], motion: dict[str, Any]) -> str:
        parts: list[str] = []
        if objects:
            visible = []
            for obj in objects[:5]:
                label = obj.get("label", "object")
                zone = obj.get("zone", "unknown")
                conf = obj.get("confidence")
                conf_text = f" {float(conf):.2f}" if isinstance(conf, (int, float)) else ""
                visible.append(f"{label} in {zone}{conf_text}")
            parts.append("Objects: " + ", ".join(visible) + ".")
        else:
            detector_status = self.object_detector.status()
            if detector_status.get("model_ready"):
                parts.append("No COCO objects detected in the current frame.")
            else:
                error = detector_status.get("error") or "YOLO model is not ready"
                parts.append(f"Camera frame available, but object detection is not ready: {error}.")

        if motion.get("detected"):
            parts.append(f"Motion detected in the {motion.get('zone', 'unknown')} zone.")
        else:
            parts.append("No significant motion detected.")

        close_obstacles = scene.get("close_obstacles") or []
        if close_obstacles:
            labels = ", ".join(f"{o.get('label')} in {o.get('zone')}" for o in close_obstacles[:3])
            parts.append(f"Possible close obstacle: {labels}.")
        return " ".join(parts)

    def _objects_confidence(self, objects: list[dict[str, Any]]) -> float:
        if not objects:
            return 0.0
        try:
            return max(float(obj.get("confidence", 0.0)) for obj in objects)
        except Exception:
            return 0.0


class ObjectDetector:
    """Small YOLO11 ONNX detector for Raspberry Pi CPU inference.

    The app uses the Ultralytics YOLO11n detection export. Its common ONNX
    output is [1, 84, 8400], but local exports may be [1, 8400, 84] or an
    NMS-enabled [N, 6]. The decoder accepts these layouts so object recognition
    does not silently fail when the model/export changes.
    """

    _COCO_NAMES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush",
    ]

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("vision.object_detection.enabled", True))
        self.backend = str(cfg.get("vision.object_detection.backend", "yolo")).strip().lower()
        self.model_path = Path(str(cfg.path("vision.object_detection.model_path", "models/vision/yolo11n.onnx")))
        self.confidence_threshold = float(cfg.get("vision.object_detection.confidence_threshold", 0.20))
        self.nms_threshold = float(cfg.get("vision.object_detection.nms_threshold", 0.45))
        self.max_objects = int(cfg.get("vision.object_detection.max_objects", 10))
        self.yolo_imgsz = int(cfg.get("vision.object_detection.imgsz", 640))
        self.threads = max(1, int(cfg.get("vision.object_detection.threads", 2)))
        self.debug = bool(cfg.get("vision.object_detection.debug", False))

        self.backend_name = "disabled" if not self.enabled else "yolo-not-loaded"
        self._load_error: str | None = None
        self._yolo = None
        self._yolo_input_name: str | None = None
        self._yolo_input_shape: list[Any] | None = None
        self._yolo_model_file: str | None = None
        self._last_stats: dict[str, Any] = {}

    def _yolo_candidates(self) -> list[Path]:
        # Keep project-root fallbacks because older zips placed yolo11n.onnx next
        # to app.py; the clean repository uses models/vision/yolo11n.onnx.
        return [
            self.model_path,
            self.model_path.parent / "yolo11n.onnx",
            self.model_path.parent.parent / "yolo11n.onnx",
            self.model_path.parent.parent.parent / "yolo11n.onnx",
        ]

    def _find_yolo_model(self) -> Path | None:
        for candidate in self._yolo_candidates():
            if candidate.exists():
                return candidate
        return None

    def status(self) -> dict[str, Any]:
        model_file = self._find_yolo_model() if self.backend == "yolo" else None
        return {
            "enabled": self.enabled,
            "backend": self.backend_name,
            "configured_backend": self.backend,
            "model_path": str(self.model_path),
            "resolved_model_path": str(model_file) if model_file else None,
            "model_ready": bool(model_file) if self.backend == "yolo" else False,
            "confidence_threshold": self.confidence_threshold,
            "nms_threshold": self.nms_threshold,
            "imgsz": self.yolo_imgsz,
            "threads": self.threads,
            "last_stats": self._last_stats,
            "error": self._load_error,
        }

    def detect(self, image, cv2, np) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        if self.backend != "yolo":
            self._load_error = f"Unsupported object detector backend: {self.backend}. Use backend: yolo."
            self.backend_name = "unsupported-backend"
            return []
        return self._detect_with_yolo(image, cv2, np)

    def _load_yolo(self):
        if self._yolo is not None:
            return self._yolo

        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:
            self._load_error = "onnxruntime is not installed. Run: pip install -U onnxruntime"
            self.backend_name = "onnxruntime-missing"
            raise RuntimeError(self._load_error) from exc

        model_file = self._find_yolo_model()
        if model_file is None:
            self._load_error = (
                f"YOLO model not found. Checked: {[str(c) for c in self._yolo_candidates()]}. "
                "Run: python scripts/download_models.py --skip-llm --skip-stt --tts-engine kokoro --vision yolo"
            )
            self.backend_name = "yolo-missing"
            raise FileNotFoundError(self._load_error)

        try:
            # Suppress harmless Raspberry Pi DRM/GPU discovery warnings when only
            # CPUExecutionProvider is available/desired.
            ort.set_default_logger_severity(3)
        except Exception:
            pass

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.inter_op_num_threads = self.threads
        opts.intra_op_num_threads = self.threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._yolo = ort.InferenceSession(
            str(model_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        yolo_input = self._yolo.get_inputs()[0]
        self._yolo_input_name = yolo_input.name
        self._yolo_input_shape = list(yolo_input.shape)
        self._yolo_model_file = str(model_file)
        self.backend_name = "onnxruntime-yolo"
        self._load_error = None
        return self._yolo

    def _input_size(self) -> int:
        # Prefer a fixed model input if the ONNX file has one; otherwise use config.
        try:
            shape = self._yolo_input_shape or []
            h = shape[2] if len(shape) >= 4 else None
            w = shape[3] if len(shape) >= 4 else None
            if isinstance(h, int) and isinstance(w, int) and h == w and h > 0:
                return int(h)
        except Exception:
            pass
        return int(self.yolo_imgsz)

    def _detect_with_yolo(self, image, cv2, np) -> list[dict[str, Any]]:
        started = time.perf_counter()
        try:
            session = self._load_yolo()
        except Exception:
            return []

        try:
            orig_h, orig_w = image.shape[:2]
            imgsz = self._input_size()
            blob, scale, pad_x, pad_y = self._letterbox_blob(image, cv2, np, imgsz)
            outputs = session.run(None, {self._yolo_input_name: blob})
            raw = outputs[0]
            preds = self._normalise_yolo_output(raw, np)
            decoded = self._decode_yolo_predictions(preds, np)

            max_conf_before_filter = float(np.max(decoded["confidences"])) if len(decoded["confidences"]) else 0.0
            keep = decoded["confidences"] >= self.confidence_threshold
            confidences = decoded["confidences"][keep]
            class_ids = decoded["class_ids"][keep]
            x1s = decoded["x1s"][keep]
            y1s = decoded["y1s"][keep]
            x2s = decoded["x2s"][keep]
            y2s = decoded["y2s"][keep]

            if not decoded["xyxy_in_original_image"]:
                x1s = (x1s - pad_x) / scale
                y1s = (y1s - pad_y) / scale
                x2s = (x2s - pad_x) / scale
                y2s = (y2s - pad_y) / scale

            x1s = np.clip(x1s, 0, orig_w).astype(int)
            y1s = np.clip(y1s, 0, orig_h).astype(int)
            x2s = np.clip(x2s, 0, orig_w).astype(int)
            y2s = np.clip(y2s, 0, orig_h).astype(int)

            valid = (x2s > x1s) & (y2s > y1s)
            confidences = confidences[valid]
            class_ids = class_ids[valid]
            x1s, y1s, x2s, y2s = x1s[valid], y1s[valid], x2s[valid], y2s[valid]

            objects: list[dict[str, Any]] = []
            if len(confidences):
                boxes_xywh = [
                    [int(x1s[i]), int(y1s[i]), int(x2s[i] - x1s[i]), int(y2s[i] - y1s[i])]
                    for i in range(len(confidences))
                ]
                nms_idx = cv2.dnn.NMSBoxes(
                    boxes_xywh,
                    [float(v) for v in confidences],
                    self.confidence_threshold,
                    self.nms_threshold,
                )
                nms_idx = np.array(nms_idx).reshape(-1) if len(nms_idx) else []

                for i in list(nms_idx)[: self.max_objects]:
                    cls_id = int(class_ids[int(i)])
                    label = self._COCO_NAMES[cls_id] if 0 <= cls_id < len(self._COCO_NAMES) else f"class_{cls_id}"
                    objects.append({
                        "label": label,
                        "confidence": round(float(confidences[int(i)]), 3),
                        "bbox": [int(x1s[int(i)]), int(y1s[int(i)]), int(x2s[int(i)]), int(y2s[int(i)])],
                        "class_id": cls_id,
                        "detector": "yolo11n-onnx",
                    })

            objects.sort(key=lambda o: float(o.get("confidence", 0.0)), reverse=True)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_stats = {
                "raw_shape": list(raw.shape),
                "preds_shape": list(preds.shape),
                "max_confidence_before_filter": round(max_conf_before_filter, 4),
                "candidates_above_threshold": int(len(confidences)),
                "objects_returned": int(len(objects)),
                "input_size": imgsz,
                "elapsed_ms": round(elapsed_ms, 1),
            }
            self.backend_name = "onnxruntime-yolo"
            self._load_error = None
            if self.debug:
                print(f"YOLO stats: {self._last_stats}", flush=True)
            return objects

        except Exception as exc:
            self._load_error = f"YOLO inference failed: {exc}"
            self.backend_name = "yolo-error"
            return []

    def _letterbox_blob(self, image, cv2, np, imgsz: int):
        orig_h, orig_w = image.shape[:2]
        scale = min(imgsz / max(orig_w, 1), imgsz / max(orig_h, 1))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        pad_x = (imgsz - new_w) / 2.0
        pad_y = (imgsz - new_h) / 2.0

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        left = int(round(pad_x - 0.1))
        top = int(round(pad_y - 0.1))
        canvas[top:top + new_h, left:left + new_w] = resized

        # OpenCV camera frames are BGR. YOLO expects RGB float NCHW in 0..1.
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.expand_dims(blob.transpose(2, 0, 1), axis=0)
        return blob, scale, float(left), float(top)

    def _normalise_yolo_output(self, raw, np):
        preds = np.squeeze(raw)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        if preds.ndim != 2:
            raise ValueError(f"unexpected YOLO output shape: raw={getattr(raw, 'shape', None)}, squeezed={preds.shape}")

        # Common Ultralytics export: [84, 8400] after squeeze. Some export paths
        # produce [8400, 84]. NMS-enabled exports may be [6, N] or [N, 6].
        if preds.shape[0] in (6, 84, 85) and preds.shape[1] > preds.shape[0]:
            preds = preds.T
        if preds.shape[1] not in (6, 84, 85):
            raise ValueError(f"unsupported YOLO output shape: raw={getattr(raw, 'shape', None)}, squeezed={preds.shape}")
        return preds

    def _decode_yolo_predictions(self, preds, np) -> dict[str, Any]:
        if preds.shape[1] == 6:
            x1s, y1s, x2s, y2s = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
            confidences = preds[:, 4].astype(float)
            class_ids = preds[:, 5].astype(int)
            # If an export returns normalized xyxy, convert to configured input pixels.
            if max(float(np.max(x2s)), float(np.max(y2s))) <= 1.5:
                x1s = x1s * self._input_size()
                y1s = y1s * self._input_size()
                x2s = x2s * self._input_size()
                y2s = y2s * self._input_size()
            return {
                "x1s": x1s, "y1s": y1s, "x2s": x2s, "y2s": y2s,
                "confidences": confidences, "class_ids": class_ids,
                "xyxy_in_original_image": False,
            }

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        if preds.shape[1] == 84:
            class_scores = preds[:, 4:]
            class_ids = np.argmax(class_scores, axis=1)
            confidences = class_scores[np.arange(len(class_scores)), class_ids]
        else:  # [N, 85] = cx, cy, w, h, objectness, 80 class scores
            objectness = preds[:, 4]
            class_scores = preds[:, 5:]
            class_ids = np.argmax(class_scores, axis=1)
            class_conf = class_scores[np.arange(len(class_scores)), class_ids]
            confidences = objectness * class_conf

        return {
            "x1s": cx - bw / 2.0,
            "y1s": cy - bh / 2.0,
            "x2s": cx + bw / 2.0,
            "y2s": cy + bh / 2.0,
            "confidences": confidences.astype(float),
            "class_ids": class_ids.astype(int),
            "xyxy_in_original_image": False,
        }
