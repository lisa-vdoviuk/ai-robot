from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .camera import CameraManager


VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
]

OBSTACLE_LABELS = {
    "person", "chair", "dining table", "sofa", "bottle", "cat", "dog", "car", "bus",
    "bicycle", "motorbike", "potted plant", "tv monitor",
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

        if not self._analyze_lock.acquire(timeout=0.05):
            existing = self.latest()
            if existing:
                return existing
            self._analyze_lock.acquire()
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
        return {
            "enabled": self.enabled,
            "backend": self.backend_name,
            "mode": self.backend,
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "model_ready": self.backend == "yolo" or self.model_path.exists(),
            "error": self._load_error,
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
            detector_ready = self.object_detector.status().get("model_ready")
            if detector_ready:
                parts.append("No known objects detected in the current frame.")
            else:
                parts.append("Camera frame available; object model files are missing, so only motion/face fallback may work.")

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
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.get("vision.object_detection.enabled", True))
        self.model_path = Path(str(cfg.path("vision.object_detection.model_path", "models/vision/mobilenet_iter_73000.caffemodel")))
        self.config_path = Path(str(cfg.path("vision.object_detection.config_path", "models/vision/deploy.prototxt")))
        self.confidence_threshold = float(cfg.get("vision.object_detection.confidence_threshold", 0.45))
        self.max_objects = int(cfg.get("vision.object_detection.max_objects", 6))
        self.backend_name = "disabled" if not self.enabled else "opencv-dnn"
        self._net = None
        self._load_error: str | None = None
        self._haar_loaded = False
        self._face_cascade = None
        self.backend = str(cfg.get("vision.object_detection.backend", "mobilenet")).strip().lower()
        self.yolo_imgsz = int(cfg.get("vision.object_detection.imgsz", 416))
        self._yolo = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "backend": self.backend_name,
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "model_ready": self.model_path.exists() and self.config_path.exists(),
            "error": self._load_error,
        }

    def detect(self, image, cv2, np) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        if self.backend == "yolo":
            yolo_objects = self._detect_with_yolo(image)
            if yolo_objects:
                return yolo_objects

        dnn_objects = self._detect_with_dnn(image, cv2, np)
        if dnn_objects:
            return dnn_objects

        return self._detect_fallback_faces(image, cv2)
    def _detect_with_yolo(self, image) -> list[dict[str, Any]]:
        try:
            if self._yolo is None:
                from ultralytics import YOLO  # type: ignore

                self._yolo = YOLO(str(self.model_path))
                self.backend_name = "ultralytics-yolo"

            results = self._yolo.predict(
                image,
                imgsz=self.yolo_imgsz,
                conf=self.confidence_threshold,
                max_det=self.max_objects,
                verbose=False,
                device="cpu",
            )
        except Exception as exc:
            self._load_error = f"yolo failed: {exc}"
            self.backend_name = "yolo-error"
            return []

        objects: list[dict[str, Any]] = []

        try:
            if not results:
                return []

            result = results[0]
            names = getattr(result, "names", {}) or {}

            for box in result.boxes:
                xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
                conf = float(box.conf[0].detach().cpu().item())
                cls_id = int(box.cls[0].detach().cpu().item())

                label = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else f"class_{cls_id}"

                x1, y1, x2, y2 = [int(v) for v in xyxy]

                objects.append({
                    "label": str(label),
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "detector": "yolo",
                })
        except Exception as exc:
            self._load_error = f"yolo parse failed: {exc}"
            return []

        objects.sort(key=lambda obj: float(obj.get("confidence", 0.0)), reverse=True)
        return objects[: self.max_objects]
    def _detect_with_dnn(self, image, cv2, np) -> list[dict[str, Any]]:
        if not (self.model_path.exists() and self.config_path.exists()):
            self._load_error = "object model files are missing; run: python scripts/download_models.py --skip-llm --skip-stt --skip-tts --vision mobilenet-ssd"
            return []
        try:
            if self._net is None:
                self._net = cv2.dnn.readNetFromCaffe(str(self.config_path), str(self.model_path))
                self.backend_name = "opencv-dnn"
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5), False, False)
            self._net.setInput(blob)
            detections = self._net.forward()
        except Exception as exc:
            self._load_error = str(exc)
            self._net = None
            return []

        objects: list[dict[str, Any]] = []
        try:
            for i in range(detections.shape[2]):
                confidence = float(detections[0, 0, i, 2])
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(detections[0, 0, i, 1])
                label = VOC_CLASSES[class_id] if 0 <= class_id < len(VOC_CLASSES) else f"class_{class_id}"
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                start_x, start_y, end_x, end_y = box.astype("int")
                start_x = max(0, min(int(start_x), w - 1))
                start_y = max(0, min(int(start_y), h - 1))
                end_x = max(0, min(int(end_x), w - 1))
                end_y = max(0, min(int(end_y), h - 1))
                objects.append({
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [start_x, start_y, end_x, end_y],
                })
        except Exception as exc:
            self._load_error = f"dnn parse failed: {exc}"
            return []

        objects.sort(key=lambda obj: float(obj.get("confidence", 0.0)), reverse=True)
        return objects[: self.max_objects]

    def _detect_fallback_faces(self, image, cv2) -> list[dict[str, Any]]:
        try:
            if not self._haar_loaded:
                cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
                self._face_cascade = cv2.CascadeClassifier(cascade_path)
                self._haar_loaded = True
                self.backend_name = "opencv-haar-fallback"
            if self._face_cascade is None or self._face_cascade.empty():
                return []
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(45, 45))
        except Exception:
            return []

        objects = []
        for (x, y, w, h) in list(faces)[: self.max_objects]:
            objects.append({
                "label": "face/person",
                "confidence": 0.55,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
            })
        return objects
