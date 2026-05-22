from __future__ import annotations

import math
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


@dataclass(frozen=True)
class VisionObservation:
    ok: bool
    ts: float
    iso: str
    backend: str
    summary: str
    confidence: float = 0.0
    hand: dict[str, Any] = field(default_factory=dict)
    objects: list[dict[str, Any]] = field(default_factory=list)
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
            "hand": self.hand,
            "objects": self.objects,
            "frame": self.frame,
            "error": self.error,
            "reason": self.reason,
        }

    def to_prompt_text(self) -> str:
        hand = self.hand or {}
        parts = [
            f"time={self.iso}",
            f"backend={self.backend}",
            f"summary={self.summary}",
            f"confidence={self.confidence:.2f}",
        ]
        if hand:
            parts.append(f"hand_detected={hand.get('detected')}")
            parts.append(f"gesture={hand.get('gesture')}")
            parts.append(f"finger_count={hand.get('finger_count')}")
            if hand.get("fingers"):
                parts.append(f"fingers={hand.get('fingers')}")
        if self.objects:
            compact = [
                {
                    "label": obj.get("label"),
                    "confidence": obj.get("confidence"),
                    "bbox": obj.get("bbox"),
                }
                for obj in self.objects[:6]
            ]
            parts.append(f"objects={compact}")
        if self.error:
            parts.append(f"error={self.error}")
        return "\n".join(parts)


class VisionService:
    """Turns camera frames into concise text observations for the UI and LLM prompt.

    The local LLM in this project is text-only. This service is the bridge: it converts
    the latest camera frame into structured text such as "open palm, 5 fingers" and
    basic object detections such as "person" or "bottle".
    """

    def __init__(self, cfg, camera: CameraManager, logger=None) -> None:
        self.cfg = cfg
        self.camera = camera
        self.logger = logger
        self.enabled = bool(cfg.get("vision.enabled", False))
        self.poll_interval_s = float(cfg.get("vision.poll_interval_s", 1.0))
        self.max_prompt_age_s = float(cfg.get("vision.max_prompt_age_s", 2.5))
        self.snapshot_on_turn = bool(cfg.get("vision.snapshot_on_turn", True))
        self.always_attach = bool(cfg.get("vision.always_attach_to_prompt", True))
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
            "hand", "hands", "finger", "fingers", "fist", "thumb", "gesture", "palm",
            "бач", "камера", "камер", "зір", "фото", "картин", "об'єкт", "обєкт",
            "предмет", "розпізн", "детект", "що бач", "рук", "палець", "пальц", "кулак", "жест", "долон",
        ]
        likely_visual = any(word in lower for word in vision_words)

        obs = self.latest()
        fresh = bool(obs and (time.time() - obs.ts) <= self.max_prompt_age_s)
        if self.snapshot_on_turn and (self.always_attach or likely_visual or not fresh):
            obs = self.analyze_now(reason="turn")
        if obs and (self.always_attach or likely_visual):
            return obs.to_prompt_text()
        return ""

    def status(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "enabled": self.enabled,
            "backend": self.analyzer.backend_name,
            "object_detector": self.analyzer.object_detector.status(),
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
        hand: dict[str, Any] | None = None,
        objects: list[dict[str, Any]] | None = None,
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
            hand=hand or {},
            objects=objects or [],
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
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.backend_name = "auto"
        self._mp_hands = None
        self._mp = None
        self._mediapipe_failed = False
        self.object_detector = ObjectDetector(cfg)

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
                summary="Camera frame is available. Install python3-opencv for hand and object detection.",
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

        mp_obs = self._try_mediapipe(image, cv2, now, iso, frame, objects, reason)
        if mp_obs is not None:
            return mp_obs

        return self._analyze_with_opencv_contours(image, cv2, np, now, iso, frame, objects, reason)

    def _load_cv(self):
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            return cv2, np
        except Exception:
            return None, None

    def _try_mediapipe(self, image, cv2, now: float, iso: str, frame: dict[str, Any], objects: list[dict[str, Any]], reason: str) -> VisionObservation | None:
        if self._mediapipe_failed:
            return None
        try:
            if self._mp_hands is None:
                import mediapipe as mp  # type: ignore
                self._mp = mp
                self._mp_hands = mp.solutions.hands.Hands(
                    static_image_mode=True,
                    max_num_hands=int(self.cfg.get("vision.max_hands", 1)),
                    min_detection_confidence=float(self.cfg.get("vision.min_detection_confidence", 0.55)),
                    min_tracking_confidence=float(self.cfg.get("vision.min_tracking_confidence", 0.5)),
                )
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self._mp_hands.process(rgb)
        except Exception:
            self._mediapipe_failed = True
            return None

        if not getattr(result, "multi_hand_landmarks", None):
            summary = self._combined_summary("No hand detected in the current camera frame.", objects)
            return VisionObservation(
                ok=True,
                ts=now,
                iso=iso,
                backend=self._backend_with_objects("mediapipe"),
                summary=summary,
                confidence=max(0.55, self._objects_confidence(objects)),
                hand={"detected": False, "gesture": "none", "finger_count": None},
                objects=objects,
                frame=frame,
                reason=reason,
            )

        hand_landmarks = result.multi_hand_landmarks[0]
        handedness = None
        score = 0.75
        if getattr(result, "multi_handedness", None):
            cls = result.multi_handedness[0].classification[0]
            handedness = getattr(cls, "label", None)
            score = float(getattr(cls, "score", score))

        lm = hand_landmarks.landmark
        fingers = self._fingers_from_landmarks(lm)
        finger_count = int(sum(1 for is_up in fingers.values() if is_up))
        gesture = self._gesture_from_fingers(fingers, lm)
        hand_summary = self._hand_summary(gesture, finger_count, score)
        summary = self._combined_summary(hand_summary, objects)
        hand = {
            "detected": True,
            "gesture": gesture,
            "finger_count": finger_count,
            "fingers": fingers,
            "handedness": handedness,
        }
        return VisionObservation(
            ok=True,
            ts=now,
            iso=iso,
            backend=self._backend_with_objects("mediapipe"),
            summary=summary,
            confidence=max(max(0.0, min(1.0, score)), self._objects_confidence(objects)),
            hand=hand,
            objects=objects,
            frame=frame,
            reason=reason,
        )

    def _fingers_from_landmarks(self, lm) -> dict[str, bool]:
        # Normalized landmark indices from MediaPipe Hands.
        def dist(a: int, b: int) -> float:
            return math.hypot(float(lm[a].x - lm[b].x), float(lm[a].y - lm[b].y))

        # Index/middle/ring/pinky: tip above PIP is a good camera-facing approximation.
        margin = 0.018
        index = lm[8].y < lm[6].y - margin
        middle = lm[12].y < lm[10].y - margin
        ring = lm[16].y < lm[14].y - margin
        pinky = lm[20].y < lm[18].y - margin

        # Thumb can point sideways or upward. Count it if the tip is clearly away from the palm.
        thumb_tip_far = dist(4, 0) > dist(3, 0) + 0.025
        thumb_sideways = abs(float(lm[4].x - lm[2].x)) > 0.055
        thumb_vertical = float(lm[4].y) < float(lm[2].y) - 0.045
        thumb = thumb_tip_far and (thumb_sideways or thumb_vertical)
        return {
            "thumb": bool(thumb),
            "index": bool(index),
            "middle": bool(middle),
            "ring": bool(ring),
            "pinky": bool(pinky),
        }

    def _gesture_from_fingers(self, fingers: dict[str, bool], lm) -> str:
        count = sum(1 for v in fingers.values() if v)
        if count == 0:
            return "fist"
        if count >= 4:
            return "open_palm"
        if fingers.get("thumb") and count == 1:
            # Require a mostly vertical thumb for thumbs-up; otherwise it may just be a sideways thumb.
            if float(lm[4].y) < float(lm[2].y) - 0.06:
                return "thumbs_up"
            return "thumb_only"
        if fingers.get("index") and count == 1:
            return "one_finger"
        if fingers.get("index") and fingers.get("middle") and count == 2:
            return "two_fingers"
        return f"{count}_fingers"

    def _hand_summary(self, gesture: str, finger_count: int | None, confidence: float) -> str:
        if gesture == "fist":
            return "One hand detected: fist / closed hand, 0 raised fingers."
        if gesture == "open_palm":
            return "One hand detected: open palm, about 5 raised fingers."
        if gesture == "thumbs_up":
            return "One hand detected: thumbs-up gesture."
        if finger_count is not None:
            return f"One hand detected: {gesture.replace('_', ' ')}, about {finger_count} raised finger(s)."
        return f"One hand detected, gesture uncertain. Confidence {confidence:.2f}."

    def _combined_summary(self, hand_summary: str, objects: list[dict[str, Any]]) -> str:
        if not objects:
            return hand_summary
        labels: list[str] = []
        for obj in objects[:4]:
            label = str(obj.get("label", "object"))
            conf = obj.get("confidence")
            if isinstance(conf, (int, float)):
                labels.append(f"{label} ({float(conf):.2f})")
            else:
                labels.append(label)
        return f"{hand_summary} Objects detected: {', '.join(labels)}."

    def _objects_confidence(self, objects: list[dict[str, Any]]) -> float:
        if not objects:
            return 0.0
        try:
            return max(float(obj.get("confidence", 0.0)) for obj in objects)
        except Exception:
            return 0.0

    def _backend_with_objects(self, hand_backend: str) -> str:
        detector_backend = self.object_detector.backend_name
        if detector_backend and detector_backend != "disabled":
            return f"{hand_backend}+{detector_backend}"
        return hand_backend

    def _analyze_with_opencv_contours(
        self,
        image,
        cv2,
        np,
        now: float,
        iso: str,
        frame: dict[str, Any],
        objects: list[dict[str, Any]],
        reason: str,
    ) -> VisionObservation:
        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Wide skin-color range. This is not robust across lighting/skin tones, but is a no-model fallback.
        lower1 = np.array([0, 25, 45], dtype=np.uint8)
        upper1 = np.array([25, 255, 255], dtype=np.uint8)
        lower2 = np.array([160, 25, 45], dtype=np.uint8)
        upper2 = np.array([179, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            summary = self._combined_summary("No hand-like contour detected in the current camera frame.", objects)
            return VisionObservation(
                ok=True,
                ts=now,
                iso=iso,
                backend=self._backend_with_objects("opencv-contour"),
                summary=summary,
                confidence=max(0.35, self._objects_confidence(objects)),
                hand={"detected": False, "gesture": "none", "finger_count": None},
                objects=objects,
                frame=frame,
                reason=reason,
            )

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        frame_area = float(width * height)
        if area < frame_area * float(self.cfg.get("vision.min_hand_area_ratio", 0.015)):
            summary = self._combined_summary("Only small hand-like contours detected; no reliable hand gesture yet.", objects)
            return VisionObservation(
                ok=True,
                ts=now,
                iso=iso,
                backend=self._backend_with_objects("opencv-contour"),
                summary=summary,
                confidence=max(0.3, self._objects_confidence(objects)),
                hand={"detected": False, "gesture": "none", "finger_count": None},
                objects=objects,
                frame={**frame, "largest_contour_area": round(area, 1)},
                reason=reason,
            )

        hull_points = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull_points)) if hull_points is not None else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)

        defects_count = 0
        try:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            if hull_indices is not None and len(hull_indices) >= 4:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, depth = defects[i, 0]
                        start = contour[s][0]
                        end = contour[e][0]
                        far = contour[f][0]
                        a = math.dist(start, end)
                        b = math.dist(start, far)
                        c = math.dist(end, far)
                        if b <= 1e-6 or c <= 1e-6:
                            continue
                        angle = math.degrees(math.acos(max(-1.0, min(1.0, (b * b + c * c - a * a) / (2 * b * c)))))
                        if angle < 90 and depth > 9000:
                            defects_count += 1
        except Exception:
            defects_count = 0

        finger_count = max(0, min(5, defects_count + 1 if defects_count > 0 else 0))
        aspect = h / max(w, 1)
        if defects_count == 0 and solidity > 0.78:
            gesture = "fist"
            finger_count = 0
        elif finger_count >= 4:
            gesture = "open_palm"
        elif aspect > 1.45 and finger_count <= 1:
            gesture = "thumbs_up_or_one_finger"
            finger_count = max(finger_count, 1)
        elif finger_count > 0:
            gesture = f"{finger_count}_fingers"
        else:
            gesture = "uncertain_hand"
            finger_count = None

        confidence = 0.48 if gesture in {"fist", "open_palm"} else 0.38
        hand = {
            "detected": True,
            "gesture": gesture,
            "finger_count": finger_count,
            "solidity": round(solidity, 3),
            "bounding_box": [int(x), int(y), int(w), int(h)],
            "defects_count": int(defects_count),
        }
        hand_summary = self._hand_summary(gesture, finger_count, confidence)
        if gesture == "thumbs_up_or_one_finger":
            hand_summary = "One hand-like contour detected: possibly thumbs-up or one raised finger. Confidence is low without MediaPipe."
        summary = self._combined_summary(hand_summary, objects)
        return VisionObservation(
            ok=True,
            ts=now,
            iso=iso,
            backend=self._backend_with_objects("opencv-contour"),
            summary=summary,
            confidence=max(confidence, self._objects_confidence(objects)),
            hand=hand,
            objects=objects,
            frame={**frame, "largest_contour_area": round(area, 1)},
            reason=reason,
        )


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
        dnn_objects = self._detect_with_dnn(image, cv2, np)
        if dnn_objects:
            return dnn_objects
        return self._detect_fallback_faces(image, cv2)

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
