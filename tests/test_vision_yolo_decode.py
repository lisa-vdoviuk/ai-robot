from __future__ import annotations

from pathlib import Path

import numpy as np

from voicepi.vision import ObjectDetector


class DummyConfig:
    def __init__(self) -> None:
        self.values = {
            "vision.object_detection.enabled": True,
            "vision.object_detection.backend": "yolo",
            "vision.object_detection.confidence_threshold": 0.20,
            "vision.object_detection.nms_threshold": 0.45,
            "vision.object_detection.max_objects": 10,
            "vision.object_detection.imgsz": 640,
            "vision.object_detection.threads": 2,
            "vision.object_detection.debug": False,
        }

    def get(self, dotted: str, default=None):
        return self.values.get(dotted, default)

    def path(self, dotted: str, default: str | None = None) -> Path:
        return Path(self.values.get(dotted, default or "models/vision/yolo11n.onnx"))


def test_normalise_yolo_output_accepts_ultralytics_layout() -> None:
    detector = ObjectDetector(DummyConfig())
    raw = np.zeros((1, 84, 8400), dtype=np.float32)
    preds = detector._normalise_yolo_output(raw, np)
    assert preds.shape == (8400, 84)


def test_decode_yolo_84_scores_uses_class_scores_without_objectness() -> None:
    detector = ObjectDetector(DummyConfig())
    preds = np.zeros((1, 84), dtype=np.float32)
    preds[0, 0:4] = [320, 320, 100, 80]
    preds[0, 4 + 41] = 0.77  # COCO class 41 = cup

    decoded = detector._decode_yolo_predictions(preds, np)

    assert decoded["class_ids"].tolist() == [41]
    assert decoded["confidences"].tolist() == [np.float32(0.77)]
    assert decoded["x1s"].tolist() == [270.0]
    assert decoded["x2s"].tolist() == [370.0]
