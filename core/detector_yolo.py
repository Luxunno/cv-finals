"""YOLOv8 detector adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str
    class_id: int
    score: float
    box_xyxy: tuple[int, int, int, int]
    attrs: dict


class YoloDetector:
    """Adapter for YOLOv8n detection."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = str(model_path) if model_path else "yolov8n.pt"
        self.model = None
        self.class_names: list[str] = []
        self.device = "cpu"
        self.warmed_up = False

    def load(self, device: str = "cpu") -> None:
        """Load YOLO model and class names."""
        if self.model is not None:
            return

        from ultralytics import YOLO

        self.device = device
        self.model = YOLO(self.model_path)
        self.model.to(device)
        self.class_names = self._load_class_names()

    def detect(
        self,
        image_bgr: np.ndarray,
        conf_threshold: float,
        nms_iou: float,
        max_det: int,
    ) -> List[Detection]:
        """Run inference and return sorted detections."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        results = self.model.predict(
            image_bgr,
            conf=conf_threshold,
            iou=nms_iou,
            max_det=max_det,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for res in results:
            if not hasattr(res, "boxes"):
                continue
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            scores = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for coord, score, cls_id in zip(xyxy, scores, classes):
                x1, y1, x2, y2 = self._clip_box(coord, width, height)
                label = self._label_from_id(cls_id)
                det = Detection(
                    label=label,
                    class_id=int(cls_id),
                    score=float(score),
                    box_xyxy=(x1, y1, x2, y2),
                    attrs={},
                )
                detections.append(det)
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def num_parameters(self) -> int:
        """Return total number of model parameters."""
        if self.model is None:
            return 0
        try:
            import torch
        except ImportError:
            return 0
        return int(sum(p.numel() for p in self.model.model.parameters()))

    def _label_from_id(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return str(class_id)

    def _clip_box(self, box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        x1 = int(max(0, min(x1, width - 1)))
        y1 = int(max(0, min(y1, height - 1)))
        x2 = int(max(0, min(x2, width - 1)))
        y2 = int(max(0, min(y2, height - 1)))
        return x1, y1, x2, y2

    def _load_class_names(self) -> list[str]:
        names_path = Path(__file__).resolve().parent.parent / "assets" / "coco80.names"
        if names_path.exists():
            with names_path.open("r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
                if names:
                    return names
        if self.model is not None and hasattr(self.model.model, "names"):
            names_map = self.model.model.names
            if isinstance(names_map, dict):
                return [names_map[i] for i in sorted(names_map.keys())]
            if isinstance(names_map, list):
                return list(names_map)
        return []
