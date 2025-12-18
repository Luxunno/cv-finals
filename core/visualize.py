"""Visualization utilities for drawing detections."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from core.detector_yolo import Detection


def draw_detections(image_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    """Draw detections on image and return a copy."""
    canvas = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        color = _color_for_label(det.label)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            canvas,
            text,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _color_for_label(label: str) -> tuple[int, int, int]:
    # Simple deterministic color hash
    base = abs(hash(label)) % 0xFFFFFF
    r = (base >> 16) & 0xFF
    g = (base >> 8) & 0xFF
    b = base & 0xFF
    # Avoid very dark colors
    return (b // 2 + 64, g // 2 + 64, r // 2 + 64)
