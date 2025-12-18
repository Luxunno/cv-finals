"""Traffic light state classifier using simple HSV rules."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(x1, width - 1)))
    y1 = int(max(0, min(y1, height - 1)))
    x2 = int(max(0, min(x2, width - 1)))
    y2 = int(max(0, min(y2, height - 1)))
    return x1, y1, x2, y2


def classify_traffic_light_state(image_bgr: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> str:
    """Classify traffic light state: red | green | unknown."""
    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(box_xyxy, w_img, h_img)
    if x2 <= x1 or y2 <= y1:
        return "unknown"

    roi = image_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask_bright = v > 80
    red_mask = ((h < 15) | (h > 165)) & (s > 50) & mask_bright
    green_mask = (h > 45) & (h < 90) & (s > 50) & mask_bright

    red_ratio = red_mask.sum() / max(1, red_mask.size)
    green_ratio = green_mask.sum() / max(1, green_mask.size)
    red_mean_v = v[red_mask].mean() if red_mask.any() else 0
    green_mean_v = v[green_mask].mean() if green_mask.any() else 0

    red_score = red_ratio * 0.7 + (red_mean_v / 255.0) * 0.3
    green_score = green_ratio * 0.7 + (green_mean_v / 255.0) * 0.3

    if red_score > 0.05 and red_score > green_score * 1.2:
        return "red"
    if green_score > 0.05 and green_score > red_score * 1.2:
        return "green"
    return "unknown"
