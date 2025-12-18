"""Postprocessing utilities: IoU and NMS."""

from __future__ import annotations

from typing import Iterable, List

from core.detector_yolo import Detection


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in xyxy format."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def nms(detections: Iterable[Detection], iou_thresh: float) -> List[Detection]:
    """Non-maximum suppression grouped by label."""
    kept: list[Detection] = []
    by_label: dict[str, list[Detection]] = {}
    for det in detections:
        by_label.setdefault(det.label, []).append(det)

    for label, dets in by_label.items():
        sorted_dets = sorted(dets, key=lambda d: d.score, reverse=True)
        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            remaining: list[Detection] = []
            for det in sorted_dets:
                iou = iou_xyxy(best.box_xyxy, det.box_xyxy)
                if iou < iou_thresh:
                    remaining.append(det)
            sorted_dets = remaining
    return kept
