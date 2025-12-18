"""Evaluation helpers for custom dataset recall@0.5 on small objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from core.postprocess import iou_xyxy

SMALL_AREA_THRESH = 32 * 32  # 1024
GT_CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "traffic_light"]
PRED_TO_GT_CLASS_MAP: Dict[int, int] = {0: 0, 2: 1, 1: 2, 3: 3, 5: 4, 9: 5}


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Load YOLO txt labels to absolute xyxy."""
    if not label_path.exists():
        return []
    items: list[tuple[int, tuple[float, float, float, float]]] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, cx, cy, w, h = parts
            cls_id = int(float(cid))
            cx, cy, w, h = map(float, (cx, cy, w, h))
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            items.append((cls_id, (x1, y1, x2, y2)))
    return items


def map_predictions(
    detections: Iterable[dict],
) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
    """Map COCO class predictions to custom class ids; ignore unmapped."""
    mapped: list[tuple[int, tuple[float, float, float, float], float]] = []
    for det in detections:
        cid = int(det.get("class_id", -1))
        if cid not in PRED_TO_GT_CLASS_MAP:
            continue
        target_cls = PRED_TO_GT_CLASS_MAP[cid]
        box = det.get("box_xyxy")
        score = float(det.get("score", 0.0))
        if not box or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        mapped.append((target_cls, (float(x1), float(y1), float(x2), float(y2)), score))
    return mapped


def _match_single_class(
    preds: List[Tuple[Tuple[float, float, float, float], float]],
    gts: List[Tuple[float, float, float, float]],
    iou_thresh: float,
) -> int:
    """Greedy matching by score desc, return TP count."""
    if not gts or not preds:
        return 0
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    matched_gt = [False] * len(gts)
    tp = 0
    for box_pred, _score in preds_sorted:
        for idx, gt_box in enumerate(gts):
            if matched_gt[idx]:
                continue
            if iou_xyxy(box_pred, gt_box) >= iou_thresh:
                matched_gt[idx] = True
                tp += 1
                break
    return tp


def accumulate_recall_small(
    preds: List[Tuple[int, Tuple[float, float, float, float], float]],
    gts: List[Tuple[int, Tuple[float, float, float, float]]],
    iou_thresh: float = 0.5,
) -> Tuple[List[int], List[int]]:
    """Return tp_small_per_class, gt_small_per_class counts."""
    tp_small = [0] * len(GT_CLASSES)
    gt_small = [0] * len(GT_CLASSES)

    gt_by_class: dict[int, list[tuple[float, float, float, float]]] = {i: [] for i in range(len(GT_CLASSES))}
    for cls_id, box in gts:
        x1, y1, x2, y2 = box
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < SMALL_AREA_THRESH:
            gt_small[cls_id] += 1
            gt_by_class[cls_id].append(box)

    preds_by_class: dict[int, list[tuple[tuple[float, float, float, float], float]]] = {
        i: [] for i in range(len(GT_CLASSES))
    }
    for cls_id, box, score in preds:
        if cls_id in preds_by_class:
            preds_by_class[cls_id].append((box, score))

    for cls_id in range(len(GT_CLASSES)):
        tp_small[cls_id] = _match_single_class(preds_by_class[cls_id], gt_by_class[cls_id], iou_thresh)

    return tp_small, gt_small


def save_eval_report(
    path: Path,
    overall: dict,
    per_class: dict,
    runtime: dict,
    config_snapshot: dict,
    meta: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "overall": overall,
        "per_class": per_class,
        "runtime": runtime,
        "config_snapshot": config_snapshot,
        "meta": meta,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
