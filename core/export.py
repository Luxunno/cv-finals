"""Export helpers for detections and summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from core.config import JobConfig
from core.detector_yolo import Detection
from core.io_image import imwrite_any


def _detection_to_dict(det: Detection) -> dict:
    return {
        "label": det.label,
        "class_id": det.class_id,
        "score": det.score,
        "box_xyxy": list(map(int, det.box_xyxy)),
        "attrs": det.attrs,
    }


def export_detections_json(
    path: Path,
    job_id: str,
    pipeline: str,
    model_name: str,
    num_parameters: int,
    image_shape: tuple[int, int],
    config: JobConfig,
    detections: Iterable[Detection],
) -> None:
    """Write detection results JSON."""
    data = {
        "job_id": job_id,
        "mode": "image",
        "pipeline": pipeline,
        "model": {"name": model_name, "num_parameters": num_parameters},
        "image": {"width": int(image_shape[1]), "height": int(image_shape[0])},
        "config": {
            "conf_threshold": config.conf_threshold,
            "nms_iou": config.nms_iou,
            "tile_size": config.tile_size,
            "overlap": config.overlap,
            "wbf_iou": config.wbf_iou,
        },
        "detections": [_detection_to_dict(det) for det in detections],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_summary_json(
    path: Path,
    job_id: str,
    device: str,
    baseline_ms: float,
    enhanced_ms: float,
    baseline_num_boxes: int,
    enhanced_num_boxes: int,
    num_parameters_total: int,
    notes: dict,
    model_load_ms: float | None = None,
    num_tiles: int | None = None,
    tile_infer_ms_total: float | None = None,
    global_infer_ms: float | None = None,
    export_ms: float | None = None,
    baseline_global_infer_ms: float | None = None,
    baseline_vis_ms: float | None = None,
    baseline_export_ms: float | None = None,
    enhanced_vis_ms: float | None = None,
    enhanced_export_ms: float | None = None,
    enhanced_global_infer_ms: float | None = None,
    x_starts: list[int] | None = None,
    y_starts: list[int] | None = None,
    warmup_ms: float | None = None,
    warmup_ran: bool | None = None,
) -> None:
    data = {
        "job_id": job_id,
        "device": device,
        "baseline_ms": baseline_ms,
        "enhanced_ms": enhanced_ms,
        "baseline_num_boxes": baseline_num_boxes,
        "enhanced_num_boxes": enhanced_num_boxes,
        "num_parameters_total": num_parameters_total,
        "notes": notes,
    }
    if model_load_ms is not None:
        data["model_load_ms"] = model_load_ms
    if num_tiles is not None:
        data["num_tiles"] = num_tiles
    if tile_infer_ms_total is not None:
        data["tile_infer_ms_total"] = tile_infer_ms_total
    if global_infer_ms is not None:
        data["global_infer_ms"] = global_infer_ms
    if export_ms is not None:
        data["export_ms"] = export_ms
    if baseline_global_infer_ms is not None:
        data["baseline_global_infer_ms"] = baseline_global_infer_ms
    if baseline_vis_ms is not None:
        data["baseline_vis_ms"] = baseline_vis_ms
    if baseline_export_ms is not None:
        data["baseline_export_ms"] = baseline_export_ms
    if enhanced_vis_ms is not None:
        data["enhanced_vis_ms"] = enhanced_vis_ms
    if enhanced_export_ms is not None:
        data["enhanced_export_ms"] = enhanced_export_ms
    if enhanced_global_infer_ms is not None:
        data["enhanced_global_infer_ms"] = enhanced_global_infer_ms
    if x_starts is not None:
        data["x_starts"] = x_starts
    if y_starts is not None:
        data["y_starts"] = y_starts
    if warmup_ms is not None:
        data["warmup_ms"] = warmup_ms
    if warmup_ran is not None:
        data["warmup_ran"] = warmup_ran
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def export_config(path: Path, config: JobConfig) -> None:
    cfg = {
        "conf_threshold": config.conf_threshold,
        "nms_iou": config.nms_iou,
        "tile_size": config.tile_size,
        "overlap": config.overlap,
        "wbf_iou": config.wbf_iou,
        "max_det": config.max_det,
        "img_max_side": config.img_max_side,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def export_image(path: Path, image_bgr: np.ndarray) -> None:
    imwrite_any(path, image_bgr)
