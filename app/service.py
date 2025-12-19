"""Service orchestrating baseline and enhanced pipelines."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from core.config import JobConfig
from core.detector_yolo import Detection, YoloDetector
from core.postprocess import nms
from core.export import (
    export_config,
    export_detections_json,
    export_image,
    export_summary_json,
)
from core.tiling import box_center_in_region, tile_image_padded
from core.traffic_light_state import classify_traffic_light_state
from core.utils import ensure_dir, new_job_id, setup_logger
from core.visualize import draw_detections
from core.wbf import weighted_boxes_fusion


class Service:
    """Pipeline service for processing images."""

    def __init__(self, output_root: Path | str = "outputs") -> None:
        self.output_root = Path(output_root)
        self.detector = YoloDetector()
        self._loaded = False

    def _ensure_detector(self, device: str) -> None:
        if not self._loaded:
            self.detector.load(device=device)
            self._loaded = True

    def _resolve_device(self, requested: str) -> tuple[str, str | None]:
        """Return (actual_device, fallback_reason) based on availability."""
        req = (requested or "cpu").lower()
        if req != "cuda":
            return req, None
        try:
            import torch
        except Exception as exc:  # noqa: BLE001
            return "cpu", f"torch import failed: {exc}"
        if not torch.cuda.is_available():
            return "cpu", "cuda not available"
        return "cuda", None

    def run_job(
        self,
        image: Any,
        config: JobConfig | None = None,
        device: str = "cpu",
    ) -> dict:
        """Run baseline and enhanced pipelines, write outputs, and return paths."""
        cfg = config or JobConfig()
        requested_device = device
        actual_device, fallback_reason = self._resolve_device(requested_device)
        load_ms = 0.0
        if not self._loaded:
            t_load = time.perf_counter()
            self._ensure_detector(actual_device)
            load_ms = (time.perf_counter() - t_load) * 1000
        else:
            self._ensure_detector(actual_device)
        job_id = new_job_id()
        job_dir = ensure_dir(self.output_root / job_id)
        logger = setup_logger(job_dir / "run.log")
        logger.info(
            "Job start %s on requested_device=%s actual_device=%s",
            job_id,
            requested_device,
            actual_device,
        )

        image_bgr = self._to_bgr(image)
        h, w = image_bgr.shape[:2]

        warmup_ms = 0.0
        warmup_ran = False
        if not self.detector.warmed_up:
            t_warm = time.perf_counter()
            _ = self.detector.detect(
                image_bgr=image_bgr,
                conf_threshold=cfg.conf_threshold,
                nms_iou=cfg.nms_iou,
                max_det=cfg.max_det,
            )
            warmup_ms = (time.perf_counter() - t_warm) * 1000
            self.detector.warmed_up = True
            warmup_ran = True

        # Baseline
        t0 = time.perf_counter()
        t_base_infer = time.perf_counter()
        dets_baseline = self.detector.detect(
            image_bgr=image_bgr,
            conf_threshold=cfg.global_conf_threshold,
            nms_iou=cfg.nms_iou,
            max_det=cfg.max_det,
        )
        baseline_global_infer_ms = (time.perf_counter() - t_base_infer) * 1000
        dets_baseline = self._attach_traffic_light_states(image_bgr, dets_baseline)
        t_base_vis = time.perf_counter()
        baseline_img = draw_detections(image_bgr, dets_baseline)
        baseline_vis_ms = (time.perf_counter() - t_base_vis) * 1000
        t_base_export = time.perf_counter()
        export_image(job_dir / "baseline_boxed.png", baseline_img)
        num_params = self.detector.num_parameters()
        export_detections_json(
            job_dir / "baseline.json",
            job_id=job_id,
            pipeline="baseline",
            model_name="yolov8n",
            num_parameters=num_params,
            image_shape=(h, w),
            config=cfg,
            detections=dets_baseline,
        )
        baseline_export_ms = (time.perf_counter() - t_base_export) * 1000
        baseline_ms = (time.perf_counter() - t0) * 1000

        # Enhanced
        t1 = time.perf_counter()
        t_global = time.perf_counter()
        dets_global = self.detector.detect(
            image_bgr=image_bgr,
            conf_threshold=cfg.global_conf_threshold,
            nms_iou=cfg.nms_iou,
            max_det=cfg.max_det,
        )
        global_infer_ms = (time.perf_counter() - t_global) * 1000
        dets_tiles: list[Detection] = []
        tiles, x_starts, y_starts = tile_image_padded(
            image_bgr,
            tile_size=cfg.tile_size,
            overlap=cfg.overlap,
            pad_px=cfg.tile_pad_px,
        )
        tile_infer_ms_total = 0.0
        for tile in tiles:
            crop_bgr = tile.crop_bgr
            if cfg.tile_scale and cfg.tile_scale != 1.0:
                crop_bgr = cv2.resize(
                    crop_bgr,
                    None,
                    fx=cfg.tile_scale,
                    fy=cfg.tile_scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            t_tile = time.perf_counter()
            tile_dets = self.detector.detect(
                image_bgr=crop_bgr,
                conf_threshold=cfg.tile_conf_threshold,
                nms_iou=cfg.nms_iou,
                max_det=cfg.max_det,
            )
            tile_infer_ms_total += (time.perf_counter() - t_tile) * 1000

            for det in tile_dets:
                x1, y1, x2, y2 = det.box_xyxy
                if cfg.tile_scale and cfg.tile_scale != 1.0:
                    x1 = int(round(x1 / cfg.tile_scale))
                    y1 = int(round(y1 / cfg.tile_scale))
                    x2 = int(round(x2 / cfg.tile_scale))
                    y2 = int(round(y2 / cfg.tile_scale))
                gx1 = x1 + tile.crop_x0
                gy1 = y1 + tile.crop_y0
                gx2 = x2 + tile.crop_x0
                gy2 = y2 + tile.crop_y0

                gx1 = max(0, min(gx1, w - 1))
                gy1 = max(0, min(gy1, h - 1))
                gx2 = max(0, min(gx2, w - 1))
                gy2 = max(0, min(gy2, h - 1))
                if gx2 <= gx1 or gy2 <= gy1:
                    continue

                if not box_center_in_region(
                    (gx1, gy1, gx2, gy2),
                    tile.eff_x0,
                    tile.eff_y0,
                    tile.eff_x1,
                    tile.eff_y1,
                ):
                    continue

                dets_tiles.append(
                    Detection(
                        label=det.label,
                        class_id=det.class_id,
                        score=det.score,
                        box_xyxy=(int(gx1), int(gy1), int(gx2), int(gy2)),
                        attrs={},
                    )
                )
        dets_all = list(dets_global) + dets_tiles
        dets_enhanced = weighted_boxes_fusion(
            dets_all,
            iou_thresh=cfg.wbf_iou_normal,
            small_area_thresh=cfg.small_area_thresh,
            iou_small=cfg.wbf_iou_small,
            iou_normal=cfg.wbf_iou_normal,
        )
        dets_enhanced = nms(dets_enhanced, iou_thresh=cfg.nms_iou)
        dets_enhanced = self._attach_traffic_light_states(image_bgr, dets_enhanced)
        t_enh_vis = time.perf_counter()
        enhanced_img = draw_detections(image_bgr, dets_enhanced)
        enhanced_vis_ms = (time.perf_counter() - t_enh_vis) * 1000
        t_enh_export = time.perf_counter()
        export_image(job_dir / "enhanced_boxed.png", enhanced_img)
        export_detections_json(
            job_dir / "enhanced.json",
            job_id=job_id,
            pipeline="enhanced",
            model_name="yolov8n",
            num_parameters=num_params,
            image_shape=(h, w),
            config=cfg,
            detections=dets_enhanced,
        )
        enhanced_export_ms = (time.perf_counter() - t_enh_export) * 1000
        enhanced_ms = (time.perf_counter() - t1) * 1000

        # Shared exports
        t_export = time.perf_counter()
        export_config(job_dir / "config.json", cfg, device=actual_device)
        export_summary_json(
            job_dir / "summary.json",
            job_id=job_id,
            device=actual_device,
            requested_device=requested_device,
            actual_device=actual_device,
            device_fallback_reason=fallback_reason,
            baseline_ms=baseline_ms,
            enhanced_ms=enhanced_ms,
            baseline_num_boxes=len(dets_baseline),
            enhanced_num_boxes=len(dets_enhanced),
            num_parameters_total=num_params,
            notes={
                "tiling": len(tiles) > 1,
                "wbf": True,
                "global_fallback": True,
            },
            model_load_ms=load_ms,
            num_tiles=len(tiles),
            tile_infer_ms_total=tile_infer_ms_total,
            global_infer_ms=global_infer_ms,
            export_ms=(time.perf_counter() - t_export) * 1000,
            baseline_global_infer_ms=baseline_global_infer_ms,
            baseline_vis_ms=baseline_vis_ms,
            baseline_export_ms=baseline_export_ms,
            enhanced_vis_ms=enhanced_vis_ms,
            enhanced_export_ms=enhanced_export_ms,
            enhanced_global_infer_ms=global_infer_ms,
            x_starts=x_starts,
            y_starts=y_starts,
            warmup_ms=warmup_ms,
            warmup_ran=warmup_ran,
            tile_pad_px=cfg.tile_pad_px,
            tile_scale=cfg.tile_scale,
            global_conf_threshold=cfg.global_conf_threshold,
            tile_conf_threshold=cfg.tile_conf_threshold,
        )

        logger.info(
            "Job %s done. Baseline %.1f ms (%d boxes), Enhanced %.1f ms (%d boxes)",
            job_id,
            baseline_ms,
            len(dets_baseline),
            enhanced_ms,
            len(dets_enhanced),
        )
        logger.debug(
            "Job %s tiling detail: num_tiles=%d x_starts=%s y_starts=%s",
            job_id,
            len(tiles),
            x_starts,
            y_starts,
        )

        return {
            "job_id": job_id,
            "output_dir": str(job_dir),
            "baseline_image": str(job_dir / "baseline_boxed.png"),
            "enhanced_image": str(job_dir / "enhanced_boxed.png"),
            "baseline_json": str(job_dir / "baseline.json"),
            "enhanced_json": str(job_dir / "enhanced.json"),
            "summary_json": str(job_dir / "summary.json"),
            "config_json": str(job_dir / "config.json"),
            "baseline_ms": baseline_ms,
            "enhanced_ms": enhanced_ms,
            "requested_device": requested_device,
            "actual_device": actual_device,
            "device_fallback_reason": fallback_reason,
        }

    def _to_bgr(self, image: Any) -> np.ndarray:
        """Convert input (PIL/np) to BGR uint8."""
        if image is None:
            raise ValueError("Input image is None")
        if isinstance(image, np.ndarray):
            arr = image
        else:
            try:
                from PIL import Image
            except ImportError as exc:
                raise ValueError("Unsupported image type and PIL not available") from exc
            if isinstance(image, Image.Image):
                arr = np.array(image)
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {arr.shape}")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    def _attach_traffic_light_states(
        self, image_bgr: np.ndarray, detections: Iterable[Detection]
    ) -> list[Detection]:
        result: list[Detection] = []
        for det in detections:
            attrs = dict(det.attrs)
            if det.label == "traffic light":
                attrs["state"] = classify_traffic_light_state(image_bgr, det.box_xyxy)
            result.append(
                Detection(
                    label=det.label,
                    class_id=det.class_id,
                    score=det.score,
                    box_xyxy=det.box_xyxy,
                    attrs=attrs,
                )
            )
        return result
