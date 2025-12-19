"""Batch baseline vs enhanced comparison with optional GT evaluation.

Inputs:
  --images: image folder
  --labels: optional GT label folder (YOLO txt)
Outputs:
  outputs/batch_<timestamp>/
    per-image folders with baseline/enhanced/diff/json/summary/config
    batch_report.json / batch_report.csv
    batch_scatter.png (summary visualization)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import Service
from core.config import JobConfig
from core.eval import accumulate_recall_small, load_yolo_labels, map_predictions
from core.postprocess import iou_xyxy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compare baseline vs enhanced.")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--labels", default="", help="Optional labels directory (YOLO txt)")
    parser.add_argument("--device", default="cpu", help='Device string, default "cpu"')
    parser.add_argument(
        "--profile",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Config profile: fast|balanced|quality",
    )
    parser.add_argument(
        "--config_json",
        default="",
        help="Optional config json containing effective_config (overrides profile)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output directory (default outputs/batch_<timestamp>)",
    )
    return parser.parse_args()


def _hash_dataset(image_paths: list[Path]) -> str:
    items = [f"{p.name}|{p.stat().st_size}" for p in image_paths]
    joined = "\n".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_config_json(path: Path) -> JobConfig:
    data = _load_json(path)
    cfg_data = data.get("effective_config", data)
    cfg = JobConfig()
    updates = {k: v for k, v in cfg_data.items() if hasattr(cfg, k)}
    cfg = dataclasses.replace(cfg, **updates)
    if hasattr(cfg, "global_conf_threshold"):
        cfg = dataclasses.replace(cfg, conf_threshold=cfg.global_conf_threshold)
    if hasattr(cfg, "wbf_iou_normal"):
        cfg = dataclasses.replace(cfg, wbf_iou=cfg.wbf_iou_normal)
    return cfg


def _apply_profile(cfg: JobConfig, profile: str) -> JobConfig:
    p = profile.lower().strip()
    if p == "fast":
        return dataclasses.replace(
            cfg,
            global_conf_threshold=0.25,
            tile_conf_threshold=0.20,
            tile_size=640,
            overlap=0.18,
            tile_pad_px=32,
            tile_scale=1.0,
            wbf_iou_normal=0.62,
            conf_threshold=0.25,
            wbf_iou=0.62,
        )
    if p == "quality":
        return dataclasses.replace(
            cfg,
            global_conf_threshold=0.25,
            tile_conf_threshold=0.16,
            tile_size=512,
            overlap=0.20,
            tile_pad_px=64,
            tile_scale=1.5,
            wbf_iou_normal=0.58,
            conf_threshold=0.25,
            wbf_iou=0.58,
        )
    return dataclasses.replace(
        cfg,
        global_conf_threshold=0.25,
        tile_conf_threshold=0.18,
        tile_size=512,
        overlap=0.20,
        tile_pad_px=48,
        tile_scale=1.0,
        wbf_iou_normal=0.58,
        conf_threshold=0.25,
        wbf_iou=0.58,
    )


def _match_counts(
    baseline_dets: list[dict],
    enhanced_dets: list[dict],
    iou_thresh: float = 0.5,
) -> tuple[int, int, int]:
    baseline_by_label: dict[str, list[dict]] = {}
    for d in baseline_dets:
        baseline_by_label.setdefault(str(d.get("label", "")), []).append(d)
    enhanced_by_label: dict[str, list[dict]] = {}
    for d in enhanced_dets:
        enhanced_by_label.setdefault(str(d.get("label", "")), []).append(d)

    added = missing = matched = 0
    for label in sorted(set(baseline_by_label) | set(enhanced_by_label)):
        b_list = baseline_by_label.get(label, [])
        e_list = enhanced_by_label.get(label, [])
        b_used = [False] * len(b_list)
        for e in e_list:
            e_box = e.get("box_xyxy")
            best_iou = 0.0
            best_idx = -1
            if isinstance(e_box, list) and len(e_box) == 4:
                for idx, b in enumerate(b_list):
                    b_box = b.get("box_xyxy")
                    if not (isinstance(b_box, list) and len(b_box) == 4):
                        continue
                    iou = iou_xyxy(tuple(map(float, e_box)), tuple(map(float, b_box)))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
            if best_iou >= iou_thresh and best_idx >= 0 and not b_used[best_idx]:
                b_used[best_idx] = True
                matched += 1
            else:
                added += 1
        missing += sum(1 for used in b_used if not used)
    return added, missing, matched


def _draw_scatter(
    points: list[tuple[float, float]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    w, h = 800, 600
    margin = 60
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    if not points:
        draw.text((10, 10), "No points to plot", fill=(0, 0, 0))
        img.save(out_path)
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    def _map_x(x: float) -> float:
        return margin + (x - min_x) / (max_x - min_x) * (w - 2 * margin)

    def _map_y(y: float) -> float:
        return h - margin - (y - min_y) / (max_y - min_y) * (h - 2 * margin)

    draw.line([(margin, margin), (margin, h - margin)], fill=(0, 0, 0), width=2)
    draw.line([(margin, h - margin), (w - margin, h - margin)], fill=(0, 0, 0), width=2)
    draw.text((10, 10), title, fill=(0, 0, 0))
    draw.text((w // 2 - 40, h - 40), xlabel, fill=(0, 0, 0))
    draw.text((10, h // 2 - 10), ylabel, fill=(0, 0, 0))

    for x, y in points:
        px, py = _map_x(x), _map_y(y)
        r = 3
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 120, 255))

    img.save(out_path)


def run_batch(
    *,
    images: Path,
    labels: Path | None,
    device: str,
    profile: str,
    config_json: Path | None,
    config: JobConfig | None,
    output: Path | None,
) -> tuple[Path, dict]:
    img_dir = Path(images)
    label_dir = Path(labels) if labels else None

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    if output:
        batch_dir = Path(output)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        batch_dir = Path("outputs") / f"batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    dataset_hash = _hash_dataset(image_paths)
    dataset_list_path = batch_dir / "dataset_list.txt"
    dataset_list_path.write_text("\n".join(str(p) for p in image_paths), encoding="utf-8")

    cfg = JobConfig()
    if config is not None:
        cfg = config
    elif config_json:
        cfg = _load_config_json(Path(config_json))
    else:
        cfg = _apply_profile(cfg, profile)

    svc = Service()
    rows: list[dict[str, Any]] = []
    scatter_points: list[tuple[float, float]] = []
    fallback_reasons: dict[str, int] = {}
    actual_devices: list[str] = []

    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        result = svc.run_job(image=image, config=cfg, device=device)
        summary = _load_json(Path(result["summary_json"]))
        actual_device = summary.get("actual_device") or summary.get("device") or device
        actual_devices.append(str(actual_device))
        fallback_reason = summary.get("device_fallback_reason")
        if fallback_reason:
            fallback_reasons[fallback_reason] = fallback_reasons.get(fallback_reason, 0) + 1

        job_dir = Path(result["output_dir"])
        per_dir = batch_dir / f"{img_path.stem}_{result['job_id']}"
        per_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "baseline_boxed.png",
            "enhanced_boxed.png",
            "baseline.json",
            "enhanced.json",
            "summary.json",
            "config.json",
        ]:
            src = job_dir / name
            if src.exists():
                (per_dir / name).write_bytes(src.read_bytes())

        baseline_data = _load_json(Path(result["baseline_json"]))
        enhanced_data = _load_json(Path(result["enhanced_json"]))
        baseline_dets = baseline_data.get("detections", [])
        enhanced_dets = enhanced_data.get("detections", [])

        added, missing, matched = _match_counts(baseline_dets, enhanced_dets, iou_thresh=0.5)

        baseline_boxes = len(baseline_dets)
        enhanced_boxes = len(enhanced_dets)
        boxes_growth = enhanced_boxes - baseline_boxes
        baseline_ms = float(result["baseline_ms"])
        enhanced_ms = float(result["enhanced_ms"])
        latency_ratio = (enhanced_ms / baseline_ms) if baseline_ms > 0 else 0.0

        baseline_recall = None
        enhanced_recall = None
        gain = None
        has_gt = False

        if label_dir is not None:
            label_path = label_dir / f"{img_path.stem}.txt"
            w, h = image.size
            gts = load_yolo_labels(label_path, w, h)
            preds_baseline = map_predictions(baseline_dets)
            preds_enhanced = map_predictions(enhanced_dets)
            tp_b, gt_small = accumulate_recall_small(preds_baseline, gts, iou_thresh=0.5)
            tp_e, _ = accumulate_recall_small(preds_enhanced, gts, iou_thresh=0.5)
            gt_small_total = sum(gt_small)
            if gt_small_total > 0:
                baseline_recall = sum(tp_b) / gt_small_total
                enhanced_recall = sum(tp_e) / gt_small_total
                gain = enhanced_recall - baseline_recall
                has_gt = True

        rows.append(
            {
                "image": img_path.name,
                "job_id": result["job_id"],
                "output_dir": str(per_dir),
                "requested_device": device,
                "actual_device": actual_device,
                "device_fallback_reason": fallback_reason,
                "baseline_ms": baseline_ms,
                "enhanced_ms": enhanced_ms,
                "latency_ratio": latency_ratio,
                "baseline_boxes": baseline_boxes,
                "enhanced_boxes": enhanced_boxes,
                "boxes_growth": boxes_growth,
                "added": added,
                "missing": missing,
                "matched": matched,
                "baseline_recall_small": baseline_recall,
                "enhanced_recall_small": enhanced_recall,
                "gain": gain,
                "has_gt": has_gt,
            }
        )

        if has_gt:
            scatter_points.append((latency_ratio, gain if gain is not None else 0.0))
        else:
            scatter_points.append((latency_ratio, boxes_growth))

    def _avg(values: Iterable[float]) -> float:
        vals = list(values)
        return sum(vals) / len(vals) if vals else 0.0

    baseline_ms_avg = _avg([r["baseline_ms"] for r in rows])
    enhanced_ms_avg = _avg([r["enhanced_ms"] for r in rows])
    latency_ratio_avg = (enhanced_ms_avg / baseline_ms_avg) if baseline_ms_avg > 0 else 0.0
    baseline_boxes_avg = _avg([r["baseline_boxes"] for r in rows])
    enhanced_boxes_avg = _avg([r["enhanced_boxes"] for r in rows])
    boxes_growth_avg = enhanced_boxes_avg - baseline_boxes_avg

    gains = [r["gain"] for r in rows if r["gain"] is not None]
    baseline_recalls = [r["baseline_recall_small"] for r in rows if r["baseline_recall_small"] is not None]
    enhanced_recalls = [r["enhanced_recall_small"] for r in rows if r["enhanced_recall_small"] is not None]
    gain_avg = _avg([g for g in gains if g is not None]) if gains else None

    report = {
        "meta": {
            "image_count": len(rows),
            "requested_device": device,
            "actual_device": actual_devices[0] if actual_devices else device,
            "fallback_count": sum(1 for r in rows if r.get("device_fallback_reason")),
            "fallback_reasons": fallback_reasons,
            "profile": "" if config_json else profile,
            "dataset_hash": dataset_hash,
            "dataset_list": str(dataset_list_path),
            "has_gt": bool(label_dir),
        },
        "config_snapshot": cfg.effective_snapshot(device=device),
        "overall": {
            "baseline_ms_avg": baseline_ms_avg,
            "enhanced_ms_avg": enhanced_ms_avg,
            "latency_ratio": latency_ratio_avg,
            "baseline_boxes_avg": baseline_boxes_avg,
            "enhanced_boxes_avg": enhanced_boxes_avg,
            "boxes_growth_avg": boxes_growth_avg,
            "baseline_recall_small@0.5": _avg(baseline_recalls) if baseline_recalls else None,
            "enhanced_recall_small@0.5": _avg(enhanced_recalls) if enhanced_recalls else None,
            "gain": gain_avg,
            "no_gt_notice": "no GT provided; recall fields are None" if not label_dir else "",
        },
        "rows": rows,
    }

    report_path = batch_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = batch_dir / "batch_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image",
                "job_id",
                "baseline_ms",
                "enhanced_ms",
                "latency_ratio",
                "baseline_boxes",
                "enhanced_boxes",
                "boxes_growth",
                "added",
                "missing",
                "matched",
                "baseline_recall_small",
                "enhanced_recall_small",
                "gain",
                "has_gt",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r["image"],
                    r["job_id"],
                    r["baseline_ms"],
                    r["enhanced_ms"],
                    r["latency_ratio"],
                    r["baseline_boxes"],
                    r["enhanced_boxes"],
                    r["boxes_growth"],
                    r["added"],
                    r["missing"],
                    r["matched"],
                    r["baseline_recall_small"],
                    r["enhanced_recall_small"],
                    r["gain"],
                    r["has_gt"],
                ]
            )

    scatter_path = batch_dir / "batch_scatter.png"
    if label_dir:
        _draw_scatter(scatter_points, "latency_ratio", "gain", "gain vs latency_ratio", scatter_path)
    else:
        _draw_scatter(scatter_points, "latency_ratio", "boxes_growth", "boxes_growth vs latency_ratio", scatter_path)

    return batch_dir, report


def main() -> None:
    args = parse_args()
    batch_dir, _report = run_batch(
        images=Path(args.images),
        labels=Path(args.labels) if args.labels else None,
        device=args.device,
        profile=args.profile,
        config_json=Path(args.config_json) if args.config_json else None,
        config=None,
        output=Path(args.output) if args.output else None,
    )
    print(f"Batch report written to {batch_dir / 'batch_report.json'}")
    print(f"CSV report written to {batch_dir / 'batch_report.csv'}")
    print(f"Scatter plot written to {batch_dir / 'batch_scatter.png'}")


if __name__ == "__main__":
    main()
