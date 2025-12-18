"""Benchmark custom dataset recall_small@0.5 for baseline vs enhanced."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.service import Service
from core.config import JobConfig
from core.eval import (
    GT_CLASSES,
    accumulate_recall_small,
    load_yolo_labels,
    map_predictions,
    save_eval_report,
)
from core.io_image import imread_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom dataset benchmark for small object recall@0.5")
    parser.add_argument("--device", default="cpu", help='Device string, default "cpu"')
    parser.add_argument(
        "--images",
        default=str(ROOT / "data" / "custom" / "images"),
        help="Directory of images",
    )
    parser.add_argument(
        "--labels",
        default=str(ROOT / "data" / "custom" / "labels"),
        help="Directory of YOLO txt labels",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "custom_eval.json"),
        help="Path to evaluation output json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_path = Path(args.output)

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    svc = Service()
    cfg = JobConfig()

    tp_baseline = [0] * len(GT_CLASSES)
    tp_enhanced = [0] * len(GT_CLASSES)
    gt_small_total = [0] * len(GT_CLASSES)
    baseline_ms_list: list[float] = []
    enhanced_ms_list: list[float] = []

    for img_path in image_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        image = Image.open(img_path).convert("RGB")
        image_np = imread_any(img_path)
        h, w = image_np.shape[:2]

        result = svc.run_job(image=image, config=cfg, device=args.device)
        with Path(result["baseline_json"]).open("r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        with Path(result["enhanced_json"]).open("r", encoding="utf-8") as f:
            enhanced_data = json.load(f)

        preds_baseline = map_predictions(baseline_data.get("detections", []))
        preds_enhanced = map_predictions(enhanced_data.get("detections", []))
        gts = load_yolo_labels(label_path, w, h)

        tp_b, gt_small = accumulate_recall_small(preds_baseline, gts, iou_thresh=0.5)
        tp_e, _gt_unused = accumulate_recall_small(preds_enhanced, gts, iou_thresh=0.5)

        tp_baseline = [a + b for a, b in zip(tp_baseline, tp_b)]
        tp_enhanced = [a + b for a, b in zip(tp_enhanced, tp_e)]
        gt_small_total = [a + b for a, b in zip(gt_small_total, gt_small)]

        baseline_ms_list.append(float(result["baseline_ms"]))
        enhanced_ms_list.append(float(result["enhanced_ms"]))

    def safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    overall_gt_small = sum(gt_small_total)
    overall_baseline_tp = sum(tp_baseline)
    overall_enhanced_tp = sum(tp_enhanced)

    overall = {
        "baseline_recall_small@0.5": safe_div(overall_baseline_tp, overall_gt_small),
        "enhanced_recall_small@0.5": safe_div(overall_enhanced_tp, overall_gt_small),
        "delta_small_recall": safe_div(overall_enhanced_tp, overall_gt_small)
        - safe_div(overall_baseline_tp, overall_gt_small),
    }

    per_class = {}
    for idx, name in enumerate(GT_CLASSES):
        per_class[name] = {
            "baseline_recall_small@0.5": safe_div(tp_baseline[idx], gt_small_total[idx]),
            "enhanced_recall_small@0.5": safe_div(tp_enhanced[idx], gt_small_total[idx]),
            "gt_small_count": gt_small_total[idx],
        }

    runtime = {
        "baseline_ms_avg": safe_div(sum(baseline_ms_list), len(baseline_ms_list)),
        "enhanced_ms_avg": safe_div(sum(enhanced_ms_list), len(enhanced_ms_list)),
    }

    config_snapshot = {
        "conf_threshold": cfg.conf_threshold,
        "nms_iou": cfg.nms_iou,
        "tile_size": cfg.tile_size,
        "overlap": cfg.overlap,
        "wbf_iou": cfg.wbf_iou,
    }

    meta = {"num_images": len(image_paths)}

    save_eval_report(out_path, overall, per_class, runtime, config_snapshot, meta)

    print(f"Eval done on {len(image_paths)} images.")
    print(f"baseline_recall_small@0.5={overall['baseline_recall_small@0.5']:.4f} "
          f"enhanced_recall_small@0.5={overall['enhanced_recall_small@0.5']:.4f} "
          f"delta={overall['delta_small_recall']:.4f}")


if __name__ == "__main__":
    main()
