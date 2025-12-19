"""Benchmark custom dataset recall_small@0.5 for baseline vs enhanced."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import dataclasses
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
)
from core.io_image import imread_any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom dataset benchmark for small object recall@0.5")
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
        help="Path to config json containing effective_config (overrides profile)",
    )
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
    parser.add_argument(
        "--subset_k",
        type=int,
        default=0,
        help="Evaluate only K images (0 means full set)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subset sampling",
    )
    return parser.parse_args()


def apply_profile(cfg: JobConfig, profile: str) -> JobConfig:
    """Apply fixed profile parameters (must match README table)."""
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
    # balanced (default)
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


def _hash_dataset(image_paths: list[Path]) -> str:
    items = [f"{p.name}|{p.stat().st_size}" for p in image_paths]
    joined = "\n".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def load_config_json(path: Path) -> JobConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "effective_config" in data:
        cfg_data = data["effective_config"]
    else:
        cfg_data = data
    cfg = JobConfig()
    updates = {key: value for key, value in cfg_data.items() if hasattr(cfg, key)}
    cfg = dataclasses.replace(cfg, **updates)
    if hasattr(cfg, "global_conf_threshold"):
        cfg = dataclasses.replace(cfg, conf_threshold=cfg.global_conf_threshold)
    if hasattr(cfg, "wbf_iou_normal"):
        cfg = dataclasses.replace(cfg, wbf_iou=cfg.wbf_iou_normal)
    return cfg


def run_benchmark(
    image_paths: list[Path],
    label_dir: Path,
    cfg: JobConfig,
    device: str,
    seed: int,
) -> dict:
    svc = Service()

    tp_baseline = [0] * len(GT_CLASSES)
    tp_enhanced = [0] * len(GT_CLASSES)
    gt_small_total = [0] * len(GT_CLASSES)
    baseline_ms_list: list[float] = []
    enhanced_ms_list: list[float] = []
    baseline_boxes_list: list[int] = []
    enhanced_boxes_list: list[int] = []

    for img_path in image_paths:
        label_path = label_dir / (img_path.stem + ".txt")
        image = Image.open(img_path).convert("RGB")
        image_np = imread_any(img_path)
        h, w = image_np.shape[:2]

        result = svc.run_job(image=image, config=cfg, device=device)
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
        baseline_boxes_list.append(len(baseline_data.get("detections", [])))
        enhanced_boxes_list.append(len(enhanced_data.get("detections", [])))

    def safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    overall_gt_small = sum(gt_small_total)
    overall_baseline_tp = sum(tp_baseline)
    overall_enhanced_tp = sum(tp_enhanced)

    overall = {
        "baseline_recall_small@0.5": safe_div(overall_baseline_tp, overall_gt_small),
        "enhanced_recall_small@0.5": safe_div(overall_enhanced_tp, overall_gt_small),
        "gain": safe_div(overall_enhanced_tp, overall_gt_small) - safe_div(overall_baseline_tp, overall_gt_small),
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
    runtime["latency_ratio"] = safe_div(runtime["enhanced_ms_avg"], runtime["baseline_ms_avg"])

    boxes = {
        "baseline_boxes_avg": safe_div(sum(baseline_boxes_list), len(baseline_boxes_list)),
        "enhanced_boxes_avg": safe_div(sum(enhanced_boxes_list), len(enhanced_boxes_list)),
    }
    boxes["boxes_growth"] = boxes["enhanced_boxes_avg"] - boxes["baseline_boxes_avg"]

    config_snapshot = cfg.effective_snapshot(device=device)
    meta = {
        "image_count": len(image_paths),
        "device": device,
        "seed": seed,
        "profile": "",
        "git_commit": "",
        "dataset_hash": _hash_dataset(image_paths),
    }

    return {
        "overall": overall,
        "per_class": per_class,
        "runtime": runtime,
        "boxes": boxes,
        "config_snapshot": config_snapshot,
        "meta": meta,
    }


def main() -> None:
    args = parse_args()
    img_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_path = Path(args.output)

    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    if args.config_json:
        cfg = load_config_json(Path(args.config_json))
        profile = ""
    else:
        cfg = apply_profile(JobConfig(), args.profile)
        profile = args.profile

    if args.subset_k and args.subset_k > 0:
        rng = random.Random(args.seed)
        image_paths = rng.sample(image_paths, k=min(args.subset_k, len(image_paths)))

    report = run_benchmark(image_paths, label_dir, cfg, args.device, args.seed)
    report["meta"]["profile"] = profile

    try:
        import subprocess

        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["meta"]["git_commit"] = commit
    except Exception:
        report["meta"]["git_commit"] = ""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Eval done on {len(image_paths)} images.")
    print(
        f"baseline_recall_small@0.5={report['overall']['baseline_recall_small@0.5']:.4f} "
        f"enhanced_recall_small@0.5={report['overall']['enhanced_recall_small@0.5']:.4f} "
        f"gain={report['overall']['gain']:.4f}"
    )
    print(
        f"baseline_boxes_avg={report['boxes']['baseline_boxes_avg']:.2f} "
        f"enhanced_boxes_avg={report['boxes']['enhanced_boxes_avg']:.2f} "
        f"boxes_growth={report['boxes']['boxes_growth']:.2f}"
    )


if __name__ == "__main__":
    main()
