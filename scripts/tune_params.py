"""Auto parameter tuning with audit and reproducible outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import math
import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import JobConfig
from scripts.benchmark_custom import apply_profile, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto tune parameters for enhanced recall.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device string")
    parser.add_argument(
        "--profile_base",
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Base profile for initial config",
    )
    parser.add_argument("--trials", type=int, default=20, help="Max number of trials")
    parser.add_argument("--time_budget_sec", type=int, default=0, help="Stop after time budget (0 disables)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset_k", type=int, default=10, help="Subset size for coarse screening")
    parser.add_argument("--images", default=str(ROOT / "data" / "custom" / "images"))
    parser.add_argument("--labels", default=str(ROOT / "data" / "custom" / "labels"))
    parser.add_argument("--output_dir", default=str(ROOT / "reports" / "tuning"))
    parser.add_argument("--latency_ratio_max", type=float, required=True)
    parser.add_argument("--boxes_growth_max", type=float, required=True)
    parser.add_argument("--min_gain", type=float, required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from existing trials.jsonl")
    parser.add_argument(
        "--tune_space",
        default=str(ROOT / "tune_space.json"),
        help="Path to tune space json",
    )
    parser.add_argument("--reject_noisy_gain", action="store_true", help="Reject noisy gains")
    parser.add_argument("--noisy_boxes_ratio", type=float, default=1.5)
    parser.add_argument("--noisy_gain_ratio", type=float, default=0.5)
    return parser.parse_args()


def load_tune_space(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "parameters" not in data:
        raise ValueError("tune_space.json missing 'parameters'")
    return data


def _dataset_hash(paths: List[Path]) -> str:
    items = [f"{p.name}|{p.stat().st_size}" for p in paths]
    joined = "\n".join(items).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _write_list(path: Path, items: Iterable[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p) for p in items]
    path.write_text("\n".join(lines), encoding="utf-8")


def _config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _condition_ok(conditions: Dict[str, Any], profile: str, cfg: JobConfig) -> bool:
    if not conditions:
        return True
    if "profile" in conditions:
        allowed = conditions["profile"]
        if isinstance(allowed, str):
            allowed = [allowed]
        if profile not in allowed:
            return False
    if "param_equals" in conditions:
        for key, expected in conditions["param_equals"].items():
            value = getattr(cfg, key, None)
            if isinstance(expected, list):
                if value not in expected:
                    return False
            else:
                if value != expected:
                    return False
    return True


def sample_params(rng: random.Random, space: Dict[str, Any], profile: str, cfg: JobConfig) -> Dict[str, Any]:
    params = {}
    for name, spec in space["parameters"].items():
        if not _condition_ok(spec.get("conditions", {}), profile, cfg):
            continue
        ptype = spec.get("type")
        if ptype == "discrete":
            values = spec.get("values", [])
            if not values:
                continue
            params[name] = rng.choice(values)
        elif ptype == "continuous":
            min_v = float(spec.get("min"))
            max_v = float(spec.get("max"))
            sample = spec.get("sample", "uniform")
            if sample == "log_uniform":
                if min_v <= 0 or max_v <= 0:
                    raise ValueError(f"log_uniform requires positive min/max for {name}")
                val = 10 ** rng.uniform(
                    math.log10(min_v),
                    math.log10(max_v),
                )
            else:
                val = rng.uniform(min_v, max_v)
            params[name] = val
        else:
            continue
    return params


def apply_params(cfg: JobConfig, params: Dict[str, Any]) -> JobConfig:
    updates = {key: value for key, value in params.items() if hasattr(cfg, key)}
    cfg = dataclasses.replace(cfg, **updates)
    if hasattr(cfg, "global_conf_threshold"):
        cfg = dataclasses.replace(cfg, conf_threshold=cfg.global_conf_threshold)
    if hasattr(cfg, "wbf_iou_normal"):
        cfg = dataclasses.replace(cfg, wbf_iou=cfg.wbf_iou_normal)
    return cfg


def shrink_space(space: Dict[str, Any], best_cfg: Dict[str, Any], factor: float = 0.5) -> Dict[str, Any]:
    new_space = json.loads(json.dumps(space))
    for name, spec in new_space.get("parameters", {}).items():
        if spec.get("type") != "continuous":
            continue
        if name not in best_cfg:
            continue
        min_v = float(spec["min"])
        max_v = float(spec["max"])
        center = float(best_cfg[name])
        span = (max_v - min_v) * factor
        spec["min"] = max(min_v, center - span / 2)
        spec["max"] = min(max_v, center + span / 2)
    return new_space


def compute_score(gain: float, latency_ratio: float, boxes_growth: float, args: argparse.Namespace) -> float:
    score = gain
    if args.latency_ratio_max > 0:
        score -= max(0.0, latency_ratio - 1.0) / args.latency_ratio_max
    if args.boxes_growth_max > 0:
        score -= max(0.0, boxes_growth) / args.boxes_growth_max
    return score


def audit_trial(
    report: Dict[str, Any],
    trial_cfg: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str]]:
    reasons: list[str] = []
    if not isinstance(report, dict):
        return False, ["json_invalid"]

    if "config_snapshot" not in report or not isinstance(report["config_snapshot"], dict):
        reasons.append("json_invalid")
    else:
        if report["config_snapshot"] != trial_cfg:
            reasons.append("config_mismatch")

    overall = report.get("overall", {})
    runtime = report.get("runtime", {})
    boxes = report.get("boxes", {})

    if not isinstance(overall, dict) or not isinstance(runtime, dict) or not isinstance(boxes, dict):
        reasons.append("json_invalid")
        return False, reasons

    gain = float(overall.get("gain", 0.0))
    latency_ratio = float(runtime.get("latency_ratio", 0.0))
    boxes_growth = float(boxes.get("boxes_growth", 0.0))

    if gain < args.min_gain:
        reasons.append("no_gain")
    if latency_ratio > args.latency_ratio_max:
        reasons.append("too_slow")
    if boxes_growth > args.boxes_growth_max:
        reasons.append("box_explosion")
    if args.reject_noisy_gain:
        if boxes_growth > args.boxes_growth_max * args.noisy_boxes_ratio and gain < args.min_gain * args.noisy_gain_ratio:
            reasons.append("noisy_gain")

    return len(reasons) == 0, reasons


def run_stage(
    stage_name: str,
    rng: random.Random,
    space: Dict[str, Any],
    base_profile: str,
    image_paths: List[Path],
    label_dir: Path,
    args: argparse.Namespace,
    output_dir: Path,
    seen_hashes: set,
    trial_start: int,
    max_trials: int,
) -> Tuple[int, List[Dict[str, Any]]]:
    trials: list[dict] = []
    trial_id = trial_start
    for _ in range(max_trials):
        cfg = apply_profile(JobConfig(), base_profile)
        params = sample_params(rng, space, base_profile, cfg)
        cfg = apply_params(cfg, params)
        eff_cfg = cfg.effective_snapshot(device=args.device)
        cfg_hash = _config_hash(eff_cfg)
        if cfg_hash in seen_hashes:
            continue
        seen_hashes.add(cfg_hash)

        report = run_benchmark(image_paths, label_dir, cfg, args.device, args.seed)
        report["meta"]["profile"] = base_profile
        report_path = output_dir / "evals" / f"trial_{trial_id:04d}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        accepted, reasons = audit_trial(report, eff_cfg, args)
        gain = float(report["overall"]["gain"])
        latency_ratio = float(report["runtime"]["latency_ratio"])
        boxes_growth = float(report["boxes"]["boxes_growth"])
        score = compute_score(gain, latency_ratio, boxes_growth, args)

        trial_record = {
            "trial_id": trial_id,
            "stage": stage_name,
            "seed": args.seed,
            "profile_base": base_profile,
            "effective_config": eff_cfg,
            "metrics": report["overall"],
            "runtime": report["runtime"],
            "boxes": report["boxes"],
            "score": score,
            "accepted": accepted,
            "reject_reason": reasons,
            "config_hash": cfg_hash,
            "report_path": str(report_path),
        }
        trials.append(trial_record)
        trial_id += 1

        if args.time_budget_sec > 0 and time.time() - START_TIME > args.time_budget_sec:
            break
        if trial_id - trial_start >= max_trials:
            break
    return trial_id, trials


def write_trials(path: Path, trials: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")


def summarize_trials(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    reject_counts: Dict[str, int] = {}
    for t in trials:
        for r in t.get("reject_reason", []):
            reject_counts[r] = reject_counts.get(r, 0) + 1
    best = None
    for t in trials:
        if best is None or t["score"] > best["score"]:
            best = t
    return {"reject_reason_counts": reject_counts, "best_trial": best}


def load_existing_trials(path: Path) -> Tuple[List[Dict[str, Any]], set, set]:
    if not path.exists():
        return [], set(), set()
    existing: list[dict] = []
    seen: set = set()
    seen_full: set = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            existing.append(item)
            if "config_hash" in item:
                seen.add(item["config_hash"])
                if item.get("stage") in {"full_eval", "local_search"}:
                    seen_full.add(item["config_hash"])
    return existing, seen, seen_full


def select_subset(paths: List[Path], k: int, seed: int) -> List[Path]:
    if k <= 0 or k >= len(paths):
        return paths
    rng = random.Random(seed)
    return rng.sample(paths, k=k)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    space = load_tune_space(Path(args.tune_space))
    img_dir = Path(args.images)
    label_dir = Path(args.labels)
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not image_paths:
        raise RuntimeError(f"No images found in {img_dir}")

    subset_paths = select_subset(image_paths, args.subset_k, args.seed)
    _write_list(output_dir / "dataset_full.txt", image_paths)
    _write_list(output_dir / "dataset_subset.txt", subset_paths)

    trials_path = output_dir / "trials.jsonl"
    existing_trials, seen_hashes, seen_full_hashes = (
        load_existing_trials(trials_path) if args.resume else ([], set(), set())
    )
    trial_id = 0
    if existing_trials:
        trial_id = max(t["trial_id"] for t in existing_trials) + 1

    rng = random.Random(args.seed)
    all_trials: list[dict] = list(existing_trials)

    stage_a_trials = max(1, int(args.trials * 0.5))
    stage_b_trials = max(1, int(args.trials * 0.3))
    stage_c_trials = max(0, args.trials - stage_a_trials - stage_b_trials)

    trial_id, trials_a = run_stage(
        "coarse_subset",
        rng,
        space,
        args.profile_base,
        subset_paths,
        label_dir,
        args,
        output_dir,
        seen_hashes,
        trial_id,
        stage_a_trials,
    )
    write_trials(trials_path, trials_a)
    all_trials.extend(trials_a)

    ranked_a = sorted(trials_a, key=lambda t: t["score"], reverse=True)
    top_k = ranked_a[: min(5, len(ranked_a))]
    trials_b: list[dict] = []
    for t in top_k[:stage_b_trials]:
        cfg = JobConfig()
        cfg = apply_params(cfg, t["effective_config"])
        eff_cfg = cfg.effective_snapshot(device=args.device)
        cfg_hash = _config_hash(eff_cfg)
        if cfg_hash in seen_full_hashes:
            continue
        seen_full_hashes.add(cfg_hash)

        report = run_benchmark(image_paths, label_dir, cfg, args.device, args.seed)
        report["meta"]["profile"] = args.profile_base
        report_path = output_dir / "evals" / f"trial_{trial_id:04d}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        accepted, reasons = audit_trial(report, eff_cfg, args)
        gain = float(report["overall"]["gain"])
        latency_ratio = float(report["runtime"]["latency_ratio"])
        boxes_growth = float(report["boxes"]["boxes_growth"])
        score = compute_score(gain, latency_ratio, boxes_growth, args)

        trial_record = {
            "trial_id": trial_id,
            "stage": "full_eval",
            "seed": args.seed,
            "profile_base": args.profile_base,
            "effective_config": eff_cfg,
            "metrics": report["overall"],
            "runtime": report["runtime"],
            "boxes": report["boxes"],
            "score": score,
            "accepted": accepted,
            "reject_reason": reasons,
            "config_hash": cfg_hash,
            "report_path": str(report_path),
        }
        trials_b.append(trial_record)
        trial_id += 1

        if args.time_budget_sec > 0 and time.time() - START_TIME > args.time_budget_sec:
            break
    write_trials(trials_path, trials_b)
    all_trials.extend(trials_b)

    best_trial = None
    for t in all_trials:
        if best_trial is None or t["score"] > best_trial["score"]:
            best_trial = t

    trials_c: list[dict] = []
    if best_trial and stage_c_trials > 0:
        local_space = shrink_space(space, best_trial["effective_config"], factor=0.5)
        trial_id, trials_c = run_stage(
            "local_search",
            rng,
            local_space,
            args.profile_base,
            image_paths,
            label_dir,
            args,
            output_dir,
            seen_hashes,
            trial_id,
            stage_c_trials,
        )
        write_trials(trials_path, trials_c)
        all_trials.extend(trials_c)

    summary = summarize_trials(all_trials)
    best = summary["best_trial"]
    best_config = {}
    if best:
        best_config = {
            "effective_config": best["effective_config"],
            "meta": {
                "seed": args.seed,
                "device": args.device,
                "profile_base": args.profile_base,
                "dataset_hash_full": _dataset_hash(image_paths),
                "dataset_hash_subset": _dataset_hash(subset_paths),
                "image_count_full": len(image_paths),
                "image_count_subset": len(subset_paths),
            },
        }

    best_path = output_dir / "best_config.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best_config, f, ensure_ascii=False, indent=2)

    tune_summary = {
        "trial_count": len(all_trials),
        "accepted_count": sum(1 for t in all_trials if t.get("accepted")),
        "reject_reason_counts": summary["reject_reason_counts"],
        "best_score": best["score"] if best else None,
        "best_metrics": best["metrics"] if best else None,
        "best_runtime": best["runtime"] if best else None,
        "best_boxes": best["boxes"] if best else None,
        "profile_base": args.profile_base,
        "seed": args.seed,
        "device": args.device,
        "dataset_hash_full": _dataset_hash(image_paths),
        "dataset_hash_subset": _dataset_hash(subset_paths),
    }
    summary_path = output_dir / "tune_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(tune_summary, f, ensure_ascii=False, indent=2)

    print(f"Trials: {len(all_trials)} (accepted {tune_summary['accepted_count']})")
    if best:
        print(
            f"Best gain={best['metrics']['gain']:.4f} "
            f"latency_ratio={best['runtime']['latency_ratio']:.2f} "
            f"boxes_growth={best['boxes']['boxes_growth']:.2f}"
        )
        print(f"Best config written to {best_path}")


START_TIME = time.time()


if __name__ == "__main__":
    main()
