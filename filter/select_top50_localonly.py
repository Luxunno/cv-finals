from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


TARGET_SYNONYMS = {
    "person": ["person"],
    "car": ["car"],
    "bicycle": ["bicycle", "bike"],
    "motorcycle": ["motorcycle", "motorbike"],
    "bus": ["bus"],
    "traffic light": ["traffic light", "traffic_light"],
}
TARGET_ORDER = ["person", "car", "bicycle", "motorcycle", "bus", "traffic light"]
TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_ORDER)}


def default_paths() -> Tuple[Path, List[Path], Path]:
    base = Path(__file__).resolve().parent
    json_path = base / "patches" / "zhiyuan_objv2_val.json"
    patch_dirs = [base / "patches" / f"patch{i}" for i in range(4)]
    output_root = base.parent / "data" / "custom"
    return json_path, patch_dirs, output_root


def parse_args() -> argparse.Namespace:
    json_path, patch_dirs, output_root = default_paths()
    parser = argparse.ArgumentParser(
        description="Select top-k local-only dense small-object images from Objects365 val and export YOLO dataset."
    )
    parser.add_argument("--json_path", type=Path, default=json_path, help="Path to COCO annotation JSON.")
    parser.add_argument(
        "--patch_dirs",
        type=Path,
        nargs="*",
        default=patch_dirs,
        help="Patch directories that actually exist locally.",
    )
    parser.add_argument("--output_root", type=Path, default=output_root, help="Output root directory.")
    parser.add_argument("--k", type=int, default=50, help="Number of images to select.")
    parser.add_argument("--small_area", type=float, default=1024, help="Area threshold for small objects.")
    parser.add_argument(
        "--min_images_per_class",
        type=int,
        default=5,
        help="Greedy coverage target per class before filling to k.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only produce reports/meta; do not copy images or write labels.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for tie-breaking.")
    return parser.parse_args()


def scan_local_images(patch_dirs: List[Path]) -> Tuple[Set[str], Dict[str, Path]]:
    local_set: Set[str] = set()
    path_map: Dict[str, Path] = {}
    for patch_dir in patch_dirs:
        if not patch_dir.exists():
            continue
        for img_path in patch_dir.glob("*.jpg"):
            basename = img_path.name
            local_set.add(basename)
            path_map.setdefault(basename, img_path)
    return local_set, path_map


def load_coco(json_path: Path) -> Dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_maps(categories: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    warnings: List[str] = []
    canonical_to_id: Dict[str, int] = {}
    synonyms_lower = {k: {s.lower() for s in v} for k, v in TARGET_SYNONYMS.items()}
    for cat in categories:
        name_raw = str(cat.get("name", "")).strip()
        lower = name_raw.lower()
        for canonical, pool in synonyms_lower.items():
            if lower in pool:
                if canonical not in canonical_to_id:
                    canonical_to_id[canonical] = cat["id"]
                break
    missing = [c for c in TARGET_ORDER if c not in canonical_to_id]
    if missing:
        warnings.append(f"Missing categories in annotations: {', '.join(missing)}")
    id_to_canonical = {v: k for k, v in canonical_to_id.items()}
    return canonical_to_id, id_to_canonical, warnings


def extract_patch_hint(file_name: str) -> Optional[str]:
    for part in Path(file_name).parts:
        if part.lower().startswith("patch"):
            return part
    return None


def collect_class_sets_for_all_images(
    coco: Dict, id_to_canonical: Dict[int, str]
) -> Dict[int, Set[str]]:
    class_sets: Dict[int, Set[str]] = defaultdict(set)
    for ann in coco.get("annotations", []):
        cat_id = ann.get("category_id")
        img_id = ann.get("image_id")
        if cat_id not in id_to_canonical:
            continue
        class_sets[img_id].add(id_to_canonical[cat_id])
    return class_sets


def prepare_stats_local_only(
    coco: Dict,
    id_to_canonical: Dict[int, str],
    local_set: Set[str],
    small_area: float,
) -> Dict[int, Dict]:
    stats: Dict[int, Dict] = {}
    for img in coco.get("images", []):
        file_name = img.get("file_name", "")
        if Path(file_name).name not in local_set:
            continue
        stats[img["id"]] = {
            "id": img["id"],
            "file_name": file_name,
            "width": img.get("width"),
            "height": img.get("height"),
            "class_counts": Counter(),
            "small_counts": Counter(),
            "class_set": set(),
            "small_count": 0,
            "total_count": 0,
            "annotations": [],
        }
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        if img_id not in stats:
            continue
        cat_id = ann.get("category_id")
        if cat_id not in id_to_canonical:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            continue
        class_name = id_to_canonical[cat_id]
        w, h = float(bbox[2]), float(bbox[3])
        area = w * h
        s = stats[img_id]
        s["total_count"] += 1
        s["class_counts"][class_name] += 1
        if area < small_area:
            s["small_count"] += 1
            s["small_counts"][class_name] += 1
        s["class_set"].add(class_name)
        s["annotations"].append({"bbox": bbox, "class_name": class_name})
    return stats


def greedy_select(
    stats: Dict[int, Dict],
    k: int,
    min_images_per_class: int,
    rng: random.Random,
) -> Tuple[List[int], Counter]:
    selected: Set[int] = set()
    coverage = Counter()
    candidates = [i for i, s in stats.items() if s["total_count"] > 0]
    rng.shuffle(candidates)

    def update(img_id: int) -> None:
        for cls in TARGET_ORDER:
            if stats[img_id]["class_counts"].get(cls, 0) > 0:
                coverage[cls] += 1

    # coverage phase
    while True:
        missing = [c for c in TARGET_ORDER if coverage[c] < min_images_per_class]
        if not missing:
            break
        pool = []
        for img_id in candidates:
            if img_id in selected:
                continue
            present = [c for c in missing if stats[img_id]["class_counts"].get(c, 0) > 0]
            if not present:
                continue
            pool.append(
                (
                    len(present),
                    stats[img_id]["small_count"],
                    stats[img_id]["total_count"],
                    rng.random(),
                    img_id,
                )
            )
        if not pool:
            break
        pool.sort(reverse=True)
        chosen = pool[0][4]
        selected.add(chosen)
        update(chosen)
        if len(selected) >= k:
            break

    remaining = [
        (
            stats[i]["small_count"],
            stats[i]["total_count"],
            rng.random(),
            i,
        )
        for i in candidates
        if i not in selected
    ]
    remaining.sort(reverse=True)
    for _, _, _, img_id in remaining:
        if len(selected) >= k:
            break
        selected.add(img_id)
    # final coverage recompute
    coverage = Counter()
    for img_id in selected:
        for cls in TARGET_ORDER:
            if stats[img_id]["class_counts"].get(cls, 0) > 0:
                coverage[cls] += 1
    return list(selected), coverage


def coco_to_yolo(bbox: List[float], width: float, height: float) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2) / width
    cy = (y + h / 2) / height
    return cx, cy, w / width, h / height


def write_labels(
    label_path: Path,
    annotations: List[Dict],
    width: float,
    height: float,
    class_index_map: Dict[str, int],
) -> None:
    lines = []
    for ann in annotations:
        idx = class_index_map[ann["class_name"]]
        cx, cy, w_norm, h_norm = coco_to_yolo(ann["bbox"], width, height)
        lines.append(f"{idx} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def export(
    selected_ids: List[int],
    stats: Dict[int, Dict],
    basename_to_path: Dict[str, Path],
    output_root: Path,
    dry_run: bool,
) -> List[Dict]:
    images_dir = output_root / "images"
    labels_dir = output_root / "labels"
    if not dry_run:
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    iterator = selected_ids
    if tqdm:
        iterator = tqdm(selected_ids, desc="Exporting")
    for img_id in iterator:
        s = stats[img_id]
        basename = Path(s["file_name"]).name
        img_path = basename_to_path.get(basename)
        if img_path is None:
            raise RuntimeError(f"Missing local file for selected image: {basename}")
        rec = {
            "image_id": img_id,
            "file_name": s["file_name"],
            "basename": basename,
            "small_count": s["small_count"],
            "total_count": s["total_count"],
            "class_set": sorted(s["class_set"]),
        }
        records.append(rec)
        if dry_run:
            continue
        shutil.copy2(img_path, images_dir / basename)
        label_path = labels_dir / f"{Path(basename).stem}.txt"
        write_labels(label_path, s["annotations"], s["width"], s["height"], TARGET_INDEX)
    return records


def dump_selection_report(
    report_path: Path,
    params: Dict,
    local_pool_size: int,
    selected_count: int,
    coverage: Counter,
    selected_records: List[Dict],
    status: str,
    shortfall: Dict[str, int],
    patch_suggestions: Dict[str, Dict[str, int]],
) -> None:
    report = {
        "generated_at": datetime.now().isoformat(),
        "params": params,
        "status": status,
        "local_pool_size": local_pool_size,
        "selected_count": selected_count,
        "coverage": {cls: int(coverage.get(cls, 0)) for cls in TARGET_ORDER},
        "selected": selected_records,
        "shortfall": shortfall,
        "patch_suggestions": patch_suggestions,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def dump_meta_yaml(
    meta_path: Path,
    json_path: Path,
    patch_dirs: List[Path],
    params: Dict,
    canonical_to_id: Dict[str, int],
    selected_records: List[Dict],
) -> None:
    meta_content = {
        "generated_at": datetime.now().isoformat(),
        "input_json": str(json_path),
        "patch_dirs": [str(p) for p in patch_dirs],
        "params": params,
        "classes": [
            {"id": TARGET_INDEX[name], "name": name, "source_category_id": canonical_to_id.get(name)}
            for name in TARGET_ORDER
        ],
        "selected_images": [rec["basename"] for rec in selected_records],
    }
    if yaml:
        meta_path.write_text(yaml.safe_dump(meta_content, sort_keys=False, allow_unicode=True), encoding="utf-8")
    else:
        meta_path.write_text(json.dumps(meta_content, indent=2, ensure_ascii=False), encoding="utf-8")


def suggest_patches_for_missing(
    coco: Dict,
    id_to_canonical: Dict[int, str],
    local_set: Set[str],
    needed_classes: List[str],
) -> Dict[str, Dict[str, int]]:
    class_patch_counts: Dict[str, Counter] = {cls: Counter() for cls in needed_classes}
    # Build class set per image (all images)
    class_sets = collect_class_sets_for_all_images(coco, id_to_canonical)
    for img in coco.get("images", []):
        basename = Path(img.get("file_name", "")).name
        if basename in local_set:
            continue
        img_classes = class_sets.get(img.get("id"), set())
        if not img_classes:
            continue
        patch_hint = extract_patch_hint(img.get("file_name", ""))
        for cls in needed_classes:
            if cls in img_classes:
                class_patch_counts[cls][patch_hint or "unknown"] += 1
    return {cls: dict(counter.most_common(5)) for cls, counter in class_patch_counts.items()}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    print("Scanning local patches...")
    local_set, basename_to_path = scan_local_images([Path(p) for p in args.patch_dirs])
    print(f"Local images found: {len(local_set)}")

    print(f"Loading COCO annotations from: {args.json_path}")
    coco = load_coco(args.json_path)
    canonical_to_id, id_to_canonical, warnings = build_category_maps(coco.get("categories", []))
    if not canonical_to_id:
        print("No target categories found. Abort.")
        sys.exit(1)

    stats = prepare_stats_local_only(coco, id_to_canonical, local_set, args.small_area)
    local_pool_size = len(stats)
    available_with_targets = sum(1 for s in stats.values() if s["total_count"] > 0)
    print(f"Local pool (with targets): {available_with_targets}/{local_pool_size}")

    selected_ids, coverage = greedy_select(
        stats, k=args.k, min_images_per_class=args.min_images_per_class, rng=rng
    )
    selected_ids.sort(
        key=lambda x: (stats[x]["small_count"], stats[x]["total_count"]),
        reverse=True,
    )

    coverage_shortfall = {
        cls: max(0, args.min_images_per_class - coverage.get(cls, 0)) for cls in TARGET_ORDER
    }
    coverage_missing_classes = [cls for cls, v in coverage_shortfall.items() if v > 0]
    k_shortfall = max(0, args.k - len(selected_ids))
    has_shortfall = k_shortfall > 0 or coverage_missing_classes

    reports_dir = Path(args.output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    selection_report_path = reports_dir / "selection_report.json"

    params = {
        "k": args.k,
        "small_area": args.small_area,
        "min_images_per_class": args.min_images_per_class,
        "seed": args.seed,
        "dry_run": args.dry_run,
    }

    patch_suggestions: Dict[str, Dict[str, int]] = {}
    if has_shortfall and coverage_missing_classes:
        patch_suggestions = suggest_patches_for_missing(coco, id_to_canonical, local_set, coverage_missing_classes)

    if has_shortfall:
        shortfall_info = {"k_shortfall": k_shortfall, "coverage_shortfall": coverage_shortfall}
        dump_selection_report(
            selection_report_path,
            params,
            local_pool_size,
            len(selected_ids),
            coverage,
            [],
            status="insufficient",
            shortfall=shortfall_info,
            patch_suggestions=patch_suggestions,
        )
        print("----- Error -----")
        print(f"Local images with targets: {available_with_targets}, need at least {args.k}")
        if k_shortfall > 0:
            print(f"Short of {k_shortfall} images to reach k={args.k}")
        for cls in TARGET_ORDER:
            deficit = coverage_shortfall.get(cls, 0)
            if deficit > 0:
                print(f"Coverage shortfall for {cls}: need {args.min_images_per_class}, got {coverage.get(cls, 0)}")
                hint = patch_suggestions.get(cls, {})
                if hint:
                    print(f"  Suggest downloading patches: {hint}")
        print(f"Report written (insufficient) to: {selection_report_path}")
        sys.exit(1)

    print(f"Selected {len(selected_ids)} images; exporting to {args.output_root}")
    selected_records = export(selected_ids, stats, basename_to_path, Path(args.output_root), args.dry_run)

    meta_path = Path(args.output_root) / "meta.yaml"
    dump_selection_report(
        selection_report_path,
        params,
        local_pool_size,
        len(selected_records),
        coverage,
        selected_records,
        status="ok",
        shortfall={"k_shortfall": 0, "coverage_shortfall": {}},
        patch_suggestions={},
    )
    dump_meta_yaml(meta_path, args.json_path, [Path(p) for p in args.patch_dirs], params, canonical_to_id, selected_records)

    print("----- Summary -----")
    print(f"Total selected: {len(selected_records)} (k={args.k})")
    print(f"Per-class coverage: {', '.join([f'{c}:{coverage.get(c,0)}' for c in TARGET_ORDER])}")
    print(f"Local pool size: {local_pool_size}, available with targets: {available_with_targets}")
    print(f"Reports written to: {selection_report_path}")
    print(f"Meta written to: {meta_path}")
    if args.dry_run:
        print("Dry run enabled: images and labels were not written.")


if __name__ == "__main__":
    main()
