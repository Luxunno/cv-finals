"""Gradio UI for PromptGuard image-only pipeline.

UI-only features:
- Parameter panel -> JobConfig (must be effective, verified via outputs/<job_id>/config.json)
- Baseline/Enhanced diff visualization and explainable lists
- Optional UI-side filter to only show 6 classes (re-render boxed images)
- Run twice (discard first) to avoid warmup bias in displayed result

Entry:
  python -m app.ui --host 127.0.0.1 --port 7860
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import gradio as gr
from PIL import Image, ImageDraw

from app.service import Service
from core.config import JobConfig
from core.postprocess import iou_xyxy


SIX_CLASS_LABELS = {"person", "car", "bicycle", "motorcycle", "bus", "traffic light"}


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return 0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_detections(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = _load_json(path)
    dets = data.get("detections", [])
    return dets if isinstance(dets, list) else []


def _filter_to_six(dets: list[dict], only_six: bool) -> list[dict]:
    if not only_six:
        return dets
    return [d for d in dets if d.get("label") in SIX_CLASS_LABELS]


def build_config_from_ui(
    global_conf_threshold: float,
    tile_size: int,
    overlap: float,
    wbf_iou_normal: float,
    max_det: int,
    *,
    tile_conf_threshold: float,
    tile_pad_px: int,
    tile_scale: float,
    wbf_iou_small: float,
    small_area_thresh: int,
) -> JobConfig:
    base = JobConfig()
    return dataclasses.replace(
        base,
        conf_threshold=float(global_conf_threshold),
        global_conf_threshold=float(global_conf_threshold),
        tile_conf_threshold=float(tile_conf_threshold),
        tile_size=int(tile_size),
        overlap=float(overlap),
        tile_pad_px=int(tile_pad_px),
        tile_scale=float(tile_scale),
        wbf_iou=float(wbf_iou_normal),
        wbf_iou_normal=float(wbf_iou_normal),
        wbf_iou_small=float(wbf_iou_small),
        small_area_thresh=int(small_area_thresh),
        max_det=int(max_det),
    )


def _config_snapshot_from_ui(cfg: JobConfig, device: str) -> dict:
    return {
        "device": device,
        "conf_threshold": cfg.conf_threshold,
        "nms_iou": cfg.nms_iou,
        "tile_size": cfg.tile_size,
        "overlap": cfg.overlap,
        "wbf_iou": cfg.wbf_iou,
        "max_det": cfg.max_det,
        "img_max_side": cfg.img_max_side,
    }


def _load_effective_config(job_dir: Path) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[Path]]:
    cfg_path = job_dir / "config.json"
    if not cfg_path.exists():
        return None, None, None, None
    data = _load_json(cfg_path)
    if "effective_config" in data or "full_config" in data:
        effective = data.get("effective_config")
        full = data.get("full_config")
        deprecated_mismatch = data.get("deprecated_mismatch")
        return effective, full, deprecated_mismatch, cfg_path
    # Backward-compatible: config.json is flat
    return data, data, {}, cfg_path


def _compare_config(ui_snapshot: dict, effective: dict) -> list[str]:
    mismatches: list[str] = []
    keys = [
        "global_conf_threshold",
        "tile_conf_threshold",
        "tile_size",
        "overlap",
        "tile_pad_px",
        "tile_scale",
        "wbf_iou_normal",
        "wbf_iou_small",
        "small_area_thresh",
        "max_det",
    ]
    for k in keys:
        if k not in effective:
            mismatches.append(f"{k}: missing in config.json")
            continue
        u = ui_snapshot.get(k)
        e = effective.get(k)
        if isinstance(u, float) or isinstance(e, float):
            if abs(_safe_float(u) - _safe_float(e)) > 1e-6:
                mismatches.append(f"{k}: UI({u}) != config.json({e})")
        else:
            if _safe_int(u) != _safe_int(e):
                mismatches.append(f"{k}: UI({u}) != config.json({e})")
    return mismatches


def _draw_boxes(image_rgb: Image.Image, dets: list[dict], color: tuple[int, int, int]) -> Image.Image:
    canvas = image_rgb.copy()
    draw = ImageDraw.Draw(canvas)
    for d in dets:
        box = d.get("box_xyxy")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = str(d.get("label", ""))
        score = _safe_float(d.get("score", 0.0))
        draw.text((x1, max(0, y1 - 12)), f"{label} {score:.2f}", fill=color)
    return canvas


def _match_diff(
    baseline_dets: list[dict],
    enhanced_dets: list[dict],
    diff_iou_thresh: float,
) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
    """Per-label greedy matching: enhanced sorted by score desc."""
    baseline_by_label: dict[str, list[dict]] = {}
    for d in baseline_dets:
        baseline_by_label.setdefault(str(d.get("label", "")), []).append(dict(d))
    enhanced_by_label: dict[str, list[dict]] = {}
    for d in enhanced_dets:
        enhanced_by_label.setdefault(str(d.get("label", "")), []).append(dict(d))

    added: list[dict] = []
    missing: list[dict] = []
    matched: list[tuple[dict, dict]] = []

    for label in sorted(set(baseline_by_label) | set(enhanced_by_label)):
        b_list = sorted(
            baseline_by_label.get(label, []),
            key=lambda x: _safe_float(x.get("score")),
            reverse=True,
        )
        e_list = sorted(
            enhanced_by_label.get(label, []),
            key=lambda x: _safe_float(x.get("score")),
            reverse=True,
        )
        b_used = [False] * len(b_list)
        e_used = [False] * len(e_list)

        for e_idx, e in enumerate(e_list):
            e_box = e.get("box_xyxy")
            best_iou_all = 0.0
            best_iou_unused = 0.0
            best_idx = -1

            if isinstance(e_box, list) and len(e_box) == 4:
                for idx, b in enumerate(b_list):
                    b_box = b.get("box_xyxy")
                    if not (isinstance(b_box, list) and len(b_box) == 4):
                        continue
                    iou = iou_xyxy(tuple(map(float, e_box)), tuple(map(float, b_box)))
                    best_iou_all = max(best_iou_all, iou)
                    if b_used[idx]:
                        continue
                    if iou > best_iou_unused:
                        best_iou_unused = iou
                        best_idx = idx

            # For Added: show best IoU against any baseline (even matched)
            e["iou_match"] = best_iou_all
            if best_iou_unused >= diff_iou_thresh and best_idx >= 0:
                b_used[best_idx] = True
                e_used[e_idx] = True
                b_list[best_idx]["iou_match"] = best_iou_unused
                e["iou_match"] = best_iou_unused
                matched.append((b_list[best_idx], e))
            else:
                added.append(e)

        for idx, b in enumerate(b_list):
            if b_used[idx]:
                continue
            b_box = b.get("box_xyxy")
            best_iou = 0.0
            if isinstance(b_box, list) and len(b_box) == 4:
                for e_idx, e in enumerate(e_list):
                    if e_used[e_idx]:
                        continue
                    e_box = e.get("box_xyxy")
                    if not (isinstance(e_box, list) and len(e_box) == 4):
                        continue
                    best_iou = max(best_iou, iou_xyxy(tuple(map(float, b_box)), tuple(map(float, e_box))))
            b["iou_match"] = best_iou
            missing.append(b)

    return added, missing, matched


def _draw_diff_image(
    image_rgb: Image.Image,
    added: list[dict],
    missing: list[dict],
    matched: list[tuple[dict, dict]],
) -> Image.Image:
    canvas = image_rgb.copy()
    draw = ImageDraw.Draw(canvas)

    for b, _e in matched:
        box = b.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(160, 160, 160), width=1)

    for d in added:
        box = d.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)

    for d in missing:
        box = d.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(220, 0, 0), width=3)

    return canvas


def _rows_for_dets(dets: list[dict]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for d in dets:
        label = str(d.get("label", ""))
        score = round(_safe_float(d.get("score")), 4)
        box = d.get("box_xyxy", [])
        iou_match = round(_safe_float(d.get("iou_match", 0.0)), 2)
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
        else:
            x1 = y1 = x2 = y2 = ""
        rows.append([label, score, x1, y1, x2, y2, iou_match])
    return rows


def _metrics_7_rows(
    summary: dict,
    added_count: int,
    matched_count: int,
    missing_count: int,
) -> list[list[Any]]:
    baseline_ms = _safe_float(summary.get("baseline_ms"))
    enhanced_ms = _safe_float(summary.get("enhanced_ms"))
    ratio = (enhanced_ms / baseline_ms) if baseline_ms > 0 else 0.0
    num_tiles = _safe_int(summary.get("num_tiles"))
    tile_infer_ms_total = _safe_float(summary.get("tile_infer_ms_total"))
    baseline_boxes = _safe_int(summary.get("baseline_num_boxes"))
    enhanced_boxes = _safe_int(summary.get("enhanced_num_boxes"))

    return [
        ["baseline_ms", baseline_ms],
        ["enhanced_ms", enhanced_ms],
        ["ratio(enhanced/baseline)", round(ratio, 3)],
        ["num_tiles", num_tiles],
        ["tile_infer_ms_total", tile_infer_ms_total],
        ["added/matched/missing", f"{added_count}/{matched_count}/{missing_count}"],
        ["num_boxes(baseline/enhanced)", f"{baseline_boxes}/{enhanced_boxes}"],
    ]


def _warn_text(summary: dict, discard_first: bool) -> str:
    warmup_ran = summary.get("warmup_ran")
    warmup_ms = summary.get("warmup_ms")
    num_tiles = _safe_int(summary.get("num_tiles"))
    tile_ms = _safe_float(summary.get("tile_infer_ms_total"))
    bms = _safe_float(summary.get("baseline_ms"))
    ems = _safe_float(summary.get("enhanced_ms"))

    msg = f"warmup_ran={warmup_ran}, warmup_ms={warmup_ms}, discard_first={discard_first}"
    warn = ""
    if num_tiles <= 1:
        warn += "\n\n**[WARNING] Enhanced degenerated: num_tiles=1；请减小 tile_size 或使用更大分辨率图片。**"
    if tile_ms > 3000 or (bms > 0 and ems > 3 * bms):
        warn += "\n\n**[WARNING] 当前参数导致 Enhanced 过慢（仅用于调试），建议使用 Preset B 或增大 tile_size/提高 conf/降低 overlap。**"
    return msg + warn


def run_job_ui(
    image_path: str,
    device: str,
    mode: str,
    only_six: bool,
    run_twice_discard_first: bool,
    conf_threshold: float,
    tile_size: int,
    overlap: float,
    wbf_iou: float,
    diff_iou_thresh: float,
    max_det: int,
    tile_conf_threshold: float,
    tile_pad_px: int,
    tile_scale: float,
    wbf_iou_small: float,
    small_area_thresh: int,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    str,
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    list[list[Any]],
    list[list[Any]],
    list[list[Any]],
    str,
    str,
    str,
    str,
]:
    if not image_path:
        raise gr.Error("请上传图片")
    src = Path(image_path)
    if not src.exists():
        raise gr.Error(f"文件不存在: {src}")

    image_rgb = Image.open(src).convert("RGB")
    svc = Service()
    cfg = build_config_from_ui(
        conf_threshold,
        tile_size,
        overlap,
        wbf_iou,
        max_det,
        tile_conf_threshold=tile_conf_threshold,
        tile_pad_px=tile_pad_px,
        tile_scale=tile_scale,
        wbf_iou_small=wbf_iou_small,
        small_area_thresh=small_area_thresh,
    )

    ui_snapshot = _config_snapshot_from_ui(cfg, device)
    discarded_dir = ""

    first = svc.run_job(image=image_rgb, config=cfg, device=device)
    final = first
    if run_twice_discard_first:
        discarded_dir = str(first["output_dir"])
        final = svc.run_job(image=image_rgb, config=cfg, device=device)

    job_dir = Path(final["output_dir"])
    summary_path = Path(final["summary_json"])
    summary = _load_json(summary_path) if summary_path.exists() else {}

    # Validate config actually effective
    effective_cfg, full_cfg, deprecated_mismatch, cfg_path = _load_effective_config(job_dir)
    ui_cfg_snapshot_path = None
    if effective_cfg is None:
        ui_cfg_snapshot_path = job_dir / "ui_config_snapshot.json"
        ui_cfg_snapshot_path.write_text(json.dumps(ui_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    mismatches = []
    if effective_cfg is not None:
        mismatches = _compare_config(ui_snapshot, effective_cfg)

    # Load dets for diff and optional 6-class re-rendering
    baseline_json_path = Path(final["baseline_json"])
    enhanced_json_path = Path(final["enhanced_json"])
    baseline_dets_all = _load_detections(baseline_json_path)
    enhanced_dets_all = _load_detections(enhanced_json_path)
    baseline_dets = _filter_to_six(baseline_dets_all, only_six=only_six)
    enhanced_dets = _filter_to_six(enhanced_dets_all, only_six=only_six)

    # Determine images to show
    baseline_img: Optional[str] = str(final["baseline_image"]) if mode in ("Baseline", "Both") else None
    enhanced_img: Optional[str] = str(final["enhanced_image"]) if mode in ("Enhanced", "Both") else None
    baseline_png: Optional[str] = baseline_img
    enhanced_png: Optional[str] = enhanced_img

    if only_six:
        if mode in ("Baseline", "Both"):
            b_img6 = _draw_boxes(image_rgb, baseline_dets, color=(255, 165, 0))
            b_path6 = job_dir / "baseline_6_boxed.png"
            b_img6.save(b_path6)
            baseline_img = str(b_path6)
            baseline_png = str(b_path6)
        if mode in ("Enhanced", "Both"):
            e_img6 = _draw_boxes(image_rgb, enhanced_dets, color=(0, 140, 255))
            e_path6 = job_dir / "enhanced_6_boxed.png"
            e_img6.save(e_path6)
            enhanced_img = str(e_path6)
            enhanced_png = str(e_path6)

    diff_img = None
    diff_png = None
    diff_json = None
    diff_json_file = None
    added_rows: list[list[Any]] = []
    missing_rows: list[list[Any]] = []
    added_count = matched_count = missing_count = 0

    if mode == "Both":
        added, missing, matched = _match_diff(baseline_dets, enhanced_dets, diff_iou_thresh=float(diff_iou_thresh))
        added_count, missing_count, matched_count = len(added), len(missing), len(matched)
        diff_image = _draw_diff_image(image_rgb, added=added, missing=missing, matched=matched)
        diff_path = job_dir / "diff_boxed.png"
        diff_image.save(diff_path)
        diff_img = str(diff_path)
        diff_png = str(diff_path)

        diff_payload = {
            "only_six_classes": only_six,
            "diff_iou_thresh": float(diff_iou_thresh),
            "added_count": added_count,
            "matched_count": matched_count,
            "missing_count": missing_count,
            "added": added,
            "missing": missing,
        }
        diff_json_path = job_dir / "diff.json"
        diff_json_path.write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        diff_json = str(diff_json_path)
        diff_json_file = str(diff_json_path)
        added_rows = _rows_for_dets(added)
        missing_rows = _rows_for_dets(missing)

    baseline_json = str(final["baseline_json"]) if mode in ("Baseline", "Both") else None
    enhanced_json = str(final["enhanced_json"]) if mode in ("Enhanced", "Both") else None
    summary_json = str(final["summary_json"])

    # Metrics (fixed 7 rows)
    metrics = _metrics_7_rows(summary, added_count, matched_count, missing_count)

    # Advanced/Debug
    advanced = {
        "ui_snapshot": ui_snapshot,
        "effective_config_json": effective_cfg,
        "full_config_json": full_cfg,
        "deprecated_mismatch": deprecated_mismatch,
        "mismatches": mismatches,
        "discarded_output_dir": discarded_dir,
    }

    status = "运行成功"
    if mismatches:
        status = "**[ERROR] 参数未生效：**\n" + "\n".join([f"- {m}" for m in mismatches])
        if deprecated_mismatch:
            status += (
                "\n\n**[ERROR] Deprecated 不同步：**\n"
                + "\n".join([f"- {k}: {v}" for k, v in (deprecated_mismatch or {}).items()])
            )
        status += "\n\n请检查 build_config_from_ui 或 service 导出逻辑。"

    status += "\n\n" + _warn_text(summary, discard_first=run_twice_discard_first)

    run_log_path = str(job_dir / "run.log")
    config_file = str(cfg_path) if cfg_path is not None else (str(ui_cfg_snapshot_path) if ui_cfg_snapshot_path else None)

    return (
        baseline_img,
        enhanced_img,
        diff_img,
        baseline_json,
        enhanced_json,
        summary_json,
        baseline_png,
        enhanced_png,
        diff_png,
        diff_json_file,
        config_file,
        metrics,
        added_rows,
        missing_rows,
        status,
        json.dumps(advanced, ensure_ascii=False, indent=2),
        run_log_path,
        str(job_dir),
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="PromptGuard UI") as demo:
        gr.Markdown("## PromptGuard（Baseline vs Enhanced）")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="上传图片")
            with gr.Column():
                device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="Device")
                mode = gr.Radio(choices=["Both", "Baseline", "Enhanced"], value="Both", label="运行模式")
                only_six = gr.Checkbox(value=False, label="只展示/对比 6 类（UI 侧过滤并重绘）")
                run_twice = gr.Checkbox(value=True, label="Run twice (discard first)")

        with gr.Row():
            profile = gr.Radio(choices=["Fast", "Balanced", "Quality", "Custom"], value="Balanced", label="Profile")
            conf_threshold = gr.Slider(0.05, 0.25, step=0.01, value=0.25, label="conf_threshold (global)")
            tile_size = gr.Dropdown([384, 512, 640, 768], value=640, label="tile_size")
            overlap = gr.Slider(0.1, 0.4, step=0.01, value=0.2, label="overlap")
            wbf_iou = gr.Slider(0.45, 0.7, step=0.01, value=0.58, label="wbf_iou_normal")
            diff_iou_thresh = gr.Slider(0.3, 0.7, step=0.01, value=0.5, label="diff_iou_thresh")
            max_det = gr.Slider(50, 300, step=10, value=300, label="max_det")

        with gr.Accordion("Advanced parameters", open=False):
            tile_conf_threshold = gr.Slider(0.05, 0.3, step=0.01, value=0.18, label="tile_conf_threshold")
            tile_pad_px = gr.Slider(0, 96, step=4, value=48, label="tile_pad_px")
            tile_scale = gr.Dropdown([1.0, 1.5, 2.0], value=1.0, label="tile_scale")
            wbf_iou_small = gr.Slider(0.3, 0.55, step=0.01, value=0.45, label="wbf_iou_small")
            small_area_thresh = gr.Slider(256, 4096, step=128, value=1024, label="small_area_thresh")

        run_btn = gr.Button("Run", variant="primary")

        def apply_profile(p: str):
            if p == "Fast":
                return 0.25, 640, 0.18, 0.62, 0.5, 0.20, 32, 1.0, 0.45, 1024
            if p == "Balanced":
                return 0.25, 512, 0.20, 0.58, 0.5, 0.18, 48, 1.0, 0.45, 1024
            if p == "Quality":
                return 0.25, 512, 0.20, 0.58, 0.5, 0.16, 64, 1.5, 0.45, 1024
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )

        profile.change(
            fn=apply_profile,
            inputs=[profile],
            outputs=[
                conf_threshold,
                tile_size,
                overlap,
                wbf_iou,
                diff_iou_thresh,
                tile_conf_threshold,
                tile_pad_px,
                tile_scale,
                wbf_iou_small,
                small_area_thresh,
            ],
        )

        with gr.Row():
            baseline_img = gr.Image(type="filepath", label="Baseline")
            enhanced_img = gr.Image(type="filepath", label="Enhanced")
            diff_img = gr.Image(type="filepath", label="Diff（绿=新增，红=消失，灰=共有）")

        with gr.Row():
            baseline_json = gr.File(label="baseline.json")
            enhanced_json = gr.File(label="enhanced.json")
            summary_json = gr.File(label="summary.json")
            config_json = gr.File(label="config.json / ui_config_snapshot.json")
            baseline_png = gr.File(label="baseline_boxed.png / baseline_6_boxed.png")
            enhanced_png = gr.File(label="enhanced_boxed.png / enhanced_6_boxed.png")
            diff_png = gr.File(label="diff_boxed.png")
            diff_json = gr.File(label="diff.json")

        metrics = gr.Dataframe(
            headers=["metric", "value"],
            datatype=["str", "str"],
            label="Metrics（固定 7 行）",
        )

        with gr.Row():
            added_df = gr.Dataframe(
                headers=["label", "score", "x1", "y1", "x2", "y2", "iou_match"],
                datatype=["str", "number", "number", "number", "number", "number", "number"],
                label="Added in Enhanced",
            )
            missing_df = gr.Dataframe(
                headers=["label", "score", "x1", "y1", "x2", "y2", "iou_match"],
                datatype=["str", "number", "number", "number", "number", "number", "number"],
                label="Missing in Enhanced",
            )

        status = gr.Markdown("")

        with gr.Accordion("Advanced / Debug", open=False):
            debug_box = gr.Textbox(label="Debug JSON (includes mismatches/discarded dir)", lines=14)
            run_log_box = gr.Textbox(label="run.log 路径", lines=1)
            output_dir_box = gr.Textbox(label="outputs/<job_id> 目录", lines=1)

        def on_run(
            img_path: str,
            dev: str,
            md: str,
            only6: bool,
            twice: bool,
            conf: float,
            ts: int,
            ov: float,
            wi: float,
            diou: float,
            mdmax: int,
            tconf: float,
            tpad: int,
            tscale: float,
            wbf_small: float,
            sath: int,
        ):
            try:
                return run_job_ui(
                    img_path,
                    dev,
                    md,
                    only6,
                    twice,
                    conf,
                    ts,
                    ov,
                    wi,
                    diou,
                    mdmax,
                    tconf,
                    tpad,
                    tscale,
                    wbf_small,
                    sath,
                )
            except Exception as exc:  # noqa: BLE001
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    [],
                    [],
                    [],
                    f"运行失败: {exc}",
                    traceback.format_exc(),
                    "",
                    "",
                )

        run_btn.click(
            fn=on_run,
            inputs=[
                image_input,
                device,
                mode,
                only_six,
                run_twice,
                conf_threshold,
                tile_size,
                overlap,
                wbf_iou,
                diff_iou_thresh,
                max_det,
                tile_conf_threshold,
                tile_pad_px,
                tile_scale,
                wbf_iou_small,
                small_area_thresh,
            ],
            outputs=[
                baseline_img,
                enhanced_img,
                diff_img,
                baseline_json,
                enhanced_json,
                summary_json,
                baseline_png,
                enhanced_png,
                diff_png,
                diff_json,
                config_json,
                metrics,
                added_df,
                missing_df,
                status,
                debug_box,
                run_log_box,
                output_dir_box,
            ],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch PromptGuard Gradio UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host")
    parser.add_argument("--port", type=int, default=7860, help="Port")
    args = parser.parse_args()
    demo = build_interface()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
