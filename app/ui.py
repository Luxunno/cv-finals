"""Gradio UI for PromptGuard demo.

Responsibilities:
- Provide a minimal UI to compare Baseline vs Enhanced (single image or folder batch).
- Ensure UI parameters become JobConfig and are verifiable via outputs/config.json.
- Show summary metrics and batch tables for demonstration.

Key constraints:
- UI must not change inference logic (service/core handles inference).
- Effective config must be checked against outputs/config.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import traceback
from pathlib import Path
from typing import Any, Optional

import gradio as gr
from PIL import Image, ImageDraw

from app.service import Service
from core.config import JobConfig
from core.postprocess import iou_xyxy
from scripts.batch_compare import run_batch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def _build_config_from_ui(
    profile: str,
    custom_enabled: bool,
    *,
    global_conf_threshold: float,
    tile_conf_threshold: float,
    tile_size: int,
    overlap: float,
    tile_pad_px: int,
    tile_scale: float,
    wbf_iou_normal: float,
    wbf_iou_small: float,
    small_area_thresh: int,
    max_det: int,
) -> JobConfig:
    cfg = JobConfig()
    if not custom_enabled:
        cfg = _apply_profile(cfg, profile)
        return cfg
    cfg = dataclasses.replace(
        cfg,
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
    return cfg


def _match_diff(
    baseline_dets: list[dict],
    enhanced_dets: list[dict],
    diff_iou_thresh: float = 0.5,
) -> tuple[list[dict], list[dict], list[tuple[dict, dict]]]:
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
            if best_iou >= diff_iou_thresh and best_idx >= 0 and not b_used[best_idx]:
                b_used[best_idx] = True
                matched.append((b_list[best_idx], e))
            else:
                added.append(e)
        for idx, used in enumerate(b_used):
            if not used:
                missing.append(b_list[idx])
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
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=3)
    for d in missing:
        box = d.get("box_xyxy")
        if isinstance(box, list) and len(box) == 4:
            x1, y1, x2, y2 = map(int, box)
            draw.rectangle([x1, y1, x2, y2], outline=(200, 0, 0), width=3)
    return canvas


def _metrics_panel(
    baseline_ms_avg: float,
    enhanced_ms_avg: float,
    latency_ratio: float,
    baseline_boxes_avg: float,
    enhanced_boxes_avg: float,
    boxes_growth: float,
    baseline_recall: Optional[float],
    enhanced_recall: Optional[float],
    gain: Optional[float],
    added: Optional[int],
    missing: Optional[int],
) -> list[list[Any]]:
    rows = [
        ["baseline_ms_avg", round(baseline_ms_avg, 3)],
        ["enhanced_ms_avg", round(enhanced_ms_avg, 3)],
        ["latency_ratio", round(latency_ratio, 3)],
        ["baseline_boxes_avg", round(baseline_boxes_avg, 3)],
        ["enhanced_boxes_avg", round(enhanced_boxes_avg, 3)],
        ["boxes_growth", round(boxes_growth, 3)],
    ]
    if baseline_recall is not None and enhanced_recall is not None and gain is not None:
        rows.append(["baseline_recall_small@0.5", round(baseline_recall, 4)])
        rows.append(["enhanced_recall_small@0.5", round(enhanced_recall, 4)])
        rows.append(["gain", round(gain, 4)])
    else:
        rows.append(["baseline_recall_small@0.5", "N/A (no GT)"])
        rows.append(["enhanced_recall_small@0.5", "N/A (no GT)"])
        rows.append(["gain", "N/A (no GT)"])
    if added is not None and missing is not None:
        rows.append(["added/missing", f"{added}/{missing}"])
    return rows


def _open_path(path: str) -> None:
    if not path:
        return
    try:
        os.startfile(path)  # noqa: S606, S607
    except Exception:
        pass


def _single_preview_from_dir(per_dir: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    baseline_img = per_dir / "baseline_boxed.png"
    enhanced_img = per_dir / "enhanced_boxed.png"
    diff_img = per_dir / "diff_boxed.png"
    return (
        str(baseline_img) if baseline_img.exists() else None,
        str(enhanced_img) if enhanced_img.exists() else None,
        str(diff_img) if diff_img.exists() else None,
    )


def _build_batch_table(rows: list[dict]) -> list[list[Any]]:
    table = []
    for r in rows:
        table.append(
            [
                r.get("image"),
                r.get("baseline_boxes"),
                r.get("enhanced_boxes"),
                r.get("added"),
                r.get("missing"),
                r.get("latency_ratio"),
                r.get("baseline_recall_small"),
                r.get("enhanced_recall_small"),
                r.get("gain"),
            ]
        )
    return table


def run_ui(
    input_mode: str,
    image_path: str,
    folder_path: str,
    labels_path: str,
    device: str,
    profile: str,
    use_custom: bool,
    global_conf_threshold: float,
    tile_conf_threshold: float,
    tile_size: int,
    overlap: float,
    tile_pad_px: int,
    tile_scale: float,
    wbf_iou_normal: float,
    wbf_iou_small: float,
    small_area_thresh: int,
    max_det: int,
) -> tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    list[list[Any]],
    list[list[Any]],
    str,
    str,
    str,
    Any,
    dict,
    dict,
]:
    if input_mode == "Single":
        if not image_path:
            raise gr.Error("请上传图片")
        img_path = Path(image_path)
        if not img_path.exists():
            raise gr.Error(f"文件不存在: {img_path}")

        image_rgb = Image.open(img_path).convert("RGB")
        cfg = _build_config_from_ui(
            profile,
            use_custom,
            global_conf_threshold=global_conf_threshold,
            tile_conf_threshold=tile_conf_threshold,
            tile_size=tile_size,
            overlap=overlap,
            tile_pad_px=tile_pad_px,
            tile_scale=tile_scale,
            wbf_iou_normal=wbf_iou_normal,
            wbf_iou_small=wbf_iou_small,
            small_area_thresh=small_area_thresh,
            max_det=max_det,
        )
        svc = Service()
        result = svc.run_job(image=image_rgb, config=cfg, device=device)
        job_dir = Path(result["output_dir"])

        baseline_data = _load_json(Path(result["baseline_json"]))
        enhanced_data = _load_json(Path(result["enhanced_json"]))
        added, missing, matched = _match_diff(
            baseline_data.get("detections", []),
            enhanced_data.get("detections", []),
            diff_iou_thresh=0.5,
        )

        diff_img = job_dir / "diff_boxed.png"
        diff = _draw_diff_image(image_rgb, added, missing, matched)
        diff.save(diff_img)

        summary = _load_json(Path(result["summary_json"]))
        baseline_ms = _safe_float(summary.get("baseline_ms"))
        enhanced_ms = _safe_float(summary.get("enhanced_ms"))
        latency_ratio = (enhanced_ms / baseline_ms) if baseline_ms > 0 else 0.0
        baseline_boxes = _safe_int(summary.get("baseline_num_boxes"))
        enhanced_boxes = _safe_int(summary.get("enhanced_num_boxes"))

        metrics = _metrics_panel(
            baseline_ms,
            enhanced_ms,
            latency_ratio,
            baseline_boxes,
            enhanced_boxes,
            enhanced_boxes - baseline_boxes,
            None,
            None,
            None,
            len(added),
            len(missing),
        )

        effective_cfg, _full, deprecated_mismatch, cfg_path = _load_effective_config(job_dir)
        ui_snapshot = cfg.effective_snapshot(device=device)
        mismatches = _compare_config(ui_snapshot, effective_cfg or {})
        status = "运行成功"
        actual_device = summary.get("actual_device") or summary.get("device") or device
        if actual_device != device:
            reason = summary.get("device_fallback_reason") or "unknown"
            status += f"\n\n**[WARNING] device fallback: {device} -> {actual_device} ({reason})**"
        if mismatches:
            status = "**[ERROR] 参数未生效：**\n" + "\n".join([f"- {m}" for m in mismatches])
            if deprecated_mismatch:
                status += (
                    "\n\n**[ERROR] Deprecated 不同步：**\n"
                    + "\n".join([f"- {k}: {v}" for k, v in (deprecated_mismatch or {}).items()])
                )

        return (
            str(job_dir / "baseline_boxed.png"),
            str(job_dir / "enhanced_boxed.png"),
            str(diff_img),
            str(job_dir / "summary.json"),
            str(cfg_path) if cfg_path else None,
            str(job_dir),
            metrics,
            [],
            status,
            "",
            "",
            gr.update(choices=[], value=None),
            {},
            {},
        )

    # Batch
    if not folder_path:
        raise gr.Error("请输入文件夹路径")
    img_dir = Path(folder_path)
    if not img_dir.exists():
        raise gr.Error(f"文件夹不存在: {img_dir}")
    label_dir = Path(labels_path) if labels_path else None

    cfg = _build_config_from_ui(
        profile,
        use_custom,
        global_conf_threshold=global_conf_threshold,
        tile_conf_threshold=tile_conf_threshold,
        tile_size=tile_size,
        overlap=overlap,
        tile_pad_px=tile_pad_px,
        tile_scale=tile_scale,
        wbf_iou_normal=wbf_iou_normal,
        wbf_iou_small=wbf_iou_small,
        small_area_thresh=small_area_thresh,
        max_det=max_det,
    )
    batch_dir, report = run_batch(
        images=img_dir,
        labels=label_dir,
        device=device,
        profile=profile,
        config_json=None,
        config=cfg,
        output=None,
    )

    rows = report.get("rows", [])
    first_dir = Path(rows[0]["output_dir"]) if rows else None
    baseline_img, enhanced_img, diff_img = _single_preview_from_dir(first_dir) if first_dir else (None, None, None)

    overall = report.get("overall", {})
    metrics = _metrics_panel(
        _safe_float(overall.get("baseline_ms_avg")),
        _safe_float(overall.get("enhanced_ms_avg")),
        _safe_float(overall.get("latency_ratio")),
        _safe_float(overall.get("baseline_boxes_avg")),
        _safe_float(overall.get("enhanced_boxes_avg")),
        _safe_float(overall.get("boxes_growth_avg")),
        overall.get("baseline_recall_small@0.5"),
        overall.get("enhanced_recall_small@0.5"),
        overall.get("gain"),
        None,
        None,
    )
    table = _build_batch_table(rows)

    cfg_snap = report.get("config_snapshot", {})
    ui_snapshot = cfg.effective_snapshot(device=device)
    mismatches = _compare_config(ui_snapshot, cfg_snap or {})
    status = "运行成功"
    if report.get("meta", {}).get("requested_device") != report.get("meta", {}).get("actual_device"):
        reason_counts = report.get("meta", {}).get("fallback_reasons", {})
        status += f"\n\n**[WARNING] device fallback: {report['meta'].get('requested_device')} -> {report['meta'].get('actual_device')} ({reason_counts})**"
    if mismatches:
        status = "**[ERROR] 参数未生效：**\n" + "\n".join([f"- {m}" for m in mismatches])

    choices = [r.get("image") for r in rows if r.get("image")]
    first_choice = choices[0] if choices else None

    return (
        baseline_img,
        enhanced_img,
        diff_img,
        str(Path(batch_dir) / "batch_report.json"),
        str(Path(batch_dir) / "batch_report.csv"),
        str(batch_dir),
        metrics,
        table,
        status,
        str(Path(batch_dir) / "batch_scatter.png"),
        "",
        gr.update(choices=choices, value=first_choice),
        report,
        {"batch_dir": str(batch_dir)},
    )


def select_batch_image(report: dict, image_name: str) -> tuple[Optional[str], Optional[str]]:
    rows = report.get("rows", [])
    for r in rows:
        if r.get("image") == image_name:
            per_dir = Path(r.get("output_dir"))
            baseline_img = per_dir / "baseline_boxed.png"
            enhanced_img = per_dir / "enhanced_boxed.png"
            return (
                str(baseline_img) if baseline_img.exists() else None,
                str(enhanced_img) if enhanced_img.exists() else None,
            )
    return None, None


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="PromptGuard UI") as demo:
        gr.Markdown("## PromptGuard（Baseline vs Enhanced）")

        with gr.Row():
            input_mode = gr.Radio(choices=["Single", "Folder"], value="Single", label="输入模式")
            device = gr.Dropdown(choices=["cpu", "cuda"], value="cpu", label="Device")
            profile = gr.Dropdown(choices=["fast", "balanced", "quality"], value="balanced", label="Profile")
            run_btn = gr.Button("Run", variant="primary")

        with gr.Row():
            image_input = gr.Image(type="filepath", label="图片（Single）")
            folder_input = gr.Textbox(label="文件夹路径（Folder）", placeholder="D:\\data\\images")
            labels_input = gr.Textbox(label="Labels 路径（可选）", placeholder="D:\\data\\labels")

        with gr.Accordion("Advanced parameters (Custom only)", open=False):
            use_custom = gr.Checkbox(value=False, label="启用自定义参数")
            global_conf_threshold = gr.Slider(0.05, 0.3, step=0.01, value=0.25, label="global_conf_threshold")
            tile_conf_threshold = gr.Slider(0.05, 0.3, step=0.01, value=0.18, label="tile_conf_threshold")
            tile_size = gr.Dropdown([384, 512, 640, 768], value=512, label="tile_size")
            overlap = gr.Slider(0.1, 0.4, step=0.01, value=0.2, label="overlap")
            tile_pad_px = gr.Slider(0, 96, step=4, value=48, label="tile_pad_px")
            tile_scale = gr.Dropdown([1.0, 1.5, 2.0], value=1.0, label="tile_scale")
            wbf_iou_normal = gr.Slider(0.45, 0.7, step=0.01, value=0.58, label="wbf_iou_normal")
            wbf_iou_small = gr.Slider(0.3, 0.55, step=0.01, value=0.45, label="wbf_iou_small")
            small_area_thresh = gr.Slider(256, 4096, step=128, value=1024, label="small_area_thresh")
            max_det = gr.Slider(50, 300, step=10, value=300, label="max_det")

        with gr.Row():
            baseline_img = gr.Image(type="filepath", label="Baseline")
            enhanced_img = gr.Image(type="filepath", label="Enhanced")
            diff_img = gr.Image(type="filepath", label="Diff (Single only)")

        batch_select = gr.Dropdown(choices=[], value=None, label="批量选择图片（仅 Folder）")

        scatter_img = gr.Image(type="filepath", label="Batch Scatter")

        with gr.Row():
            report_json = gr.File(label="batch_report.json / summary.json")
            report_csv = gr.File(label="batch_report.csv / config.json")

        metrics = gr.Dataframe(
            headers=["metric", "value"],
            datatype=["str", "str"],
            label="Summary Metrics",
        )

        batch_table = gr.Dataframe(
            headers=[
                "image",
                "baseline_boxes",
                "enhanced_boxes",
                "added",
                "missing",
                "latency_ratio",
                "baseline_recall_small",
                "enhanced_recall_small",
                "gain",
            ],
            datatype=["str", "number", "number", "number", "number", "number", "number", "number", "number"],
            label="Batch Table",
        )

        with gr.Row():
            output_dir = gr.Textbox(label="输出目录", lines=1)
            open_dir_btn = gr.Button("打开输出目录")

        status = gr.Markdown("")

        report_state = gr.State({})
        meta_state = gr.State({})

        def on_run(
            mode: str,
            img_path: str,
            folder: str,
            labels: str,
            dev: str,
            prof: str,
            custom: bool,
            gconf: float,
            tconf: float,
            tsize: int,
            ov: float,
            tpad: int,
            tscale: float,
            wi: float,
            wis: float,
            sath: int,
            mdmax: int,
        ):
            try:
                return run_ui(
                    mode,
                    img_path,
                    folder,
                    labels,
                    dev,
                    prof,
                    custom,
                    gconf,
                    tconf,
                    tsize,
                    ov,
                    tpad,
                    tscale,
                    wi,
                    wis,
                    sath,
                    mdmax,
                )
            except Exception as exc:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    [],
                    [],
                    f"运行失败: {exc}\n\n{traceback.format_exc()}",
                    None,
                    None,
                    gr.update(choices=[], value=None),
                    {},
                    {},
                )

        run_btn.click(
            fn=on_run,
            inputs=[
                input_mode,
                image_input,
                folder_input,
                labels_input,
                device,
                profile,
                use_custom,
                global_conf_threshold,
                tile_conf_threshold,
                tile_size,
                overlap,
                tile_pad_px,
                tile_scale,
                wbf_iou_normal,
                wbf_iou_small,
                small_area_thresh,
                max_det,
            ],
            outputs=[
                baseline_img,
                enhanced_img,
                diff_img,
                report_json,
                report_csv,
                output_dir,
                metrics,
                batch_table,
                status,
                scatter_img,
                labels_input,
                batch_select,
                report_state,
                meta_state,
            ],
        )

        batch_select.change(
            fn=select_batch_image,
            inputs=[report_state, batch_select],
            outputs=[baseline_img, enhanced_img],
        )

        open_dir_btn.click(fn=_open_path, inputs=[output_dir], outputs=[])

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
