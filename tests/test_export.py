import json
from pathlib import Path

import numpy as np

from core.config import JobConfig
from core.export import export_config, export_detections_json, export_summary_json
from core.io_image import imread_any, imwrite_any
from core.detector_yolo import Detection


def test_imread_imwrite_unicode_path(tmp_path):
    img = np.full((8, 9, 3), 127, dtype=np.uint8)
    subdir = tmp_path / "子目录"
    path = subdir / "测试图像.png"

    imwrite_any(path, img)
    loaded = imread_any(path)

    assert loaded.shape == img.shape
    assert np.array_equal(loaded, img)


def _dummy_det():
    return [
        Detection(
            label="person",
            class_id=0,
            score=0.9,
            box_xyxy=(1, 2, 3, 4),
            attrs={"state": "red"},
        )
    ]


def test_export_jsons(tmp_path):
    job_id = "job123"
    config = JobConfig()
    export_config(tmp_path / "config.json", config)
    export_detections_json(
        tmp_path / "baseline.json",
        job_id=job_id,
        pipeline="baseline",
        model_name="yolov8n",
        num_parameters=1,
        image_shape=(10, 20),
        config=config,
        detections=_dummy_det(),
    )
    export_summary_json(
        tmp_path / "summary.json",
        job_id=job_id,
        device="cpu",
        baseline_ms=1.0,
        enhanced_ms=2.0,
        baseline_num_boxes=1,
        enhanced_num_boxes=2,
        num_parameters_total=1,
        notes={"tiling": True},
    )
    for name in ["config.json", "baseline.json", "summary.json"]:
        path = tmp_path / name
        assert path.exists()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data
