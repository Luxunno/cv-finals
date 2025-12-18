"""Configuration definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JobConfig:
    """Job-level inference configuration."""

    conf_threshold: float = 0.25
    nms_iou: float = 0.5
    tile_size: int = 640
    overlap: float = 0.2
    wbf_iou: float = 0.55
    max_det: int = 300
    img_max_side: int = 1280
