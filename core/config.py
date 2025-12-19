"""Configuration definitions."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class JobConfig:
    """Job-level inference configuration."""

    conf_threshold: float = 0.25
    global_conf_threshold: float = 0.25
    tile_conf_threshold: float = 0.18
    nms_iou: float = 0.5
    tile_size: int = 640
    overlap: float = 0.2
    tile_pad_px: int = 48
    tile_scale: float = 1.0
    wbf_iou: float = 0.55
    small_area_thresh: int = 1024
    wbf_iou_small: float = 0.45
    wbf_iou_normal: float = 0.55
    max_det: int = 300
    img_max_side: int = 1280

    def to_dict(self) -> Dict[str, Any]:
        """Return full config including deprecated fields."""
        return asdict(self)

    def effective_snapshot(self, device: Optional[str] = None) -> Dict[str, Any]:
        """Return effective fields used by inference."""
        snap = {
            "global_conf_threshold": self.global_conf_threshold,
            "tile_conf_threshold": self.tile_conf_threshold,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "tile_pad_px": self.tile_pad_px,
            "tile_scale": self.tile_scale,
            "wbf_iou_normal": self.wbf_iou_normal,
            "wbf_iou_small": self.wbf_iou_small,
            "small_area_thresh": self.small_area_thresh,
            "nms_iou": self.nms_iou,
            "max_det": self.max_det,
        }
        if device is not None:
            snap["device"] = device
        return snap

    def deprecated_mismatch(self) -> Dict[str, Dict[str, Any]]:
        """Detect mismatches between deprecated and effective fields."""
        mismatches: Dict[str, Dict[str, Any]] = {}
        if self.conf_threshold != self.global_conf_threshold:
            mismatches["conf_threshold"] = {
                "deprecated": self.conf_threshold,
                "effective": self.global_conf_threshold,
            }
        if self.wbf_iou != self.wbf_iou_normal:
            mismatches["wbf_iou"] = {
                "deprecated": self.wbf_iou,
                "effective": self.wbf_iou_normal,
            }
        return mismatches
