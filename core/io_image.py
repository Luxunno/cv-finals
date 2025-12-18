"""Image IO helpers with Unicode path support."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None")
    if image.dtype != np.uint8:
        return image.astype(np.uint8)
    return image


def imread_any(path: Path) -> np.ndarray:
    """Read image from any (Unicode) path, returning BGR np.uint8."""
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def imwrite_any(path: Path, image_bgr: np.ndarray) -> None:
    """Write image to any (Unicode) path using OpenCV encode."""
    path = Path(path)
    image_bgr = _ensure_uint8_image(image_bgr)
    ext = path.suffix or ".png"
    success, buf = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError(f"Failed to encode image with ext {ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(path)
