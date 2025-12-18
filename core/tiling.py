"""Image tiling and box remapping."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def _compute_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    starts.append(length - tile_size)
    # remove duplicates then sort
    starts = sorted(set(starts))
    return starts


def tile_image(image_bgr: np.ndarray, tile_size: int, overlap: float) -> List[tuple[np.ndarray, int, int]]:
    """Split image into overlapping tiles covering the whole image.

    Returns list of (tile_image, x0, y0) where (x0, y0) is top-left in original image.
    """
    h, w = image_bgr.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))

    xs = _compute_starts(w, tile_size, stride)
    ys = _compute_starts(h, tile_size, stride)

    tiles: list[tuple[np.ndarray, int, int]] = []
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            tile = image_bgr[y0:y1, x0:x1].copy()
            tiles.append((tile, x0, y0))
    return tiles


def remap_box_xyxy(box_xyxy: tuple[int, int, int, int], x0: int, y0: int) -> tuple[int, int, int, int]:
    """Map tile-local box back to global coordinates."""
    x1, y1, x2, y2 = box_xyxy
    return x1 + x0, y1 + y0, x2 + x0, y2 + y0
