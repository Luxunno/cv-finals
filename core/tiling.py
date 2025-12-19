"""Image tiling and box remapping."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TileCrop:
    crop_bgr: np.ndarray
    crop_x0: int
    crop_y0: int
    eff_x0: int
    eff_y0: int
    eff_x1: int
    eff_y1: int
    tile_x0: int
    tile_y0: int


def tile_image_padded(
    image_bgr: np.ndarray, tile_size: int, overlap: float, pad_px: int
) -> tuple[list[TileCrop], list[int], list[int]]:
    """Create padded crops for each tile with an 'effective' (unpadded) region.

    Returns (tiles, x_starts, y_starts) where x_starts/y_starts are tile starts.
    """
    h, w = image_bgr.shape[:2]
    stride = max(1, int(tile_size * (1 - overlap)))
    x_starts = _compute_starts(w, tile_size, stride)
    y_starts = _compute_starts(h, tile_size, stride)

    tiles: list[TileCrop] = []
    for y0 in y_starts:
        for x0 in x_starts:
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)

            crop_x0 = max(0, x0 - pad_px)
            crop_y0 = max(0, y0 - pad_px)
            crop_x1 = min(w, x1 + pad_px)
            crop_y1 = min(h, y1 + pad_px)

            crop = image_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            tiles.append(
                TileCrop(
                    crop_bgr=crop,
                    crop_x0=crop_x0,
                    crop_y0=crop_y0,
                    eff_x0=x0,
                    eff_y0=y0,
                    eff_x1=x1,
                    eff_y1=y1,
                    tile_x0=x0,
                    tile_y0=y0,
                )
            )
    return tiles, x_starts, y_starts


def box_center_in_region(
    box_xyxy: tuple[float, float, float, float],
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> bool:
    """Return True if box center is inside [x0,x1]x[y0,y1]."""
    bx1, by1, bx2, by2 = box_xyxy
    cx = (bx1 + bx2) / 2.0
    cy = (by1 + by2) / 2.0
    return (x0 <= cx <= x1) and (y0 <= cy <= y1)
