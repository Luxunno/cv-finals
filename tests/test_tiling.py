import numpy as np

from core.tiling import remap_box_xyxy, tile_image


def test_tile_image_full_coverage_and_overlap():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    tiles = tile_image(img, tile_size=64, overlap=0.2)
    assert tiles  # non-empty
    covered = np.zeros((100, 120), dtype=np.uint8)
    for _, x0, y0 in tiles:
        x1 = min(x0 + 64, 120)
        y1 = min(y0 + 64, 100)
        covered[y0:y1, x0:x1] = 1
    assert covered.all()


def test_tile_image_small_image_single_tile():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    tiles = tile_image(img, tile_size=64, overlap=0.2)
    assert len(tiles) == 1
    tile, x0, y0 = tiles[0]
    assert x0 == 0 and y0 == 0
    assert tile.shape[0] == 10 and tile.shape[1] == 10


def test_remap_box_within_bounds():
    local_box = (5, 6, 20, 30)
    x0, y0 = 40, 50
    global_box = remap_box_xyxy(local_box, x0, y0)
    assert global_box == (45, 56, 60, 80)


def test_tile_image_no_duplicate_starts_expected_two_tiles():
    img = np.zeros((512, 683, 3), dtype=np.uint8)
    tiles = tile_image(img, tile_size=640, overlap=0.2)
    starts = {(x0, y0) for _, x0, y0 in tiles}
    assert len(starts) == len(tiles)
    # Height<=tile_size -> only y0=0; Width slightly>tile_size -> expect x0 in {0, 43}
    assert len(tiles) == 2
