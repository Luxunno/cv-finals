import numpy as np

from core.tiling import box_center_in_region, remap_box_xyxy, tile_image, tile_image_padded


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


def test_tile_image_padded_crop_and_effective_region():
    img = np.zeros((50, 60, 3), dtype=np.uint8)
    tiles, xs, ys = tile_image_padded(img, tile_size=40, overlap=0.2, pad_px=8)
    assert xs == sorted(set(xs))
    assert ys == sorted(set(ys))
    assert len(tiles) == len(xs) * len(ys)
    for t in tiles:
        # Crop top-left should be <= effective top-left
        assert t.crop_x0 <= t.eff_x0
        assert t.crop_y0 <= t.eff_y0
        # Effective region should be within image bounds
        assert 0 <= t.eff_x0 <= t.eff_x1 <= 60
        assert 0 <= t.eff_y0 <= t.eff_y1 <= 50
        # Crop image shape matches crop coordinates
        crop_h, crop_w = t.crop_bgr.shape[:2]
        assert crop_w == (min(60, t.eff_x1 + 8) - max(0, t.eff_x0 - 8))
        assert crop_h == (min(50, t.eff_y1 + 8) - max(0, t.eff_y0 - 8))


def test_box_center_in_region_filter():
    # box center at (5,5) should be inside region [0,0]-[10,10]
    assert box_center_in_region((4.0, 4.0, 6.0, 6.0), 0.0, 0.0, 10.0, 10.0)
    # box center at (15,15) outside
    assert not box_center_in_region((14.0, 14.0, 16.0, 16.0), 0.0, 0.0, 10.0, 10.0)
