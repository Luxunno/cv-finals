import numpy as np

from core.traffic_light_state import classify_traffic_light_state


def _make_img(color_bgr):
    img = np.zeros((40, 20, 3), dtype=np.uint8)
    img[:, :] = color_bgr
    return img


def test_classify_red():
    red_bgr = (0, 0, 255)
    img = _make_img(red_bgr)
    state = classify_traffic_light_state(img, (0, 0, 20, 40))
    assert state == "red"


def test_classify_green():
    green_bgr = (0, 255, 0)
    img = _make_img(green_bgr)
    state = classify_traffic_light_state(img, (0, 0, 20, 40))
    assert state == "green"


def test_classify_unknown_for_invalid_box():
    img = _make_img((0, 0, 0))
    state = classify_traffic_light_state(img, (10, 10, 5, 5))
    assert state == "unknown"
