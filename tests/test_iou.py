from core.detector_yolo import Detection
from core.postprocess import iou_xyxy, nms


def make_det(x1, y1, x2, y2, score=1.0, label="person"):
    return Detection(
        label=label,
        class_id=0,
        score=score,
        box_xyxy=(x1, y1, x2, y2),
        attrs={},
    )


def test_iou_symmetry_and_bounds():
    a = (0.0, 0.0, 2.0, 2.0)
    b = (1.0, 1.0, 3.0, 3.0)
    iou_ab = iou_xyxy(a, b)
    iou_ba = iou_xyxy(b, a)
    assert iou_ab == iou_ba
    assert 0.0 <= iou_ab <= 1.0


def test_iou_no_overlap_and_full_overlap():
    a = (0.0, 0.0, 1.0, 1.0)
    b = (2.0, 2.0, 3.0, 3.0)
    c = (0.0, 0.0, 1.0, 1.0)
    assert iou_xyxy(a, b) == 0.0
    assert iou_xyxy(a, c) == 1.0


def test_nms_suppresses_same_label():
    dets = [
        make_det(0, 0, 2, 2, score=0.9, label="person"),
        make_det(0, 0, 2, 2, score=0.5, label="person"),
    ]
    kept = nms(dets, iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].score == 0.9


def test_nms_keeps_different_labels():
    dets = [
        make_det(0, 0, 2, 2, score=0.9, label="person"),
        make_det(0, 0, 2, 2, score=0.8, label="car"),
    ]
    kept = nms(dets, iou_thresh=0.1)
    assert len(kept) == 2
