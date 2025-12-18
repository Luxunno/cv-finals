from core.detector_yolo import Detection
from core.wbf import weighted_boxes_fusion


def det(x1, y1, x2, y2, score, label):
    return Detection(
        label=label,
        class_id=0,
        score=score,
        box_xyxy=(x1, y1, x2, y2),
        attrs={},
    )


def test_wbf_fuses_overlapping_same_label():
    detections = [
        det(0, 0, 10, 10, 0.9, "person"),
        det(1, 1, 11, 11, 0.6, "person"),
    ]
    fused = weighted_boxes_fusion(detections, iou_thresh=0.5)
    assert len(fused) == 1
    assert fused[0].label == "person"
    assert fused[0].score == 0.9


def test_wbf_keeps_non_overlapping():
    detections = [
        det(0, 0, 5, 5, 0.9, "person"),
        det(20, 20, 25, 25, 0.8, "person"),
    ]
    fused = weighted_boxes_fusion(detections, iou_thresh=0.5)
    assert len(fused) == 2


def test_wbf_different_labels_not_fused():
    detections = [
        det(0, 0, 10, 10, 0.9, "person"),
        det(1, 1, 11, 11, 0.8, "car"),
    ]
    fused = weighted_boxes_fusion(detections, iou_thresh=0.5)
    assert len(fused) == 2
