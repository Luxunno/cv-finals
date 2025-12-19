"""Weighted Boxes Fusion for detections."""

from __future__ import annotations

from typing import Iterable, List

from core.detector_yolo import Detection
from core.postprocess import iou_xyxy


def _box_area_xyxy(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def weighted_boxes_fusion(
    detections: Iterable[Detection],
    iou_thresh: float,
    *,
    small_area_thresh: int | None = None,
    iou_small: float | None = None,
    iou_normal: float | None = None,
) -> List[Detection]:
    """Fuse detections per label using weighted boxes fusion.

    If small_area_thresh/iou_small/iou_normal provided, uses a lower IoU threshold for small boxes.
    """
    if iou_normal is None:
        iou_normal = iou_thresh

    def pair_thresh(a: Detection, b: Detection) -> float:
        if small_area_thresh is None or iou_small is None:
            return iou_normal
        if min(_box_area_xyxy(a.box_xyxy), _box_area_xyxy(b.box_xyxy)) < small_area_thresh:
            return iou_small
        return iou_normal

    clusters_by_label: dict[str, list[list[Detection]]] = {}
    for det in detections:
        clusters = clusters_by_label.setdefault(det.label, [])
        matched = False
        for cluster in clusters:
            if _cluster_matches(det, cluster, pair_thresh):
                cluster.append(det)
                matched = True
                break
        if not matched:
            clusters.append([det])

    fused: list[Detection] = []
    for label, clusters in clusters_by_label.items():
        for cluster in clusters:
            fused.append(_fuse_cluster(cluster))
    fused.sort(key=lambda d: d.score, reverse=True)
    return fused


def _cluster_matches(det: Detection, cluster: list[Detection], pair_thresh) -> bool:
    for other in cluster:
        t = pair_thresh(det, other)
        if iou_xyxy(det.box_xyxy, other.box_xyxy) >= t:
            return True
    return False


def _fuse_cluster(cluster: list[Detection]) -> Detection:
    total_score = sum(det.score for det in cluster)
    if total_score == 0:
        total_score = 1e-9
    weighted_coords = [0.0, 0.0, 0.0, 0.0]
    for det in cluster:
        for i, v in enumerate(det.box_xyxy):
            weighted_coords[i] += det.score * v
    fused_coords = tuple(int(round(c / total_score)) for c in weighted_coords)  # type: ignore
    fused_score = max(det.score for det in cluster)
    first = cluster[0]
    return Detection(
        label=first.label,
        class_id=first.class_id,
        score=fused_score,
        box_xyxy=fused_coords,  # type: ignore
        attrs={},
    )
