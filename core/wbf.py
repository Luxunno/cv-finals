"""Weighted Boxes Fusion for detections."""

from __future__ import annotations

from typing import Iterable, List

from core.detector_yolo import Detection
from core.postprocess import iou_xyxy


def weighted_boxes_fusion(detections: Iterable[Detection], iou_thresh: float) -> List[Detection]:
    """Fuse detections per label using weighted boxes fusion."""
    clusters_by_label: dict[str, list[list[Detection]]] = {}
    for det in detections:
        clusters = clusters_by_label.setdefault(det.label, [])
        matched = False
        for cluster in clusters:
            if _max_iou_with_cluster(det, cluster) >= iou_thresh:
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


def _max_iou_with_cluster(det: Detection, cluster: list[Detection]) -> float:
    return max(iou_xyxy(det.box_xyxy, other.box_xyxy) for other in cluster)


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
