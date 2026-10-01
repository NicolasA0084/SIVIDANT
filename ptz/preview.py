"""Anotaciones breves sobre video en vivo, independientes de la inferencia."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ptz.detector import Detection, draw_detections
from ptz.target_tracker import TrackedTarget, TrackState

ANNOTATION_TTL_S = 0.25


@dataclass(frozen=True)
class PreviewAnnotations:
    frame_shape: tuple[int, ...]
    received_at: float
    detections: tuple[Detection, ...]
    target: TrackedTarget | None
    state: TrackState
    dead_zone: tuple[float, float]


def draw_target(
    frame: NDArray[np.uint8], target: TrackedTarget | None, state: TrackState,
) -> None:
    if target is None:
        return
    detection = target.detection
    lost = state == TrackState.TEMPORARILY_LOST
    color = (0, 140, 255) if lost else (0, 215, 255)
    label = (f"TRACK {target.track_id} LOST {target.missed_frames}"
             if lost else f"TRACK {target.track_id}")
    cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 4)
    cv2.putText(
        frame, label, (detection.x1, min(frame.shape[0] - 8, detection.y2 + 24)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA,
    )


def draw_dead_zone(frame: NDArray[np.uint8], dead_zone: tuple[float, float]) -> None:
    height, width = frame.shape[:2]
    half_width, half_height = round(dead_zone[0] * width / 2), round(dead_zone[1] * height / 2)
    cx, cy = width // 2, height // 2
    cv2.rectangle(frame, (cx - half_width, cy - half_height),
                  (cx + half_width, cy + half_height), (255, 180, 0), 2)


def render_preview(
    frame: NDArray[np.uint8], annotations: PreviewAnnotations | None,
    *, timestamp: float,
) -> NDArray[np.uint8]:
    """Nunca reutiliza cajas caducadas ni de otra resolución."""
    if annotations is None:
        return frame
    fresh = (
        frame.shape == annotations.frame_shape
        and 0.0 <= timestamp - annotations.received_at <= ANNOTATION_TTL_S
    )
    display = draw_detections(frame, annotations.detections) if fresh else frame.copy()
    if fresh:
        draw_target(display, annotations.target, annotations.state)
    draw_dead_zone(display, annotations.dead_zone)
    return display
