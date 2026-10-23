"""Desplazamiento de fondo con Lucas–Kanade y consenso robusto de OpenCV C++.

No se confunde el movimiento del dron con movimiento de la cámara: se enmascaran
las cajas y se exige un conjunto suficiente de puntos de fondo coherentes.
En cielo sin textura el resultado se marca desconocido; no se inventa odometría.
Sólo se estima traslación local, no una pose PTZ absoluta ni zoom calibrado.
"""
from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np


@dataclass(frozen=True)
class BackgroundShift:
    dx: float = 0.0
    dy: float = 0.0
    reliable: bool = False
    inliers: int = 0


class CameraMotionEstimator:
    def __init__(self, max_width: int = 480, min_points: int = 10) -> None:
        self.max_width = max_width
        self.min_points = min_points
        self._previous = None
        self._points = None
        self._shape = None
        self._timestamp = None

    def reset(self) -> None:
        self._previous = self._points = self._shape = self._timestamp = None

    def update(self, frame, *, captured_at: float, boxes=()) -> BackgroundShift:
        height, width = frame.shape[:2]
        ratio = min(1.0, self.max_width / width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if ratio < 1:
            gray = cv2.resize(gray, (round(width*ratio), round(height*ratio)))
        result = BackgroundShift()
        valid_interval = (self._timestamp is not None and 0 < captured_at-self._timestamp <= .6)
        if (self._previous is not None and self._shape == frame.shape and valid_interval
                and self._points is not None and len(self._points) >= self.min_points):
            current, status, _error = cv2.calcOpticalFlowPyrLK(
                self._previous, gray, self._points, None, winSize=(21, 21), maxLevel=3)
            if current is not None and status is not None:
                good = status.reshape(-1).astype(bool)
                old = self._points.reshape(-1, 2)[good]
                new = current.reshape(-1, 2)[good]
                if len(old) >= self.min_points:
                    deltas = new - old
                    median = np.median(deltas, axis=0)
                    residual = np.linalg.norm(deltas-median, axis=1)
                    inliers = residual < 2.5
                    count = int(inliers.sum())
                    if count >= self.min_points and count / len(old) >= .6:
                        dx, dy = np.median(deltas[inliers], axis=0) / ratio
                        if isfinite(dx) and isfinite(dy) and abs(dx) < width*.35 and abs(dy) < height*.35:
                            result = BackgroundShift(float(dx), float(dy), True, count)
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = (box.x1, box.y1, box.x2, box.y2)
            cv2.rectangle(mask, (max(0, round(x1*ratio)-8), max(0, round(y1*ratio)-8)),
                          (round(x2*ratio)+8, round(y2*ratio)+8), 0, -1)
        self._points = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=.02,
                                             minDistance=10, mask=mask)
        self._previous, self._shape, self._timestamp = gray, frame.shape, captured_at
        return result
