"""Seguimiento temporal simple de una Detection, sin control PTZ ni video."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import logging
from math import hypot, isfinite
import os
from time import monotonic
from typing import Callable, Sequence

from ptz.detector import Detection

LOGGER = logging.getLogger(__name__)
DEFAULT_MAX_MISSED_FRAMES = 10
DEFAULT_MIN_IOU = 0.10
DEFAULT_MAX_CENTER_DISTANCE = 0.15
DEFAULT_MIN_ASSOCIATION_SCORE = 0.30
DEFAULT_IOU_WEIGHT = 0.65
DEFAULT_DISTANCE_WEIGHT = 0.30
DEFAULT_CONFIDENCE_WEIGHT = 0.05


class TargetTrackerConfigurationError(ValueError):
    """La configuración o las dimensiones recibidas no son válidas."""


class TrackState(str, Enum):
    """Estados posibles de una identidad local seguida."""

    NO_TARGET = "NO_TARGET"
    TRACKED = "TRACKED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"


@dataclass(frozen=True)
class AssociationWeights:
    """Pesos para relacionar la detección actual con el track anterior."""

    iou: float
    distance: float
    confidence: float


@dataclass(frozen=True)
class TargetAssociation:
    """Medidas que justifican una asociación temporal aceptada."""

    detection: Detection
    score: float
    iou: float
    normalized_center_distance: float


@dataclass(frozen=True)
class TrackedTarget:
    """Estado temporal del objetivo, incluida su velocidad observada."""

    track_id: int
    detection: Detection
    age_frames: int
    visible_frames: int
    missed_frames: int
    last_seen_timestamp: float
    previous_center: tuple[float, float] | None
    velocity_x: float
    velocity_y: float
    # velocity_x/y siguen siendo desplazamiento en píxeles por actualización.
    # Este intervalo permite convertirlos correctamente a píxeles/segundo.
    velocity_dt_s: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center_x, self.detection.center_y


def intersection_over_union(
    box_a: Detection | Sequence[float], box_b: Detection | Sequence[float]
) -> float:
    """Devuelve IoU en 0..1; una caja inválida produce 0.0."""
    first = _box_coordinates(box_a)
    second = _box_coordinates(box_b)
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height
    union = ((ax2 - ax1) * (ay2 - ay1)
             + (bx2 - bx1) * (by2 - by1) - intersection)
    if union <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def normalized_center_distance(
    first: Detection, second: Detection, frame_shape: Sequence[int]
) -> float:
    """Distancia de centros dividida entre la diagonal completa del frame."""
    height, width = _frame_dimensions(frame_shape)
    diagonal = hypot(width, height)
    distance = hypot(first.center_x - second.center_x,
                     first.center_y - second.center_y)
    return min(1.0, max(0.0, distance / diagonal))


class TargetTracker:
    """Mantiene un track local mientras asociaciones cercanas sean válidas."""

    def __init__(
        self,
        *,
        max_missed_frames: int = DEFAULT_MAX_MISSED_FRAMES,
        min_iou: float = DEFAULT_MIN_IOU,
        max_center_distance: float = DEFAULT_MAX_CENTER_DISTANCE,
        min_association_score: float = DEFAULT_MIN_ASSOCIATION_SCORE,
        iou_weight: float = DEFAULT_IOU_WEIGHT,
        distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
        confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
        allow_class_change: bool = False,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if isinstance(max_missed_frames, bool):
            raise TargetTrackerConfigurationError(
                "max_missed_frames debe ser un entero no negativo."
            )
        try:
            self._max_missed_frames = int(max_missed_frames)
        except (TypeError, ValueError) as error:
            raise TargetTrackerConfigurationError(
                "max_missed_frames debe ser un entero no negativo."
            ) from error
        if self._max_missed_frames != max_missed_frames or self._max_missed_frames < 0:
            raise TargetTrackerConfigurationError(
                "max_missed_frames debe ser un entero no negativo."
            )
        self._min_iou = self._unit(min_iou, "min_iou")
        self._max_center_distance = self._unit(
            max_center_distance, "max_center_distance"
        )
        self._min_association_score = self._unit(
            min_association_score, "min_association_score"
        )
        self._weights = self._normalized_weights(
            iou_weight, distance_weight, confidence_weight
        )
        if not isinstance(allow_class_change, bool):
            raise TargetTrackerConfigurationError("allow_class_change debe ser booleano.")
        self._allow_class_change = allow_class_change
        self._clock = clock
        self._current_target: TrackedTarget | None = None
        self._state = TrackState.NO_TARGET
        self._next_track_id = 1
        self._last_association: TargetAssociation | None = None
        LOGGER.info(
            "TargetTracker configurado: missed=%d, min_iou=%.2f, "
            "max_distance=%.2f, min_score=%.2f, class_change=%s",
            self._max_missed_frames, self._min_iou,
            self._max_center_distance, self._min_association_score,
            self._allow_class_change,
        )

    @classmethod
    def from_environment(cls, **overrides: object) -> "TargetTracker":
        options = dict(overrides)
        options.setdefault("max_missed_frames", cls._env_int(
            "DETECTCAM_TRACK_MAX_MISSED_FRAMES", DEFAULT_MAX_MISSED_FRAMES
        ))
        for key, variable, default in (
            ("min_iou", "DETECTCAM_TRACK_MIN_IOU", DEFAULT_MIN_IOU),
            ("max_center_distance", "DETECTCAM_TRACK_MAX_CENTER_DISTANCE",
             DEFAULT_MAX_CENTER_DISTANCE),
            ("min_association_score", "DETECTCAM_TRACK_MIN_ASSOCIATION_SCORE",
             DEFAULT_MIN_ASSOCIATION_SCORE),
            ("iou_weight", "DETECTCAM_TRACK_IOU_WEIGHT", DEFAULT_IOU_WEIGHT),
            ("distance_weight", "DETECTCAM_TRACK_DISTANCE_WEIGHT",
             DEFAULT_DISTANCE_WEIGHT),
            ("confidence_weight", "DETECTCAM_TRACK_CONFIDENCE_WEIGHT",
             DEFAULT_CONFIDENCE_WEIGHT),
        ):
            options.setdefault(key, cls._env_float(variable, default))
        options.setdefault("allow_class_change", cls._env_bool(
            "DETECTCAM_TRACK_ALLOW_CLASS_CHANGE", False
        ))
        return cls(**options)

    @property
    def current_target(self) -> TrackedTarget | None:
        return self._current_target

    @property
    def has_target(self) -> bool:
        return self._current_target is not None

    @property
    def state(self) -> TrackState:
        return self._state

    @property
    def max_missed_frames(self) -> int:
        return self._max_missed_frames

    @property
    def weights(self) -> AssociationWeights:
        return self._weights

    @property
    def last_association(self) -> TargetAssociation | None:
        return self._last_association

    def update(
        self,
        detections: Sequence[Detection],
        selected_candidate: Detection | None,
        frame_shape: Sequence[int],
        *,
        timestamp: float | None = None,
    ) -> TrackedTarget | None:
        """Actualiza el track; el candidato sólo se usa cuando no hay target."""
        height, width = _frame_dimensions(frame_shape)
        now = self._clock() if timestamp is None else self._timestamp(timestamp)
        valid_detections = [
            detection for detection in detections
            if _valid_detection(detection, width, height)
        ]

        if self._current_target is None:
            self._last_association = None
            if selected_candidate is None or not _valid_detection(
                selected_candidate, width, height
            ):
                self._state = TrackState.NO_TARGET
                return None
            self._current_target = TrackedTarget(
                track_id=self._next_track_id,
                detection=selected_candidate,
                age_frames=1,
                visible_frames=1,
                missed_frames=0,
                last_seen_timestamp=now,
                previous_center=None,
                velocity_x=0.0,
                velocity_y=0.0,
                velocity_dt_s=None,
            )
            self._next_track_id += 1
            self._state = TrackState.TRACKED
            LOGGER.info("Nuevo track local: id=%d, clase=%s",
                        self._current_target.track_id,
                        self._current_target.detection.class_name)
            return self._current_target

        association = self._best_association(valid_detections, frame_shape)
        if association is not None and self._passes_gating(association):
            previous = self._current_target
            detection = association.detection
            elapsed = now - previous.last_seen_timestamp
            velocity_dt_s = elapsed if isfinite(elapsed) and elapsed > 0.0 else None
            self._current_target = TrackedTarget(
                track_id=previous.track_id,
                detection=detection,
                age_frames=previous.age_frames + 1,
                visible_frames=previous.visible_frames + 1,
                missed_frames=0,
                last_seen_timestamp=now,
                previous_center=previous.center,
                velocity_x=detection.center_x - previous.center[0],
                velocity_y=detection.center_y - previous.center[1],
                velocity_dt_s=velocity_dt_s,
            )
            self._last_association = association
            self._state = TrackState.TRACKED
            return self._current_target

        missed = self._current_target.missed_frames + 1
        self._last_association = None
        if missed > self._max_missed_frames:
            lost_id = self._current_target.track_id
            self._current_target = None
            self._state = TrackState.NO_TARGET
            LOGGER.info("Track %d perdido definitivamente", lost_id)
            return None
        self._current_target = replace(
            self._current_target,
            age_frames=self._current_target.age_frames + 1,
            missed_frames=missed,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_dt_s=None,
        )
        self._state = TrackState.TEMPORARILY_LOST
        return self._current_target

    def reset(self) -> None:
        """Descarta el target actual sin reutilizar identificadores anteriores."""
        self._current_target = None
        self._last_association = None
        self._state = TrackState.NO_TARGET

    def _best_association(
        self, detections: Sequence[Detection], frame_shape: Sequence[int]
    ) -> TargetAssociation | None:
        assert self._current_target is not None
        associations: list[tuple[int, TargetAssociation]] = []
        for index, detection in enumerate(detections):
            if not self._compatible_class(
                self._current_target.detection, detection
            ):
                continue
            overlap = intersection_over_union(
                self._current_target.detection, detection
            )
            distance = normalized_center_distance(
                self._current_target.detection, detection, frame_shape
            )
            score = (
                self._weights.iou * overlap
                + self._weights.distance * (1.0 - distance)
                + self._weights.confidence * detection.confidence
            )
            associations.append((index, TargetAssociation(
                detection=detection,
                score=score,
                iou=overlap,
                normalized_center_distance=distance,
            )))
        if not associations:
            return None
        # El índice negativo hace que un empate exacto conserve el primer input.
        return max(associations, key=lambda item: (
            item[1].score, item[1].iou,
            -item[1].normalized_center_distance,
            item[1].detection.confidence, -item[0],
        ))[1]

    def _passes_gating(self, association: TargetAssociation) -> bool:
        geometric_gate = (
            association.iou >= self._min_iou
            or association.normalized_center_distance <= self._max_center_distance
        )
        return geometric_gate and association.score >= self._min_association_score

    def _compatible_class(self, previous: Detection, current: Detection) -> bool:
        return self._allow_class_change or previous.class_id == current.class_id

    @staticmethod
    def _normalized_weights(iou: float, distance: float,
                            confidence: float) -> AssociationWeights:
        values = tuple(TargetTracker._nonnegative(value, name) for value, name in (
            (iou, "iou_weight"), (distance, "distance_weight"),
            (confidence, "confidence_weight"),
        ))
        total = sum(values)
        if total <= 0:
            raise TargetTrackerConfigurationError(
                "Al menos un peso de asociación debe ser mayor que cero."
            )
        return AssociationWeights(*(value / total for value in values))

    @staticmethod
    def _unit(value: float, name: str) -> float:
        result = TargetTracker._number(value, name)
        if not 0.0 <= result <= 1.0:
            raise TargetTrackerConfigurationError(
                f"{name} debe estar entre 0.0 y 1.0."
            )
        return result

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        result = TargetTracker._number(value, name)
        if result < 0:
            raise TargetTrackerConfigurationError(f"{name} no puede ser negativo.")
        return result

    @staticmethod
    def _number(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise TargetTrackerConfigurationError(f"{name} debe ser numérico.") from error
        if not isfinite(result):
            raise TargetTrackerConfigurationError(f"{name} debe ser finito.")
        return result

    @staticmethod
    def _timestamp(value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise TargetTrackerConfigurationError("timestamp debe ser numérico.") from error
        if not isfinite(result) or result < 0:
            raise TargetTrackerConfigurationError(
                "timestamp debe ser finito y no negativo."
            )
        return result

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError as error:
            raise TargetTrackerConfigurationError(f"{name} debe ser entero.") from error

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError as error:
            raise TargetTrackerConfigurationError(f"{name} debe ser numérico.") from error

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        normalized = value.strip().casefold()
        if normalized in ("1", "true", "yes", "on", "si", "sí"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise TargetTrackerConfigurationError(f"{name} debe ser true o false.")


def _box_coordinates(
    value: Detection | Sequence[float],
) -> tuple[float, float, float, float] | None:
    try:
        if isinstance(value, Detection):
            coordinates = (value.x1, value.y1, value.x2, value.y2)
        else:
            if len(value) != 4:
                return None
            coordinates = tuple(value)
        x1, y1, x2, y2 = (float(item) for item in coordinates)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _frame_dimensions(frame_shape: Sequence[int]) -> tuple[int, int]:
    if len(frame_shape) < 2:
        raise TargetTrackerConfigurationError(
            "frame_shape debe incluir alto y ancho."
        )
    try:
        height, width = int(frame_shape[0]), int(frame_shape[1])
    except (TypeError, ValueError) as error:
        raise TargetTrackerConfigurationError(
            "El alto y ancho del frame deben ser enteros."
        ) from error
    if height <= 0 or width <= 0:
        raise TargetTrackerConfigurationError(
            "El frame debe tener dimensiones positivas."
        )
    return height, width


def _valid_detection(detection: Detection, width: int, height: int) -> bool:
    if not isinstance(detection, Detection):
        return False
    if not 0.0 <= detection.confidence <= 1.0:
        return False
    return (
        detection.x1 >= 0 and detection.y1 >= 0
        and detection.x2 > detection.x1 and detection.y2 > detection.y1
        and detection.x2 <= width and detection.y2 <= height
        and detection.area > 0
    )
