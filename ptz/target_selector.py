"""Selección independiente de un objetivo a partir de detecciones de un frame."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from math import hypot, sqrt
import os
from typing import Iterable, Sequence

from ptz.detector import Detection

LOGGER = logging.getLogger(__name__)
DEFAULT_MIN_CONFIDENCE = 0.50
DEFAULT_CONFIDENCE_WEIGHT = 0.90
DEFAULT_CENTER_WEIGHT = 0.00
DEFAULT_SIZE_WEIGHT = 0.10


class TargetSelectorConfigurationError(ValueError):
    """La configuración o las dimensiones del frame no son válidas."""


@dataclass(frozen=True)
class SelectionWeights:
    """Pesos normalizados usados para ordenar candidatos."""

    confidence: float
    center: float
    size: float


@dataclass(frozen=True)
class TargetSelection:
    """Resultado explicable del último proceso de selección."""

    detection: Detection
    score: float
    center_distance: float
    relative_area: float


class TargetSelector:
    """Elige por score un candidato sin mantener estado temporal del objeto."""

    def __init__(
        self,
        *,
        allowed_classes: Iterable[str] | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        confidence_weight: float = DEFAULT_CONFIDENCE_WEIGHT,
        center_weight: float = DEFAULT_CENTER_WEIGHT,
        size_weight: float = DEFAULT_SIZE_WEIGHT,
    ) -> None:
        self._min_confidence = self._unit_value(min_confidence, "min_confidence")
        self._allowed_classes = self._normalize_classes(allowed_classes)
        self._weights = self._normalize_weights(
            confidence_weight, center_weight, size_weight
        )
        self._last_selection: TargetSelection | None = None
        self._last_candidate_count = 0
        LOGGER.info(
            "TargetSelector configurado: min_confidence=%.2f, clases=%s, "
            "pesos=(%.2f, %.2f, %.2f)",
            self._min_confidence,
            "ALL" if self._allowed_classes is None else sorted(self._allowed_classes),
            self._weights.confidence, self._weights.center, self._weights.size,
        )

    @classmethod
    def from_environment(
        cls, *, default_classes: Iterable[str] | None = None, **overrides: object
    ) -> "TargetSelector":
        raw_classes = os.getenv("DETECTCAM_TARGET_CLASSES")
        if raw_classes is None:
            allowed_classes = default_classes
        else:
            allowed_classes = [item.strip() for item in raw_classes.split(",") if item.strip()]
            if not allowed_classes:
                raise TargetSelectorConfigurationError(
                    "DETECTCAM_TARGET_CLASSES no puede estar vacía."
                )
        options = dict(overrides)
        options.setdefault("allowed_classes", allowed_classes)
        options.setdefault("min_confidence", cls._environment_float(
            "DETECTCAM_TARGET_MIN_CONFIDENCE", DEFAULT_MIN_CONFIDENCE))
        options.setdefault("confidence_weight", cls._environment_float(
            "DETECTCAM_TARGET_CONFIDENCE_WEIGHT", DEFAULT_CONFIDENCE_WEIGHT))
        options.setdefault("center_weight", cls._environment_float(
            "DETECTCAM_TARGET_CENTER_WEIGHT", DEFAULT_CENTER_WEIGHT))
        options.setdefault("size_weight", cls._environment_float(
            "DETECTCAM_TARGET_SIZE_WEIGHT", DEFAULT_SIZE_WEIGHT))
        return cls(**options)

    @property
    def allowed_classes(self) -> frozenset[str] | None:
        return self._allowed_classes

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    def configure_tracking(self, *, min_confidence: float, allowed_classes: Iterable[str]) -> None:
        """Aplica ajustes en el hilo de visión; ninguna clase pausa la selección."""
        confidence = self._unit_value(min_confidence, "min_confidence")
        classes = tuple(allowed_classes)
        normalized = self._normalize_classes(classes) if classes else frozenset()
        self._min_confidence = confidence
        self._allowed_classes = normalized
        self._last_selection = None
        self._last_candidate_count = 0

    @property
    def weights(self) -> SelectionWeights:
        return self._weights

    @property
    def last_selection(self) -> TargetSelection | None:
        return self._last_selection

    @property
    def last_score(self) -> float | None:
        return None if self._last_selection is None else self._last_selection.score

    @property
    def last_candidate_count(self) -> int:
        return self._last_candidate_count

    def select(
        self, detections: Sequence[Detection], frame_shape: Sequence[int]
    ) -> Detection | None:
        height, width = self._frame_dimensions(frame_shape)
        frame_area = width * height
        half_diagonal = hypot(width / 2, height / 2)
        selections: list[TargetSelection] = []

        for detection in detections:
            if not self._is_candidate(detection, width, height):
                continue
            center_distance = min(1.0, hypot(
                detection.center_x - width / 2,
                detection.center_y - height / 2,
            ) / half_diagonal)
            relative_area = min(1.0, detection.area / frame_area)
            score = (
                self._weights.confidence * detection.confidence
                + self._weights.center * (1.0 - center_distance)
                + self._weights.size * sqrt(relative_area)
            )
            selection = TargetSelection(
                detection=detection,
                score=score,
                center_distance=center_distance,
                relative_area=relative_area,
            )
            selections.append(selection)
            LOGGER.debug("Candidato %s: score=%.4f", detection.class_name, score)

        self._last_candidate_count = len(selections)
        self._last_selection = max(selections, key=lambda item: item.score, default=None)
        return None if self._last_selection is None else self._last_selection.detection

    def _is_candidate(self, detection: Detection, width: int, height: int) -> bool:
        if detection.confidence < self._min_confidence:
            return False
        if (self._allowed_classes is not None
                and detection.class_name.casefold() not in self._allowed_classes):
            return False
        return (
            detection.x1 >= 0 and detection.y1 >= 0
            and detection.x2 > detection.x1 and detection.y2 > detection.y1
            and detection.x2 <= width and detection.y2 <= height
            and detection.area > 0
        )

    @staticmethod
    def _frame_dimensions(frame_shape: Sequence[int]) -> tuple[int, int]:
        if len(frame_shape) < 2:
            raise TargetSelectorConfigurationError("frame_shape debe incluir alto y ancho.")
        try:
            height, width = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError) as error:
            raise TargetSelectorConfigurationError(
                "El alto y ancho del frame deben ser enteros."
            ) from error
        if height <= 0 or width <= 0:
            raise TargetSelectorConfigurationError("El frame debe tener dimensiones positivas.")
        return height, width

    @staticmethod
    def _normalize_classes(classes: Iterable[str] | None) -> frozenset[str] | None:
        if classes is None:
            return None
        normalized = frozenset(str(item).strip().casefold() for item in classes if str(item).strip())
        if not normalized:
            raise TargetSelectorConfigurationError("allowed_classes no puede estar vacío.")
        return normalized

    @classmethod
    def _normalize_weights(cls, confidence: float, center: float, size: float) -> SelectionWeights:
        values = tuple(cls._nonnegative(value, name) for value, name in (
            (confidence, "confidence_weight"), (center, "center_weight"),
            (size, "size_weight")))
        total = sum(values)
        if total <= 0:
            raise TargetSelectorConfigurationError("Al menos un peso debe ser mayor que cero.")
        return SelectionWeights(*(value / total for value in values))

    @staticmethod
    def _unit_value(value: float, name: str) -> float:
        result = TargetSelector._number(value, name)
        if not 0.0 <= result <= 1.0:
            raise TargetSelectorConfigurationError(f"{name} debe estar entre 0.0 y 1.0.")
        return result

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        result = TargetSelector._number(value, name)
        if result < 0:
            raise TargetSelectorConfigurationError(f"{name} no puede ser negativo.")
        return result

    @staticmethod
    def _number(value: float, name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as error:
            raise TargetSelectorConfigurationError(f"{name} debe ser numérico.") from error

    @staticmethod
    def _environment_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError as error:
            raise TargetSelectorConfigurationError(f"{name} debe ser numérico.") from error
