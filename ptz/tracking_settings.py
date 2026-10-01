"""Ajustes de seguimiento compartidos entre la interfaz, visión y control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import threading
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TrackingSettings:
    min_confidence: float
    allowed_classes: frozenset[str]
    revision: int = 0


class TrackingOptions:
    """Publica una configuración completa y coherente por cada aplicación."""

    def __init__(
        self, class_names: Mapping[int, str] | None = None, *,
        min_confidence: float = 0.5, allowed_classes: Iterable[str] | None = None,
    ) -> None:
        self._class_names = dict(class_names or {})
        self._known = frozenset(name.strip().casefold() for name in self._class_names.values())
        selected = self._known if allowed_classes is None else frozenset(
            name.strip().casefold() for name in allowed_classes
        ) & self._known
        self._lock = threading.Lock()
        self._snapshot = TrackingSettings(self._confidence(min_confidence), selected)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def snapshot(self) -> TrackingSettings:
        with self._lock:
            return self._snapshot

    def update(self, *, min_confidence: float, allowed_classes: Iterable[str]) -> TrackingSettings:
        confidence = self._confidence(min_confidence)
        selected = frozenset(name.strip().casefold() for name in allowed_classes)
        if not selected <= self._known:
            raise ValueError("Selecciona únicamente clases disponibles en el modelo.")
        with self._lock:
            current = self._snapshot
            if (confidence, selected) != (current.min_confidence, current.allowed_classes):
                self._snapshot = TrackingSettings(confidence, selected, current.revision + 1)
            return self._snapshot

    @staticmethod
    def _confidence(value: float) -> float:
        result = float(value)
        if not isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError("La confianza mínima debe estar entre 0 y 1.")
        return result
