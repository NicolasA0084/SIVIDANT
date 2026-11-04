"""Detección en CPU con ventanas multiescala, HOG/HSV y un SVM calibrado.

La búsqueda genera cajas sin YOLO. La confianza procede exclusivamente de
predict_proba del modelo calibrado, nunca de una transformación manual de márgenes.
"""

from __future__ import annotations

import logging
from math import isfinite
import os
from time import perf_counter
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from ptz.detector import Detection, DetectorConfigurationError, InferenceError, ModelLoadError
from ptz.joblib_classifier import JoblibClassifier, load_feature_extractor, load_labels
from ptz.target_tracker import intersection_over_union

LOGGER = logging.getLogger(__name__)


class SVMDetector:
    """Clasifica un lote de ventanas y elimina fondo y cajas redundantes."""

    def __init__(
        self,
        classifier: JoblibClassifier,
        feature_extractor: Callable[[NDArray[np.uint8]], NDArray],
        *,
        confidence: float = 0.5,
        scan_width: int = 320,
        window_sizes: Sequence[int] = (32, 64, 128),
        stride_ratio: float = 0.5,
        nms_iou: float = 0.35,
        max_detections: int = 30,
    ) -> None:
        if not classifier.has_probabilities:
            raise ModelLoadError(
                "El detector SVM necesita probabilidades calibradas; el LinearSVC "
                "original sólo devuelve márgenes. Ejecute python -m ptz.calibrate_svm "
                "y configure DETECTCAM_CLASSIFIER_PATH con el modelo calibrado."
            )
        names = classifier.class_names
        self._background_ids = {
            class_id for class_id, name in names.items()
            if name.strip().casefold() == "background"
        }
        self._class_names = {
            class_id: name for class_id, name in names.items()
            if class_id not in self._background_ids
        }
        if not self._background_ids or not self._class_names:
            raise ModelLoadError("El detector SVM necesita una clase background y clases de objetos.")
        if not callable(feature_extractor):
            raise DetectorConfigurationError("El extractor de características debe ser una función.")
        self._classifier = classifier
        self._extract_features = feature_extractor
        self._confidence = self._unit(confidence, "confidence")
        self._scan_width = self._positive_integer(scan_width, "scan_width")
        try:
            if isinstance(window_sizes, (str, bytes)):
                raise TypeError
            self._window_sizes = tuple(dict.fromkeys(
                self._positive_integer(size, "window_sizes") for size in window_sizes
            ))
        except TypeError as error:
            raise DetectorConfigurationError("window_sizes debe ser una secuencia de enteros positivos.") from error
        if not self._window_sizes:
            raise DetectorConfigurationError("window_sizes no puede estar vacío.")
        self._stride_ratio = self._unit(stride_ratio, "stride_ratio")
        if self._stride_ratio == 0.0:
            raise DetectorConfigurationError("stride_ratio debe ser mayor que cero.")
        self._nms_iou = self._unit(nms_iou, "nms_iou")
        self._max_detections = self._positive_integer(max_detections, "max_detections")
        self._last_inference_ms: float | None = None
        LOGGER.info(
            "Detector SVM calibrado en CPU: ancho de búsqueda=%d, ventanas=%s, confianza=%.2f",
            self._scan_width, self._window_sizes, self._confidence,
        )

    @classmethod
    def from_environment(cls, **overrides: Any) -> "SVMDetector":
        options = dict(overrides)
        try:
            if "confidence" not in options:
                options["confidence"] = float(os.getenv("DETECTCAM_CONFIDENCE", "0.25"))
            if "scan_width" not in options:
                options["scan_width"] = int(os.getenv("DETECTCAM_SVM_SCAN_WIDTH", "320"))
            if "window_sizes" not in options:
                options["window_sizes"] = tuple(
                    int(value.strip()) for value in os.getenv(
                        "DETECTCAM_SVM_WINDOW_SIZES", "32,64,128",
                    ).split(",")
                )
        except ValueError as error:
            raise DetectorConfigurationError(
                "Revise DETECTCAM_CONFIDENCE (número), DETECTCAM_SVM_SCAN_WIDTH "
                "(entero) y DETECTCAM_SVM_WINDOW_SIZES (enteros separados por comas)."
            ) from error
        if "classifier" not in options:
            model_path = os.getenv("DETECTCAM_CLASSIFIER_PATH", "").strip()
            labels_path = os.getenv("DETECTCAM_CLASSIFIER_LABELS_PATH", "").strip()
            if not model_path or not labels_path:
                raise ModelLoadError(
                    "Configure DETECTCAM_CLASSIFIER_PATH y DETECTCAM_CLASSIFIER_LABELS_PATH "
                    "para usar el detector SVM."
                )
            options["classifier"] = JoblibClassifier(model_path, labels=load_labels(labels_path))
        if "feature_extractor" not in options:
            features_path = os.getenv("DETECTCAM_CLASSIFIER_FEATURES_PATH", "").strip()
            if not features_path:
                raise ModelLoadError("Configure DETECTCAM_CLASSIFIER_FEATURES_PATH para usar el detector SVM.")
            options["feature_extractor"] = load_feature_extractor(features_path)
        return cls(**options)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def confidence(self) -> float:
        return self._confidence

    def set_confidence(self, value: float) -> None:
        self._confidence = self._unit(value, "confidence")

    @property
    def last_inference_ms(self) -> float | None:
        return self._last_inference_ms

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
            raise DetectorConfigurationError("El frame debe ser una imagen BGR uint8 no vacía (alto, ancho, 3).")
        started = perf_counter()
        try:
            height, width = frame.shape[:2]
            scan_width = min(width, self._scan_width)
            scan_height = max(1, round(height * scan_width / width))
            image = (frame if scan_width == width else cv2.resize(
                frame, (scan_width, scan_height), interpolation=cv2.INTER_AREA,
            ))
            scale_x, scale_y = width / scan_width, height / scan_height
            windows: list[tuple[int, int, int, int]] = []
            vectors = []
            # Para cuadros menores que una escala se usa el mayor cuadrado que
            # cabe; deduplicar evita repetirlo para las escalas restantes.
            sizes = dict.fromkeys(min(size, scan_width, scan_height) for size in self._window_sizes)
            for size in sizes:
                stride = max(1, round(size * self._stride_ratio))
                for y in self._starts(scan_height, size, stride):
                    for x in self._starts(scan_width, size, stride):
                        vector = np.asarray(self._extract_features(image[y:y + size, x:x + size].copy()))
                        if vector.shape != (self._classifier.n_features,):
                            raise InferenceError(
                                f"El extractor SVM debe devolver ({self._classifier.n_features},); "
                                f"devolvió {vector.shape}."
                            )
                        vectors.append(vector)
                        windows.append((x, y, x + size, y + size))
            if not windows:
                return []
            results = self._classifier.predict_features(np.stack(vectors))
            if len(results) != len(windows):
                raise InferenceError("El clasificador no devolvió una predicción por ventana.")
            detections = []
            for (x1, y1, x2, y2), result in zip(windows, results):
                probability = result.confidence
                if probability is None or not isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise InferenceError("El modelo SVM no devolvió una probabilidad válida por ventana.")
                if result.class_id in self._background_ids or probability < self._confidence:
                    continue
                detections.append(Detection(
                    class_id=result.class_id, class_name=result.class_name,
                    confidence=float(probability),
                    x1=round(x1 * scale_x), y1=round(y1 * scale_y),
                    x2=min(width, round(x2 * scale_x)), y2=min(height, round(y2 * scale_y)),
                    classification_margin=result.margin,
                ))
            return self._suppress(detections)
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("No fue posible completar la detección SVM por ventanas.") from error
        finally:
            self._last_inference_ms = (perf_counter() - started) * 1000.0

    def _suppress(self, detections: Sequence[Detection]) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                detection.class_id == previous.class_id
                and intersection_over_union(detection, previous) > self._nms_iou
                for previous in selected
            ):
                continue
            selected.append(detection)
            if len(selected) == self._max_detections:
                break
        return selected

    @staticmethod
    def _starts(length: int, size: int, stride: int) -> list[int]:
        last = length - size
        starts = list(range(0, last + 1, stride))
        if starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _positive_integer(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise DetectorConfigurationError(f"{name} debe ser un entero positivo.")
        return int(value)

    @staticmethod
    def _unit(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise DetectorConfigurationError(f"{name} debe ser numérico entre 0 y 1.") from error
        if not isfinite(result) or not 0.0 <= result <= 1.0:
            raise DetectorConfigurationError(f"{name} debe estar entre 0 y 1.")
        return result
