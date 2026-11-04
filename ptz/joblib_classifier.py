"""Clasificación opcional de recortes YOLO con características del entrenamiento.

No se deduce un extractor a partir de n_features_in_: coincidir en longitud no
garantiza que los valores signifiquen lo mismo que durante el entrenamiento.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import importlib.util
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from numpy.typing import NDArray

from ptz.detector import Detection, InferenceError, ModelLoadError, PROJECT_ROOT, YOLODetector

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Classification:
    class_id: int
    label: str
    class_name: str
    margin: float | None
    confidence: float | None = None


class JoblibClassifier:
    """Carga una vez el pipeline y valida sus entradas antes de predecir."""

    def __init__(
        self, model_path: str | Path, *,
        labels: Mapping[str, str] | None = None,
        model_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ModelLoadError(f"No se encontró el clasificador: {self.model_path}.")
        try:
            if model_loader is None:
                import joblib
                from sklearn.exceptions import InconsistentVersionWarning

                with warnings.catch_warnings():
                    warnings.simplefilter("error", InconsistentVersionWarning)
                    self._model = joblib.load(self.model_path)
            else:
                self._model = model_loader(str(self.model_path))
        except ImportError as error:
            raise ModelLoadError(
                "Faltan dependencias del clasificador. Ejecute "
                "python -m pip install -r requirements-joblib.txt."
            ) from error
        except Exception as error:
            raise ModelLoadError(
                "No se pudo cargar el clasificador; use las versiones del "
                "entrenamiento (requirements-joblib.txt para el modelo proporcionado)."
            ) from error

        classes = np.asarray(getattr(self._model, "classes_", []))
        features = getattr(self._model, "n_features_in_", None)
        if (not callable(getattr(self._model, "predict", None))
                or classes.ndim != 1 or classes.size < 2
                or not isinstance(features, (int, np.integer)) or features <= 0):
            raise ModelLoadError(
                "El .joblib debe contener un clasificador entrenado con predict, "
                "classes_ y n_features_in_."
            )
        self.n_features = int(features)
        self.labels = tuple(str(value) for value in classes.tolist())
        if len(set(self.labels)) != len(self.labels):
            raise ModelLoadError("El modelo contiene identificadores de clase ambiguos.")
        if labels is not None:
            if (not isinstance(labels, Mapping) or set(labels) != set(self.labels)
                    or any(not isinstance(name, str) or not name.strip()
                           for name in labels.values())):
                raise ModelLoadError(
                    f"El mapa JSON debe asignar un nombre a cada clase: {', '.join(self.labels)}."
                )
        self._class_names = {
            index: str(label if labels is None else labels[label]).strip()
            for index, label in enumerate(self.labels)
        }
        self._class_ids = {label: index for index, label in enumerate(self.labels)}
        LOGGER.info("Clasificador cargado: %d características; clases=%s; CPU",
                    self.n_features, self._class_names)

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def has_probabilities(self) -> bool:
        return callable(getattr(self._model, "predict_proba", None))

    def describe(self) -> dict[str, Any]:
        return {
            "model_path": str(self.model_path),
            "type": type(self._model).__name__,
            "steps": [
                {"name": name, "type": type(step).__name__}
                for name, step in getattr(self._model, "steps", [])
            ],
            "n_features": self.n_features,
            "classes": [
                {"label": label, "name": self._class_names[index]}
                for index, label in enumerate(self.labels)
            ],
            "has_image_extractor": False,
            "scores_are_probabilities": self.has_probabilities,
        }

    def predict_features(self, features: NDArray) -> list[Classification]:
        try:
            matrix = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise InferenceError("Las características deben ser numéricas.") from error
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != self.n_features:
            raise InferenceError(
                f"El clasificador necesita una matriz (N, {self.n_features}); "
                f"recibió {matrix.shape}. Use el extractor original del entrenamiento."
            )
        if not np.isfinite(matrix).all():
            raise InferenceError("Las características contienen NaN o infinito.")
        try:
            predictions = np.asarray(self._model.predict(matrix))
            if predictions.shape != (matrix.shape[0],):
                raise ValueError("Número de predicciones inesperado")
            margins: list[float | None] = [None] * len(predictions)
            confidences: list[float | None] = [None] * len(predictions)
            if self.has_probabilities:
                probabilities = np.asarray(self._model.predict_proba(matrix), dtype=float)
                if (probabilities.shape != (len(predictions), len(self.labels))
                        or not np.isfinite(probabilities).all()
                        or np.any(probabilities < 0.0) or np.any(probabilities > 1.0)
                        or not np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-6, atol=1e-6)):
                    raise ValueError("El modelo devolvió probabilidades no válidas")
                confidences = [float(probabilities[row, self._class_ids[str(label)]])
                               for row, label in enumerate(predictions)]
            if callable(getattr(self._model, "decision_function", None)):
                scores = np.asarray(self._model.decision_function(matrix), dtype=float)
                if not np.isfinite(scores).all():
                    raise ValueError("Puntuaciones no finitas")
                if len(self.labels) == 2 and scores.shape == (len(predictions),):
                    margins = [float(score if str(label) == self.labels[1] else -score)
                               for score, label in zip(scores, predictions)]
                elif scores.shape == (len(predictions), len(self.labels)):
                    margins = [float(scores[row, self._class_ids[str(label)]])
                               for row, label in enumerate(predictions)]
                else:
                    raise ValueError("Formato de decision_function no compatible")
            return [
                Classification(
                    self._class_ids[str(label)], str(label),
                    self._class_names[self._class_ids[str(label)]], margin, confidence,
                )
                for label, margin, confidence in zip(predictions, margins, confidences)
            ]
        except Exception as error:
            raise InferenceError("No fue posible clasificar las características del recorte.") from error


def load_feature_extractor(path: str | Path) -> Callable[[NDArray[np.uint8]], NDArray]:
    """Carga extract_features(image_bgr) de un archivo local configurado."""
    source = Path(path).expanduser().resolve()
    try:
        spec = importlib.util.spec_from_file_location("detectcam_user_features", source)
        if spec is None or spec.loader is None:
            raise ValueError("Se necesita un archivo Python")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        extractor = getattr(module, "extract_features", None)
        if not callable(extractor):
            raise ValueError("Falta extract_features(image_bgr)")
        return extractor
    except Exception as error:
        raise ModelLoadError(
            f"No se pudo cargar {source}; debe definir extract_features(image_bgr). "
            "Compruebe sus dependencias con: python -m pip install -r requirements-joblib.txt."
        ) from error


def load_labels(path: str | Path) -> dict[str, str]:
    try:
        labels = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(labels, dict):
            raise ValueError("Se necesita un objeto JSON")
        return labels
    except (OSError, ValueError) as error:
        raise ModelLoadError("No se pudo leer el mapa de clases JSON del clasificador.") from error


class ClassifyingDetector:
    """YOLO aporta las cajas y su confianza; el clasificador aporta la clase.

    El margen SVM se muestra por separado y nunca se convierte a probabilidad.
    Sólo se clasifican objetos que YOLO haya localizado.
    """

    def __init__(
        self, detector: YOLODetector, classifier: JoblibClassifier,
        feature_extractor: Callable[[NDArray[np.uint8]], NDArray],
    ) -> None:
        self._detector = detector
        self._classifier = classifier
        self._extract_features = feature_extractor
        self._last_inference_ms: float | None = None

    @property
    def class_names(self) -> dict[int, str]:
        return {class_id: name for class_id, name in self._classifier.class_names.items()
                if name.strip().casefold() != "background"}

    @property
    def confidence(self) -> float:
        return self._detector.confidence

    def set_confidence(self, value: float) -> None:
        self._detector.set_confidence(value)

    @property
    def last_inference_ms(self) -> float | None:
        return self._last_inference_ms

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        started = perf_counter()
        try:
            detections = self._detector.detect(frame)
            height, width = frame.shape[:2]
            boxes: list[Detection] = []
            features = []
            for detection in detections:
                x1, x2 = max(0, min(width, detection.x1)), max(0, min(width, detection.x2))
                y1, y2 = max(0, min(height, detection.y1)), max(0, min(height, detection.y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                vector = np.asarray(self._extract_features(frame[y1:y2, x1:x2].copy()))
                if vector.shape != (self._classifier.n_features,):
                    raise InferenceError(
                        f"extract_features debe devolver ({self._classifier.n_features},); "
                        f"devolvió {vector.shape}. Use la función original del entrenamiento."
                    )
                features.append(vector)
                boxes.append(replace(detection, x1=x1, y1=y1, x2=x2, y2=y2))
            if not boxes:
                return []
            predictions = self._classifier.predict_features(np.stack(features))
            return [
                replace(box, class_id=result.class_id, class_name=result.class_name,
                        classification_margin=result.margin)
                for box, result in zip(boxes, predictions)
                if result.class_name.strip().casefold() != "background"
            ]
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError("Falló la extracción de características del clasificador.") from error
        finally:
            self._last_inference_ms = (perf_counter() - started) * 1000.0


def with_optional_classifier(detector: YOLODetector) -> YOLODetector | ClassifyingDetector:
    model_path = os.getenv("DETECTCAM_CLASSIFIER_PATH", "").strip()
    if not model_path:
        return detector
    features_path = os.getenv("DETECTCAM_CLASSIFIER_FEATURES_PATH", "").strip()
    labels_path = os.getenv("DETECTCAM_CLASSIFIER_LABELS_PATH", "").strip()
    if not features_path or not labels_path:
        raise ModelLoadError(
            "Configure DETECTCAM_CLASSIFIER_FEATURES_PATH y DETECTCAM_CLASSIFIER_LABELS_PATH. "
            "Consulte docs/MODELO_JOBLIB.md."
        )
    classifier = JoblibClassifier(model_path, labels=load_labels(labels_path))
    return ClassifyingDetector(detector, classifier, load_feature_extractor(features_path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspecciona un .joblib o clasifica un recorte de imagen o características .npy.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model" / "model.joblib")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--features", type=Path, help="Matriz .npy calculada con el extractor original")
    inputs.add_argument("--image", type=Path, help="Imagen de un objeto o recorte para clasificar")
    parser.add_argument("--extractor", type=Path, default=PROJECT_ROOT / "ptz" / "hog_hsv_features.py",
                        help="Extractor para --image; por defecto HOG_HSV_V1")
    parser.add_argument("--labels", type=Path, help="Mapa JSON de identificadores a nombres reales")
    arguments = parser.parse_args()
    try:
        classifier = JoblibClassifier(
            arguments.model, labels=load_labels(arguments.labels) if arguments.labels else None,
        )
        output = classifier.describe()
        matrix = None
        if arguments.image:
            import cv2

            image = cv2.imread(str(arguments.image), cv2.IMREAD_COLOR)
            if image is None:
                raise InferenceError(f"No se pudo leer la imagen: {arguments.image}.")
            matrix = load_feature_extractor(arguments.extractor)(image)
            output["image_path"] = str(arguments.image.resolve())
            output["image_extractor_path"] = str(arguments.extractor.resolve())
        elif arguments.features:
            matrix = np.load(arguments.features, allow_pickle=False)
        if matrix is not None:
            output["predictions"] = [
                {"label": result.label, "class_name": result.class_name,
                 "svm_margin": result.margin, "confidence": result.confidence}
                for result in classifier.predict_features(matrix)
            ]
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except (ModelLoadError, InferenceError, OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
