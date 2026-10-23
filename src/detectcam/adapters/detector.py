"""Inferencia YOLO independiente de RTSP y del control PTZ."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIDENCE = 0.25
DEFAULT_DEVICE = "cpu"
DEFAULT_IMGSZ = 640
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "best.pt"


class DetectorError(RuntimeError):
    """Error base del detector."""


class DetectorConfigurationError(ValueError):
    """Configuración inválida del detector."""


class ModelLoadError(DetectorError):
    """El modelo no existe o no pudo cargarse."""


class InferenceError(DetectorError):
    """La inferencia no pudo completarse."""


@dataclass(frozen=True)
class Detection:
    """Caja detectada expresada en píxeles y con confianza normalizada."""

    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height


class YOLODetector:
    """Carga una vez un modelo Ultralytics y normaliza sus detecciones."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        confidence: float = DEFAULT_CONFIDENCE,
        device: str | int = DEFAULT_DEVICE,
        imgsz: int = DEFAULT_IMGSZ,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        configured_path = model_path or os.getenv("DETECTCAM_MODEL_PATH") or DEFAULT_MODEL_PATH
        self._model_path = Path(configured_path).expanduser().resolve()
        if not self._model_path.is_file():
            raise ModelLoadError(
                f"No se encontró el modelo YOLO en: {self._model_path}. "
                "Configure DETECTCAM_MODEL_PATH."
            )
        self._confidence = self._validate_confidence(confidence)
        self._device = self._validate_device(device)
        self._imgsz = self._validate_imgsz(imgsz)
        self._last_inference_ms: float | None = None
        self._total_inference_ms = 0.0
        self._inference_count = 0

        if model_factory is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise ModelLoadError(
                    "Ultralytics no está instalado en el entorno Python activo."
                ) from error
            model_factory = YOLO
        LOGGER.info("Cargando modelo YOLO desde %s", self._model_path)
        try:
            self._model = model_factory(str(self._model_path))
            self._class_names = self._normalize_names(getattr(self._model, "names", {}))
        except Exception as error:
            LOGGER.error("No se pudo cargar el modelo YOLO (%s)", type(error).__name__)
            raise ModelLoadError("No se pudo cargar el modelo YOLO configurado.") from error
        LOGGER.info(
            "Modelo YOLO cargado una vez; dispositivo=%s, imgsz=%d, confidence=%.2f",
            self._device, self._imgsz, self._confidence,
        )

    @classmethod
    def from_environment(cls, **overrides: Any) -> "YOLODetector":
        try:
            confidence = float(os.getenv("DETECTCAM_CONFIDENCE", str(DEFAULT_CONFIDENCE)))
        except ValueError as error:
            raise DetectorConfigurationError("DETECTCAM_CONFIDENCE debe ser numérico.") from error
        try:
            imgsz = int(os.getenv("DETECTCAM_IMGSZ", str(DEFAULT_IMGSZ)))
        except ValueError as error:
            raise DetectorConfigurationError("DETECTCAM_IMGSZ debe ser un entero.") from error
        return cls(
            model_path=os.getenv("DETECTCAM_MODEL_PATH") or None,
            confidence=confidence,
            device=os.getenv("DETECTCAM_DEVICE", DEFAULT_DEVICE),
            imgsz=imgsz,
            **overrides,
        )

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def confidence(self) -> float:
        return self._confidence

    def set_confidence(self, value: float) -> None:
        """Actualiza el umbral de inferencia desde el hilo de visión."""
        self._confidence = self._validate_confidence(value)

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def device(self) -> str | int:
        return self._device

    @property
    def last_inference_ms(self) -> float | None:
        return self._last_inference_ms

    @property
    def average_inference_ms(self) -> float | None:
        if not self._inference_count:
            return None
        return self._total_inference_ms / self._inference_count

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        if frame is None:
            raise DetectorConfigurationError("El frame no puede ser None.")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise DetectorConfigurationError("El frame debe ser un arreglo NumPy no vacío.")
        if frame.ndim not in (2, 3):
            raise DetectorConfigurationError("El frame debe tener dos o tres dimensiones.")

        started = perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                conf=self._confidence,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
        except Exception as error:
            LOGGER.error("Falló la inferencia YOLO (%s)", type(error).__name__)
            raise InferenceError("No fue posible completar la inferencia YOLO.") from error
        detections: list[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            # Una sola transferencia de todas las cajas evita sincronizar GPU
            # por cada .item() de clase/confianza y cada .tolist() de coordenadas.
            data = getattr(boxes, "data", None)
            if data is not None:
                rows = data.detach().cpu().numpy()
                if rows.ndim != 2 or rows.shape[1] != 6:
                    raise InferenceError("El detector debe devolver cajas xyxy, confianza y clase")
                for x1, y1, x2, y2, confidence, class_value in rows:
                    class_id = int(class_value)
                    detections.append(Detection(
                        class_id, self._class_names.get(class_id, str(class_id)), float(confidence),
                        int(round(float(x1))), int(round(float(y1))),
                        int(round(float(x2))), int(round(float(y2))),
                    ))
                continue
            for box in boxes:
                try:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
                except (AttributeError, IndexError, TypeError, ValueError) as error:
                    raise InferenceError("Ultralytics devolvió una detección no válida.") from error
                detections.append(Detection(
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    confidence=confidence,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        # Incluye postprocesado/transferencia; no ocultar sincronizaciones GPU
        # detrás de una cifra que sólo mide la llamada predict().
        elapsed_ms = (perf_counter() - started) * 1000
        self._last_inference_ms = elapsed_ms
        self._total_inference_ms += elapsed_ms
        self._inference_count += 1
        return detections

    @staticmethod
    def _normalize_names(names: Mapping[int, str] | Sequence[str]) -> dict[int, str]:
        if isinstance(names, Mapping):
            return {int(key): str(value) for key, value in names.items()}
        return {index: str(value) for index, value in enumerate(names)}

    @staticmethod
    def _validate_confidence(value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise DetectorConfigurationError("confidence debe ser numérico.") from error
        if not 0.0 <= result <= 1.0:
            raise DetectorConfigurationError("confidence debe estar entre 0.0 y 1.0.")
        return result

    @staticmethod
    def _validate_device(value: str | int) -> str | int:
        if isinstance(value, int):
            return value
        if not isinstance(value, str) or not value.strip():
            raise DetectorConfigurationError("device no puede estar vacío.")
        device = value.strip()
        return int(device) if device.isdigit() else device

    @staticmethod
    def _validate_imgsz(value: int) -> int:
        if isinstance(value, bool):
            raise DetectorConfigurationError("imgsz debe ser un entero positivo.")
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise DetectorConfigurationError("imgsz debe ser un entero positivo.") from error
        if result <= 0:
            raise DetectorConfigurationError("imgsz debe ser un entero positivo.")
        return result


def draw_detections(
    frame: NDArray[np.uint8], detections: Sequence[Detection]
) -> NDArray[np.uint8]:
    """Devuelve una copia del frame con cajas y etiquetas dibujadas."""
    import cv2

    output = frame.copy()
    for detection in detections:
        cv2.rectangle(output, (detection.x1, detection.y1),
                      (detection.x2, detection.y2), (0, 255, 0), 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(output, label, (detection.x1, max(20, detection.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return output
