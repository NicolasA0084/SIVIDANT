"""Validación compartida de pesos y exportaciones locales de YOLO."""

from pathlib import Path


class ModelArtifactError(ValueError):
    """El archivo o directorio del modelo no está completo."""


def validate_model_artifact(path: Path) -> None:
    """Acepta archivos existentes o un directorio NCNN exportado completo.

    Ultralytics identifica NCNN por el nombre ``*_ncnn_model`` y carga el primer
    ``*.param`` del directorio junto con su ``*.bin`` homónimo. Exigir un solo
    par evita seleccionar por accidente archivos intermedios de PNNX.
    """
    if path.is_file():
        if path.suffix.lower() == ".joblib":
            raise ModelArtifactError(
                "Un archivo .joblib de clasificación no proporciona cajas YOLO. "
                "Conserve DETECTCAM_MODEL_PATH con sus pesos YOLO y configure "
                "DETECTCAM_CLASSIFIER_PATH para el clasificador. "
                "Consulte docs/MODELO_JOBLIB.md."
            )
        return
    if not path.is_dir():
        raise ModelArtifactError(
            f"No se encontró el modelo YOLO en {path}. "
            "Configure DETECTCAM_MODEL_PATH."
        )
    if not path.name.endswith("_ncnn_model"):
        raise ModelArtifactError(
            f"El directorio del modelo debe ser una exportación NCNN con "
            f"nombre *_ncnn_model: {path}."
        )
    parameters = [entry for entry in path.glob("*.param") if entry.is_file()]
    if len(parameters) != 1:
        raise ModelArtifactError(
            f"La exportación NCNN en {path} debe contener exactamente un "
            "archivo .param y su archivo .bin homónimo."
        )
    weights = parameters[0].with_suffix(".bin")
    if not weights.is_file():
        raise ModelArtifactError(
            f"La exportación NCNN está incompleta; falta {weights}."
        )
