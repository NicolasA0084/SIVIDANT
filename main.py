"""Punto de entrada estable de Detectcam: ``python main.py``."""

from __future__ import annotations

import logging

from ptz.config import ConfigurationError, bootstrap_environment, configure_logging


LOGGER = logging.getLogger("detectcam")


def main() -> int:
    configure_logging()
    try:
        configuration = bootstrap_environment()
    except ConfigurationError as error:
        LOGGER.error("No se puede iniciar Detectcam: %s", error)
        return 2

    LOGGER.info(
        "Configuración validada; modelo=%s; PTZ físico=%s",
        configuration.model_path,
        "habilitado (inicia desarmado)" if configuration.hardware_enabled else "deshabilitado",
    )
    # Importación diferida: el preflight termina antes de cargar OpenCV, NumPy,
    # Ultralytics o cualquier backend de cámara.
    from ptz.main import main as run_detectcam

    return run_detectcam()


if __name__ == "__main__":
    raise SystemExit(main())
