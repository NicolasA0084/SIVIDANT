"""Diagnóstico ONVIF manual; consulta datos y no envía movimiento PTZ."""

from __future__ import annotations

import logging
import os

from onvif import ONVIFCamera

from ptz.config import ConfigurationError, bootstrap_environment, configure_logging


LOGGER = logging.getLogger(__name__)


def main() -> int:
    configure_logging()
    try:
        bootstrap_environment()
        port = int(os.getenv("DETECTCAM_ONVIF_PORT", "80"))
        camera = ONVIFCamera(
            os.environ["DETECTCAM_CAMERA_IP"],
            port,
            os.environ["DETECTCAM_CAMERA_USER"],
            os.environ["DETECTCAM_CAMERA_PASSWORD"],
        )
        info = camera.create_devicemgmt_service().GetDeviceInformation()
    except ConfigurationError as error:
        LOGGER.error("Configuración inválida: %s", error)
        return 2
    except Exception as error:
        LOGGER.error(
            "No se pudo consultar la cámara por ONVIF (%s). Revise red, puerto y "
            "credenciales.",
            type(error).__name__,
        )
        return 1

    LOGGER.info("Conexión ONVIF exitosa")
    LOGGER.info("Fabricante: %s", info.Manufacturer)
    LOGGER.info("Modelo: %s", info.Model)
    LOGGER.info("Firmware: %s", info.FirmwareVersion)
    # El número de serie no se imprime para evitar filtrar un identificador del equipo.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
