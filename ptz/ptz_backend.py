"""Interfaz común de backends PTZ y fábrica compatible con ISAPI/ONVIF."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable, Protocol, runtime_checkable

LOGGER = logging.getLogger(__name__)
SUPPORTED_BACKENDS = ("auto", "isapi", "onvif")


class PTZBackendError(RuntimeError):
    """Error genérico de una implementación física PTZ."""


class PTZBackendCapabilityError(PTZBackendError, NotImplementedError):
    """La operación solicitada no está anunciada por el backend."""


@dataclass(frozen=True)
class PTZCapabilities:
    """Capacidades comunes sin exponer rangos propios de un fabricante."""

    continuous_pan_tilt: bool = False
    continuous_zoom: bool = False
    absolute_pan_tilt: bool = False
    absolute_zoom: bool = False
    position3d: bool = False
    get_status: bool = False
    presets: bool = False


@runtime_checkable
class PTZBackend(Protocol):
    """Contrato estructural usado por el ejecutor seguro.

    ``move_velocity`` recibe coordenadas de pantalla normalizadas: pan positivo
    significa derecha y tilt positivo significa abajo. Cada backend traduce esa
    convención a su protocolo físico.
    """

    @property
    def backend_capabilities(self) -> PTZCapabilities: ...

    def connect(self) -> "PTZBackend": ...
    def is_connected(self) -> bool: ...
    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None: ...
    def stop(self) -> None: ...
    def get_status(self) -> object | None: ...
    def move_absolute(self, **coordinates: float) -> None: ...
    def position3d(self, **coordinates: int) -> object | None: ...
    def close(self) -> None: ...


def create_ptz_controller(
    backend: str | None = None,
    *,
    isapi_factory: Callable[[], Any] | None = None,
    onvif_factory: Callable[[], Any] | None = None,
) -> PTZBackend:
    """Crea y conecta un backend sin emitir movimiento.

    ``auto`` prueba ISAPI mediante lecturas de capacidades y usa ONVIF si esa
    conexión inicial falla. Nunca reintenta una orden de movimiento en otro
    backend, pues un timeout no permite saber con certeza si la cámara la recibió.
    Las importaciones son diferidas para que el modo simulado no cargue drivers.
    """
    from .isapi_ptz import ISAPIPTZController, ISAPIPTZError
    from .ptz_controller import PTZController

    selected = (backend or os.getenv("DETECTCAM_PTZ_BACKEND", "auto")).strip().lower()
    if selected not in SUPPORTED_BACKENDS:
        raise ValueError("DETECTCAM_PTZ_BACKEND debe ser auto, isapi u onvif.")
    make_isapi = isapi_factory or ISAPIPTZController.from_environment
    make_onvif = onvif_factory or PTZController.from_environment

    if selected in ("auto", "isapi"):
        controller = make_isapi()
        try:
            return controller.connect()
        except ISAPIPTZError:
            controller.close()
            if selected == "isapi":
                raise
            LOGGER.warning(
                "ISAPI no disponible durante la conexión inicial; usando ONVIF."
            )
    return make_onvif().connect()
