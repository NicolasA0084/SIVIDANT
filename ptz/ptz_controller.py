"""Control PTZ ONVIF reutilizable, sin video, inferencia ni tracking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from typing import Any, Callable

from onvif import ONVIFCamera
from zeep.transports import Transport

from .ptz_backend import PTZBackendCapabilityError, PTZCapabilities

LOGGER = logging.getLogger(__name__)
DEFAULT_CAMERA_IP = "192.168.1.100"
DEFAULT_ONVIF_PORT = 80
DEFAULT_PAN_SPEED = 0.5
DEFAULT_TILT_SPEED = 0.5
DEFAULT_ZOOM_SPEED = 0.5
MIN_VELOCITY, MAX_VELOCITY = -1.0, 1.0


class PTZError(RuntimeError):
    """Error base de control PTZ."""


class PTZConfigurationError(ValueError):
    """Configuración PTZ inválida."""


class PTZConnectionError(PTZError):
    """Fallo de conexión ONVIF."""


class PTZCommandError(PTZError):
    """Fallo al ejecutar un comando PTZ."""


@dataclass(frozen=True)
class MotionCommand:
    """Última velocidad normalizada solicitada al backend ONVIF."""

    pan: float
    tilt: float
    zoom: float


class PTZController:
    """Encapsula movimiento continuo y parada mediante ONVIF."""

    def __init__(self, *, username: str, password: str,
                 camera_ip: str = DEFAULT_CAMERA_IP, port: int = DEFAULT_ONVIF_PORT,
                 pan_speed: float = DEFAULT_PAN_SPEED,
                 tilt_speed: float = DEFAULT_TILT_SPEED,
                 zoom_speed: float = DEFAULT_ZOOM_SPEED,
                 connection_timeout: float = 5.0, operation_timeout: float = 5.0,
                 camera_factory: Callable[..., Any] = ONVIFCamera) -> None:
        self._username = self._required(username, "usuario")
        self._password = self._required(password, "contraseña", strip=False)
        self._camera_ip = self._validate_host(camera_ip)
        if not 1 <= port <= 65535:
            raise PTZConfigurationError("El puerto ONVIF debe estar entre 1 y 65535.")
        if connection_timeout <= 0 or operation_timeout <= 0:
            raise PTZConfigurationError("Los timeouts ONVIF deben ser positivos.")
        self._port = port
        self._pan_speed = self._speed(pan_speed, "pan_speed")
        self._tilt_speed = self._speed(tilt_speed, "tilt_speed")
        self._zoom_speed = self._speed(zoom_speed, "zoom_speed")
        self._connection_timeout = connection_timeout
        self._operation_timeout = operation_timeout
        self._camera_factory = camera_factory
        self._lock = threading.RLock()
        self._camera = self._service = self._token = None
        self._connected = False
        self._last_command: MotionCommand | None = None

    @classmethod
    def from_environment(cls, **overrides: Any) -> "PTZController":
        user = os.getenv("DETECTCAM_CAMERA_USER", "")
        password = os.getenv("DETECTCAM_CAMERA_PASSWORD", "")
        missing = [name for name, value in (
            ("DETECTCAM_CAMERA_USER", user.strip()),
            ("DETECTCAM_CAMERA_PASSWORD", password),
        ) if not value]
        if missing:
            raise PTZConfigurationError("Faltan variables de entorno: " + ", ".join(missing))
        try:
            port = int(os.getenv("DETECTCAM_ONVIF_PORT", str(DEFAULT_ONVIF_PORT)))
        except ValueError as error:
            raise PTZConfigurationError("DETECTCAM_ONVIF_PORT debe ser un entero.") from error
        return cls(username=user, password=password,
                   camera_ip=os.getenv("DETECTCAM_CAMERA_IP", DEFAULT_CAMERA_IP),
                   port=port, **overrides)

    def connect(self) -> "PTZController":
        with self._lock:
            if self._connected:
                return self
            LOGGER.info("Conectando PTZ ONVIF en %s:%d", self._camera_ip, self._port)
            try:
                transport = Transport(timeout=self._connection_timeout,
                                      operation_timeout=self._operation_timeout)
                camera = self._camera_factory(self._camera_ip, self._port,
                                              self._username, self._password,
                                              transport=transport)
                media = camera.create_media_service()
                service = camera.create_ptz_service()
                profiles = media.GetProfiles()
                if not profiles or not getattr(profiles[0], "token", None):
                    raise PTZConnectionError("La cámara no devolvió un perfil ONVIF válido.")
            except PTZConnectionError:
                raise
            except Exception as error:
                LOGGER.error("Falló la conexión PTZ ONVIF (%s)", type(error).__name__)
                raise PTZConnectionError(
                    "No se pudo conectar por ONVIF; revise red, puerto y credenciales."
                ) from error
            self._camera, self._service, self._token = camera, service, profiles[0].token
            self._connected = True
            LOGGER.info("Control PTZ ONVIF conectado")
        return self

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def backend_capabilities(self) -> PTZCapabilities:
        return PTZCapabilities(
            continuous_pan_tilt=True,
            continuous_zoom=True,
        )

    @property
    def last_command(self) -> MotionCommand | None:
        with self._lock:
            return self._last_command

    @property
    def speeds(self) -> MotionCommand:
        return MotionCommand(self._pan_speed, self._tilt_speed, self._zoom_speed)

    def move(self, *, pan: float = 0.0, tilt: float = 0.0, zoom: float = 0.0) -> None:
        command = MotionCommand(self._velocity(pan, "pan"), self._velocity(tilt, "tilt"),
                                self._velocity(zoom, "zoom"))
        with self._lock:
            service, token = self._require_connection()
            try:
                request = service.create_type("ContinuousMove")
                request.ProfileToken = token
                request.Velocity = {"PanTilt": {"x": command.pan, "y": command.tilt},
                                    "Zoom": {"x": command.zoom}}
                service.ContinuousMove(request)
            except Exception as error:
                LOGGER.error("Falló ContinuousMove (%s)", type(error).__name__)
                self._emergency_stop(service, token)
                raise PTZCommandError(
                    "Falló el movimiento PTZ; se intentó detener la cámara."
                ) from error
            self._last_command = command

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        """Traduce pantalla (+Y abajo) a ONVIF (+Y arriba)."""
        self.move(pan=pan, tilt=-tilt, zoom=zoom)

    def get_status(self) -> object | None:
        raise PTZBackendCapabilityError(
            "El backend ONVIF actual no expone lectura de status común."
        )

    def move_absolute(self, **coordinates: float) -> None:
        del coordinates
        raise PTZBackendCapabilityError(
            "El backend ONVIF actual no implementa movimiento absoluto común."
        )

    def position3d(self, **coordinates: int) -> object | None:
        del coordinates
        raise PTZBackendCapabilityError("ONVIF no implementa Position3D Hikvision.")

    def move_left(self, speed: float | None = None) -> None:
        self.move(pan=-self._direction(speed, self._pan_speed, "pan"))
    def move_right(self, speed: float | None = None) -> None:
        self.move(pan=self._direction(speed, self._pan_speed, "pan"))
    def move_up(self, speed: float | None = None) -> None:
        self.move(tilt=self._direction(speed, self._tilt_speed, "tilt"))
    def move_down(self, speed: float | None = None) -> None:
        self.move(tilt=-self._direction(speed, self._tilt_speed, "tilt"))
    def zoom_in(self, speed: float | None = None) -> None:
        self.move(zoom=self._direction(speed, self._zoom_speed, "zoom"))
    def zoom_out(self, speed: float | None = None) -> None:
        self.move(zoom=-self._direction(speed, self._zoom_speed, "zoom"))
    def move_up_left(self) -> None: self.move(pan=-self._pan_speed, tilt=self._tilt_speed)
    def move_up_right(self) -> None: self.move(pan=self._pan_speed, tilt=self._tilt_speed)
    def move_down_left(self) -> None: self.move(pan=-self._pan_speed, tilt=-self._tilt_speed)
    def move_down_right(self) -> None: self.move(pan=self._pan_speed, tilt=-self._tilt_speed)

    def stop(self) -> None:
        """Detiene explícitamente pan, tilt y zoom."""
        with self._lock:
            service, token = self._require_connection()
            try:
                service.Stop({"ProfileToken": token, "PanTilt": True, "Zoom": True})
            except Exception as error:
                LOGGER.error("Falló PTZ Stop (%s)", type(error).__name__)
                raise PTZCommandError("No se pudo confirmar la detención PTZ.") from error
            self._last_command = MotionCommand(0.0, 0.0, 0.0)

    def close(self) -> None:
        with self._lock:
            if not self._connected:
                return
            self._emergency_stop(self._service, self._token)
            self._connected = False
            self._camera = self._service = self._token = None
            LOGGER.info("Control PTZ ONVIF cerrado")

    def __enter__(self) -> "PTZController": return self.connect()
    def __exit__(self, *_: object) -> None: self.close()

    def _require_connection(self) -> tuple[Any, str]:
        if not self._connected or self._service is None or self._token is None:
            raise PTZConnectionError("PTZ no está conectado; ejecute connect() primero.")
        return self._service, self._token

    def _emergency_stop(self, service: Any, token: str) -> None:
        try:
            service.Stop({"ProfileToken": token, "PanTilt": True, "Zoom": True})
            self._last_command = MotionCommand(0.0, 0.0, 0.0)
        except Exception as error:
            self._last_command = None
            LOGGER.error("Falló también la parada de seguridad (%s)", type(error).__name__)

    @staticmethod
    def _required(value: str, label: str, *, strip: bool = True) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PTZConfigurationError(f"Falta configurar {label}.")
        return value.strip() if strip else value

    @staticmethod
    def _validate_host(value: str) -> str:
        host = PTZController._required(value, "IP o host")
        if any(c.isspace() for c in host) or any(c in host for c in "/@?#"):
            raise PTZConfigurationError("La IP o el host no es válido.")
        return host

    @staticmethod
    def _velocity(value: float, label: str) -> float:
        try: result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZConfigurationError(f"La velocidad {label} debe ser numérica.") from error
        if not MIN_VELOCITY <= result <= MAX_VELOCITY:
            raise PTZConfigurationError(f"La velocidad {label} debe estar entre -1.0 y 1.0.")
        return result

    @classmethod
    def _speed(cls, value: float, label: str) -> float:
        result = cls._velocity(value, label)
        if result < 0: raise PTZConfigurationError(f"{label} debe estar entre 0.0 y 1.0.")
        return result

    @classmethod
    def _direction(cls, value: float | None, default: float, label: str) -> float:
        return default if value is None else cls._speed(value, f"velocidad {label}")
