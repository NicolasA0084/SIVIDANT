"""Backend de control PTZ Hikvision mediante ISAPI y HTTP Digest."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from typing import Any
from urllib.parse import urljoin
import xml.etree.ElementTree as ET

import requests
from requests.auth import HTTPDigestAuth

from .ptz_controller import MotionCommand, PTZConfigurationError, PTZError
from .ptz_backend import PTZCapabilities

LOGGER = logging.getLogger(__name__)
ISAPI_NAMESPACE = "http://www.isapi.org/ver20/XMLSchema"
DEFAULT_CAMERA_IP = "192.168.1.100"
DEFAULT_ISAPI_PORT = 80
DEFAULT_CHANNEL_ID = 1
DEFAULT_SPEED = 0.5
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 10.0


class ISAPIPTZError(PTZError):
    """Error base del backend PTZ ISAPI."""


class ISAPIPTZConnectionError(ISAPIPTZError):
    """No fue posible preparar el backend PTZ ISAPI."""


class ISAPIPTZCommandError(ISAPIPTZError):
    """La cámara rechazó o no confirmó una orden PTZ ISAPI."""


class ISAPIPTZNotSupportedError(ISAPIPTZError):
    """La capacidad solicitada no fue anunciada por la cámara."""


@dataclass(frozen=True)
class ValueRange:
    """Intervalo numérico anunciado por las capabilities ISAPI."""

    minimum: float
    maximum: float

    def validate(self, value: float, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZConfigurationError(f"{label} debe ser numérico.") from error
        if not self.minimum <= result <= self.maximum:
            raise PTZConfigurationError(
                f"{label} debe estar entre {self.minimum:g} y {self.maximum:g}."
            )
        return result


@dataclass(frozen=True)
class ISAPIPTZCapabilities:
    """Capacidades y rangos PTZ descubiertos al conectar con la cámara."""

    continuous_pan: ValueRange | None
    continuous_tilt: ValueRange | None
    continuous_zoom: ValueRange | None
    absolute_azimuth: ValueRange | None
    absolute_elevation: ValueRange | None
    absolute_zoom: ValueRange | None
    position_3d: bool

    @property
    def supports_continuous(self) -> bool:
        return all((self.continuous_pan, self.continuous_tilt,
                    self.continuous_zoom))

    @property
    def supports_absolute(self) -> bool:
        return all((self.absolute_azimuth, self.absolute_elevation,
                    self.absolute_zoom))


@dataclass(frozen=True)
class ISAPIPTZStatus:
    """Posición reportada en grados y factor de zoom óptico."""

    azimuth: float
    elevation: float
    absolute_zoom: float


@dataclass(frozen=True)
class ISAPICommandResult:
    """Confirmación HTTP/ISAPI de una orden de escritura."""

    status_code: int
    status_string: str | None
    sub_status_code: str | None


class ISAPIPTZController:
    """Control PTZ ISAPI separado de video, detección y tracking.

    La API pública usa velocidades normalizadas de -1.0 a 1.0, igual que el
    backend ONVIF existente. La cámara anuncia y recibe -100..100 por ISAPI.
    """

    def __init__(
        self,
        *,
        username: str,
        password: str,
        camera_ip: str = DEFAULT_CAMERA_IP,
        port: int = DEFAULT_ISAPI_PORT,
        channel_id: int = DEFAULT_CHANNEL_ID,
        pan_speed: float = DEFAULT_SPEED,
        tilt_speed: float = DEFAULT_SPEED,
        zoom_speed: float = DEFAULT_SPEED,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        session: requests.Session | None = None,
    ) -> None:
        self._username = self._required(username, "usuario")
        self._password = self._required(password, "contraseña", strip=False)
        self._camera_ip = self._validate_host(camera_ip)
        if not 1 <= int(port) <= 65535:
            raise PTZConfigurationError("El puerto ISAPI debe estar entre 1 y 65535.")
        if int(channel_id) < 1:
            raise PTZConfigurationError("El canal PTZ ISAPI debe ser positivo.")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise PTZConfigurationError("Los timeouts ISAPI deben ser positivos.")

        self._port = int(port)
        self._channel_id = int(channel_id)
        self._base_url = f"http://{self._camera_ip}:{self._port}/"
        self._timeout = (float(connect_timeout), float(read_timeout))
        self._pan_speed = self._speed(pan_speed, "pan_speed")
        self._tilt_speed = self._speed(tilt_speed, "tilt_speed")
        self._zoom_speed = self._speed(zoom_speed, "zoom_speed")
        self._session = session or requests.Session()
        self._session.auth = HTTPDigestAuth(self._username, self._password)
        self._session.headers.update({"Accept": "application/xml"})
        self._lock = threading.RLock()
        self._connected = False
        self._moving = False
        self._last_command: MotionCommand | None = None
        self._capabilities: ISAPIPTZCapabilities | None = None

    @classmethod
    def from_environment(cls, **overrides: Any) -> "ISAPIPTZController":
        username = os.getenv("DETECTCAM_CAMERA_USER", "")
        password = os.getenv("DETECTCAM_CAMERA_PASSWORD", "")
        missing = [name for name, value in (
            ("DETECTCAM_CAMERA_USER", username.strip()),
            ("DETECTCAM_CAMERA_PASSWORD", password),
        ) if not value]
        if missing:
            raise PTZConfigurationError(
                "Faltan variables de entorno: " + ", ".join(missing)
            )
        try:
            port = int(os.getenv("DETECTCAM_ISAPI_PORT", str(DEFAULT_ISAPI_PORT)))
            channel_id = int(os.getenv(
                "DETECTCAM_PTZ_CHANNEL", str(DEFAULT_CHANNEL_ID)
            ))
        except ValueError as error:
            raise PTZConfigurationError(
                "DETECTCAM_ISAPI_PORT y DETECTCAM_PTZ_CHANNEL deben ser enteros."
            ) from error
        return cls(
            username=username,
            password=password,
            camera_ip=os.getenv("DETECTCAM_CAMERA_IP", DEFAULT_CAMERA_IP),
            port=port,
            channel_id=channel_id,
            **overrides,
        )

    def connect(self) -> "ISAPIPTZController":
        """Lee capacidades; no envía ninguna orden de movimiento."""
        with self._lock:
            if self._connected:
                return self
            base = self._get_xml(self._path("capabilities"), required=True)
            continuous_pan, continuous_tilt = self._xy_ranges(
                base, "ContinuousPanTiltSpace"
            )
            continuous_zoom = self._single_range(base, "ContinuousZoomSpace")
            position_3d = self._bool_element(base, "isSupportPosition3D")

            absolute_azimuth = absolute_elevation = absolute_zoom = None
            absolute_response = self._request(
                "GET", self._path("absoluteEx/capabilities")
            )
            if absolute_response.status_code == 200:
                absolute_root = self._parse_xml(
                    absolute_response.content, "capacidades absoluteEx"
                )
                absolute_elevation = self._attribute_range(
                    absolute_root, "elevation"
                )
                absolute_azimuth = self._attribute_range(absolute_root, "azimuth")
                absolute_zoom = self._attribute_range(
                    absolute_root, "absoluteZoom"
                )
            elif absolute_response.status_code not in (400, 404, 405):
                self._raise_http_error(
                    absolute_response, self._path("absoluteEx/capabilities")
                )

            self._capabilities = ISAPIPTZCapabilities(
                continuous_pan=continuous_pan,
                continuous_tilt=continuous_tilt,
                continuous_zoom=continuous_zoom,
                absolute_azimuth=absolute_azimuth,
                absolute_elevation=absolute_elevation,
                absolute_zoom=absolute_zoom,
                position_3d=position_3d,
            )
            if not self._capabilities.supports_continuous:
                raise ISAPIPTZNotSupportedError(
                    "La cámara no anunció los tres ejes de movimiento continuo ISAPI."
                )
            self._connected = True
            LOGGER.info("Control PTZ ISAPI conectado en %s:%d, canal %d",
                        self._camera_ip, self._port, self._channel_id)
            return self

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def backend_capabilities(self) -> PTZCapabilities:
        with self._lock:
            capabilities = self._require_connection()
            return PTZCapabilities(
                continuous_pan_tilt=(
                    capabilities.continuous_pan is not None
                    and capabilities.continuous_tilt is not None
                ),
                continuous_zoom=capabilities.continuous_zoom is not None,
                absolute_pan_tilt=(
                    capabilities.absolute_azimuth is not None
                    and capabilities.absolute_elevation is not None
                ),
                absolute_zoom=capabilities.absolute_zoom is not None,
                position3d=capabilities.position_3d,
                get_status=True,
                presets=False,
            )

    @property
    def capabilities(self) -> ISAPIPTZCapabilities:
        with self._lock:
            self._require_connection()
            assert self._capabilities is not None
            return self._capabilities

    @property
    def last_command(self) -> MotionCommand | None:
        with self._lock:
            return self._last_command

    @property
    def speeds(self) -> MotionCommand:
        return MotionCommand(self._pan_speed, self._tilt_speed, self._zoom_speed)

    def move(self, *, pan: float = 0.0, tilt: float = 0.0,
             zoom: float = 0.0) -> None:
        command = MotionCommand(
            self._velocity(pan, "pan"), self._velocity(tilt, "tilt"),
            self._velocity(zoom, "zoom")
        )
        with self._lock:
            capabilities = self._require_connection()
            assert capabilities.continuous_pan is not None
            assert capabilities.continuous_tilt is not None
            assert capabilities.continuous_zoom is not None
            payload = self._ptz_data_xml(
                pan=self._map_velocity(command.pan, capabilities.continuous_pan),
                tilt=self._map_velocity(command.tilt, capabilities.continuous_tilt),
                zoom=self._map_velocity(command.zoom, capabilities.continuous_zoom),
            )
            try:
                self._put_xml(self._path("continuous"), payload)
            except ISAPIPTZError:
                self._emergency_stop()
                raise
            self._last_command = command
            self._moving = command != MotionCommand(0.0, 0.0, 0.0)

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0) -> None:
        """Traduce pantalla (+Y abajo) al eje ISAPI existente (+Y arriba)."""
        self.move(pan=pan, tilt=-tilt, zoom=zoom)

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

    def move_up_left(self) -> None:
        self.move(pan=-self._pan_speed, tilt=self._tilt_speed)

    def move_up_right(self) -> None:
        self.move(pan=self._pan_speed, tilt=self._tilt_speed)

    def move_down_left(self) -> None:
        self.move(pan=-self._pan_speed, tilt=-self._tilt_speed)

    def move_down_right(self) -> None:
        self.move(pan=self._pan_speed, tilt=-self._tilt_speed)

    def stop(self) -> None:
        """Detiene simultáneamente pan, tilt y zoom."""
        with self._lock:
            self._require_connection()
            self._put_xml(
                self._path("continuous"), self._ptz_data_xml(pan=0, tilt=0, zoom=0)
            )
            self._moving = False
            self._last_command = MotionCommand(0.0, 0.0, 0.0)

    def get_status(self) -> ISAPIPTZStatus:
        """Lee la posición actual sin mover la cámara."""
        with self._lock:
            self._require_connection()
            root = self._get_xml(self._path("status"), required=True)
            absolute = self._find(root, "AbsoluteHigh")
            if absolute is None:
                raise ISAPIPTZCommandError(
                    "El status ISAPI no contiene AbsoluteHigh."
                )
            return ISAPIPTZStatus(
                azimuth=self._number_text(absolute, "azimuth") / 10.0,
                elevation=self._number_text(absolute, "elevation") / 10.0,
                absolute_zoom=self._number_text(absolute, "absoluteZoom") / 10.0,
            )

    def move_absolute(self, *, azimuth: float, elevation: float,
                      absolute_zoom: float) -> None:
        """Mueve a una posición absoluta usando el endpoint absoluteEx confirmado."""
        with self._lock:
            capabilities = self._require_connection()
            if not capabilities.supports_absolute:
                raise ISAPIPTZNotSupportedError(
                    "La cámara no confirmó /absoluteEx/capabilities."
                )
            assert capabilities.absolute_azimuth is not None
            assert capabilities.absolute_elevation is not None
            assert capabilities.absolute_zoom is not None
            values = {
                "azimuth": capabilities.absolute_azimuth.validate(azimuth, "azimuth"),
                "elevation": capabilities.absolute_elevation.validate(
                    elevation, "elevation"
                ),
                "absoluteZoom": capabilities.absolute_zoom.validate(
                    absolute_zoom, "absolute_zoom"
                ),
            }
            self._put_xml(self._path("absoluteEx"), self._xml("PTZAbsoluteEx", values))

    def position_3d(self, *, start_x: int, start_y: int,
                    end_x: int | None = None,
                    end_y: int | None = None) -> ISAPICommandResult:
        """Centra un punto o encuadra un rectángulo normalizado de 0 a 255.

        Un punto usa inicio y fin iguales. En rectángulos, X inicial menor que X
        final acerca el zoom; el orden inverso lo aleja.
        """
        with self._lock:
            capabilities = self._require_connection()
            if not capabilities.position_3d:
                raise ISAPIPTZNotSupportedError(
                    "La cámara no anunció isSupportPosition3D=true."
                )
            payload = self.build_position_3d_payload(
                start_x=start_x, start_y=start_y, end_x=end_x, end_y=end_y
            )
            LOGGER.debug("Payload Position3D: %s", payload.decode("utf-8"))
            result = self._put_xml(self._path("position3D"), payload)
            LOGGER.info("Position3D enviado; HTTP %d, status=%s",
                        result.status_code, result.status_string or "sin cuerpo")
            return result

    def position3d(self, **coordinates: int) -> ISAPICommandResult:
        """Alias de interfaz común; permanece bloqueado por el ejecutor 7B-0."""
        return self.position_3d(**coordinates)

    @classmethod
    def build_position_3d_payload(
        cls,
        *,
        start_x: int,
        start_y: int,
        end_x: int | None = None,
        end_y: int | None = None,
    ) -> bytes:
        """Construye el XML Position3D sin conectarse ni enviar una solicitud."""
        x1 = cls._coordinate(start_x, "start_x")
        y1 = cls._coordinate(start_y, "start_y")
        x2 = x1 if end_x is None else cls._coordinate(end_x, "end_x")
        y2 = y1 if end_y is None else cls._coordinate(end_y, "end_y")
        root = cls._root("Position3D")
        for name, x, y in (("StartPoint", x1, y1), ("EndPoint", x2, y2)):
            point = ET.SubElement(root, name)
            ET.SubElement(point, "positionX").text = str(x)
            ET.SubElement(point, "positionY").text = str(y)
        return cls._serialize(root)

    def close(self) -> None:
        with self._lock:
            if self._connected and self._moving:
                self._emergency_stop()
            self._connected = False
            self._moving = False
            self._capabilities = None
            self._session.close()
            LOGGER.info("Control PTZ ISAPI cerrado")

    def __enter__(self) -> "ISAPIPTZController":
        return self.connect()

    def __exit__(self, *_: object) -> None:
        self.close()

    def _path(self, suffix: str) -> str:
        return f"/ISAPI/PTZCtrl/channels/{self._channel_id}/{suffix}"

    def _request(self, method: str, path: str, data: bytes | None = None) -> Any:
        url = urljoin(self._base_url, path.lstrip("/"))
        try:
            if method == "GET":
                return self._session.get(url, timeout=self._timeout)
            return self._session.put(
                url, data=data, headers={"Content-Type": "application/xml"},
                timeout=self._timeout
            )
        except requests.Timeout as error:
            raise ISAPIPTZConnectionError(f"Timeout en {method} {path}.") from error
        except requests.RequestException as error:
            raise ISAPIPTZConnectionError(
                f"No fue posible ejecutar {method} {path}."
            ) from error

    def _get_xml(self, path: str, *, required: bool) -> ET.Element:
        response = self._request("GET", path)
        if response.status_code != 200:
            if required:
                self._raise_http_error(response, path)
            raise ISAPIPTZNotSupportedError(f"Endpoint no disponible: {path}")
        return self._parse_xml(response.content, path)

    def _put_xml(self, path: str, payload: bytes) -> ISAPICommandResult:
        response = self._request("PUT", path, payload)
        if not 200 <= response.status_code < 300:
            self._raise_http_error(response, path)
        if not response.content:
            return ISAPICommandResult(response.status_code, None, None)
        root = self._parse_xml(response.content, f"respuesta de {path}")
        status_code = self._optional_text(root, "statusCode")
        status_string = self._optional_text(root, "statusString")
        sub_status_code = self._optional_text(root, "subStatusCode")
        if status_code not in (None, "0", "1") or (
            status_string is not None and status_string.casefold() != "ok"
        ):
            detail = sub_status_code or status_string
            raise ISAPIPTZCommandError(
                f"La cámara rechazó {path}: {detail or 'respuesta ISAPI no válida'}."
            )
        return ISAPICommandResult(
            response.status_code, status_string, sub_status_code
        )

    @staticmethod
    def _raise_http_error(response: Any, path: str) -> None:
        descriptions = {
            400: "solicitud no válida", 401: "autenticación fallida",
            403: "permisos insuficientes", 404: "endpoint no soportado",
            405: "método no permitido",
        }
        detail = descriptions.get(response.status_code, "error HTTP")
        raise ISAPIPTZCommandError(
            f"{path}: HTTP {response.status_code} ({detail})."
        )

    def _emergency_stop(self) -> None:
        try:
            response = self._request(
                "PUT", self._path("continuous"),
                self._ptz_data_xml(pan=0, tilt=0, zoom=0)
            )
            if not 200 <= response.status_code < 300:
                LOGGER.error("La parada de seguridad ISAPI devolvió HTTP %d",
                             response.status_code)
            self._moving = False
            self._last_command = MotionCommand(0.0, 0.0, 0.0)
        except ISAPIPTZError as error:
            self._moving = False
            self._last_command = None
            LOGGER.error("Falló también la parada de seguridad ISAPI: %s", error)

    def _require_connection(self) -> ISAPIPTZCapabilities:
        if not self._connected or self._capabilities is None:
            raise ISAPIPTZConnectionError(
                "PTZ ISAPI no está conectado; ejecute connect() primero."
            )
        return self._capabilities

    @classmethod
    def _ptz_data_xml(cls, *, pan: int, tilt: int, zoom: int) -> bytes:
        return cls._xml("PTZData", {"pan": pan, "tilt": tilt, "zoom": zoom})

    @classmethod
    def _xml(cls, root_name: str, values: dict[str, float | int]) -> bytes:
        root = cls._root(root_name)
        for name, value in values.items():
            ET.SubElement(root, name).text = f"{value:g}"
        return cls._serialize(root)

    @staticmethod
    def _root(name: str) -> ET.Element:
        ET.register_namespace("", ISAPI_NAMESPACE)
        return ET.Element(f"{{{ISAPI_NAMESPACE}}}{name}", {"version": "2.0"})

    @staticmethod
    def _serialize(root: ET.Element) -> bytes:
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _find(cls, root: ET.Element, name: str) -> ET.Element | None:
        return next((item for item in root.iter()
                     if cls._local_name(item.tag) == name), None)

    @classmethod
    def _optional_text(cls, root: ET.Element, name: str) -> str | None:
        element = cls._find(root, name)
        if element is None or element.text is None:
            return None
        value = element.text.strip()
        return value or None

    @classmethod
    def _number_text(cls, root: ET.Element, name: str) -> float:
        value = cls._optional_text(root, name)
        if value is None:
            raise ISAPIPTZCommandError(
                f"El status no contiene el campo {name}."
            )
        try:
            return float(value)
        except ValueError as error:
            raise ISAPIPTZCommandError(
                f"El status contiene un valor inválido para {name}."
            ) from error

    @classmethod
    def _bool_element(cls, root: ET.Element, name: str) -> bool:
        return (cls._optional_text(root, name) or "").casefold() == "true"

    @classmethod
    def _xy_ranges(cls, root: ET.Element, name: str
                   ) -> tuple[ValueRange | None, ValueRange | None]:
        space = cls._find(root, name)
        if space is None:
            return None, None
        return cls._child_range(space, "XRange"), cls._child_range(space, "YRange")

    @classmethod
    def _single_range(cls, root: ET.Element, name: str) -> ValueRange | None:
        space = cls._find(root, name)
        if space is None:
            return None
        # Hikvision usa ZRange para zoom. XRange se conserva como compatibilidad
        # con respuestas antiguas observadas en algunos firmwares.
        return cls._child_range(space, "ZRange") or cls._child_range(space, "XRange")

    @classmethod
    def _child_range(cls, root: ET.Element, name: str) -> ValueRange | None:
        node = cls._find(root, name)
        if node is None:
            return None
        minimum = cls._optional_text(node, "Min")
        maximum = cls._optional_text(node, "Max")
        if minimum is None or maximum is None:
            return None
        try:
            return ValueRange(float(minimum), float(maximum))
        except ValueError:
            return None

    @classmethod
    def _attribute_range(cls, root: ET.Element, name: str) -> ValueRange | None:
        node = cls._find(root, name)
        if node is None or "min" not in node.attrib or "max" not in node.attrib:
            return None
        try:
            return ValueRange(float(node.attrib["min"]), float(node.attrib["max"]))
        except ValueError:
            return None

    @staticmethod
    def _parse_xml(content: bytes, label: str) -> ET.Element:
        try:
            return ET.fromstring(content)
        except ET.ParseError as error:
            raise ISAPIPTZCommandError(f"XML inválido en {label}.") from error

    @staticmethod
    def _map_velocity(value: float, limits: ValueRange) -> int:
        mapped = value * (limits.maximum if value >= 0 else abs(limits.minimum))
        return int(round(mapped))

    @staticmethod
    def _coordinate(value: int, label: str) -> int:
        if isinstance(value, bool):
            raise PTZConfigurationError(f"{label} debe ser un entero entre 0 y 255.")
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise PTZConfigurationError(
                f"{label} debe ser un entero entre 0 y 255."
            ) from error
        if result != value or not 0 <= result <= 255:
            raise PTZConfigurationError(f"{label} debe estar entre 0 y 255.")
        return result

    @staticmethod
    def _required(value: str, label: str, *, strip: bool = True) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PTZConfigurationError(f"Falta configurar {label}.")
        return value.strip() if strip else value

    @classmethod
    def _validate_host(cls, value: str) -> str:
        host = cls._required(value, "IP o host")
        if any(character.isspace() for character in host) or any(
            character in host for character in "/@?#:"
        ):
            raise PTZConfigurationError("La IP o el host no es válido.")
        return host

    @staticmethod
    def _velocity(value: float, label: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZConfigurationError(f"La velocidad {label} debe ser numérica.") from error
        if not -1.0 <= result <= 1.0:
            raise PTZConfigurationError(
                f"La velocidad {label} debe estar entre -1.0 y 1.0."
            )
        return result

    @classmethod
    def _speed(cls, value: float, label: str) -> float:
        result = cls._velocity(value, label)
        if result < 0:
            raise PTZConfigurationError(f"{label} debe estar entre 0.0 y 1.0.")
        return result

    @classmethod
    def _direction(cls, value: float | None, default: float, label: str) -> float:
        return default if value is None else cls._speed(value, f"velocidad {label}")
