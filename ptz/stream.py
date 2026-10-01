"""Adquisición robusta de video RTSP para Detectcam.

Este módulo se limita a recibir video. No contiene inferencia, tracking ni control
PTZ.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from time import monotonic
from urllib.parse import quote

import cv2
import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)

DEFAULT_CAMERA_IP = "192.168.1.100"
RTSP_PORT = 554
RTSP_CHANNEL = "101"
MAX_READ_FAILURES = 10
RECONNECT_DELAY = 2.0
OPEN_TIMEOUT = 5.0
READ_TIMEOUT = 5.0
STOP_TIMEOUT = 6.0
LOW_LATENCY_OPTIONS = (
    "fflags;nobuffer|flags;low_delay|analyzeduration;500000|probesize;262144"
)


@dataclass(frozen=True)
class ReceivedFrame:
    """Cuadro y hora local de decodificación leídos de manera atómica."""

    frame_id: int
    frame: NDArray[np.uint8]
    received_at: float


class StreamConfigurationError(ValueError):
    """Indica que falta configuración o que uno de sus valores no es válido."""


class RTSPStream:
    """Mantiene en segundo plano el frame más reciente de una fuente RTSP."""

    def __init__(
        self,
        *,
        username: str,
        password: str,
        camera_ip: str = DEFAULT_CAMERA_IP,
        port: int = RTSP_PORT,
        channel: str = RTSP_CHANNEL,
        max_read_failures: int = MAX_READ_FAILURES,
        reconnect_delay: float = RECONNECT_DELAY,
        open_timeout: float = OPEN_TIMEOUT,
        read_timeout: float = READ_TIMEOUT,
        low_latency: bool = True,
        transport: str = "tcp",
        decoder_threads: int = 1,
    ) -> None:
        self._username = self._required_value(username, "usuario")
        self._password = self._required_value(password, "contraseña")
        self._camera_ip = self._validate_host(camera_ip)
        self._channel = self._required_value(channel, "canal RTSP")

        if not 1 <= port <= 65535:
            raise StreamConfigurationError("El puerto RTSP debe estar entre 1 y 65535.")
        if max_read_failures < 1:
            raise StreamConfigurationError("max_read_failures debe ser mayor que cero.")
        if reconnect_delay < 0 or open_timeout <= 0 or read_timeout <= 0:
            raise StreamConfigurationError(
                "Los timeouts deben ser positivos y reconnect_delay no puede ser negativo."
            )

        if transport not in {"tcp", "udp"}:
            raise StreamConfigurationError("DETECTCAM_RTSP_TRANSPORT debe ser tcp o udp.")
        if isinstance(decoder_threads, bool) or not isinstance(decoder_threads, int) or not 0 <= decoder_threads <= 64:
            raise StreamConfigurationError("DETECTCAM_RTSP_DECODER_THREADS debe estar entre 0 y 64.")
        if not isinstance(low_latency, bool):
            raise StreamConfigurationError("DETECTCAM_RTSP_LOW_LATENCY debe ser true o false.")
        self._transport = transport
        self._low_latency = low_latency
        self._decoder_threads = decoder_threads
        self._port = port
        self._max_read_failures = max_read_failures
        self._reconnect_delay = reconnect_delay
        self._open_timeout_ms = round(open_timeout * 1000)
        self._read_timeout_ms = round(read_timeout * 1000)

        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: NDArray[np.uint8] | None = None
        self._last_frame_time: float | None = None
        self._frame_id = 0
        self._connected = False

    @classmethod
    def from_environment(cls, **overrides: object) -> "RTSPStream":
        """Crea el stream usando las variables de entorno de Detectcam."""
        username = os.getenv("DETECTCAM_CAMERA_USER", "")
        password = os.getenv("DETECTCAM_CAMERA_PASSWORD", "")
        camera_ip = os.getenv("DETECTCAM_CAMERA_IP", DEFAULT_CAMERA_IP)

        missing = []
        if not username.strip():
            missing.append("DETECTCAM_CAMERA_USER")
        if not password:
            missing.append("DETECTCAM_CAMERA_PASSWORD")
        if missing:
            raise StreamConfigurationError(
                "Faltan variables de entorno requeridas: " + ", ".join(missing)
            )

        options = dict(overrides)
        options.setdefault("channel", os.getenv("DETECTCAM_RTSP_CHANNEL", RTSP_CHANNEL))
        options.setdefault("transport", os.getenv("DETECTCAM_RTSP_TRANSPORT", "tcp").strip().lower())
        low_latency = os.getenv("DETECTCAM_RTSP_LOW_LATENCY", "true").strip().lower()
        if low_latency not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise StreamConfigurationError("DETECTCAM_RTSP_LOW_LATENCY debe ser true o false.")
        options.setdefault("low_latency", low_latency in {"true", "1", "yes", "on"})
        try:
            options.setdefault("decoder_threads", int(os.getenv("DETECTCAM_RTSP_DECODER_THREADS", "1")))
        except ValueError as error:
            raise StreamConfigurationError("DETECTCAM_RTSP_DECODER_THREADS debe ser entero.") from error
        return cls(username=username, password=password, camera_ip=camera_ip, **options)

    def start(self) -> "RTSPStream":
        """Inicia la adquisición sin bloquear al hilo que realiza la llamada."""
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._capture_loop,
                name="detectcam-rtsp",
                daemon=True,
            )
            self._thread.start()
        return self

    def read(self, *, copy: bool = True) -> NDArray[np.uint8] | None:
        """Devuelve el frame más reciente, o ``None`` si aún no hay uno disponible."""
        with self._state_lock:
            frame = self._latest_frame
            if frame is None:
                return None
            return frame.copy() if copy else frame

    def read_with_id(
        self, *, copy: bool = True, after_id: int | None = None
    ) -> tuple[int, NDArray[np.uint8] | None]:
        """Devuelve el frame más reciente solo si es posterior a ``after_id``."""
        with self._state_lock:
            frame = self._latest_frame
            if frame is None or self._frame_id == after_id:
                return self._frame_id, None
            return self._frame_id, frame.copy() if copy else frame

    def read_packet(
        self, *, copy: bool = True, after_id: int | None = None,
    ) -> ReceivedFrame | None:
        """No mezcla la imagen de un cuadro con la hora del siguiente.

        Con copy=False el consumidor debe tratar el ndarray como sólo lectura.
        received_at no incluye la demora previa de cámara, red ni decodificador.
        """
        with self._state_lock:
            if self._latest_frame is None or self._last_frame_time is None or self._frame_id == after_id:
                return None
            return ReceivedFrame(
                self._frame_id,
                self._latest_frame.copy() if copy else self._latest_frame,
                self._last_frame_time,
            )

    @property
    def frame_id(self) -> int:
        """Contador monotónico de frames válidos recibidos por esta instancia."""
        with self._state_lock:
            return self._frame_id

    def is_connected(self) -> bool:
        """Indica si la captura está abierta y no ha agotado sus reintentos de lectura."""
        with self._state_lock:
            return self._connected

    @property
    def last_frame_age(self) -> float | None:
        """Edad local desde decodificación; no mide demora cámara-pantalla."""
        with self._state_lock:
            if self._last_frame_time is None:
                return None
            return monotonic() - self._last_frame_time

    def stop(self, timeout: float = STOP_TIMEOUT) -> None:
        """Solicita la parada, libera la captura y espera un tiempo limitado."""
        self._stop_event.set()
        with self._state_lock:
            capture = self._capture
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(timeout, 0.0))
            if thread.is_alive():
                LOGGER.warning(
                    "El hilo RTSP no terminó dentro del timeout; liberando la captura"
                )
                if capture is not None:
                    capture.release()
                thread.join(timeout=0.5)
        elif capture is not None:
            capture.release()
        with self._state_lock:
            self._connected = False
            if thread is None or not thread.is_alive():
                self._thread = None
        LOGGER.info("Stream RTSP detenido")

    def __enter__(self) -> "RTSPStream":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    capture = self._open_capture()
                except Exception as error:
                    LOGGER.error(
                        "Error al abrir el stream RTSP: %s; se reintentará",
                        type(error).__name__,
                    )
                    capture = None
                if capture is None:
                    if self._stop_event.wait(self._reconnect_delay):
                        break
                    continue

                failures = 0
                while not self._stop_event.is_set():
                    try:
                        ok, frame = capture.read()
                    except Exception as error:
                        LOGGER.warning(
                            "Error al leer el stream RTSP: %s",
                            type(error).__name__,
                        )
                        ok, frame = False, None
                    if ok and frame is not None:
                        failures = 0
                        with self._state_lock:
                            self._latest_frame = frame
                            self._last_frame_time = monotonic()
                            self._frame_id += 1
                        continue

                    failures += 1
                    if failures == 1:
                        LOGGER.warning("Lectura de frame RTSP fallida; se volverá a intentar")
                    if failures < self._max_read_failures:
                        if self._stop_event.wait(0.05):
                            break
                        continue

                    LOGGER.warning(
                        "Se alcanzaron %d lecturas fallidas; intentando reconexión",
                        self._max_read_failures,
                    )
                    break

                self._release_capture(capture, clear_frame=True)
                if not self._stop_event.is_set():
                    self._stop_event.wait(self._reconnect_delay)
        except Exception:
            LOGGER.exception("Error inesperado en el hilo de adquisición RTSP")
        finally:
            with self._state_lock:
                capture = self._capture
            if capture is not None:
                self._release_capture(capture, clear_frame=True)

    def _open_capture(self) -> cv2.VideoCapture | None:
        LOGGER.info(
            "Intentando conexión RTSP con %s:%d, canal %s",
            self._camera_ip,
            self._port,
            self._channel,
        )
        # El backend FFmpeg de OpenCV interpreta esta opción y no expone la URL.
        options = f"rtsp_transport;{self._transport}"
        if self._low_latency:
            options += "|" + LOW_LATENCY_OPTIONS
        custom_options = bool(os.getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS"))
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", options)
        params = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self._open_timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self._read_timeout_ms,
            cv2.CAP_PROP_N_THREADS,
            self._decoder_threads,
        ]
        capture = cv2.VideoCapture(self._build_url(), cv2.CAP_FFMPEG, params)
        if not capture.isOpened():
            capture.release()
            LOGGER.error(
                "No se pudo abrir el stream RTSP de %s:%d; se reintentará",
                self._camera_ip,
                self._port,
            )
            return None

        # Algunos backends ignoran esta propiedad; en esos casos se mantiene la
        # estrategia principal de conservar solo el frame compartido más reciente.
        buffer_supported = capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        LOGGER.info(
            "RTSP: perfil=%s, transporte=%s, hilos decoder=%d, buffersize soportado=%s",
            "externo" if custom_options else ("baja latencia" if self._low_latency else "estándar"),
            self._transport, self._decoder_threads, buffer_supported,
        )
        with self._state_lock:
            if self._stop_event.is_set():
                capture.release()
                return None
            self._capture = capture
            self._connected = True
        LOGGER.info("Conexión RTSP establecida")
        return capture

    def _release_capture(
        self,
        capture: cv2.VideoCapture,
        *,
        clear_frame: bool,
    ) -> None:
        capture.release()
        with self._state_lock:
            if self._capture is capture:
                self._capture = None
            self._connected = False
            if clear_frame:
                self._latest_frame = None
                self._last_frame_time = None

    def _build_url(self) -> str:
        username = quote(self._username, safe="")
        password = quote(self._password, safe="")
        host = self._camera_ip
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return (
            f"rtsp://{username}:{password}@{host}:{self._port}"
            f"/Streaming/Channels/{quote(self._channel, safe='')}"
        )

    @staticmethod
    def _required_value(value: str, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise StreamConfigurationError(f"Falta configurar {label}.")
        return value.strip() if label != "contraseña" else value

    @staticmethod
    def _validate_host(value: str) -> str:
        host = RTSPStream._required_value(value, "IP o host de la cámara")
        if any(character.isspace() for character in host) or any(
            character in host for character in "/@?#"
        ):
            raise StreamConfigurationError("La IP o el host de la cámara no es válido.")
        return host
