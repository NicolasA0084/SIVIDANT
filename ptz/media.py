"""Capturas y grabación local fuera de los hilos de Tk y de inferencia."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
import logging
from math import isfinite
import os
from pathlib import Path
import threading
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np
from numpy.typing import NDArray

from ptz.stream import RTSPStream, ReceivedFrame

LOGGER = logging.getLogger(__name__)
MAX_FRAME_AGE = 0.5


@dataclass(frozen=True)
class MediaStatus:
    recording: bool = False
    stopping: bool = False
    started_at: float | None = None
    path: Path | None = None
    message: str = "Capturas JPG · video MP4"


class MediaStore:
    """Guarda imágenes completas y video limpio a los FPS de la cámara.

    El grabador consume RTSP directamente; no depende de los FPS de YOLO ni de
    los redibujados de la ventana. Repite cuadros cuando es necesario para
    conservar la duración real en un archivo de frecuencia constante.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        configured = directory or os.getenv("DETECTCAM_MEDIA_DIR")
        self._directory = Path(configured or Path.home() / "Videos" / "Detectcam").expanduser().resolve()
        self._lock = threading.Lock()
        self._status = MediaStatus()
        self._closed = False
        self._images = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detectcam-capturas")
        self._image_slots = threading.BoundedSemaphore(4)
        self._record_stop = threading.Event()
        self._record_thread: threading.Thread | None = None

    @property
    def directory(self) -> Path:
        with self._lock:
            return self._directory

    @property
    def status(self) -> MediaStatus:
        with self._lock:
            return self._status

    def set_directory(self, directory: str | Path) -> None:
        destination = Path(directory).expanduser().resolve()
        with self._lock:
            if self._status.recording:
                raise ValueError("Detén la grabación antes de cambiar de carpeta")
            self._directory = destination
        LOGGER.info("Carpeta de capturas y videos: %s", destination)

    def _path(self, subdirectory: str, prefix: str, extension: str) -> Path:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S_%f")
        return self._directory / subdirectory / f"{prefix}_{stamp}_{uuid4().hex[:8]}{extension}"

    def _message(self, message: str) -> None:
        with self._lock:
            self._status = replace(self._status, message=message)

    def capture(self, frame: NDArray[np.uint8]) -> Path | None:
        with self._lock:
            if self._closed:
                return None
            if not self._image_slots.acquire(blocking=False):
                LOGGER.warning("Captura omitida: todavía se están guardando imágenes")
                return None
            path = self._path("capturas", "captura", ".jpg")
            try:
                self._images.submit(self._save_image, path, frame.copy())
            except Exception:
                self._image_slots.release()
                raise
        return path

    def _save_image(self, path: Path, frame: NDArray[np.uint8]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), frame):
                raise OSError("No se pudo escribir la imagen")
            self._message(f"Captura guardada: {path.name}")
            LOGGER.info("Captura guardada: %s", path)
        except Exception as error:
            self._message("Error al guardar captura · consulta Eventos")
            LOGGER.error("No se pudo guardar la captura en %s: %s", path, error)
        finally:
            self._image_slots.release()

    def start_recording(self, source: RTSPStream) -> bool:
        first = source.read_packet(copy=True)
        if (not source.is_connected() or first is None
                or monotonic() - first.received_at > MAX_FRAME_AGE):
            self._message("No se puede grabar: cámara sin señal reciente")
            LOGGER.warning("No se puede iniciar la grabación: cámara sin señal reciente")
            return False
        rate = source.frame_rate
        fps = float(rate) if isinstance(rate, (int, float)) and isfinite(rate) and 0 < rate <= 240 else 25.0
        with self._lock:
            if self._closed or self._status.recording:
                return False
            path = self._path("grabaciones", "video", ".mp4")
            self._record_stop = threading.Event()
            self._status = MediaStatus(recording=True, path=path, message="Preparando grabación…")
            self._record_thread = threading.Thread(
                target=self._record, args=(source, first, path, fps, self._record_stop),
                name="detectcam-grabacion", daemon=True,
            )
            self._record_thread.start()
        return True

    def stop_recording(self) -> None:
        with self._lock:
            if self._status.recording:
                self._status = replace(self._status, stopping=True, message="Guardando video…")
                self._record_stop.set()

    def _record(
        self, source: RTSPStream, first: ReceivedFrame, path: Path,
        fps: float, stop: threading.Event,
    ) -> None:
        writer = None
        frames_written = 0
        message = "Grabación cancelada"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = first.frame
            shape = frame.shape
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (shape[1], shape[0]),
            )
            if not writer.isOpened():
                raise OSError("No se pudo abrir el codificador MP4")
            started_at = monotonic()
            with self._lock:
                self._status = replace(self._status, started_at=started_at, message="Grabando video")
            LOGGER.info("Grabación iniciada: %s (%.2f FPS, %d×%d)", path, fps, shape[1], shape[0])
            frame_id, received_at = first.frame_id, first.received_at
            while not stop.is_set():
                packet = source.read_packet(copy=False, after_id=frame_id)
                if packet is not None:
                    if packet.frame.shape != shape:
                        LOGGER.warning("Grabación detenida por cambio de resolución")
                        break
                    frame, frame_id, received_at = packet.frame, packet.frame_id, packet.received_at
                now = monotonic()
                if not source.is_connected() or now - received_at > MAX_FRAME_AGE:
                    LOGGER.warning("Grabación detenida por pérdida de señal; cerrando el archivo")
                    break
                due_frames = int((now - started_at) * fps) + 1
                while frames_written < due_frames and not stop.is_set():
                    writer.write(frame)
                    frames_written += 1
                stop.wait(max(0.001, min(0.01, started_at + frames_written / fps - monotonic())))
            message = f"Video guardado: {path.name}" if frames_written else "Grabación cancelada"
        except Exception as error:
            message = "Error al grabar video · consulta Eventos"
            LOGGER.error("No se pudo grabar video en %s: %s", path, error)
        finally:
            if writer is not None:
                try:
                    writer.release()
                except Exception as error:
                    message = "Error al cerrar video · consulta Eventos"
                    LOGGER.error("No se pudo cerrar el video %s: %s", path, error)
            if frames_written:
                LOGGER.info("Grabación finalizada: %s (%.2f s)", path, frames_written / fps)
            else:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._lock:
                self._status = MediaStatus(path=path if frames_written else None, message=message)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self.stop_recording()
        if self._record_thread is not None:
            self._record_thread.join()
        self._images.shutdown(wait=True)
