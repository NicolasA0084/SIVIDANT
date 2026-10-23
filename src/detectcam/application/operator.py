"""Arbitraje de operador fuera de Tk y de YOLO, sin I/O de cámara.

El runtime recibe sólo la intención vigente. Su estado se lee como snapshot;
ninguna propiedad de la interfaz adquiere un bloqueo que incluya red. El reloj
del operador avanza aunque la inferencia tarde o no haya nuevas detecciones.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from math import isfinite
import threading
from time import monotonic
from typing import Callable

from .tracking_settings import TrackingOptions

LOGGER = logging.getLogger(__name__)
ARM_CONFIRMATION = "MOVER"
MAX_ALLOWED_MANUAL_ZOOM = .30


@dataclass(frozen=True)
class OperatorStatus:
    hardware: bool = False
    armed: bool = False
    mode: str = "manual"
    state: str = "DISARMED"
    reason: str = "Listo · selecciona un modo y arma el PTZ"
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    zoom_supported: bool = False
    phase: str = "IDLE"
    command_id: int = 0
    backend_latency_ms: float | None = None


class OperatorControl:
    def __init__(self, executor, video_age: Callable[[], float | None], *,
                 controller=None, tracking_options: TrackingOptions | None = None,
                 on_emergency: Callable[[], None] = lambda: None,
                 clock: Callable[[], float] = monotonic, telemetry=None,
                 max_video_age_s: float = .5) -> None:
        self.executor = executor  # Alias de compatibilidad para la vista anterior.
        self.controller = controller
        self.tracking_options = tracking_options
        self._video_age = video_age
        self._clock = clock
        self._on_emergency = on_emergency
        self._telemetry = telemetry
        self._max_video_age_s = max_video_age_s
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._actions = deque(maxlen=32)
        self._generation = 0
        self._manual = (0., 0., 0., float("-inf"))
        self._heartbeat = float("-inf")
        self._preview_at = None
        self._automatic = None
        self._tracking_revision = None
        self._mode = "manual"
        self._status = OperatorStatus()
        self._thread = None
        self._last_intent = None

    @property
    def status(self) -> OperatorStatus:
        with self._lock:
            return self._status

    @property
    def last_intent(self):
        with self._lock:
            return self._last_intent

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeat = self._clock()

    def preview_seen(self, captured_at: float | None) -> None:
        if captured_at is not None and (not isfinite(captured_at) or captured_at > self._clock()):
            captured_at = None
        with self._lock:
            self._preview_at = captured_at

    def set_manual(self, pan: float, tilt: float, zoom: float = 0.) -> None:
        values = tuple(float(v) for v in (pan, tilt, zoom))
        if not all(isfinite(v) for v in values):
            raise ValueError("Entrada manual no finita")
        with self._lock:
            self._manual = (*values, self._clock())
        self._wake.set()

    def request(self, action: str, value: str = "") -> None:
        if action not in {"arm", "disarm", "emergency", "mode", "recover"}:
            raise ValueError("Acción desconocida")
        if action == "mode" and value not in {"manual", "automatic"}:
            raise ValueError("Modo desconocido")
        with self._lock:
            if self._closing.is_set():
                return
            if action in {"disarm", "emergency", "mode", "recover"}:
                self._generation += 1
                self._actions.clear()
                self._automatic = None
            self._manual = (0., 0., 0., float("-inf"))
            self._actions.append((action, value, self._generation))
        # STOP no espera al próximo tick ni a que termine un request de movimiento.
        if self.executor is not None:
            if action == "emergency":
                self.executor.emergency_stop()
            elif action in {"disarm", "mode", "recover"}:
                self.executor.disarm()
        self._wake.set()

    def submit_snapshot(self, snapshot, *, settings_revision: int = 0) -> None:
        with self._lock:
            self._automatic = (snapshot, settings_revision)
        self._wake.set()

    def start(self) -> None:
        if self.executor is not None:
            self.executor.start()
        self._thread = threading.Thread(target=self._run, name="detectcam-operator", daemon=True)
        self._thread.start()

    def step(self) -> None:
        now = self._clock()
        with self._lock:
            actions = list(self._actions)
            self._actions.clear()
        for action, value, generation in actions:
            # Armado y cancelación comparten este breve lock. arm() no hace red:
            # una emergencia que llegue durante el tick invalida armados viejos.
            with self._lock:
                if generation != self._generation:
                    continue
                if action == "arm" and value == ARM_CONFIRMATION and self.executor is not None:
                    self.executor.arm()
            if action == "mode":
                self._mode = value
            if self.controller is not None and action in {"mode", "disarm", "emergency", "recover"}:
                self.controller.reset()
            if action == "emergency":
                self._on_emergency()
            if self.executor is None:
                continue
            if action == "recover":
                self.executor.recover()
        settings = self.tracking_options.snapshot if self.tracking_options else None
        revision = settings.revision if settings else 0
        if revision != self._tracking_revision:
            if self.controller is not None:
                self.controller.reset()
            if self.executor is not None:
                self.executor.stop("Ajustes actualizados; esperando evidencia nueva")
            self._tracking_revision = revision
        with self._lock:
            pan, tilt, zoom, pressed_at = self._manual
            heartbeat, preview_at = self._heartbeat, self._preview_at
            automatic = self._automatic
            pending = bool(self._actions)
        runtime = self.executor
        intent = None
        reason = "Simulación · hardware deshabilitado"
        if runtime is not None:
            physical = runtime.snapshot
            if isfinite(heartbeat) and 0 <= now-heartbeat <= .5:
                # Presencia del operador no renueva la evidencia del movimiento.
                runtime.heartbeat()
            if not pending and physical.armed:
                video_age = self._video_age()
                if now-heartbeat > .5:
                    runtime.disarm()
                    reason = "Interfaz sin respuesta · desarmado"
                elif (video_age is None or not isfinite(video_age) or video_age < 0
                      or video_age > self._max_video_age_s):
                    runtime.stop("Video sin señal reciente")
                    reason = "Video sin señal reciente"
                elif self._mode == "manual":
                    display_age = float("inf") if preview_at is None else now-preview_at
                    if (not isfinite(display_age) or not 0 <= display_age <= self._max_video_age_s
                            or not 0 <= now-pressed_at <= .25):
                        runtime.stop("Entrada manual o vista previa vencida")
                    else:
                        runtime.submit_velocity(pan, tilt, zoom, expires_at=min(
                            pressed_at+.25, preview_at+self._max_video_age_s, now+.12))
                elif (automatic is not None and automatic[1] == revision
                      and (settings is None or bool(settings.allowed_classes)) and self.controller is not None):
                    intent = self.controller.command(automatic[0], now=now, video_age_s=video_age)
                    if intent.expires_at <= now or (intent.pan == 0 and intent.tilt == 0 and intent.zoom == 0):
                        runtime.stop(intent.reason)
                    else:
                        runtime.submit_velocity(intent.pan, intent.tilt, intent.zoom,
                            expires_at=intent.expires_at, frame_id=intent.source_frame_id,
                            settings_revision=revision)
                else:
                    runtime.stop("Esperando detección válida con los ajustes actuales")
            physical = runtime.snapshot
            reason = physical.reason if reason == "Simulación · hardware deshabilitado" else reason
            status = OperatorStatus(
                hardware=physical.connected, armed=physical.armed, mode=self._mode,
                state=str(getattr(physical.state, "value", physical.state)), reason=reason,
                pan=physical.pan, tilt=physical.tilt, zoom=physical.zoom,
                zoom_supported=physical.zoom_supported,
                phase=str(getattr(intent.phase, "value", intent.phase)) if intent else "IDLE",
                command_id=physical.command_id, backend_latency_ms=physical.last_backend_latency_ms,
            )
        else:
            # Simulación conserva el cálculo de seguimiento y muestra su intención.
            if (self._mode == "automatic" and automatic is not None and automatic[1] == revision
                    and self.controller is not None and (settings is None or settings.allowed_classes)):
                intent = self.controller.command(automatic[0], now=now, video_age_s=self._video_age())
            status = OperatorStatus(mode=self._mode, reason=reason,
                phase=str(getattr(intent.phase, "value", intent.phase)) if intent else "IDLE")
        with self._lock:
            self._status, self._last_intent = status, intent

    def close(self) -> None:
        if self._closing.is_set():
            return
        self._closing.set()
        if self.executor is not None:
            self.executor.emergency_stop()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.)
        if self.executor is not None:
            self.executor.close()

    def _run(self) -> None:
        try:
            while not self._closing.is_set():
                self._wake.clear()
                self.step()
                self._wake.wait(.02)
        except Exception:
            LOGGER.exception("Falló el arbitraje del operador")
            if self.executor is not None:
                self.executor.emergency_stop()
            with self._lock:
                self._status = OperatorStatus(mode=self._mode, state="FAULT", reason="Control detenido")
