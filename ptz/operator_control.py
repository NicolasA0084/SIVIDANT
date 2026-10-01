"""Arbitraje manual/automático en un hilo separado de Tk y de YOLO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import logging
import threading
from time import monotonic
from typing import Callable

from ptz.ptz_decision import PTZDecision
from ptz.ptz_execution_policy import PTZExecutionContext
from ptz.ptz_executor import PTZCommandExecutor
from ptz.tracking_settings import TrackingOptions

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OperatorStatus:
    hardware: bool = False
    armed: bool = False
    mode: str = "manual"
    state: str = "DISARMED"
    reason: str = "Listo · selecciona un modo y arma el PTZ"
    pan: float = 0.0
    tilt: float = 0.0


class OperatorControl:
    """Un único consumidor serializa todas las órdenes hacia el ejecutor.

    La UI renueva su presencia; las entradas manuales caducan a los 250 ms.
    Una parada elimina armados pendientes. Cambiar de modo siempre desarma.
    """

    def __init__(
        self, executor: PTZCommandExecutor | None,
        video_age: Callable[[], float | None], *,
        on_emergency: Callable[[], None] = lambda: None,
        clock: Callable[[], float] = monotonic,
        tracking_options: TrackingOptions | None = None,
    ) -> None:
        self.executor = executor
        self.tracking_options = tracking_options
        self._tracking_revision: int | None = None
        self._video_age = video_age
        self._on_emergency = on_emergency
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closing = threading.Event()
        self._actions: deque[tuple[str, str]] = deque()
        self._manual = (0.0, 0.0, float("-inf"))
        self._heartbeat = float("-inf")
        self._preview_at: float | None = None
        self._automatic: tuple[PTZDecision, PTZExecutionContext, float, int | None] | None = None
        self._mode = "manual"
        self._reason = "Listo · selecciona un modo y arma el PTZ"
        self._status = OperatorStatus(hardware=executor is not None)
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> OperatorStatus:
        with self._lock:
            return self._status

    def heartbeat(self) -> None:
        with self._lock:
            self._heartbeat = self._clock()

    def preview_seen(self, captured_at: float | None) -> None:
        with self._lock:
            self._preview_at = captured_at

    def request(self, action: str, value: str = "") -> None:
        if action not in {"arm", "disarm", "emergency", "mode"}:
            raise ValueError("Acción de interfaz desconocida")
        if action == "mode" and value not in {"manual", "automatic"}:
            raise ValueError("Modo de control desconocido")
        with self._lock:
            if self._closing.is_set():
                return
            if action in {"disarm", "emergency"}:
                self._actions.clear()
            self._manual = (0.0, 0.0, float("-inf"))
            self._actions.append((action, value))
        self._wake.set()

    def set_manual(self, pan: float, tilt: float) -> None:
        with self._lock:
            self._manual = (pan, tilt, self._clock())
        self._wake.set()

    def submit_automatic(
        self, decision: PTZDecision, context: PTZExecutionContext, *,
        settings_revision: int | None = None,
    ) -> None:
        with self._lock:
            self._automatic = (decision, context, self._clock(), settings_revision)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="detectcam-controls", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._closing.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        elif self.executor is not None:
            self.executor.emergency_stop()
            self.executor.close()

    def step(self) -> None:
        """Un ciclo determinista, también verificable con un backend falso."""
        with self._lock:
            actions = list(self._actions)
            self._actions.clear()
        executor = self.executor
        settings = None if self.tracking_options is None else self.tracking_options.snapshot
        if settings is not None and settings.revision != self._tracking_revision:
            if executor is not None:
                executor.set_min_track_confidence(settings.min_confidence)
            self._tracking_revision = settings.revision
        for action, value in actions:
            if action == "mode":
                if executor is not None:
                    executor.disarm()
                self._mode = value
                self._reason = "Modo cambiado · PTZ desarmado"
                with self._lock:
                    self._automatic = None
                    self._manual = (0.0, 0.0, float("-inf"))
                LOGGER.info("Modo %s seleccionado; PTZ desarmado", value)
            elif executor is None:
                self._reason = "Simulación · hardware PTZ deshabilitado"
            elif action == "arm":
                try:
                    if not executor.armed:
                        executor.arm(value)
                    self._reason = "PTZ armado"
                except Exception as error:
                    self._reason = str(error)
                    LOGGER.warning("Armado rechazado: %s", error)
            elif action == "disarm":
                self._reason = executor.disarm().reason
            elif action == "emergency":
                self._reason = executor.emergency_stop().reason
                self._on_emergency()

        with self._lock:
            pan, tilt, pressed_at = self._manual
            heartbeat = self._heartbeat
            preview_at = self._preview_at
            automatic = self._automatic
            pending_actions = bool(self._actions) or self._closing.is_set()
        now = self._clock()
        if settings is not None and (
            not settings.allowed_classes
            or (automatic is not None and automatic[3] != settings.revision)
        ):
            # No ejecutar una inferencia pendiente con los filtros anteriores.
            automatic = None
        if executor is not None and executor.armed and not pending_actions:
            if now - heartbeat > 0.5:
                executor.disarm()
                self._reason = "Interfaz sin respuesta · PTZ desarmado"
            elif self._mode == "manual":
                age = self._video_age()
                age = None if age is None or preview_at is None else max(age, now - preview_at)
                result = executor.execute_manual(
                    pan, tilt, frame_age_ms=None if age is None else age * 1000.0,
                    input_age_ms=max(0.0, (now - pressed_at) * 1000.0),
                )
                self._reason = result.reason
            elif automatic is not None:
                decision, context, received_at, _revision = automatic
                elapsed = max(0.0, (now - received_at) * 1000.0)
                context = replace(
                    context,
                    target_age_ms=None if context.target_age_ms is None else context.target_age_ms + elapsed,
                    frame_age_ms=None if context.frame_age_ms is None else context.frame_age_ms + elapsed,
                )
                if self._video_age() is None:
                    context = replace(context, frame_age_ms=None)
                self._reason = executor.execute(decision, context).reason
            else:
                self._reason = executor.execute_manual(
                    0.0, 0.0, frame_age_ms=None, input_age_ms=0.0,
                ).reason
        if executor is None:
            status = OperatorStatus(mode=self._mode, reason="Simulación · hardware PTZ deshabilitado")
        else:
            pan, tilt, _ = executor.last_physical_velocity
            status = OperatorStatus(
                hardware=True, armed=executor.armed, mode=self._mode,
                state=executor.state.value, reason=self._reason, pan=pan, tilt=tilt,
            )
        with self._lock:
            self._status = status

    def _run(self) -> None:
        failed = False
        try:
            while not self._closing.is_set():
                self._wake.clear()
                self.step()
                self._wake.wait(0.04)
        except Exception:
            failed = True
            LOGGER.exception("Falló el controlador de la interfaz")
        finally:
            self._closing.set()
            if self.executor is not None:
                try:
                    self.executor.emergency_stop()
                finally:
                    self.executor.close()
            with self._lock:
                self._status = replace(
                    self._status, armed=False,
                    state="FAULT" if failed or (self.executor and self.executor.state.value == "FAULT") else "DISARMED",
                    reason="Control detenido", pan=0.0, tilt=0.0,
                )
