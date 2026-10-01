"""Ejecutor PTZ con armado explícito, watchdog y parada segura."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import logging
import os
import threading
from time import monotonic, perf_counter
from typing import Callable

from ptz.ptz_backend import PTZBackend, PTZCapabilities
from ptz.ptz_decision import DecisionAction, PTZDecision
from ptz.ptz_execution_policy import (
    PTZExecutionContext,
    PTZExecutionPolicy,
    PTZPolicyResult,
)

DEFAULT_MAX_CONTINUOUS_MS = 400
DEFAULT_EXECUTOR_HEARTBEAT_MS = 500
DEFAULT_PHYSICAL_MAX_PAN = 0.30
DEFAULT_PHYSICAL_MAX_TILT = 0.30
DEFAULT_PHYSICAL_MAX_ZOOM = 0.00
MAX_ALLOWED_PHYSICAL_PAN_TILT = 0.30
MAX_ALLOWED_MANUAL_PAN_TILT = 0.75
ARM_CONFIRMATION = "MOVER"
LOGGER = logging.getLogger(__name__)


class PTZExecutorState(str, Enum):
    """Estados físicos y de seguridad del ejecutor."""

    DISARMED = "DISARMED"
    ARMED_IDLE = "ARMED_IDLE"
    MOVING_CONTINUOUS = "MOVING_CONTINUOUS"
    POSITIONING = "POSITIONING"
    FAULT = "FAULT"


class PTZExecutorError(RuntimeError):
    """Error base durante la ejecución segura de una orden."""


class PTZArmError(PTZExecutorError):
    """El usuario no confirmó el armado o el sistema no puede armarse."""


class PTZExecutorConfigurationError(ValueError):
    """Los límites físicos o temporales del ejecutor son inválidos."""


@dataclass(frozen=True)
class PTZExecutionResult:
    """Resultado observable de una solicitud hecha al ejecutor."""

    executed: bool
    state: PTZExecutorState
    reason: str


class PTZCommandExecutor:
    """Único componente autorizado a llamar movimiento sobre un PTZBackend.

    Construirlo no conecta, no arma y no mueve. El watchdog sólo se inicia tras
    un armado explícito. Position3D nunca se ejecuta: sus candidatos usan el
    mismo movimiento continuo seguro para alcanzar objetivos en los bordes.
    """

    def __init__(
        self,
        backend: PTZBackend,
        policy: PTZExecutionPolicy,
        *,
        max_continuous_ms: int = DEFAULT_MAX_CONTINUOUS_MS,
        heartbeat_ms: int = DEFAULT_EXECUTOR_HEARTBEAT_MS,
        physical_max_pan: float = DEFAULT_PHYSICAL_MAX_PAN,
        physical_max_tilt: float = DEFAULT_PHYSICAL_MAX_TILT,
        physical_max_zoom: float = DEFAULT_PHYSICAL_MAX_ZOOM,
        clock: Callable[[], float] = monotonic,
        start_watchdog: bool = True,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self._max_continuous_ms = self._positive_int(
            max_continuous_ms, "max_continuous_ms"
        )
        self._heartbeat_ms = self._positive_int(heartbeat_ms, "heartbeat_ms")
        self._physical_max_pan = self._unit_limit(
            physical_max_pan, "physical_max_pan"
        )
        self._physical_max_tilt = self._unit_limit(
            physical_max_tilt, "physical_max_tilt"
        )
        self._physical_max_zoom = self._unit_limit(
            physical_max_zoom, "physical_max_zoom"
        )
        if self._physical_max_pan > MAX_ALLOWED_PHYSICAL_PAN_TILT:
            raise PTZExecutorConfigurationError(
                "El perfil SAFE limita physical_max_pan a 0.30."
            )
        if self._physical_max_tilt > MAX_ALLOWED_PHYSICAL_PAN_TILT:
            raise PTZExecutorConfigurationError(
                "El perfil SAFE limita physical_max_tilt a 0.30."
            )
        if self._physical_max_zoom != 0.0:
            raise PTZExecutorConfigurationError(
                "Esta versión exige DETECTCAM_PTZ_PHYSICAL_MAX_ZOOM=0.0."
            )
        self._clock = clock
        self._watchdog_enabled = bool(start_watchdog)
        self._state = PTZExecutorState.DISARMED
        self._armed = False
        self._closed = False
        self._last_command_at: float | None = None
        self._last_command_interval_ms: float | None = None
        self._last_valid_update_at: float | None = None
        self._last_decision: PTZDecision | None = None
        self._last_requested_velocity = (0.0, 0.0, 0.0)
        self._last_physical_velocity = (0.0, 0.0, 0.0)
        self._last_backend_latency_ms: float | None = None
        self._deadman_activations = 0
        self._watchdog_activations = 0
        self._stop_commands = 0
        self._fault_reason: str | None = None
        self._lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @classmethod
    def from_environment(
        cls,
        backend: PTZBackend,
        *,
        policy: PTZExecutionPolicy | None = None,
        **overrides: object,
    ) -> "PTZCommandExecutor":
        selected_policy = policy or PTZExecutionPolicy.from_environment()
        options = dict(overrides)
        options.setdefault("max_continuous_ms", cls._env_int(
            "DETECTCAM_PTZ_MAX_CONTINUOUS_MS", DEFAULT_MAX_CONTINUOUS_MS
        ))
        options.setdefault("heartbeat_ms", cls._env_int(
            "DETECTCAM_PTZ_EXECUTOR_HEARTBEAT_MS", DEFAULT_EXECUTOR_HEARTBEAT_MS
        ))
        options.setdefault("physical_max_pan", cls._env_float(
            "DETECTCAM_PTZ_PHYSICAL_MAX_PAN", DEFAULT_PHYSICAL_MAX_PAN
        ))
        options.setdefault("physical_max_tilt", cls._env_float(
            "DETECTCAM_PTZ_PHYSICAL_MAX_TILT", DEFAULT_PHYSICAL_MAX_TILT
        ))
        options.setdefault("physical_max_zoom", cls._env_float(
            "DETECTCAM_PTZ_PHYSICAL_MAX_ZOOM", DEFAULT_PHYSICAL_MAX_ZOOM
        ))
        return cls(backend, selected_policy, **options)

    @property
    def state(self) -> PTZExecutorState:
        with self._lock:
            return self._state

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def hardware_enabled(self) -> bool:
        return self._policy.hardware_enabled

    @property
    def fault_reason(self) -> str | None:
        with self._lock:
            return self._fault_reason

    @property
    def last_requested_velocity(self) -> tuple[float, float, float]:
        with self._lock:
            return self._last_requested_velocity

    @property
    def last_physical_velocity(self) -> tuple[float, float, float]:
        with self._lock:
            return self._last_physical_velocity

    @property
    def heartbeat_age_ms(self) -> float | None:
        with self._lock:
            if self._last_valid_update_at is None:
                return None
            return max(0.0, (self._clock() - self._last_valid_update_at) * 1000.0)

    @property
    def deadman_active(self) -> bool:
        with self._lock:
            return self._state == PTZExecutorState.MOVING_CONTINUOUS

    @property
    def last_backend_latency_ms(self) -> float | None:
        with self._lock:
            return self._last_backend_latency_ms

    @property
    def last_command_interval_ms(self) -> float | None:
        with self._lock:
            return self._last_command_interval_ms

    @property
    def deadman_activations(self) -> int:
        with self._lock:
            return self._deadman_activations

    @property
    def watchdog_activations(self) -> int:
        with self._lock:
            return self._watchdog_activations

    @property
    def stop_commands(self) -> int:
        with self._lock:
            return self._stop_commands

    def set_min_track_confidence(self, value: float) -> None:
        """Actualiza el umbral de seguimiento sin cambiar el armado."""
        validated = PTZExecutionPolicy._unit(value, "min_track_confidence")
        with self._lock:
            self._policy.min_track_confidence = validated

    def arm(self, confirmation: str) -> PTZExecutionResult:
        """Arma sólo con flag físico, backend conectado y texto ``MOVER``."""
        with self._lock:
            if self._closed:
                raise PTZArmError("El ejecutor está cerrado.")
            if not self._policy.hardware_enabled:
                raise PTZArmError("DETECTCAM_PTZ_HARDWARE_ENABLED=false")
            if confirmation.strip() != ARM_CONFIRMATION:
                raise PTZArmError("Confirmación inválida; escriba MOVER.")
            if self._state == PTZExecutorState.FAULT:
                raise PTZArmError("El ejecutor está en FAULT; recupérelo primero.")
            if not self._backend.is_connected():
                raise PTZArmError("El backend PTZ no está conectado.")
            self._armed = True
            self._state = PTZExecutorState.ARMED_IDLE
            self._last_valid_update_at = self._clock()
            self._start_watchdog_locked()
            LOGGER.info("Ejecutor PTZ armado explícitamente")
            return PTZExecutionResult(True, self._state, "executor armed")

    def execute(
        self,
        decision: PTZDecision,
        context: PTZExecutionContext,
        *,
        timestamp: float | None = None,
    ) -> PTZExecutionResult:
        """Evalúa interlocks y, si autorizan movimiento, llama move_velocity."""
        now = self._clock() if timestamp is None else self._timestamp(timestamp)
        with self._lock:
            if self._closed:
                return PTZExecutionResult(False, self._state, "executor closed")
            self._last_decision = decision
            self._last_requested_velocity = (
                decision.pan, decision.tilt, decision.zoom
            )
            connected = False
            capabilities = PTZCapabilities()
            should_inspect_backend = (
                self._policy.hardware_enabled
                and self._armed
                and self._state != PTZExecutorState.FAULT
            )
            if should_inspect_backend:
                try:
                    connected = self._backend.is_connected()
                    if connected:
                        capabilities = self._backend.backend_capabilities
                except Exception as error:
                    self._enter_fault_locked(
                        f"backend inspection failed: {type(error).__name__}"
                    )
                    return PTZExecutionResult(
                        False, self._state,
                        self._fault_reason or "backend inspection failed",
                    )
                if not connected:
                    self._enter_fault_locked("backend disconnected")
                    return PTZExecutionResult(
                        False, self._state, "backend disconnected"
                    )
            result = self._policy.evaluate(
                decision,
                context,
                armed=self._armed,
                faulted=self._state == PTZExecutorState.FAULT,
                backend_connected=connected,
                capabilities=capabilities,
                now=now,
                last_command_at=self._last_command_at,
            )
            if result.allowed:
                return self._execute_velocity_locked(decision, now)
            if result.deferred:
                self._last_valid_update_at = now
                return PTZExecutionResult(False, self._state, result.reason)
            if result.reason == "backend disconnected" and self._armed:
                self._enter_fault_locked("backend disconnected")
                return PTZExecutionResult(False, self._state, "backend disconnected")
            if result.stop_required and self._state in (
                PTZExecutorState.MOVING_CONTINUOUS,
                PTZExecutorState.POSITIONING,
            ):
                if not self._stop_locked(result.reason):
                    return PTZExecutionResult(False, self._state, self._fault_reason or result.reason)
            if self._state != PTZExecutorState.FAULT:
                self._state = (
                    PTZExecutorState.ARMED_IDLE
                    if self._armed else PTZExecutorState.DISARMED
                )
                if connected and self._armed:
                    self._last_valid_update_at = now
            return PTZExecutionResult(False, self._state, result.reason)


    def execute_manual(
        self, pan: float, tilt: float, *, frame_age_ms: float | None,
        input_age_ms: float | None,
    ) -> PTZExecutionResult:
        """Movimiento manual hasta 75 %, con interlocks y watchdog vigentes."""
        now = self._clock()
        with self._lock:
            if self._closed:
                return PTZExecutionResult(False, self._state, "executor closed")
            connected = False
            capabilities = PTZCapabilities()
            if self._policy.hardware_enabled and self._armed and self._state != PTZExecutorState.FAULT:
                try:
                    connected = self._backend.is_connected()
                    if connected:
                        capabilities = self._backend.backend_capabilities
                except Exception as error:
                    self._enter_fault_locked(f"backend inspection failed: {type(error).__name__}")
                    return PTZExecutionResult(False, self._state, self._fault_reason or "backend fault")
                if not connected:
                    self._enter_fault_locked("backend disconnected")
                    return PTZExecutionResult(False, self._state, "backend disconnected")
            result = self._policy.evaluate_manual(
                pan, tilt, armed=self._armed,
                faulted=self._state == PTZExecutorState.FAULT,
                backend_connected=connected, capabilities=capabilities,
                frame_age_ms=frame_age_ms, input_age_ms=input_age_ms,
                now=now, last_command_at=self._last_command_at,
            )
            self._last_requested_velocity = (pan, tilt, 0.0)
            if result.allowed:
                decision = PTZDecision(
                    action=DecisionAction.MOVE, pan=pan, tilt=tilt, zoom=0.0,
                    target_error_x=0.0, target_error_y=0.0,
                    inside_dead_zone=False, safe_to_move=True, reason="manual control",
                )
                return self._execute_velocity_locked(decision, now, manual=True)
            if result.deferred:
                self._last_valid_update_at = now
            elif result.stop_required and self._state in (
                PTZExecutorState.MOVING_CONTINUOUS, PTZExecutorState.POSITIONING,
            ):
                if not self._stop_locked(result.reason):
                    return PTZExecutionResult(False, self._state, self._fault_reason or result.reason)
            return PTZExecutionResult(False, self._state, result.reason)

    def check_timeouts(self, *, timestamp: float | None = None) -> PTZExecutionResult:
        """Ejecuta dead-man/heartbeat; es usado por el hilo y por tests."""
        now = self._clock() if timestamp is None else self._timestamp(timestamp)
        with self._lock:
            if self._state != PTZExecutorState.MOVING_CONTINUOUS:
                return PTZExecutionResult(False, self._state, "executor not moving")
            heartbeat_elapsed = self._elapsed_ms(now, self._last_valid_update_at)
            command_elapsed = self._elapsed_ms(now, self._last_command_at)
            reason: str | None = None
            if heartbeat_elapsed > self._heartbeat_ms:
                reason = "executor heartbeat timeout"
            elif command_elapsed > self._max_continuous_ms:
                reason = "continuous movement dead-man timeout"
            if reason is None:
                return PTZExecutionResult(False, self._state, "timeouts not reached")
            if reason == "executor heartbeat timeout":
                self._watchdog_activations += 1
            else:
                self._deadman_activations += 1
            stopped = self._stop_locked(reason)
            return PTZExecutionResult(stopped, self._state, reason)

    def disarm(self) -> PTZExecutionResult:
        """Detiene una operación activa y bloquea decisiones posteriores."""
        with self._lock:
            if self._state in (
                PTZExecutorState.MOVING_CONTINUOUS,
                PTZExecutorState.POSITIONING,
            ) and not self._stop_locked("executor disarmed"):
                return PTZExecutionResult(False, self._state, self._fault_reason or "stop failed")
            self._armed = False
            if self._state != PTZExecutorState.FAULT:
                self._state = PTZExecutorState.DISARMED
            result = PTZExecutionResult(True, self._state, "executor disarmed")
            LOGGER.info("Ejecutor PTZ desarmado")
        self._stop_watchdog()
        return result

    def emergency_stop(self) -> PTZExecutionResult:
        """Envía STOP explícito y desarma; un fallo de STOP conserva FAULT."""
        with self._lock:
            if self._closed:
                return PTZExecutionResult(False, self._state, "executor closed")
            try:
                request_started = perf_counter()
                self._backend.stop()
            except Exception as error:
                self._last_backend_latency_ms = (
                    perf_counter() - request_started
                ) * 1000.0
                self._enter_fault_locked(
                    f"emergency stop failed: {type(error).__name__}",
                    try_stop=False,
                )
                LOGGER.error("Falló la parada de emergencia PTZ")
                return PTZExecutionResult(
                    False, self._state,
                    self._fault_reason or "emergency stop failed",
                )
            self._last_backend_latency_ms = (
                perf_counter() - request_started
            ) * 1000.0
            self._stop_commands += 1
            self._armed = False
            self._state = PTZExecutorState.DISARMED
            self._last_command_at = None
            self._last_valid_update_at = None
            self._last_physical_velocity = (0.0, 0.0, 0.0)
            result = PTZExecutionResult(True, self._state, "emergency stop")
            LOGGER.info("Parada de emergencia PTZ ejecutada; sistema desarmado")
        self._stop_watchdog()
        return result

    def recover(self) -> PTZExecutionResult:
        """Reconecta, envía STOP y vuelve desarmado; nunca reanuda una orden."""
        with self._lock:
            if self._closed:
                return PTZExecutionResult(False, self._state, "executor closed")
            if self._state != PTZExecutorState.FAULT:
                return PTZExecutionResult(False, self._state, "executor is not faulted")
            try:
                self._backend.connect()
                if not self._backend.is_connected():
                    raise PTZExecutorError("backend reconnect did not become connected")
                self._backend.stop()
                self._stop_commands += 1
            except Exception as error:
                self._fault_reason = f"backend recovery failed: {type(error).__name__}"
                return PTZExecutionResult(False, self._state, self._fault_reason)
            self._armed = False
            self._state = PTZExecutorState.DISARMED
            self._fault_reason = None
            self._last_command_at = None
            self._last_valid_update_at = None
            self._last_physical_velocity = (0.0, 0.0, 0.0)
            LOGGER.info("Backend PTZ recuperado, detenido y desarmado")
            return PTZExecutionResult(True, self._state, "backend recovered and stopped")

    def close(self) -> None:
        """Detiene si corresponde, apaga watchdog y cierra el backend."""
        with self._lock:
            if self._closed:
                return
            if self._state in (
                PTZExecutorState.MOVING_CONTINUOUS,
                PTZExecutorState.POSITIONING,
            ):
                self._stop_locked("executor closing")
            self._armed = False
            if self._state != PTZExecutorState.FAULT:
                self._state = PTZExecutorState.DISARMED
            self._closed = True
        self._stop_watchdog()
        try:
            self._backend.close()
        except Exception:
            with self._lock:
                self._state = PTZExecutorState.FAULT
                self._fault_reason = "backend close failed"

    def _execute_velocity_locked(
        self, decision: PTZDecision, now: float, *, manual: bool = False
    ) -> PTZExecutionResult:
        pan_limit = MAX_ALLOWED_MANUAL_PAN_TILT if manual else self._physical_max_pan
        tilt_limit = MAX_ALLOWED_MANUAL_PAN_TILT if manual else self._physical_max_tilt
        physical_pan = self._clamp(decision.pan, pan_limit)
        physical_tilt = self._clamp(decision.tilt, tilt_limit)
        physical_zoom = 0.0
        LOGGER.debug(
            "PTZ solicitado pan=%.3f tilt=%.3f; físico pan=%.3f tilt=%.3f",
            decision.pan, decision.tilt, physical_pan, physical_tilt,
        )
        try:
            request_started = perf_counter()
            self._backend.move_velocity(
                physical_pan, physical_tilt, physical_zoom
            )
        except Exception as error:
            self._last_backend_latency_ms = (
                perf_counter() - request_started
            ) * 1000.0
            self._enter_fault_locked(f"backend movement failed: {type(error).__name__}")
            return PTZExecutionResult(False, self._state, self._fault_reason or "backend fault")
        self._last_backend_latency_ms = (
            perf_counter() - request_started
        ) * 1000.0
        previous_command_at = self._last_command_at
        self._last_physical_velocity = (
            physical_pan, physical_tilt, physical_zoom
        )
        self._last_command_at = now
        self._last_command_interval_ms = (
            None if previous_command_at is None
            else max(0.0, (now - previous_command_at) * 1000.0)
        )
        self._last_valid_update_at = now
        self._state = PTZExecutorState.MOVING_CONTINUOUS
        LOGGER.info(
            "ContinuousMove PTZ enviado: pan=%.3f tilt=%.3f zoom=0.000",
            physical_pan, physical_tilt,
        )
        LOGGER.debug("Latencia de request PTZ: %.1f ms", self._last_backend_latency_ms)
        return PTZExecutionResult(True, self._state, "continuous movement executed")

    def _stop_locked(self, reason: str) -> bool:
        try:
            request_started = perf_counter()
            self._backend.stop()
        except Exception as error:
            self._last_backend_latency_ms = (
                perf_counter() - request_started
            ) * 1000.0
            self._enter_fault_locked(f"backend stop failed: {type(error).__name__}", try_stop=False)
            return False
        self._last_backend_latency_ms = (
            perf_counter() - request_started
        ) * 1000.0
        self._stop_commands += 1
        self._last_command_at = None
        self._last_physical_velocity = (0.0, 0.0, 0.0)
        self._state = (
            PTZExecutorState.ARMED_IDLE if self._armed else PTZExecutorState.DISARMED
        )
        LOGGER.info("STOP PTZ enviado: %s", reason)
        return True

    def _enter_fault_locked(self, reason: str, *, try_stop: bool = True) -> None:
        self._fault_reason = reason
        self._armed = False
        self._state = PTZExecutorState.FAULT
        self._last_command_at = None
        self._last_physical_velocity = (0.0, 0.0, 0.0)
        LOGGER.error("Ejecutor PTZ entró en FAULT: %s", reason)
        if try_stop:
            try:
                self._backend.stop()
                self._stop_commands += 1
            except Exception:
                pass

    def _start_watchdog_locked(self) -> None:
        if not self._watchdog_enabled:
            return
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            return
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="detectcam-ptz-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def _stop_watchdog(self) -> None:
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._watchdog_thread = None

    def _watchdog_loop(self) -> None:
        period = max(
            0.01,
            min(0.05, min(self._max_continuous_ms, self._heartbeat_ms) / 4000.0),
        )
        while not self._watchdog_stop.wait(period):
            self.check_timeouts()

    @staticmethod
    def _elapsed_ms(now: float, previous: float | None) -> float:
        return float("inf") if previous is None else max(0.0, (now - previous) * 1000.0)

    @staticmethod
    def _timestamp(value: float) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZExecutorConfigurationError("timestamp debe ser numérico.") from error
        if not isfinite(result) or result < 0:
            raise PTZExecutorConfigurationError("timestamp debe ser finito y no negativo.")
        return result

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise PTZExecutorConfigurationError(f"{name} debe ser entero positivo.")
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise PTZExecutorConfigurationError(f"{name} debe ser entero positivo.") from error
        if result != value or result <= 0:
            raise PTZExecutorConfigurationError(f"{name} debe ser entero positivo.")
        return result

    @staticmethod
    def _unit_limit(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZExecutorConfigurationError(
                f"{name} debe ser numérico."
            ) from error
        if not isfinite(result) or not 0.0 <= result <= 1.0:
            raise PTZExecutorConfigurationError(
                f"{name} debe estar entre 0.0 y 1.0."
            )
        return result

    @staticmethod
    def _clamp(value: float, maximum: float) -> float:
        return min(maximum, max(-maximum, value))

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZExecutorConfigurationError(f"{name} debe ser entero.") from error

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZExecutorConfigurationError(f"{name} debe ser numérico.") from error
