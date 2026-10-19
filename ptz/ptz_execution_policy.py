"""Política pura de interlocks para decisiones PTZ; no realiza I/O."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import os

from ptz.ptz_backend import PTZCapabilities
from ptz.ptz_decision import DecisionAction, PTZDecision
from ptz.target_tracker import TrackState

DEFAULT_HARDWARE_ENABLED = False
DEFAULT_POSITION3D_ENABLED = False
DEFAULT_AUTO_ZOOM_ENABLED = False
DEFAULT_MAX_TARGET_AGE_MS = 300
DEFAULT_MAX_FRAME_AGE_MS = 500
DEFAULT_COMMAND_INTERVAL_MS = 200
DEFAULT_MIN_TRACK_CONFIDENCE = 0.50
DEFAULT_MIN_VISIBLE_FRAMES = 3
DEFAULT_POSITION3D_ENTER_THRESHOLD = 0.65
DEFAULT_POSITION3D_EXIT_THRESHOLD = 0.55
DEFAULT_LOST_COAST_MS = 300


class PTZExecutionPolicyConfigurationError(ValueError):
    """La configuración de interlocks de seguridad no es válida."""


@dataclass(frozen=True)
class PTZExecutionContext:
    """Evidencia fresca que acompaña una decisión simulada."""

    tracker_state: TrackState
    target_age_ms: float | None
    frame_age_ms: float | None
    confidence: float | None
    visible_frames: int


@dataclass(frozen=True)
class PTZPolicyResult:
    """Autorización o rechazo emitido por los interlocks."""

    allowed: bool
    reason: str
    stop_required: bool = False
    deferred: bool = False


class PTZExecutionPolicy:
    """Evalúa seguridad sin conocer ni llamar el backend físico."""

    def __init__(
        self,
        *,
        hardware_enabled: bool = DEFAULT_HARDWARE_ENABLED,
        position3d_enabled: bool = DEFAULT_POSITION3D_ENABLED,
        auto_zoom_enabled: bool = DEFAULT_AUTO_ZOOM_ENABLED,
        max_target_age_ms: int = DEFAULT_MAX_TARGET_AGE_MS,
        max_frame_age_ms: int = DEFAULT_MAX_FRAME_AGE_MS,
        command_interval_ms: int = DEFAULT_COMMAND_INTERVAL_MS,
        min_track_confidence: float = DEFAULT_MIN_TRACK_CONFIDENCE,
        min_visible_frames: int = DEFAULT_MIN_VISIBLE_FRAMES,
        lost_coast_ms: int = DEFAULT_LOST_COAST_MS,
        position3d_enter_threshold: float = DEFAULT_POSITION3D_ENTER_THRESHOLD,
        position3d_exit_threshold: float = DEFAULT_POSITION3D_EXIT_THRESHOLD,
    ) -> None:
        if not isinstance(hardware_enabled, bool):
            raise PTZExecutionPolicyConfigurationError(
                "hardware_enabled debe ser booleano."
            )
        if position3d_enabled:
            raise PTZExecutionPolicyConfigurationError(
                "Position3D automático no está habilitado en esta versión."
            )
        if auto_zoom_enabled:
            raise PTZExecutionPolicyConfigurationError(
                "El zoom físico automático no está habilitado en esta versión."
            )
        self.hardware_enabled = hardware_enabled
        self.position3d_enabled = False
        self.auto_zoom_enabled = False
        self.max_target_age_ms = self._integer(
            max_target_age_ms, "max_target_age_ms", minimum=1
        )
        self.max_frame_age_ms = self._integer(
            max_frame_age_ms, "max_frame_age_ms", minimum=1
        )
        self.command_interval_ms = self._integer(
            command_interval_ms, "command_interval_ms", minimum=0
        )
        self.min_track_confidence = self._unit(
            min_track_confidence, "min_track_confidence"
        )
        self.min_visible_frames = self._integer(
            min_visible_frames, "min_visible_frames", minimum=1
        )
        self.lost_coast_ms = self._integer(
            lost_coast_ms, "lost_coast_ms", minimum=0
        )
        self.position3d_enter_threshold = self._unit(
            position3d_enter_threshold, "position3d_enter_threshold"
        )
        self.position3d_exit_threshold = self._unit(
            position3d_exit_threshold, "position3d_exit_threshold"
        )
        if self.position3d_exit_threshold >= self.position3d_enter_threshold:
            raise PTZExecutionPolicyConfigurationError(
                "El umbral EXIT de Position3D debe ser menor que ENTER."
            )

    @classmethod
    def from_environment(cls, **overrides: object) -> "PTZExecutionPolicy":
        options = dict(overrides)
        for key, name, default in (
            ("hardware_enabled", "DETECTCAM_PTZ_HARDWARE_ENABLED", False),
            ("position3d_enabled", "DETECTCAM_PTZ_POSITION3D_ENABLED", False),
            ("auto_zoom_enabled", "DETECTCAM_PTZ_AUTO_ZOOM_ENABLED", False),
        ):
            options.setdefault(key, cls._env_bool(name, default))
        for key, name, default in (
            ("max_target_age_ms", "DETECTCAM_PTZ_MAX_TARGET_AGE_MS", DEFAULT_MAX_TARGET_AGE_MS),
            ("max_frame_age_ms", "DETECTCAM_PTZ_MAX_FRAME_AGE_MS", DEFAULT_MAX_FRAME_AGE_MS),
            ("command_interval_ms", "DETECTCAM_PTZ_COMMAND_INTERVAL_MS", DEFAULT_COMMAND_INTERVAL_MS),
            ("min_visible_frames", "DETECTCAM_PTZ_MIN_VISIBLE_FRAMES", DEFAULT_MIN_VISIBLE_FRAMES),
            ("lost_coast_ms", "DETECTCAM_PTZ_LOST_COAST_MS", DEFAULT_LOST_COAST_MS),
        ):
            options.setdefault(key, cls._env_int(name, default))
        for key, name, default in (
            ("min_track_confidence", "DETECTCAM_PTZ_MIN_TRACK_CONFIDENCE", DEFAULT_MIN_TRACK_CONFIDENCE),
            ("position3d_enter_threshold", "DETECTCAM_PTZ_POSITION3D_ENTER", DEFAULT_POSITION3D_ENTER_THRESHOLD),
            ("position3d_exit_threshold", "DETECTCAM_PTZ_POSITION3D_EXIT", DEFAULT_POSITION3D_EXIT_THRESHOLD),
        ):
            options.setdefault(key, cls._env_float(name, default))
        return cls(**options)

    def evaluate(
        self,
        decision: PTZDecision,
        context: PTZExecutionContext,
        *,
        armed: bool,
        faulted: bool,
        backend_connected: bool,
        capabilities: PTZCapabilities,
        now: float,
        last_command_at: float | None,
    ) -> PTZPolicyResult:
        """Devuelve autorización o rechazo sin producir efectos secundarios."""
        if decision.action == DecisionAction.HOLD:
            return PTZPolicyResult(False, decision.reason, stop_required=True)
        if not self.hardware_enabled:
            return PTZPolicyResult(False, "PTZ hardware disabled", stop_required=True)
        if not armed:
            return PTZPolicyResult(False, "executor disarmed", stop_required=True)
        if faulted:
            return PTZPolicyResult(False, "executor faulted", stop_required=True)
        if not backend_connected:
            return PTZPolicyResult(False, "backend disconnected", stop_required=True)
        temporarily_lost = context.tracker_state == TrackState.TEMPORARILY_LOST
        if temporarily_lost and not decision.coasting:
            return PTZPolicyResult(False, "target temporarily lost", stop_required=True)
        if temporarily_lost and not self._fresh(
            context.target_age_ms, self.lost_coast_ms
        ):
            return PTZPolicyResult(
                False, "target loss coast expired", stop_required=True
            )
        if context.tracker_state not in (
            TrackState.TRACKED,
            TrackState.TEMPORARILY_LOST,
        ):
            return PTZPolicyResult(False, "no target", stop_required=True)
        if not decision.safe_to_move:
            return PTZPolicyResult(False, "decision is not safe to move", stop_required=True)
        if not self._fresh(context.target_age_ms, self.max_target_age_ms):
            return PTZPolicyResult(False, "target stale", stop_required=True)
        if not self._fresh(context.frame_age_ms, self.max_frame_age_ms):
            return PTZPolicyResult(False, "frame stale", stop_required=True)
        if (
            context.confidence is None
            or not isfinite(context.confidence)
            or not 0.0 <= context.confidence <= 1.0
            or context.confidence < self.min_track_confidence
        ):
            return PTZPolicyResult(False, "track confidence below safety threshold", True)
        if context.visible_frames < self.min_visible_frames:
            return PTZPolicyResult(False, "track has too few visible frames", True)
        if temporarily_lost and decision.action != DecisionAction.MOVE:
            return PTZPolicyResult(False, "invalid lost-target coast", True)
        if abs(decision.zoom) > 1e-12:
            return PTZPolicyResult(False, "physical auto zoom not enabled", True)
        if decision.action not in (
            DecisionAction.MOVE,
            DecisionAction.POSITION3D_CANDIDATE,
        ):
            return PTZPolicyResult(False, "unsupported PTZ decision", True)
        values = (decision.pan, decision.tilt, decision.zoom)
        if not all(isfinite(value) and -1.0 <= value <= 1.0 for value in values):
            return PTZPolicyResult(False, "command outside normalized limits", True)
        if not capabilities.continuous_pan_tilt:
            return PTZPolicyResult(False, "backend lacks continuous pan/tilt", True)
        if last_command_at is not None:
            elapsed_ms = (now - last_command_at) * 1000.0
            if elapsed_ms + 1e-9 < self.command_interval_ms:
                return PTZPolicyResult(
                    False, "executor command cooldown active", deferred=True
                )
        if decision.action == DecisionAction.POSITION3D_CANDIDATE:
            return PTZPolicyResult(
                True,
                "Position3D disabled; continuous movement fallback permitted",
            )
        return PTZPolicyResult(True, "continuous movement permitted")


    def evaluate_manual(
        self, pan: float, tilt: float, *, zoom: float = 0.0, armed: bool, faulted: bool,
        backend_connected: bool, capabilities: PTZCapabilities,
        frame_age_ms: float | None, input_age_ms: float | None,
        now: float, last_command_at: float | None,
    ) -> PTZPolicyResult:
        """Autoriza al operador con video fresco y una pulsación renovada."""
        for rejected, reason in (
            (not self.hardware_enabled, "PTZ hardware disabled"),
            (not armed, "executor disarmed"),
            (faulted, "executor faulted"),
            (not backend_connected, "backend disconnected"),
            (not self._fresh(frame_age_ms, self.max_frame_age_ms), "frame stale"),
            (not self._fresh(input_age_ms, 250), "manual input expired"),
            (not all(isfinite(v) and -1.0 <= v <= 1.0 for v in (pan, tilt, zoom)),
             "command outside normalized limits"),
            ((pan != 0.0 or tilt != 0.0) and not capabilities.continuous_pan_tilt,
             "backend lacks continuous pan/tilt"),
            (zoom != 0.0 and not capabilities.continuous_zoom, "backend lacks continuous zoom"),
            (pan == 0.0 and tilt == 0.0 and zoom == 0.0, "manual control idle"),
        ):
            if rejected:
                return PTZPolicyResult(False, reason, stop_required=True)
        if last_command_at is not None:
            if (now - last_command_at) * 1000.0 + 1e-9 < self.command_interval_ms:
                return PTZPolicyResult(False, "executor command cooldown active", deferred=True)
        return PTZPolicyResult(True, "manual movement permitted")

    @staticmethod
    def _fresh(age_ms: float | None, maximum_ms: int) -> bool:
        return (
            age_ms is not None and isfinite(age_ms)
            and 0.0 <= age_ms <= maximum_ms
        )

    @staticmethod
    def _integer(value: int, name: str, *, minimum: int) -> int:
        if isinstance(value, bool):
            raise PTZExecutionPolicyConfigurationError(f"{name} debe ser entero.")
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise PTZExecutionPolicyConfigurationError(
                f"{name} debe ser entero."
            ) from error
        if result != value or result < minimum:
            raise PTZExecutionPolicyConfigurationError(
                f"{name} debe ser mayor o igual que {minimum}."
            )
        return result

    @staticmethod
    def _unit(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZExecutionPolicyConfigurationError(
                f"{name} debe ser numérico."
            ) from error
        if not isfinite(result) or not 0.0 <= result <= 1.0:
            raise PTZExecutionPolicyConfigurationError(
                f"{name} debe estar entre 0.0 y 1.0."
            )
        return result

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        normalized = value.strip().casefold()
        if normalized in ("1", "true", "yes", "on", "si", "sí"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
        raise PTZExecutionPolicyConfigurationError(f"{name} debe ser true o false.")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZExecutionPolicyConfigurationError(f"{name} debe ser entero.") from error

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZExecutionPolicyConfigurationError(f"{name} debe ser numérico.") from error
