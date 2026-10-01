"""Control PTZ simulado avanzado, sin acceso a backends ni hardware."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, pi
import os
from time import monotonic
from typing import Callable, Sequence

from ptz.target_tracker import TrackedTarget, TrackState

DEFAULT_DEAD_ZONE_X = 0.10
DEFAULT_DEAD_ZONE_Y = 0.10
DEFAULT_DEAD_ZONE_HYSTERESIS = 0.02
DEFAULT_KP_PAN = 0.50
DEFAULT_KP_TILT = 0.50
DEFAULT_KD_PAN = 0.04
DEFAULT_KD_TILT = 0.04
DEFAULT_KV_PAN = 0.10
DEFAULT_KV_TILT = 0.10
DEFAULT_FILTER_MIN_CUTOFF = 1.0
DEFAULT_FILTER_BETA = 0.05
DEFAULT_FILTER_D_CUTOFF = 1.0
DEFAULT_LEAD_TIME_S = 0.15
DEFAULT_MAX_PREDICTION_ERROR = 0.90
DEFAULT_MAX_COMMAND_DELTA_PER_S = 1.50
DEFAULT_RESPONSE_POWER = 1.20
DEFAULT_OSCILLATION_GAIN = 0.35
DEFAULT_OSCILLATION_DECAY = 0.85
DEFAULT_OSCILLATION_MAX = 6.0
DEFAULT_STALE_TARGET_MS = 300
DEFAULT_MAX_FRAME_AGE_MS = 500
DEFAULT_MAX_PAN_COMMAND = 0.35
DEFAULT_MAX_TILT_COMMAND = 0.35
DEFAULT_MAX_ZOOM_COMMAND = 0.15
DEFAULT_MIN_TRACK_CONFIDENCE = 0.50
DEFAULT_MIN_VISIBLE_FRAMES = 3
DEFAULT_COMMAND_INTERVAL_MS = 200
DEFAULT_POSITION3D_THRESHOLD = 0.65
DEFAULT_LOST_COAST_MS = 300


class PTZDecisionConfigurationError(ValueError):
    """La configuración es inválida o intenta salir de simulación."""


class DecisionAction(str, Enum):
    """Acciones lógicas producidas sin tocar el hardware."""

    HOLD = "HOLD"
    MOVE = "MOVE"
    POSITION3D_CANDIDATE = "POSITION3D_CANDIDATE"


@dataclass(frozen=True)
class PTZDecision:
    """Intención matemática que todavía no constituye una orden física."""

    action: DecisionAction
    pan: float
    tilt: float
    zoom: float
    target_error_x: float
    target_error_y: float
    inside_dead_zone: bool
    safe_to_move: bool
    reason: str
    position3d_candidate: bool = False
    raw_error_x: float = 0.0
    raw_error_y: float = 0.0
    filtered_error_x: float = 0.0
    filtered_error_y: float = 0.0
    predicted_error_x: float = 0.0
    predicted_error_y: float = 0.0
    derivative_x: float = 0.0
    derivative_y: float = 0.0
    normalized_velocity_x: float = 0.0
    normalized_velocity_y: float = 0.0
    feedforward_x: float = 0.0
    feedforward_y: float = 0.0
    oscillation_score_pan: float = 0.0
    oscillation_score_tilt: float = 0.0
    frame_age_ms: float | None = None
    coasting: bool = False


class OneEuroFilter:
    """Filtro adaptativo de primer orden con derivada suavizada.

    Los timestamps y ``dt`` están en segundos monotónicos. Un ``dt`` no finito
    o no positivo conserva la salida previa y no corrompe el estado.
    """

    def __init__(
        self,
        *,
        min_cutoff: float = DEFAULT_FILTER_MIN_CUTOFF,
        beta: float = DEFAULT_FILTER_BETA,
        d_cutoff: float = DEFAULT_FILTER_D_CUTOFF,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.min_cutoff = self._positive(min_cutoff, "min_cutoff")
        self.beta = self._nonnegative(beta, "beta")
        self.d_cutoff = self._positive(d_cutoff, "d_cutoff")
        self._clock = clock
        self.reset()

    @property
    def derivative(self) -> float:
        return self._derivative

    def reset(self) -> None:
        self._value: float | None = None
        self._raw_value: float | None = None
        self._timestamp: float | None = None
        self._derivative = 0.0

    def filter(self, value: float, *, timestamp: float | None = None) -> float:
        sample = self._finite(value, "value")
        now = self._clock() if timestamp is None else self._finite(
            timestamp, "timestamp"
        )
        if self._value is None:
            self._value = sample
            self._raw_value = sample
            self._timestamp = now
            return sample
        assert self._raw_value is not None and self._timestamp is not None
        dt = now - self._timestamp
        if not isfinite(dt) or dt <= 0.0:
            return self._value
        raw_derivative = (sample - self._raw_value) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        self._derivative += derivative_alpha * (
            raw_derivative - self._derivative
        )
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        value_alpha = self._alpha(cutoff, dt)
        self._value += value_alpha * (sample - self._value)
        self._raw_value = sample
        self._timestamp = now
        return self._value

    __call__ = filter

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _finite(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZDecisionConfigurationError(f"{name} debe ser numérico.") from error
        if not isfinite(result):
            raise PTZDecisionConfigurationError(f"{name} debe ser finito.")
        return result

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result <= 0.0:
            raise PTZDecisionConfigurationError(f"{name} debe ser positivo.")
        return result

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        result = cls._finite(value, name)
        if result < 0.0:
            raise PTZDecisionConfigurationError(f"{name} no puede ser negativo.")
        return result


class PTZDecisionEngine:
    """Pipeline avanzado que produce únicamente intenciones normalizadas."""

    def __init__(
        self,
        *,
        simulation: bool = True,
        dead_zone_x: float = DEFAULT_DEAD_ZONE_X,
        dead_zone_y: float = DEFAULT_DEAD_ZONE_Y,
        dead_zone_hysteresis: float = DEFAULT_DEAD_ZONE_HYSTERESIS,
        kp_pan: float = DEFAULT_KP_PAN,
        kp_tilt: float = DEFAULT_KP_TILT,
        kd_pan: float = DEFAULT_KD_PAN,
        kd_tilt: float = DEFAULT_KD_TILT,
        kv_pan: float = DEFAULT_KV_PAN,
        kv_tilt: float = DEFAULT_KV_TILT,
        filter_min_cutoff: float = DEFAULT_FILTER_MIN_CUTOFF,
        filter_beta: float = DEFAULT_FILTER_BETA,
        filter_d_cutoff: float = DEFAULT_FILTER_D_CUTOFF,
        lead_time_s: float = DEFAULT_LEAD_TIME_S,
        max_prediction_error: float = DEFAULT_MAX_PREDICTION_ERROR,
        max_command_delta_per_s: float = DEFAULT_MAX_COMMAND_DELTA_PER_S,
        response_power: float = DEFAULT_RESPONSE_POWER,
        oscillation_gain: float = DEFAULT_OSCILLATION_GAIN,
        oscillation_decay: float = DEFAULT_OSCILLATION_DECAY,
        oscillation_max: float = DEFAULT_OSCILLATION_MAX,
        stale_target_ms: int = DEFAULT_STALE_TARGET_MS,
        max_frame_age_ms: int = DEFAULT_MAX_FRAME_AGE_MS,
        max_pan_command: float = DEFAULT_MAX_PAN_COMMAND,
        max_tilt_command: float = DEFAULT_MAX_TILT_COMMAND,
        max_zoom_command: float = DEFAULT_MAX_ZOOM_COMMAND,
        min_track_confidence: float = DEFAULT_MIN_TRACK_CONFIDENCE,
        min_visible_frames: int = DEFAULT_MIN_VISIBLE_FRAMES,
        command_interval_ms: int = DEFAULT_COMMAND_INTERVAL_MS,
        position3d_threshold: float = DEFAULT_POSITION3D_THRESHOLD,
        lost_coast_ms: int = DEFAULT_LOST_COAST_MS,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if simulation is not True:
            raise PTZDecisionConfigurationError(
                "La capa de decisión sólo admite DETECTCAM_PTZ_SIMULATION=true."
            )
        self._dead_zone_x = self._less_than_one(dead_zone_x, "dead_zone_x")
        self._dead_zone_y = self._less_than_one(dead_zone_y, "dead_zone_y")
        self._dead_zone_hysteresis = self._unit(
            dead_zone_hysteresis, "dead_zone_hysteresis"
        )
        if max(self._dead_zone_x, self._dead_zone_y) + self._dead_zone_hysteresis > 1.0:
            raise PTZDecisionConfigurationError(
                "dead_zone + dead_zone_hysteresis no puede superar 1.0."
            )
        self._kp_pan = self._nonnegative(kp_pan, "kp_pan")
        self._kp_tilt = self._nonnegative(kp_tilt, "kp_tilt")
        self._kd_pan = self._nonnegative(kd_pan, "kd_pan")
        self._kd_tilt = self._nonnegative(kd_tilt, "kd_tilt")
        self._kv_pan = self._nonnegative(kv_pan, "kv_pan")
        self._kv_tilt = self._nonnegative(kv_tilt, "kv_tilt")
        self._filter_min_cutoff = self._positive(filter_min_cutoff, "filter_min_cutoff")
        self._filter_beta = self._nonnegative(filter_beta, "filter_beta")
        self._filter_d_cutoff = self._positive(filter_d_cutoff, "filter_d_cutoff")
        self._lead_time_s = self._nonnegative(lead_time_s, "lead_time_s")
        self._max_prediction_error = self._positive_unit(
            max_prediction_error, "max_prediction_error"
        )
        self._max_command_delta_per_s = self._positive(
            max_command_delta_per_s, "max_command_delta_per_s"
        )
        self._response_power = self._positive(response_power, "response_power")
        self._oscillation_gain = self._nonnegative(oscillation_gain, "oscillation_gain")
        self._oscillation_decay = self._unit(oscillation_decay, "oscillation_decay")
        self._oscillation_max = self._positive(oscillation_max, "oscillation_max")
        self._stale_target_ms = self._integer(stale_target_ms, "stale_target_ms", minimum=1)
        self._max_frame_age_ms = self._integer(max_frame_age_ms, "max_frame_age_ms", minimum=1)
        self._max_pan_command = self._unit(max_pan_command, "max_pan_command")
        self._max_tilt_command = self._unit(max_tilt_command, "max_tilt_command")
        self._max_zoom_command = self._unit(max_zoom_command, "max_zoom_command")
        self._min_track_confidence = self._unit(min_track_confidence, "min_track_confidence")
        self._min_visible_frames = self._integer(min_visible_frames, "min_visible_frames", minimum=1)
        self._command_interval_ms = self._integer(command_interval_ms, "command_interval_ms", minimum=0)
        self._position3d_threshold = self._unit(position3d_threshold, "position3d_threshold")
        self._lost_coast_ms = self._integer(
            lost_coast_ms, "lost_coast_ms", minimum=0
        )
        self._clock = clock
        self._filter_x = OneEuroFilter(
            min_cutoff=self._filter_min_cutoff, beta=self._filter_beta,
            d_cutoff=self._filter_d_cutoff, clock=clock,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=self._filter_min_cutoff, beta=self._filter_beta,
            d_cutoff=self._filter_d_cutoff, clock=clock,
        )
        self._active_track_id: int | None = None
        self._reset_dynamic(clear_track=False)

    @classmethod
    def from_environment(cls, **overrides: object) -> "PTZDecisionEngine":
        options = dict(overrides)
        options.setdefault("simulation", cls._env_bool("DETECTCAM_PTZ_SIMULATION", True))
        floats = (
            ("dead_zone_x", "DETECTCAM_PTZ_DEAD_ZONE_X", DEFAULT_DEAD_ZONE_X),
            ("dead_zone_y", "DETECTCAM_PTZ_DEAD_ZONE_Y", DEFAULT_DEAD_ZONE_Y),
            ("dead_zone_hysteresis", "DETECTCAM_PTZ_DEAD_ZONE_HYSTERESIS", DEFAULT_DEAD_ZONE_HYSTERESIS),
            ("kp_pan", "DETECTCAM_PTZ_KP_PAN", DEFAULT_KP_PAN),
            ("kp_tilt", "DETECTCAM_PTZ_KP_TILT", DEFAULT_KP_TILT),
            ("kd_pan", "DETECTCAM_PTZ_KD_PAN", DEFAULT_KD_PAN),
            ("kd_tilt", "DETECTCAM_PTZ_KD_TILT", DEFAULT_KD_TILT),
            ("kv_pan", "DETECTCAM_PTZ_KV_PAN", DEFAULT_KV_PAN),
            ("kv_tilt", "DETECTCAM_PTZ_KV_TILT", DEFAULT_KV_TILT),
            ("filter_min_cutoff", "DETECTCAM_PTZ_FILTER_MIN_CUTOFF", DEFAULT_FILTER_MIN_CUTOFF),
            ("filter_beta", "DETECTCAM_PTZ_FILTER_BETA", DEFAULT_FILTER_BETA),
            ("filter_d_cutoff", "DETECTCAM_PTZ_FILTER_D_CUTOFF", DEFAULT_FILTER_D_CUTOFF),
            ("lead_time_s", "DETECTCAM_PTZ_LEAD_TIME_S", DEFAULT_LEAD_TIME_S),
            ("max_prediction_error", "DETECTCAM_PTZ_MAX_PREDICTION_ERROR", DEFAULT_MAX_PREDICTION_ERROR),
            ("max_command_delta_per_s", "DETECTCAM_PTZ_MAX_COMMAND_DELTA_PER_S", DEFAULT_MAX_COMMAND_DELTA_PER_S),
            ("response_power", "DETECTCAM_PTZ_RESPONSE_POWER", DEFAULT_RESPONSE_POWER),
            ("oscillation_gain", "DETECTCAM_PTZ_OSCILLATION_GAIN", DEFAULT_OSCILLATION_GAIN),
            ("oscillation_decay", "DETECTCAM_PTZ_OSCILLATION_DECAY", DEFAULT_OSCILLATION_DECAY),
            ("oscillation_max", "DETECTCAM_PTZ_OSCILLATION_MAX", DEFAULT_OSCILLATION_MAX),
            ("max_pan_command", "DETECTCAM_PTZ_MAX_PAN_COMMAND", DEFAULT_MAX_PAN_COMMAND),
            ("max_tilt_command", "DETECTCAM_PTZ_MAX_TILT_COMMAND", DEFAULT_MAX_TILT_COMMAND),
            ("max_zoom_command", "DETECTCAM_PTZ_MAX_ZOOM_COMMAND", DEFAULT_MAX_ZOOM_COMMAND),
            ("min_track_confidence", "DETECTCAM_PTZ_MIN_TRACK_CONFIDENCE", DEFAULT_MIN_TRACK_CONFIDENCE),
            ("position3d_threshold", "DETECTCAM_PTZ_POSITION3D_THRESHOLD", DEFAULT_POSITION3D_THRESHOLD),
        )
        for key, variable, default in floats:
            options.setdefault(key, cls._env_float(variable, default))
        integers = (
            ("stale_target_ms", "DETECTCAM_PTZ_STALE_TARGET_MS", DEFAULT_STALE_TARGET_MS),
            ("max_frame_age_ms", "DETECTCAM_PTZ_MAX_FRAME_AGE_MS", DEFAULT_MAX_FRAME_AGE_MS),
            ("min_visible_frames", "DETECTCAM_PTZ_MIN_VISIBLE_FRAMES", DEFAULT_MIN_VISIBLE_FRAMES),
            ("command_interval_ms", "DETECTCAM_PTZ_COMMAND_INTERVAL_MS", DEFAULT_COMMAND_INTERVAL_MS),
            ("lost_coast_ms", "DETECTCAM_PTZ_LOST_COAST_MS", DEFAULT_LOST_COAST_MS),
        )
        for key, variable, default in integers:
            options.setdefault(key, cls._env_int(variable, default))
        return cls(**options)

    @property
    def simulation(self) -> bool:
        return True

    @property
    def dead_zone(self) -> tuple[float, float]:
        return self._dead_zone_x, self._dead_zone_y

    @property
    def min_track_confidence(self) -> float:
        return self._min_track_confidence

    def set_min_track_confidence(self, value: float) -> None:
        """Aplica la confianza del operador desde el hilo de visión."""
        self._min_track_confidence = self._unit(value, "min_track_confidence")

    def reset(self) -> None:
        """Borra toda dinámica simulada; no tiene efectos físicos."""
        self._reset_dynamic(clear_track=True)

    def decide(
        self,
        target: TrackedTarget | None,
        frame_shape: Sequence[int],
        *,
        tracker_state: TrackState | None = None,
        timestamp: float | None = None,
        frame_age_s: float | None = None,
    ) -> PTZDecision:
        """Calcula una intención avanzada sin importar ni llamar backends PTZ."""
        now = self._clock() if timestamp is None else self._valid_timestamp(timestamp)
        frame_age_ms = self._frame_age_ms(frame_age_s)
        dimensions = self._frame_dimensions(frame_shape)
        if dimensions is None:
            return self._safety_hold("invalid frame", now, frame_age_ms=frame_age_ms)
        height, width = dimensions
        state = tracker_state
        if state is None:
            if target is None:
                state = TrackState.NO_TARGET
            elif target.missed_frames > 0:
                state = TrackState.TEMPORARILY_LOST
            else:
                state = TrackState.TRACKED
        if target is None or state == TrackState.NO_TARGET:
            self._reset_dynamic(clear_track=True)
            return self._hold("no target", frame_age_ms=frame_age_ms)
        if target.track_id != self._active_track_id:
            self._reset_dynamic(clear_track=False)
            self._active_track_id = target.track_id
        if not self._valid_bbox(target, width, height):
            return self._safety_hold("invalid bounding box", now, frame_age_ms=frame_age_ms)

        raw_x = self._clamp((target.detection.center_x - width / 2.0) / (width / 2.0), 1.0)
        raw_y = self._clamp((target.detection.center_y - height / 2.0) / (height / 2.0), 1.0)
        common_raw = dict(raw_error_x=raw_x, raw_error_y=raw_y, frame_age_ms=frame_age_ms)
        target_age_ms = self._target_age_ms(target.last_seen_timestamp, now)
        if target.detection.confidence < self._min_track_confidence:
            return self._safety_hold("track confidence below safety threshold", now, **common_raw)
        if state == TrackState.TEMPORARILY_LOST or target.missed_frames > 0:
            if frame_age_ms is None:
                return self._safety_hold(
                    "frame unavailable during target loss", now, **common_raw
                )
            if frame_age_ms > self._max_frame_age_ms:
                return self._safety_hold("frame stale", now, **common_raw)
            if target_age_ms is None or target_age_ms > self._lost_coast_ms:
                return self._safety_hold(
                    "target loss coast expired", now, **common_raw
                )
            if abs(self._previous_pan) <= 1e-12 and abs(self._previous_tilt) <= 1e-12:
                return self._safety_hold(
                    "target temporarily lost without previous direction",
                    now,
                    **common_raw,
                )
            return PTZDecision(
                action=DecisionAction.MOVE,
                pan=self._previous_pan,
                tilt=self._previous_tilt,
                zoom=0.0,
                target_error_x=raw_x,
                target_error_y=raw_y,
                inside_dead_zone=False,
                safe_to_move=True,
                reason="target temporarily lost; coasting last direction",
                coasting=True,
                **common_raw,
            )
        if state != TrackState.TRACKED:
            return self._safety_hold("invalid tracker state", now, frame_age_ms=frame_age_ms)
        if target.visible_frames < self._min_visible_frames:
            return self._safety_hold("track has too few visible frames", now, **common_raw)
        if target_age_ms is None:
            return self._safety_hold("invalid target timestamp", now, **common_raw)
        self._last_valid_target_update_at = target.last_seen_timestamp
        if target_age_ms > self._stale_target_ms:
            return self._safety_hold("target stale", now, **common_raw)
        if frame_age_ms is not None and frame_age_ms > self._max_frame_age_ms:
            return self._safety_hold("frame stale", now, **common_raw)

        filtered_x = self._filter_x(raw_x, timestamp=now)
        filtered_y = self._filter_y(raw_y, timestamp=now)
        velocity_x, velocity_y = self._normalized_velocity(target, width, height)
        predicted_x = self._clamp(filtered_x + velocity_x * self._lead_time_s, self._max_prediction_error)
        predicted_y = self._clamp(filtered_y + velocity_y * self._lead_time_s, self._max_prediction_error)
        common = dict(
            raw_error_x=raw_x, raw_error_y=raw_y,
            filtered_error_x=filtered_x, filtered_error_y=filtered_y,
            predicted_error_x=predicted_x, predicted_error_y=predicted_y,
            normalized_velocity_x=velocity_x, normalized_velocity_y=velocity_y,
            frame_age_ms=frame_age_ms,
        )
        inside_x = abs(predicted_x) <= self._dead_zone_x
        inside_y = abs(predicted_y) <= self._dead_zone_y
        inside_dead_zone = inside_x and inside_y
        enter_x = self._dead_zone_x + self._dead_zone_hysteresis
        enter_y = self._dead_zone_y + self._dead_zone_hysteresis
        if self._movement_active:
            if inside_dead_zone:
                self._movement_active = False
                return self._safety_hold(
                    "target centered inside dead zone", now,
                    inside_dead_zone=True, **common,
                )
        elif abs(predicted_x) <= enter_x and abs(predicted_y) <= enter_y:
            return self._safety_hold(
                "target inside dead-zone hysteresis", now,
                inside_dead_zone=inside_dead_zone, **common,
            )
        else:
            self._movement_active = True

        position3d_candidate = (
            abs(predicted_x) > self._position3d_threshold
            or abs(predicted_y) > self._position3d_threshold
        )
        if self._cooldown_active(now):
            action = (
                DecisionAction.POSITION3D_CANDIDATE
                if position3d_candidate else DecisionAction.MOVE
            )
            return PTZDecision(
                action=action,
                pan=self._previous_pan,
                tilt=self._previous_tilt,
                zoom=0.0,
                target_error_x=predicted_x,
                target_error_y=predicted_y,
                inside_dead_zone=inside_dead_zone,
                safe_to_move=True,
                reason="simulated command cooldown active",
                position3d_candidate=position3d_candidate,
                **common,
            )

        controlled_x = self._soft_deadband(predicted_x, self._dead_zone_x)
        controlled_y = self._soft_deadband(predicted_y, self._dead_zone_y)
        derivative_x, derivative_y, control_dt = self._derivatives(controlled_x, controlled_y, now)
        feedforward_x = self._kv_pan * velocity_x
        feedforward_y = self._kv_tilt * velocity_y
        pan_control = self._kp_pan * controlled_x + self._kd_pan * derivative_x + feedforward_x
        tilt_control = self._kp_tilt * controlled_y + self._kd_tilt * derivative_y + feedforward_y
        pan_shaped = self._shape(self._clamp(pan_control, self._max_pan_command), self._max_pan_command)
        tilt_shaped = self._shape(self._clamp(tilt_control, self._max_tilt_command), self._max_tilt_command)
        pan_slewed = self._slew(self._previous_pan, pan_shaped, control_dt)
        tilt_slewed = self._slew(self._previous_tilt, tilt_shaped, control_dt)
        self._oscillation_score_pan, self._pan_sign = self._oscillation(
            pan_slewed, self._pan_sign, self._oscillation_score_pan
        )
        self._oscillation_score_tilt, self._tilt_sign = self._oscillation(
            tilt_slewed, self._tilt_sign, self._oscillation_score_tilt
        )
        pan = pan_slewed / (1.0 + self._oscillation_gain * self._oscillation_score_pan)
        tilt = tilt_slewed / (1.0 + self._oscillation_gain * self._oscillation_score_tilt)
        pan = self._clamp(pan, self._max_pan_command)
        tilt = self._clamp(tilt, self._max_tilt_command)
        self._previous_pan, self._previous_tilt = pan, tilt
        self._last_movement_decision_at = now
        action = DecisionAction.POSITION3D_CANDIDATE if position3d_candidate else DecisionAction.MOVE
        reason = (
            "target far from center; simulated Position3D candidate"
            if position3d_candidate else "advanced simulated control"
        )
        return PTZDecision(
            action=action, pan=pan, tilt=tilt, zoom=0.0,
            target_error_x=predicted_x, target_error_y=predicted_y,
            inside_dead_zone=False, safe_to_move=True, reason=reason,
            position3d_candidate=position3d_candidate,
            derivative_x=derivative_x, derivative_y=derivative_y,
            feedforward_x=feedforward_x, feedforward_y=feedforward_y,
            oscillation_score_pan=self._oscillation_score_pan,
            oscillation_score_tilt=self._oscillation_score_tilt,
            **common,
        )

    def _reset_dynamic(self, *, clear_track: bool) -> None:
        self._filter_x.reset()
        self._filter_y.reset()
        self._movement_active = False
        self._previous_controlled_x = self._previous_controlled_y = 0.0
        self._last_control_at: float | None = None
        self._previous_pan = self._previous_tilt = 0.0
        self._oscillation_score_pan = self._oscillation_score_tilt = 0.0
        self._pan_sign = self._tilt_sign = 0
        self._previous_target_seen_at: float | None = None
        self._last_valid_target_update_at: float | None = None
        self._last_movement_decision_at: float | None = None
        if clear_track:
            self._active_track_id = None

    def _safety_hold(self, reason: str, now: float, **telemetry: object) -> PTZDecision:
        self._movement_active = False
        self._previous_controlled_x = self._previous_controlled_y = 0.0
        self._previous_pan = self._previous_tilt = 0.0
        self._last_control_at = now
        telemetry.setdefault(
            "target_error_x",
            telemetry.get("predicted_error_x", telemetry.get("raw_error_x", 0.0)),
        )
        telemetry.setdefault(
            "target_error_y",
            telemetry.get("predicted_error_y", telemetry.get("raw_error_y", 0.0)),
        )
        return self._hold(reason, **telemetry)

    def _normalized_velocity(self, target: TrackedTarget, width: int, height: int) -> tuple[float, float]:
        dt = target.velocity_dt_s
        if dt is None and self._previous_target_seen_at is not None:
            candidate = target.last_seen_timestamp - self._previous_target_seen_at
            dt = candidate if isfinite(candidate) and candidate > 0.0 else None
        self._previous_target_seen_at = target.last_seen_timestamp
        if dt is None or not isfinite(dt) or dt <= 0.0:
            return 0.0, 0.0
        if not all(isfinite(value) for value in (target.velocity_x, target.velocity_y)):
            return 0.0, 0.0
        return 2.0 * target.velocity_x / width / dt, 2.0 * target.velocity_y / height / dt

    def _derivatives(self, controlled_x: float, controlled_y: float, now: float) -> tuple[float, float, float | None]:
        dt = None if self._last_control_at is None else now - self._last_control_at
        if dt is None or not isfinite(dt) or dt <= 0.0:
            derivative_x = derivative_y = 0.0
            dt = None
        else:
            derivative_x = 0.0 if controlled_x == 0.0 else (controlled_x - self._previous_controlled_x) / dt
            derivative_y = 0.0 if controlled_y == 0.0 else (controlled_y - self._previous_controlled_y) / dt
        self._previous_controlled_x, self._previous_controlled_y = controlled_x, controlled_y
        self._last_control_at = now
        return derivative_x, derivative_y, dt

    def _slew(self, previous: float, requested: float, dt: float | None) -> float:
        if dt is None:
            return requested
        if requested == 0.0:
            return 0.0
        if previous != 0.0 and requested * previous > 0.0:
            if abs(requested) <= abs(previous):
                return requested
            delta = min(abs(requested) - abs(previous), self._max_command_delta_per_s * dt)
            return self._sign(requested) * (abs(previous) + delta)
        return self._sign(requested) * min(abs(requested), self._max_command_delta_per_s * dt)

    def _oscillation(self, command: float, previous_sign: int, score: float) -> tuple[float, int]:
        sign = self._sign(command)
        if sign and previous_sign and sign != previous_sign:
            score = min(self._oscillation_max, score + 1.0)
        else:
            score *= self._oscillation_decay
        return score, sign or previous_sign

    def _shape(self, value: float, maximum: float) -> float:
        if value == 0.0 or maximum == 0.0:
            return 0.0
        normalized = min(1.0, abs(value) / maximum)
        return self._sign(value) * maximum * normalized ** self._response_power

    @staticmethod
    def _soft_deadband(value: float, threshold: float) -> float:
        magnitude = abs(value)
        if magnitude <= threshold:
            return 0.0
        return PTZDecisionEngine._sign(value) * ((magnitude - threshold) / (1.0 - threshold))

    def _cooldown_active(self, now: float) -> bool:
        if self._last_movement_decision_at is None:
            return False
        return (now - self._last_movement_decision_at) * 1000.0 + 1e-9 < self._command_interval_ms

    def _hold(
        self, reason: str, target_error_x: float = 0.0,
        target_error_y: float = 0.0, inside_dead_zone: bool = False,
        *, position3d_candidate: bool = False, **telemetry: object,
    ) -> PTZDecision:
        return PTZDecision(
            action=DecisionAction.HOLD, pan=0.0, tilt=0.0, zoom=0.0,
            target_error_x=target_error_x, target_error_y=target_error_y,
            inside_dead_zone=inside_dead_zone, safe_to_move=False,
            reason=reason, position3d_candidate=position3d_candidate,
            **telemetry,
        )

    @staticmethod
    def _target_age_ms(last_seen: float, now: float) -> float | None:
        try:
            timestamp = float(last_seen)
        except (TypeError, ValueError):
            return None
        age = (now - timestamp) * 1000.0
        return age if isfinite(timestamp) and isfinite(age) and age >= 0.0 else None

    @staticmethod
    def _frame_age_ms(frame_age_s: float | None) -> float | None:
        if frame_age_s is None:
            return None
        try:
            age = float(frame_age_s) * 1000.0
        except (TypeError, ValueError) as error:
            raise PTZDecisionConfigurationError("frame_age_s debe ser numérico.") from error
        if not isfinite(age) or age < 0.0:
            raise PTZDecisionConfigurationError("frame_age_s debe ser finito y no negativo.")
        return age

    @staticmethod
    def _valid_bbox(target: TrackedTarget, width: int, height: int) -> bool:
        detection = target.detection
        values = (detection.confidence, detection.x1, detection.y1, detection.x2,
                  detection.y2, target.last_seen_timestamp)
        try:
            finite = all(isfinite(float(value)) for value in values)
        except (TypeError, ValueError):
            return False
        return (
            finite and 0.0 <= detection.confidence <= 1.0
            and detection.x1 >= 0 and detection.y1 >= 0
            and detection.x2 > detection.x1 and detection.y2 > detection.y1
            and detection.x2 <= width and detection.y2 <= height
            and target.visible_frames >= 0
        )

    @staticmethod
    def _frame_dimensions(frame_shape: Sequence[int]) -> tuple[int, int] | None:
        try:
            if len(frame_shape) < 2:
                return None
            height, width = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError):
            return None
        return (height, width) if height > 0 and width > 0 else None

    @staticmethod
    def _sign(value: float) -> int:
        return 1 if value > 0.0 else -1 if value < 0.0 else 0

    @staticmethod
    def _clamp(value: float, maximum: float) -> float:
        return min(maximum, max(-maximum, value))

    @staticmethod
    def _number(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZDecisionConfigurationError(f"{name} debe ser numérico.") from error
        if not isfinite(result):
            raise PTZDecisionConfigurationError(f"{name} debe ser finito.")
        return result

    @classmethod
    def _unit(cls, value: float, name: str) -> float:
        result = cls._number(value, name)
        if not 0.0 <= result <= 1.0:
            raise PTZDecisionConfigurationError(f"{name} debe estar entre 0.0 y 1.0.")
        return result

    @classmethod
    def _positive_unit(cls, value: float, name: str) -> float:
        result = cls._unit(value, name)
        if result == 0.0:
            raise PTZDecisionConfigurationError(f"{name} debe ser positivo.")
        return result

    @classmethod
    def _less_than_one(cls, value: float, name: str) -> float:
        result = cls._unit(value, name)
        if result >= 1.0:
            raise PTZDecisionConfigurationError(f"{name} debe ser menor que 1.0.")
        return result

    @classmethod
    def _positive(cls, value: float, name: str) -> float:
        result = cls._number(value, name)
        if result <= 0.0:
            raise PTZDecisionConfigurationError(f"{name} debe ser positivo.")
        return result

    @classmethod
    def _nonnegative(cls, value: float, name: str) -> float:
        result = cls._number(value, name)
        if result < 0.0:
            raise PTZDecisionConfigurationError(f"{name} no puede ser negativo.")
        return result

    @staticmethod
    def _integer(value: int, name: str, *, minimum: int) -> int:
        if isinstance(value, bool):
            raise PTZDecisionConfigurationError(f"{name} debe ser entero.")
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise PTZDecisionConfigurationError(f"{name} debe ser entero.") from error
        if result != value or result < minimum:
            raise PTZDecisionConfigurationError(
                f"{name} debe ser un entero mayor o igual que {minimum}."
            )
        return result

    @staticmethod
    def _valid_timestamp(value: float) -> float:
        result = PTZDecisionEngine._number(value, "timestamp")
        if result < 0.0:
            raise PTZDecisionConfigurationError("timestamp no puede ser negativo.")
        return result

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZDecisionConfigurationError(f"{name} debe ser numérico.") from error

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError as error:
            raise PTZDecisionConfigurationError(f"{name} debe ser entero.") from error

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
        raise PTZDecisionConfigurationError(f"{name} debe ser true o false.")
