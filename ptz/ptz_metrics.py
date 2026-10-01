"""Métricas pasivas para validar el seguimiento PTZ sin modificar el control."""

from __future__ import annotations

from collections import deque
import csv
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
import json
import logging
from math import isfinite
import os
from pathlib import Path
from statistics import fmean, median
from typing import Callable, Mapping


LOGGER = logging.getLogger(__name__)
DEFAULT_METRICS_ENABLED = False
DEFAULT_RECENTER_STABLE_MS = 300
DEFAULT_RECENTER_START_ERROR = 0.20
DEFAULT_OSCILLATION_WINDOW_MS = 1_000


class PTZMetricsConfigurationError(ValueError):
    """La configuración de métricas no es válida."""


@dataclass(frozen=True)
class PTZMetricSample:
    """Muestra completa de una iteración del pipeline."""

    time_s: float
    track_id: int | None
    tracker_state: str
    confidence: float | None
    raw_ex: float
    raw_ey: float
    filtered_ex: float
    filtered_ey: float
    predicted_ex: float
    predicted_ey: float
    pan_cmd: float
    tilt_cmd: float
    pan_physical: float
    tilt_physical: float
    decision: str
    executor_state: str
    frame_age_ms: float | None
    target_age_ms: float | None
    inference_ms: float | None
    decision_processing_ms: float
    executor_processing_ms: float
    backend_latency_ms: float | None
    command_interval_ms: float | None
    deadman_activations: int
    watchdog_activations: int
    osc_pan: float
    osc_tilt: float
    inside_dead_zone: bool
    safe_to_move: bool
    reason: str


@dataclass(frozen=True)
class RecenterEvent:
    initial_error_x: float
    initial_error_y: float
    time_to_enter_ms: float
    recenter_time_ms: float
    overshoot_x: float
    overshoot_y: float


@dataclass(frozen=True)
class PTZMetricsSnapshot:
    recenter_active: bool
    pan_sign_flips: int
    tilt_sign_flips: int
    pan_saturations: int
    tilt_saturations: int
    dead_zone_chatter_events: int


class PTZMetrics:
    """Acumula muestras, eventos y un CSV opcional por sesión."""

    def __init__(
        self,
        *,
        enabled: bool = DEFAULT_METRICS_ENABLED,
        recenter_stable_ms: int = DEFAULT_RECENTER_STABLE_MS,
        recenter_start_error: float = DEFAULT_RECENTER_START_ERROR,
        oscillation_window_ms: int = DEFAULT_OSCILLATION_WINDOW_MS,
        physical_max_pan: float = 0.30,
        physical_max_tilt: float = 0.30,
        output_dir: str | Path = "runs/ptz_metrics",
        clock: Callable[[], float] | None = None,
        session_name: str | None = None,
        profile: str = "UNSPECIFIED",
        effective_configuration: Mapping[str, object] | None = None,
        write_csv: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.recenter_stable_ms = self._positive_int(
            recenter_stable_ms, "recenter_stable_ms"
        )
        self.recenter_start_error = self._positive_float(
            recenter_start_error, "recenter_start_error"
        )
        self.oscillation_window_ms = self._positive_int(
            oscillation_window_ms, "oscillation_window_ms"
        )
        self.physical_max_pan = self._positive_float(
            physical_max_pan, "physical_max_pan"
        )
        self.physical_max_tilt = self._positive_float(
            physical_max_tilt, "physical_max_tilt"
        )
        self._clock = clock
        self._output_dir = Path(output_dir)
        self._session_name = session_name or datetime.now().strftime(
            "session_%Y%m%d_%H%M%S_%f.csv"
        )
        self.profile = str(profile).strip().upper() or "UNSPECIFIED"
        self.effective_configuration = dict(effective_configuration or {})
        self._write_csv = bool(write_csv)
        self._csv_file = None
        self._writer: csv.DictWriter | None = None
        self._rows_written = 0
        self.csv_path: Path | None = None
        self.metadata_path: Path | None = None
        self._closed = False
        self._reset_state()

    @classmethod
    def from_environment(cls, **overrides: object) -> "PTZMetrics":
        options = dict(overrides)
        options.setdefault(
            "enabled", cls._env_bool(
                "DETECTCAM_PTZ_METRICS_ENABLED", DEFAULT_METRICS_ENABLED
            )
        )
        options.setdefault(
            "recenter_stable_ms", cls._env_int(
                "DETECTCAM_PTZ_RECENTER_STABLE_MS", DEFAULT_RECENTER_STABLE_MS
            )
        )
        options.setdefault(
            "physical_max_pan", cls._env_float(
                "DETECTCAM_PTZ_PHYSICAL_MAX_PAN", 0.30
            )
        )
        options.setdefault(
            "physical_max_tilt", cls._env_float(
                "DETECTCAM_PTZ_PHYSICAL_MAX_TILT", 0.30
            )
        )
        return cls(**options)

    def _reset_state(self) -> None:
        self._first_time: float | None = None
        self._last: PTZMetricSample | None = None
        self._tracked_time = self._hold_time = self._move_time = 0.0
        self._hold_entries = 0
        self._hold_durations: list[float] = []
        self._current_hold_started: float | None = None
        self._hold_reason_time: dict[str, float] = {
            "dead_zone": 0.0,
            "stale_target": 0.0,
            "stale_frame": 0.0,
            "temporarily_lost": 0.0,
            "cooldown": 0.0,
            "other": 0.0,
        }
        self.recenter_events: list[RecenterEvent] = []
        self._recenter_started: float | None = None
        self._recenter_entered: float | None = None
        self._recenter_initial = (0.0, 0.0)
        self._recenter_track_id: int | None = None
        self._overshoot_x = self._overshoot_y = 0.0
        self.pan_sign_flips = self.tilt_sign_flips = 0
        self.pan_oscillation_sequences = self.tilt_oscillation_sequences = 0
        self._last_pan_sign = self._last_tilt_sign = 0
        self._pan_sign_history: deque[tuple[float, int]] = deque(maxlen=3)
        self._tilt_sign_history: deque[tuple[float, int]] = deque(maxlen=3)
        self.max_oscillation_score = 0.0
        self.pan_saturations = self.tilt_saturations = 0
        self.move_commands = self.stop_commands = 0
        self.command_intervals_ms: list[float] = []
        self._last_move_command_at: float | None = None
        self.temporary_losses = self.same_track_recoveries = 0
        self.permanent_losses = 0
        self.loss_durations_ms: list[float] = []
        self.resume_delays_ms: list[float] = []
        self._loss_started: float | None = None
        self._loss_track_id: int | None = None
        self._recovered_at: float | None = None
        self._previous_dead_zone: bool | None = None
        self.dead_zone_entries = self.dead_zone_exits = 0
        self.dead_zone_chatter_events = 0
        self._dead_zone_transitions: deque[tuple[float, bool]] = deque(maxlen=4)
        self.deadman_activations = self.watchdog_activations = 0
        self.emergency_stops = 0
        self.backend_latencies_ms: list[float] = []
        self.decision_latencies_ms: list[float] = []
        self.executor_latencies_ms: list[float] = []
        self.frame_ages_ms: list[float] = []
        self.confidences: list[float] = []

    @property
    def snapshot(self) -> PTZMetricsSnapshot:
        return PTZMetricsSnapshot(
            recenter_active=self._recenter_started is not None,
            pan_sign_flips=self.pan_sign_flips,
            tilt_sign_flips=self.tilt_sign_flips,
            pan_saturations=self.pan_saturations,
            tilt_saturations=self.tilt_saturations,
            dead_zone_chatter_events=self.dead_zone_chatter_events,
        )

    def record(self, sample: PTZMetricSample) -> None:
        if not self.enabled or self._closed:
            return
        self._validate_sample(sample)
        backend_request_observed = (
            sample.reason == "continuous movement executed"
            or (
                self._last is not None
                and self._moving(self._last)
                and not self._moving(sample)
            )
        )
        if not backend_request_observed:
            sample = replace(sample, backend_latency_ms=None)
        if self._write_csv:
            self._ensure_csv()
            assert self._writer is not None
            self._writer.writerow(asdict(sample))
            self._rows_written += 1
            if self._rows_written % 30 == 0:
                assert self._csv_file is not None
                self._csv_file.flush()
        if self._first_time is None:
            self._first_time = sample.time_s
        self._accumulate_time(sample)
        self._track_recenter(sample)
        self._track_signs(sample)
        self._track_dead_zone(sample)
        self._track_commands(sample)
        self._track_losses(sample)
        self.max_oscillation_score = max(
            self.max_oscillation_score, sample.osc_pan, sample.osc_tilt
        )
        self.decision_latencies_ms.append(sample.decision_processing_ms)
        self.executor_latencies_ms.append(sample.executor_processing_ms)
        if sample.frame_age_ms is not None:
            self.frame_ages_ms.append(sample.frame_age_ms)
        if sample.tracker_state == "TRACKED" and sample.confidence is not None:
            self.confidences.append(sample.confidence)
        if sample.backend_latency_ms is not None:
            self.backend_latencies_ms.append(sample.backend_latency_ms)
        self.deadman_activations = max(
            self.deadman_activations, sample.deadman_activations
        )
        self.watchdog_activations = max(
            self.watchdog_activations, sample.watchdog_activations
        )
        self._last = sample

    def note_emergency_stop(self) -> None:
        if self.enabled:
            self.emergency_stops += 1

    def note_stop(self) -> None:
        if self.enabled:
            self.stop_commands += 1

    def sync_executor_counters(
        self, *, stop_commands: int, deadman_activations: int,
        watchdog_activations: int,
    ) -> None:
        if not self.enabled:
            return
        self.stop_commands = max(self.stop_commands, stop_commands)
        self.deadman_activations = max(
            self.deadman_activations, deadman_activations
        )
        self.watchdog_activations = max(
            self.watchdog_activations, watchdog_activations
        )

    def note_timeout(self, reason: str) -> None:
        if not self.enabled:
            return
        if "dead-man" in reason:
            self.deadman_activations += 1
        elif "heartbeat timeout" in reason:
            self.watchdog_activations += 1

    def close(self) -> None:
        if self._closed:
            return
        if self.enabled and self._last is not None:
            self._finish_open_hold(self._last.time_s)
        if self._csv_file is not None:
            self._csv_file.close()
        if (
            self.metadata_path is not None
            and self.metadata_path.parent.exists()
        ):
            self._write_metadata(include_summary=True)
        self._closed = True

    def summary(self) -> dict[str, object]:
        duration = 0.0
        if self._first_time is not None and self._last is not None:
            duration = max(0.0, self._last.time_s - self._first_time)
        recenter_times = [event.recenter_time_ms for event in self.recenter_events]
        confidence = sorted(self.confidences)
        max_overshoot_x = max(
            (event.overshoot_x for event in self.recenter_events), default=0.0
        )
        max_overshoot_y = max(
            (event.overshoot_y for event in self.recenter_events), default=0.0
        )
        return {
            "duration_s": duration,
            "tracked_time_s": self._tracked_time,
            "hold_time_s": self._hold_time,
            "move_time_s": self._move_time,
            "hold_entries": self._hold_entries,
            "average_hold_ms": self._mean_ms(self._hold_durations),
            "hold_reason_time_s": dict(self._hold_reason_time),
            "recenter_events": len(self.recenter_events),
            "average_recenter_ms": self._mean(recenter_times),
            "max_recenter_ms": max(recenter_times, default=0.0),
            "max_overshoot_x": max_overshoot_x,
            "max_overshoot_y": max_overshoot_y,
            "recenter_event_details": [asdict(event) for event in self.recenter_events],
            "pan_sign_flips": self.pan_sign_flips,
            "tilt_sign_flips": self.tilt_sign_flips,
            "sign_flips_per_s": (
                (self.pan_sign_flips + self.tilt_sign_flips) / duration
                if duration > 0.0 else 0.0
            ),
            "pan_oscillation_sequences": self.pan_oscillation_sequences,
            "tilt_oscillation_sequences": self.tilt_oscillation_sequences,
            "max_oscillation_score": self.max_oscillation_score,
            "pan_saturations": self.pan_saturations,
            "tilt_saturations": self.tilt_saturations,
            "move_commands": self.move_commands,
            "move_commands_per_s": self.move_commands / duration if duration > 0 else 0.0,
            "stop_commands": self.stop_commands,
            "average_command_interval_ms": self._mean(self.command_intervals_ms),
            "temporary_losses": self.temporary_losses,
            "same_track_recoveries": self.same_track_recoveries,
            "permanent_losses": self.permanent_losses,
            "average_loss_ms": self._mean(self.loss_durations_ms),
            "average_resume_delay_ms": self._mean(self.resume_delays_ms),
            "dead_zone_entries": self.dead_zone_entries,
            "dead_zone_exits": self.dead_zone_exits,
            "dead_zone_chatter_events": self.dead_zone_chatter_events,
            "deadman_activations": self.deadman_activations,
            "watchdog_activations": self.watchdog_activations,
            "emergency_stops": self.emergency_stops,
            "average_backend_latency_ms": self._mean(self.backend_latencies_ms),
            "average_decision_latency_ms": self._mean(self.decision_latencies_ms),
            "average_executor_latency_ms": self._mean(self.executor_latencies_ms),
            "average_frame_age_ms": self._mean(self.frame_ages_ms),
            "p90_frame_age_ms": self._percentile(sorted(self.frame_ages_ms), 0.90),
            "max_frame_age_ms": max(self.frame_ages_ms, default=0.0),
            "confidence_samples": len(confidence),
            "confidence_mean": self._mean(confidence),
            "confidence_median": median(confidence) if confidence else 0.0,
            "confidence_p10": self._percentile(confidence, 0.10),
            "confidence_p25": self._percentile(confidence, 0.25),
            "confidence_p75": self._percentile(confidence, 0.75),
            "confidence_p90": self._percentile(confidence, 0.90),
            "confidence_below_0_50": sum(value < 0.50 for value in confidence),
            "confidence_between_0_50_0_60": sum(
                0.50 <= value <= 0.60 for value in confidence
            ),
            "confidence_above_0_60": sum(value > 0.60 for value in confidence),
            "profile": self.profile,
            "effective_configuration": dict(self.effective_configuration),
            "csv_path": None if self.csv_path is None else str(self.csv_path),
            "metadata_path": (
                None if self.metadata_path is None else str(self.metadata_path)
            ),
        }

    def summary_text(self) -> str:
        data = self.summary()
        return (
            "Session PTZ Metrics\n"
            f"Duration: {data['duration_s']:.2f} s\n"
            f"Tracked time: {data['tracked_time_s']:.2f} s\n"
            f"HOLD time: {data['hold_time_s']:.2f} s\n"
            f"MOVE time: {data['move_time_s']:.2f} s\n"
            f"HOLD entries / average: {data['hold_entries']} / "
            f"{data['average_hold_ms']:.1f} ms\n"
            f"Recenter events: {data['recenter_events']}\n"
            f"Average recenter: {data['average_recenter_ms']:.1f} ms\n"
            f"Max recenter: {data['max_recenter_ms']:.1f} ms\n"
            f"Max overshoot X/Y: {data['max_overshoot_x']:.3f} / "
            f"{data['max_overshoot_y']:.3f}\n"
            f"Pan sign flips: {data['pan_sign_flips']}\n"
            f"Tilt sign flips: {data['tilt_sign_flips']}\n"
            f"Oscillation sequences P/T: {data['pan_oscillation_sequences']} / "
            f"{data['tilt_oscillation_sequences']}\n"
            f"Dead-zone chatter: {data['dead_zone_chatter_events']}\n"
            f"Pan saturation: {data['pan_saturations']}\n"
            f"Tilt saturation: {data['tilt_saturations']}\n"
            f"MOVE commands / rate: {data['move_commands']} / "
            f"{data['move_commands_per_s']:.2f} commands/s\n"
            f"STOP commands: {data['stop_commands']}\n"
            f"Average command interval: {data['average_command_interval_ms']:.1f} ms\n"
            f"Temporary losses: {data['temporary_losses']}\n"
            f"Recovered same track: {data['same_track_recoveries']}\n"
            f"Permanent losses: {data['permanent_losses']}\n"
            f"Dead-man activations: {data['deadman_activations']}\n"
            f"Watchdog activations: {data['watchdog_activations']}\n"
            f"Emergency stops: {data['emergency_stops']}\n"
            f"Average latency decision/executor/backend: "
            f"{data['average_decision_latency_ms']:.2f} / "
            f"{data['average_executor_latency_ms']:.2f} / "
            f"{data['average_backend_latency_ms']:.2f} ms\n"
            f"Frame age average/p90/max: {data['average_frame_age_ms']:.1f} / "
            f"{data['p90_frame_age_ms']:.1f} / {data['max_frame_age_ms']:.1f} ms\n"
            f"Confidence mean/median/p10/p25/p75/p90: "
            f"{data['confidence_mean']:.3f} / {data['confidence_median']:.3f} / "
            f"{data['confidence_p10']:.3f} / {data['confidence_p25']:.3f} / "
            f"{data['confidence_p75']:.3f} / {data['confidence_p90']:.3f}\n"
            f"Confidence samples <0.50 / 0.50..0.60 / >0.60: "
            f"{data['confidence_below_0_50']} / "
            f"{data['confidence_between_0_50_0_60']} / "
            f"{data['confidence_above_0_60']}\n"
            f"Profile: {data['profile']}\n"
            f"CSV: {data['csv_path'] or 'disabled'}\n"
            f"Effective config: {data['metadata_path'] or 'disabled'}"
        )

    def _ensure_csv(self) -> None:
        if self._writer is not None:
            return
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self._output_dir / self._session_name
        self.metadata_path = self.csv_path.with_suffix(".json")
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        names = [field.name for field in fields(PTZMetricSample)]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=names)
        self._writer.writeheader()
        self._write_metadata(include_summary=False)

    def _write_metadata(self, *, include_summary: bool) -> None:
        assert self.metadata_path is not None
        payload: dict[str, object] = {
            "schema_version": 1,
            "profile": self.profile,
            "effective_configuration": self.effective_configuration,
        }
        if include_summary:
            payload["summary"] = self.summary()
        self.metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _accumulate_time(self, sample: PTZMetricSample) -> None:
        previous = self._last
        if previous is None:
            if sample.decision == "HOLD":
                self._start_hold(sample.time_s)
            return
        dt = max(0.0, sample.time_s - previous.time_s)
        if previous.tracker_state == "TRACKED":
            self._tracked_time += dt
        if previous.decision == "HOLD":
            self._hold_time += dt
            self._hold_reason_time[self._hold_category(previous.reason)] += dt
        if self._moving(previous):
            self._move_time += dt
        if sample.decision == "HOLD" and previous.decision != "HOLD":
            self._start_hold(sample.time_s)
        elif sample.decision != "HOLD" and previous.decision == "HOLD":
            self._finish_open_hold(sample.time_s)

    def _start_hold(self, now: float) -> None:
        self._hold_entries += 1
        self._current_hold_started = now

    def _finish_open_hold(self, now: float) -> None:
        if self._current_hold_started is not None:
            self._hold_durations.append(max(0.0, now - self._current_hold_started))
            self._current_hold_started = None

    def _track_recenter(self, sample: PTZMetricSample) -> None:
        if (
            self._recenter_started is not None
            and sample.track_id != self._recenter_track_id
        ):
            self._recenter_started = self._recenter_entered = None
            self._recenter_track_id = None
        if self._recenter_started is None:
            if (
                sample.tracker_state == "TRACKED"
                and max(abs(sample.raw_ex), abs(sample.raw_ey)) > self.recenter_start_error
            ):
                self._recenter_started = sample.time_s
                self._recenter_initial = (sample.raw_ex, sample.raw_ey)
                self._recenter_track_id = sample.track_id
                self._recenter_entered = None
                self._overshoot_x = self._overshoot_y = 0.0
            return
        if sample.tracker_state != "TRACKED":
            self._recenter_entered = None
            return
        initial_x, initial_y = self._recenter_initial
        self._overshoot_x = max(
            self._overshoot_x, self._opposite_error(initial_x, sample.raw_ex)
        )
        self._overshoot_y = max(
            self._overshoot_y, self._opposite_error(initial_y, sample.raw_ey)
        )
        if not sample.inside_dead_zone:
            self._recenter_entered = None
            return
        if self._recenter_entered is None:
            self._recenter_entered = sample.time_s
            return
        stable_ms = (sample.time_s - self._recenter_entered) * 1000.0
        if stable_ms + 1e-9 < self.recenter_stable_ms:
            return
        assert self._recenter_started is not None
        event = RecenterEvent(
            initial_error_x=initial_x,
            initial_error_y=initial_y,
            time_to_enter_ms=(self._recenter_entered - self._recenter_started) * 1000.0,
            recenter_time_ms=(sample.time_s - self._recenter_started) * 1000.0,
            overshoot_x=self._overshoot_x,
            overshoot_y=self._overshoot_y,
        )
        self.recenter_events.append(event)
        LOGGER.info(
            "Recenter: initial=(%+.3f,%+.3f), enter=%.1f ms, stable=%.1f ms, "
            "overshoot=(%.3f,%.3f)",
            event.initial_error_x, event.initial_error_y,
            event.time_to_enter_ms, event.recenter_time_ms,
            event.overshoot_x, event.overshoot_y,
        )
        self._recenter_started = self._recenter_entered = None
        self._recenter_track_id = None

    def _track_signs(self, sample: PTZMetricSample) -> None:
        self._last_pan_sign, flipped = self._axis_sign(
            sample.time_s, sample.pan_physical, self._last_pan_sign,
            self._pan_sign_history,
        )
        if flipped:
            self.pan_sign_flips += 1
            if self._alternating(self._pan_sign_history):
                self.pan_oscillation_sequences += 1
        self._last_tilt_sign, flipped = self._axis_sign(
            sample.time_s, sample.tilt_physical, self._last_tilt_sign,
            self._tilt_sign_history,
        )
        if flipped:
            self.tilt_sign_flips += 1
            if self._alternating(self._tilt_sign_history):
                self.tilt_oscillation_sequences += 1

    def _axis_sign(
        self, now: float, value: float, previous: int,
        history: deque[tuple[float, int]],
    ) -> tuple[int, bool]:
        sign = 1 if value > 1e-9 else -1 if value < -1e-9 else 0
        if sign == 0:
            return previous, False
        flipped = previous != 0 and sign != previous
        if not history or history[-1][1] != sign:
            history.append((now, sign))
        return sign, flipped

    def _alternating(self, history: deque[tuple[float, int]]) -> bool:
        return (
            len(history) == 3
            and history[0][1] == history[2][1] != history[1][1]
            and (history[2][0] - history[0][0]) * 1000.0
            <= self.oscillation_window_ms
        )

    def _track_dead_zone(self, sample: PTZMetricSample) -> None:
        if sample.tracker_state != "TRACKED":
            self._previous_dead_zone = None
            self._dead_zone_transitions.clear()
            return
        current = sample.inside_dead_zone
        if self._previous_dead_zone is None:
            self._previous_dead_zone = current
            return
        if current == self._previous_dead_zone:
            return
        if current:
            self.dead_zone_entries += 1
        else:
            self.dead_zone_exits += 1
        self._dead_zone_transitions.append((sample.time_s, current))
        if (
            len(self._dead_zone_transitions) == 4
            and (self._dead_zone_transitions[-1][0] - self._dead_zone_transitions[0][0])
            * 1000.0 <= self.oscillation_window_ms
        ):
            self.dead_zone_chatter_events += 1
            self._dead_zone_transitions.clear()
        self._previous_dead_zone = current

    def _track_commands(self, sample: PTZMetricSample) -> None:
        if sample.reason == "continuous movement executed":
            self.move_commands += 1
            if self._last_move_command_at is not None:
                interval = (sample.time_s - self._last_move_command_at) * 1000.0
                if interval >= 0.0:
                    self.command_intervals_ms.append(interval)
            self._last_move_command_at = sample.time_s
            if abs(sample.pan_physical) + 1e-9 >= self.physical_max_pan:
                self.pan_saturations += 1
            if abs(sample.tilt_physical) + 1e-9 >= self.physical_max_tilt:
                self.tilt_saturations += 1
        if self._last is not None and self._moving(self._last) and not self._moving(sample):
            self.stop_commands += 1

    def _track_losses(self, sample: PTZMetricSample) -> None:
        previous = self._last
        if previous is None:
            return
        if previous.tracker_state == "TRACKED" and sample.tracker_state == "TEMPORARILY_LOST":
            self.temporary_losses += 1
            self._loss_started = sample.time_s
            self._loss_track_id = previous.track_id
        elif previous.tracker_state == "TEMPORARILY_LOST" and sample.tracker_state == "TRACKED":
            if self._loss_started is not None:
                self.loss_durations_ms.append((sample.time_s - self._loss_started) * 1000.0)
            if sample.track_id == self._loss_track_id:
                self.same_track_recoveries += 1
            self._recovered_at = sample.time_s
            self._loss_started = None
        elif previous.tracker_state == "TEMPORARILY_LOST" and sample.tracker_state == "NO_TARGET":
            self.permanent_losses += 1
            if self._loss_started is not None:
                self.loss_durations_ms.append((sample.time_s - self._loss_started) * 1000.0)
            self._loss_started = None
            self._recovered_at = None
        if self._recovered_at is not None and self._moving(sample):
            self.resume_delays_ms.append((sample.time_s - self._recovered_at) * 1000.0)
            self._recovered_at = None

    @staticmethod
    def _moving(sample: PTZMetricSample) -> bool:
        return abs(sample.pan_physical) > 1e-9 or abs(sample.tilt_physical) > 1e-9

    @staticmethod
    def _opposite_error(initial: float, current: float) -> float:
        if initial > 0.0:
            return max(0.0, -current)
        if initial < 0.0:
            return max(0.0, current)
        return 0.0

    @staticmethod
    def _hold_category(reason: str) -> str:
        lowered = reason.lower()
        if "dead zone" in lowered or "dead-zone" in lowered or "centered" in lowered:
            return "dead_zone"
        if "target stale" in lowered:
            return "stale_target"
        if "frame stale" in lowered:
            return "stale_frame"
        if "temporarily lost" in lowered:
            return "temporarily_lost"
        if "cooldown" in lowered:
            return "cooldown"
        return "other"

    @staticmethod
    def _validate_sample(sample: PTZMetricSample) -> None:
        if not isfinite(sample.time_s) or sample.time_s < 0.0:
            raise PTZMetricsConfigurationError("time_s debe ser finito y no negativo.")

    @staticmethod
    def _mean(values: list[float]) -> float:
        return fmean(values) if values else 0.0

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    @staticmethod
    def _mean_ms(values_s: list[float]) -> float:
        return PTZMetrics._mean(values_s) * 1000.0

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError) as error:
            raise PTZMetricsConfigurationError(f"{name} debe ser entero positivo.") from error
        if isinstance(value, bool) or result != value or result <= 0:
            raise PTZMetricsConfigurationError(f"{name} debe ser entero positivo.")
        return result

    @staticmethod
    def _positive_float(value: float, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PTZMetricsConfigurationError(f"{name} debe ser positivo.") from error
        if not isfinite(result) or result <= 0.0:
            raise PTZMetricsConfigurationError(f"{name} debe ser positivo.")
        return result

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise PTZMetricsConfigurationError(f"{name} debe ser true o false.")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError as error:
            raise PTZMetricsConfigurationError(f"{name} debe ser entero.") from error

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError as error:
            raise PTZMetricsConfigurationError(f"{name} debe ser numérico.") from error
