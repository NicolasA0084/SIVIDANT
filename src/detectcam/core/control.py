"""Fast proportional centering with explicit evidence and recovery budgets.

Controller produces intents only. It performs no sleeping, networking, camera
calls, or frame processing. One control thread owns it, receiving immutable
TrackSnapshots from vision. A hardware adapter must enforce each expiry too.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .models import MotionIntent, TrackPhase, TrackSnapshot


@dataclass(frozen=True, slots=True)
class ControlConfig:
    max_pan: float = .65
    max_tilt: float = .65
    kp_acquire: float = 1.8
    kp_track: float = 1.2
    far_error: float = .25
    dead_zone: float = .025
    dead_zone_hysteresis: float = .015
    min_command: float = .07
    prediction_s: float = .06
    max_prediction: float = .08
    max_frame_age_s: float = .50
    target_fresh_s: float = .30
    recovery_s: float = .85
    recovery_max_command: float = .35
    recovery_budget: float = .09
    intent_ttl_s: float = .12

    def __post_init__(self) -> None:
        for name in ("max_pan", "max_tilt", "far_error", "dead_zone",
                     "dead_zone_hysteresis", "min_command", "max_prediction",
                     "recovery_max_command"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(name + " must be finite and in 0..1")
        for name in ("kp_acquire", "kp_track", "max_frame_age_s", "target_fresh_s", "intent_ttl_s"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("prediction_s", "recovery_s", "recovery_budget"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if self.dead_zone + self.dead_zone_hysteresis >= 1:
            raise ValueError("dead zone plus hysteresis must be less than one")
        if self.min_command > min(self.max_pan, self.max_tilt, self.recovery_max_command):
            raise ValueError("min_command must not exceed movement limits")


class Controller:
    """Immediately center a confirmed target, then stop on current visual error.

    No minimum visible-frame count or 200-ms command cooldown is imposed here.
    Apparent image velocity is not a blind feedforward term: optional short
    prediction uses background-compensated object velocity only. On loss the
    goal is the last observed point, shifted by reliable background evidence.
    Without that evidence, each axis has a finite integral-of-command budget.
    Neither repeated calls nor new empty frames renew the loss deadline.
    """

    def __init__(self, config: ControlConfig | None = None) -> None:
        self.config = config or ControlConfig()
        self.reset()

    def reset(self) -> None:
        self._track_id: int | None = None
        self._last_detection_frame = -1
        self._last_detection_at: float | None = None
        self._last_evidence_frame = -1
        self._last_evidence_at: float | None = None
        self._last_now: float | None = None
        self._last_intent: MotionIntent | None = None
        self._last_was_recovery = False
        self._centered_x = self._centered_y = True
        self._remaining_x = self._remaining_y = self.config.recovery_budget
        self._recovery_sign_x = self._recovery_sign_y = 0
        self._recovery_started = False
        self._recovery_finished_x = self._recovery_finished_y = False

    def command(self, snapshot: TrackSnapshot, now: float,
                video_age_s: float | None) -> MotionIntent:
        """Evaluate current time even if vision has not published another frame.

        Expiry is min(now + short TTL, acquisition + max image age, loss
        deadline). Reusing a snapshot can never extend its evidence lifetime.
        """
        if not isfinite(now) or now < 0:
            raise ValueError("now must be finite and nonnegative")
        if self._last_now is not None and now < self._last_now:
            return self._hold(snapshot, now, "control time moved backwards")
        self._consume_budget(now)
        self._last_now = now
        # Check sequence before resetting per-track dynamics. Otherwise an
        # older snapshot with a different identity could resurrect an old aim.
        if snapshot.frame_id < self._last_evidence_frame:
            return self._hold(snapshot, now, "out-of-order snapshot")
        if (snapshot.frame_id == self._last_evidence_frame and self._last_evidence_at is not None
                and snapshot.captured_at != self._last_evidence_at):
            return self._hold(snapshot, now, "source frame timestamp changed")
        if snapshot.track_id != self._track_id:
            evidence_frame, evidence_at = self._last_evidence_frame, self._last_evidence_at
            self.reset()
            self._last_now = now
            self._track_id = snapshot.track_id
            self._last_evidence_frame, self._last_evidence_at = evidence_frame, evidence_at
        self._last_evidence_frame = snapshot.frame_id
        self._last_evidence_at = snapshot.captured_at
        if (snapshot.detection is None or snapshot.center is None
                or snapshot.last_seen_at is None or snapshot.track_id is None):
            return self._hold(snapshot, now, "no target")
        if len(snapshot.frame_shape) < 2 or min(snapshot.frame_shape[:2]) <= 0:
            return self._hold(snapshot, now, "invalid frame dimensions")
        if (not all(isfinite(value) for value in (*snapshot.center, snapshot.captured_at, snapshot.last_seen_at))
                or snapshot.captured_at > now or snapshot.last_seen_at > snapshot.captured_at):
            return self._hold(snapshot, now, "invalid evidence timestamp or point")
        source_age = now - snapshot.captured_at
        if (video_age_s is None or not isfinite(video_age_s) or video_age_s < 0
                or video_age_s >= self.config.max_frame_age_s
                or source_age >= self.config.max_frame_age_s):
            return self._hold(snapshot, now, "video evidence unavailable or stale")
        if not snapshot.confirmed or snapshot.phase == TrackPhase.ACQUIRING:
            return self._hold(snapshot, now, "target confirmation pending")
        if snapshot.phase in (TrackPhase.IDLE, TrackPhase.EXPIRED):
            return self._hold(snapshot, now, "target recovery expired")
        if (snapshot.last_seen_frame_id < self._last_detection_frame
                or snapshot.last_seen_frame_id > snapshot.frame_id):
            return self._hold(snapshot, now, "invalid detection frame sequence")
        if (snapshot.last_seen_frame_id == self._last_detection_frame
                and self._last_detection_at is not None
                and snapshot.last_seen_at != self._last_detection_at):
            return self._hold(snapshot, now, "detection timestamp changed without evidence")
        if snapshot.last_seen_frame_id != self._last_detection_frame:
            # Only an actual new target detection renews recovery resources.
            # A newer image containing no target cannot reset this budget.
            self._last_detection_frame = snapshot.last_seen_frame_id
            self._last_detection_at = snapshot.last_seen_at
            self._remaining_x = self._remaining_y = self.config.recovery_budget
            self._recovery_started = False
            self._recovery_finished_x = self._recovery_finished_y = False
        height, width = snapshot.frame_shape[:2]
        ex = (snapshot.center[0] - width * .5) / (width * .5)
        ey = (snapshot.center[1] - height * .5) / (height * .5)
        if not (-1.5 <= ex <= 1.5 and -1.5 <= ey <= 1.5):
            return self._hold(snapshot, now, "estimated point left search area")
        age = now - snapshot.last_seen_at
        recovering = snapshot.phase == TrackPhase.RECOVERING or age >= self.config.target_fresh_s
        deadline = snapshot.captured_at + self.config.max_frame_age_s
        if recovering:
            deadline = min(deadline, snapshot.last_seen_at + self.config.recovery_s)
            if snapshot.recovery_deadline is not None:
                if not isfinite(snapshot.recovery_deadline):
                    return self._hold(snapshot, now, "invalid recovery deadline")
                deadline = min(deadline, snapshot.recovery_deadline)
            if now >= deadline:
                return self._hold(snapshot, now, "bounded target recovery expired")
            if not self._recovery_started:
                self._recovery_started = True
                self._recovery_sign_x = self._sign(ex)
                self._recovery_sign_y = self._sign(ey)
            ex = self._recovery_axis(ex, "x")
            ey = self._recovery_axis(ey, "y")
            limit_x = min(self.config.max_pan, self.config.recovery_max_command)
            limit_y = min(self.config.max_tilt, self.config.recovery_max_command)
            pan = self._axis(ex, self.config.kp_track, limit_x, "x")
            tilt = self._axis(ey, self.config.kp_track, limit_y, "y")
            visual_compensation = (snapshot.camera_compensated
                                   and snapshot.phase == TrackPhase.RECOVERING
                                   and source_age < self.config.target_fresh_s)
            if not visual_compensation:
                if self._remaining_x <= 1e-9:
                    pan = 0.
                if self._remaining_y <= 1e-9:
                    tilt = 0.
                # Bound the *issued request duration* too, so a paused caller
                # cannot spend more than its remaining recovery budget.
                if pan:
                    deadline = min(deadline, now + self._remaining_x / abs(pan))
                if tilt:
                    deadline = min(deadline, now + self._remaining_y / abs(tilt))
            reason = "last observed point; visual compensation" if visual_compensation else "last observed point; finite search budget"
            phase = TrackPhase.RECOVERING.value
        else:
            if snapshot.camera_compensated and snapshot.consecutive_hits >= 2:
                # Only independent object motion has a valid interpretation
                # here; apparent velocity also includes our camera's response.
                if all(isfinite(value) for value in (snapshot.object_velocity_x, snapshot.object_velocity_y)):
                    ex += self._clamp(2. * snapshot.object_velocity_x / width * self.config.prediction_s, self.config.max_prediction)
                    ey += self._clamp(2. * snapshot.object_velocity_y / height * self.config.prediction_s, self.config.max_prediction)
            far = max(abs(ex), abs(ey)) >= self.config.far_error
            gain = self.config.kp_acquire if far else self.config.kp_track
            pan = self._axis(ex, gain, self.config.max_pan, "x")
            tilt = self._axis(ey, gain, self.config.max_tilt, "y")
            reason = "rapid acquisition correction" if far else "visual tracking correction"
            phase = TrackPhase.TRACKING.value
        expires_at = min(now + self.config.intent_ttl_s, deadline)
        if expires_at <= now:
            return self._hold(snapshot, now, "motion evidence expired")
        if not pan and not tilt:
            if recovering and not visual_compensation and (self._remaining_x <= 1e-9 or self._remaining_y <= 1e-9):
                reason = "finite recovery budget exhausted"
            else:
                reason = "last observed point reached" if recovering else "target centered"
        intent = MotionIntent(pan, tilt, 0., expires_at, snapshot.frame_id, reason, phase,
                              now, snapshot.captured_at, snapshot.track_id, ex, ey)
        self._last_intent = intent
        self._last_was_recovery = recovering
        return intent

    def _consume_budget(self, now: float) -> None:
        previous = self._last_intent
        if previous is not None and self._last_was_recovery:
            duration = max(0., min(now, previous.expires_at) - previous.created_at)
            self._remaining_x = max(0., self._remaining_x - abs(previous.pan) * duration)
            self._remaining_y = max(0., self._remaining_y - abs(previous.tilt) * duration)
        self._last_intent = None

    def _recovery_axis(self, error: float, axis: str) -> float:
        sign = getattr(self, "_recovery_sign_" + axis)
        finished = "_recovery_finished_" + axis
        if getattr(self, finished) or abs(error) <= self.config.dead_zone or self._sign(error) != sign:
            setattr(self, finished, True)
            return 0.
        return error

    def _axis(self, error: float, gain: float, limit: float, axis: str) -> float:
        attr = "_centered_" + axis
        if abs(error) <= self.config.dead_zone:
            setattr(self, attr, True)
            return 0.
        if getattr(self, attr) and abs(error) <= self.config.dead_zone + self.config.dead_zone_hysteresis:
            return 0.
        setattr(self, attr, False)
        return self._sign(error) * min(limit, max(self.config.min_command, gain * abs(error)))

    def _hold(self, snapshot: TrackSnapshot, now: float, reason: str) -> MotionIntent:
        self._last_was_recovery = False
        self._last_intent = MotionIntent(0., 0., 0., now, snapshot.frame_id,
                                        reason, "HOLD", now, snapshot.captured_at,
                                        snapshot.track_id)
        return self._last_intent

    @staticmethod
    def _sign(value: float) -> int:
        return 1 if value > 0 else -1 if value < 0 else 0

    @staticmethod
    def _clamp(value: float, limit: float) -> float:
        return max(-limit, min(limit, value))
