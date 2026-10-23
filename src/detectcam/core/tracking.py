"""Single-target association with acquisition timestamps and time-bounded loss.

This is an independent, small constant-velocity association implementation;
it does not import SORT/ByteTrack or their code. One vision thread owns Tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, hypot, isfinite
from typing import Iterable, Sequence

from .models import Detection, FrameObservation, TrackPhase, TrackSnapshot


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    acquisition_mode: str = "fast"
    confirmation_hits: int = 2
    association_distance: float = .18
    min_iou: float = .03
    prediction_horizon_s: float = .18
    velocity_tau_s: float = .12
    max_velocity_diagonals_s: float = 2.
    recovery_s: float = .85
    forget_s: float = 1.20

    def __post_init__(self) -> None:
        if self.acquisition_mode not in {"fast", "confirm"}:
            raise ValueError("acquisition_mode must be fast or confirm")
        if isinstance(self.confirmation_hits, bool) or not isinstance(self.confirmation_hits, int) or self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be a positive integer")
        for name in ("association_distance", "min_iou"):
            value = getattr(self, name)
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(name + " must be finite and in 0..1")
        for name in ("prediction_horizon_s", "velocity_tau_s", "recovery_s", "forget_s"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and nonnegative")
        if not isfinite(self.max_velocity_diagonals_s) or self.max_velocity_diagonals_s <= 0:
            raise ValueError("max_velocity_diagonals_s must be finite and positive")
        if self.forget_s < self.recovery_s:
            raise ValueError("forget_s must be at least recovery_s")


def _iou(first: Detection, second: Detection, dx: float = 0., dy: float = 0.) -> float:
    width = max(0., min(first.x2 + dx, second.x2) - max(first.x1 + dx, second.x1))
    height = max(0., min(first.y2 + dy, second.y2) - max(first.y1 + dy, second.y1))
    intersection = width * height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.


class Tracker:
    """Maintain one identity, replacing a missing identity immediately if needed.

    update() consumes strictly increasing frame IDs and acquisition times;
    duplicates/out-of-order input cannot create evidence or renew deadlines.
    predict() only ages the snapshot. It never extends acquisition timestamps.
    An allowed_classes empty iterable pauses selection; None permits all.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._snapshot = TrackSnapshot()
        self._next_id = 1
        self._camera_since_seen = (0., 0.)
        self._camera_chain_reliable = True
        self._velocity_samples = 0
        self._last_filter: tuple[float, frozenset[str] | None] | None = None

    @property
    def snapshot(self) -> TrackSnapshot:
        return self._snapshot

    def reset(self) -> None:
        self._snapshot = TrackSnapshot()
        self._camera_since_seen = (0., 0.)
        self._camera_chain_reliable = True
        self._velocity_samples = 0
        self._last_filter = None

    def observe(self, observation: FrameObservation, *, min_confidence: float = .5,
                allowed_classes: Iterable[str] | None = None) -> TrackSnapshot:
        return self.update(observation.detections, observation.frame_id,
                           observation.captured_at, observation.frame_shape,
                           min_confidence, allowed_classes, observation.camera_delta)

    def update(self, detections: Sequence[Detection], frame_id: int,
               captured_at: float, frame_shape: Sequence[int],
               min_confidence: float = .5,
               allowed_classes: Iterable[str] | None = None,
               camera_delta: tuple[float, float] | None = None) -> TrackSnapshot:
        """Consume the latest detections and reliable background shift in pixels."""
        if not isfinite(captured_at) or captured_at < 0:
            raise ValueError("captured_at must use a finite nonnegative monotonic time")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError("frame_id must be a nonnegative integer")
        shape = tuple(int(value) for value in frame_shape)
        if len(shape) < 2 or min(shape[:2]) <= 0:
            raise ValueError("frame_shape must contain positive height and width")
        if not isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in 0..1")
        classes = None if allowed_classes is None else frozenset(item.strip().casefold() for item in allowed_classes)
        filters = min_confidence, classes
        previous = self._snapshot
        if frame_id <= previous.frame_id or (previous.frame_id >= 0 and captured_at <= previous.captured_at):
            return previous
        if previous.frame_id >= 0 and (shape[:2] != previous.frame_shape[:2] or filters != self._last_filter):
            self.reset()
            previous = self._snapshot
        self._last_filter = filters
        height, width = shape[:2]
        diagonal = hypot(width, height)
        candidates = [item for item in detections if self._valid(item, width, height, min_confidence, classes)]
        active = previous.detection is not None and previous.last_seen_at is not None and captured_at - previous.last_seen_at < self.config.forget_s
        if active and previous.center is not None:
            reliable = camera_delta is not None and len(camera_delta) == 2 and all(isfinite(value) for value in camera_delta)
            dx, dy = camera_delta if reliable else (0., 0.)
            # Reject physically implausible background estimates rather than
            # turning a bad optical-flow estimate into an unbounded search.
            if hypot(dx, dy) > diagonal * .5:
                reliable, dx, dy = False, 0., 0.
            self._camera_chain_reliable = self._camera_chain_reliable and reliable
            self._camera_since_seen = self._camera_since_seen[0] + dx, self._camera_since_seen[1] + dy
            center = previous.center[0] + dx, previous.center[1] + dy
            elapsed = captured_at - previous.last_seen_at
            horizon = min(elapsed, self.config.prediction_horizon_s)
            vx = previous.object_velocity_x if self._camera_chain_reliable else previous.velocity_x
            vy = previous.object_velocity_y if self._camera_chain_reliable else previous.velocity_y
            expected = center[0] + vx * horizon, center[1] + vy * horizon
            associated = self._associate(candidates, previous.detection, expected, diagonal)
            if associated is not None:
                return self._observe_target(associated, previous, frame_id, captured_at, shape,
                                            same_identity=True, camera_reliable=self._camera_chain_reliable)
            # A valid alternative immediately takes priority over missing
            # evidence; no ten-frame wait keeps the camera tied to an old box.
            if candidates:
                return self._observe_target(self._select(candidates, width, height), previous,
                                            frame_id, captured_at, shape, same_identity=False)
            phase = TrackPhase.RECOVERING if previous.confirmed and elapsed < self.config.recovery_s else TrackPhase.EXPIRED
            self._snapshot = replace(previous, phase=phase, center=center,
                                     frame_id=frame_id, captured_at=captured_at,
                                     frame_shape=shape, consecutive_hits=0,
                                     camera_compensated=self._camera_chain_reliable,
                                     reason="target missing; bounded recovery" if phase == TrackPhase.RECOVERING else "target recovery expired")
            return self._snapshot
        if candidates:
            return self._observe_target(self._select(candidates, width, height), previous,
                                        frame_id, captured_at, shape, same_identity=False)
        self._snapshot = TrackSnapshot(frame_id=frame_id, captured_at=captured_at,
                                       frame_shape=shape, reason="no eligible detection")
        return self._snapshot

    def predict(self, now: float) -> TrackSnapshot:
        """Age existing evidence without inventing a new frame or moving a point."""
        if not isfinite(now) or now < 0:
            raise ValueError("now must be finite and nonnegative")
        current = self._snapshot
        if current.last_seen_at is None or now < current.captured_at:
            return current
        age = now - current.last_seen_at
        if age >= self.config.forget_s:
            self._snapshot = replace(current, phase=TrackPhase.EXPIRED,
                                     confirmed=False, reason="target memory expired")
        elif age >= self.config.recovery_s:
            self._snapshot = replace(current, phase=TrackPhase.EXPIRED,
                                     reason="target recovery expired")
        return self._snapshot

    def _observe_target(self, detection: Detection, previous: TrackSnapshot,
                        frame_id: int, captured_at: float, shape: tuple[int, ...], *,
                        same_identity: bool, camera_reliable: bool = False) -> TrackSnapshot:
        vx = vy = object_vx = object_vy = 0.
        if same_identity:
            assert previous.detection is not None and previous.last_seen_at is not None
            dt = captured_at - previous.last_seen_at
            measured_x = (detection.center_x - previous.detection.center_x) / dt
            measured_y = (detection.center_y - previous.detection.center_y) / dt
            limit = hypot(shape[1], shape[0]) * self.config.max_velocity_diagonals_s
            measured_x, measured_y = self._limit_vector(measured_x, measured_y, limit)
            alpha = 1. if self._velocity_samples == 0 or self.config.velocity_tau_s == 0 else 1. - exp(-dt / self.config.velocity_tau_s)
            vx = previous.velocity_x + alpha * (measured_x - previous.velocity_x)
            vy = previous.velocity_y + alpha * (measured_y - previous.velocity_y)
            if camera_reliable:
                object_x = measured_x - self._camera_since_seen[0] / dt
                object_y = measured_y - self._camera_since_seen[1] / dt
                object_x, object_y = self._limit_vector(object_x, object_y, limit)
                object_vx = previous.object_velocity_x + alpha * (object_x - previous.object_velocity_x)
                object_vy = previous.object_velocity_y + alpha * (object_y - previous.object_velocity_y)
            hits = previous.consecutive_hits + 1
            track_id = previous.track_id
            confirmed = previous.confirmed or self.config.acquisition_mode == "fast" or hits >= self.config.confirmation_hits
            self._velocity_samples += 1
        else:
            track_id = self._next_id
            self._next_id += 1
            self._velocity_samples = 0
            hits = 1
            confirmed = self.config.acquisition_mode == "fast" or hits >= self.config.confirmation_hits
        self._camera_since_seen = (0., 0.)
        self._camera_chain_reliable = True
        phase = TrackPhase.TRACKING if confirmed else TrackPhase.ACQUIRING
        self._snapshot = TrackSnapshot(
            phase=phase, track_id=track_id, detection=detection, center=detection.center,
            frame_shape=shape, frame_id=frame_id, captured_at=captured_at,
            last_seen_at=captured_at, last_seen_frame_id=frame_id,
            velocity_x=vx, velocity_y=vy, object_velocity_x=object_vx,
            object_velocity_y=object_vy, camera_compensated=camera_reliable,
            confirmed=confirmed, consecutive_hits=hits,
            recovery_deadline=captured_at + self.config.recovery_s,
            reason="target associated" if same_identity else "new target acquired",
        )
        return self._snapshot

    def _associate(self, candidates: Sequence[Detection], previous: Detection,
                   expected: tuple[float, float], diagonal: float) -> Detection | None:
        scored: list[tuple[float, Detection]] = []
        dx, dy = expected[0] - previous.center_x, expected[1] - previous.center_y
        for item in candidates:
            if item.class_id != previous.class_id:
                continue
            distance = hypot(item.center_x - expected[0], item.center_y - expected[1]) / diagonal
            overlap = _iou(previous, item, dx, dy)
            if distance > self.config.association_distance and overlap < self.config.min_iou:
                continue
            score = .55 * overlap + .35 * (1. - min(1., distance)) + .10 * item.confidence
            scored.append((score, item))
        return max(scored, key=lambda pair: pair[0])[1] if scored else None

    @staticmethod
    def _select(candidates: Sequence[Detection], width: int, height: int) -> Detection:
        return max(candidates, key=lambda item: (
            item.confidence, -hypot(item.center_x - width * .5, item.center_y - height * .5), item.area,
        ))

    @staticmethod
    def _limit_vector(x: float, y: float, limit: float) -> tuple[float, float]:
        magnitude = hypot(x, y)
        return (x * limit / magnitude, y * limit / magnitude) if magnitude > limit else (x, y)

    @staticmethod
    def _valid(item: Detection, width: int, height: int, threshold: float,
               classes: frozenset[str] | None) -> bool:
        return (all(isfinite(value) for value in (item.confidence, item.x1, item.y1, item.x2, item.y2))
                and threshold <= item.confidence <= 1.
                and (classes is None or item.class_name.strip().casefold() in classes)
                and 0 <= item.x1 < item.x2 <= width and 0 <= item.y1 < item.y2 <= height)


TargetTracker = Tracker
