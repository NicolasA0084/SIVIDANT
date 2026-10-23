"""Immutable messages shared by vision and control, with no camera dependencies.

All times use the same monotonic clock in seconds. ``captured_at`` is the
decoded frame acquisition time, never the completion time of inference.
Screen coordinates use pixels, positive X right and positive Y down.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackPhase(str, Enum):
    IDLE = "IDLE"
    ACQUIRING = "ACQUIRING"
    TRACKING = "TRACKING"
    RECOVERING = "RECOVERING"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) * .5

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) * .5

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y

    @property
    def width(self) -> float:
        return max(0., self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0., self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """One acquisition; camera_delta is background displacement since prior frame.

    Pass ``None`` when background motion cannot be estimated reliably. A real
    zero displacement is ``(0., 0.)``; it is useful evidence, unlike ``None``.
    """
    frame_id: int
    captured_at: float
    frame_shape: tuple[int, ...]
    detections: tuple[Detection, ...]
    camera_delta: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Thread-safe value published by Tracker; never contains live tracker state.

    ``detection`` is the last actual detection. ``center`` locates that point in
    the latest observed image, including reliable background displacement.
    Apparent velocity is pixels/second, and includes camera movement. Object
    velocity removes background movement only when ``camera_compensated`` is
    true. During loss, center follows the last observed point, not an infinite
    extrapolation of the object's last velocity.
    """
    phase: TrackPhase = TrackPhase.IDLE
    track_id: int | None = None
    detection: Detection | None = None
    center: tuple[float, float] | None = None
    frame_shape: tuple[int, ...] = (0, 0, 3)
    frame_id: int = -1
    captured_at: float = 0.
    last_seen_at: float | None = None
    last_seen_frame_id: int = -1
    velocity_x: float = 0.
    velocity_y: float = 0.
    object_velocity_x: float = 0.
    object_velocity_y: float = 0.
    camera_compensated: bool = False
    confirmed: bool = False
    consecutive_hits: int = 0
    recovery_deadline: float | None = None
    reason: str = "no target"

    @property
    def target(self) -> Detection | None:
        return self.detection

    @property
    def last_seen(self) -> float | None:
        return self.last_seen_at

    @property
    def velocity(self) -> tuple[float, float]:
        return self.velocity_x, self.velocity_y


@dataclass(frozen=True, slots=True)
class MotionIntent:
    """A request, not a hardware action; velocities are normalized to [-1, 1].

    ``expires_at`` never moves beyond the source evidence deadline. Runtime
    must reject expired requests and must not renew expiry from its own clock.
    Pan/tilt use screen coordinates; hardware adapters translate motor signs.
    """
    pan: float
    tilt: float
    zoom: float
    expires_at: float
    source_frame_id: int
    reason: str
    phase: str
    created_at: float = 0.
    evidence_at: float = 0.
    track_id: int | None = None
    error_x: float = 0.
    error_y: float = 0.

    @property
    def moving(self) -> bool:
        return bool(self.pan or self.tilt or self.zoom)
