"""Immutable transport models; timestamps use time.monotonic()."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class MotionLimits:
    max_pan: float = 0.75
    max_tilt: float = 0.75
    max_zoom: float = 0.30

    def __post_init__(self) -> None:
        for name in ("max_pan", "max_tilt", "max_zoom"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")

    def clamp(self, pan: float, tilt: float, zoom: float) -> tuple[float, float, float]:
        values = (pan, tilt, zoom)
        if not all(math.isfinite(v) and -1 <= v <= 1 for v in values):
            raise ValueError("velocity must be finite and normalized to [-1, 1]")
        return tuple(max(-limit, min(limit, value)) for value, limit in zip(
            values, (self.max_pan, self.max_tilt, self.max_zoom)
        ))


@dataclass(frozen=True)
class MotionCapabilities:
    pan_tilt: bool = True
    zoom: bool = False
    # Support alone does not imply the server can honor a shorter source TTL.
    server_motion_timeout: bool = False
    backend: str = "unknown"
    server_timeout_min_s: float | None = None
    server_timeout_max_s: float | None = None


@dataclass(frozen=True)
class MotionCommand:
    command_id: int
    pan: float
    tilt: float
    zoom: float
    accepted_at: float
    expires_at: float
    epoch: int
    frame_id: int | None = None
    settings_revision: int | None = None


@dataclass(frozen=True)
class MotionSnapshot:
    state: str = "CONNECTING"
    armed: bool = False
    connected: bool = False
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 0.0
    zoom_supported: bool = False
    reason: str = "Waiting for transport"
    command_id: int | None = None
    acknowledged_command_id: int | None = None
    frame_id: int | None = None
    last_backend_latency_ms: float | None = None
    last_command_interval_ms: float | None = None
    last_command_at: float | None = None
    stop_pending: bool = False
    stop_confirmed: bool = False
    inflight: bool = False
    backend: str = "unknown"
    # Protocol acknowledgement is never evidence of mechanical movement.
    motion_confirmed: bool = False
    version: int = 0


@dataclass(frozen=True)
class NetworkConfig:
    host: str
    username: str = field(repr=False)
    password: str = field(repr=False)
    backend: str = "auto"
    port: int = 80
    onvif_port: int | None = None
    channel: int = 1
    connect_timeout_s: float = 0.35
    read_timeout_s: float = 0.40
    server_motion_timeout_s: float = 0.35

    def __post_init__(self) -> None:
        if not self.host or any(c.isspace() or c in "/@?#" for c in self.host):
            raise ValueError("invalid camera host")
        if not self.username.strip() or not self.password:
            raise ValueError("camera username and password are required")
        if self.backend not in {"auto", "isapi", "onvif"}:
            raise ValueError("backend must be auto, isapi or onvif")
        if (not 1 <= self.port <= 65535 or self.channel < 1
                or (self.onvif_port is not None and not 1 <= self.onvif_port <= 65535)):
            raise ValueError("invalid camera port or channel")
        for name in ("connect_timeout_s", "read_timeout_s", "server_motion_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 5:
                raise ValueError(f"{name} must be in (0, 5] seconds")
