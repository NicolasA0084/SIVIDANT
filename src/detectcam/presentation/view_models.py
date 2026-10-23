"""Datos de presentación; el seguimiento real reside en core.tracking.

Este DTO preserva la vista previa y las capturas con anotaciones. No mantiene
identidades, no predice y nunca puede enviar movimiento al dispositivo.
"""
from dataclasses import dataclass
from enum import Enum
from detectcam.adapters.detector import Detection

class TrackState(str, Enum):
    NO_TARGET = "NO_TARGET"
    TRACKED = "TRACKED"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"


@dataclass(frozen=True)
class TrackedTarget:
    """Estado temporal del objetivo, incluida su velocidad observada."""

    track_id: int
    detection: Detection
    age_frames: int
    visible_frames: int
    missed_frames: int
    last_seen_timestamp: float
    previous_center: tuple[float, float] | None
    velocity_x: float
    velocity_y: float
    # La aplicación adapta velocidades del core en píxeles por segundo.
    # Intervalo conservado sólo para compatibilidad del DTO de presentación.
    velocity_dt_s: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.detection.center_x, self.detection.center_y
