"""Configuración reproducible del perfil SAFE validado para Detectcam."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isclose
import os


class PTZTuningConfigurationError(ValueError):
    """La selección del perfil o una constante de comparación es inválida."""


class PTZTuningProfileName(str, Enum):
    """Perfiles habilitados en la versión pública actual."""

    SAFE = "SAFE"


@dataclass(frozen=True)
class PTZTuningProfile:
    """Ganancias y límites que determinan la respuesta física del seguimiento."""

    name: PTZTuningProfileName
    kp_pan: float
    kp_tilt: float
    max_pan_command: float
    max_tilt_command: float
    physical_max_pan: float
    physical_max_tilt: float


PROFILES = {
    PTZTuningProfileName.SAFE: PTZTuningProfile(
        PTZTuningProfileName.SAFE, 0.50, 0.50, 0.35, 0.35, 0.30, 0.30
    ),
}

# Estos interlocks permanecen fijos en el perfil SAFE publicado.
FIXED_FLOATS = {
    "DETECTCAM_PTZ_MIN_TRACK_CONFIDENCE": 0.50,
    "DETECTCAM_PTZ_LOST_COAST_MS": 300.0,
    "DETECTCAM_PTZ_KD_PAN": 0.04,
    "DETECTCAM_PTZ_KD_TILT": 0.04,
    "DETECTCAM_PTZ_KV_PAN": 0.10,
    "DETECTCAM_PTZ_KV_TILT": 0.10,
    "DETECTCAM_PTZ_LEAD_TIME_S": 0.15,
    "DETECTCAM_PTZ_FILTER_MIN_CUTOFF": 1.0,
    "DETECTCAM_PTZ_FILTER_BETA": 0.05,
    "DETECTCAM_PTZ_DEAD_ZONE_HYSTERESIS": 0.02,
    "DETECTCAM_PTZ_MAX_COMMAND_DELTA_PER_S": 1.50,
    "DETECTCAM_PTZ_OSCILLATION_GAIN": 0.35,
    "DETECTCAM_PTZ_DEAD_ZONE_X": 0.10,
    "DETECTCAM_PTZ_DEAD_ZONE_Y": 0.10,
    "DETECTCAM_PTZ_PHYSICAL_MAX_ZOOM": 0.0,
}
FIXED_INTS = {"DETECTCAM_TRACK_MAX_MISSED_FRAMES": 10}
FIXED_FALSES = {
    "DETECTCAM_PTZ_POSITION3D_ENABLED",
    "DETECTCAM_PTZ_AUTO_ZOOM_ENABLED",
}
MAX_SAFE_PHYSICAL = 0.30


@dataclass(frozen=True)
class PTZTuningConfiguration:
    """Resuelve SAFE y bloquea cambios accidentales en sus interlocks."""

    profile: PTZTuningProfile

    @classmethod
    def from_environment(cls) -> "PTZTuningConfiguration":
        raw = os.getenv("DETECTCAM_PTZ_TUNING_PROFILE", "SAFE").strip().upper()
        try:
            name = PTZTuningProfileName(raw)
        except ValueError as error:
            choices = ", ".join(item.value for item in PTZTuningProfileName)
            raise PTZTuningConfigurationError(
                f"DETECTCAM_PTZ_TUNING_PROFILE debe ser {choices}."
            ) from error
        configuration = cls(PROFILES[name])
        configuration.validate_environment_constants()
        configuration.validate_profile()
        return configuration

    def validate_profile(self) -> None:
        values = (
            self.profile.kp_pan, self.profile.kp_tilt,
            self.profile.max_pan_command, self.profile.max_tilt_command,
            self.profile.physical_max_pan, self.profile.physical_max_tilt,
        )
        if any(value < 0.0 or value > 1.0 for value in values):
            raise PTZTuningConfigurationError(
                f"El perfil {self.profile.name.value} contiene valores fuera de 0..1."
            )
        if max(self.profile.physical_max_pan, self.profile.physical_max_tilt) > MAX_SAFE_PHYSICAL:
            raise PTZTuningConfigurationError(
                "El perfil SAFE no puede superar physical max 0.30."
            )

    def validate_environment_constants(self) -> None:
        for name, expected in FIXED_FLOATS.items():
            raw = os.getenv(name)
            if raw is None:
                continue
            try:
                actual = float(raw)
            except ValueError as error:
                raise PTZTuningConfigurationError(f"{name} debe ser numérico.") from error
            if not isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
                raise PTZTuningConfigurationError(
                    f"{name} debe permanecer en {expected:g} para el perfil SAFE."
                )
        for name, expected in FIXED_INTS.items():
            raw = os.getenv(name)
            if raw is not None and raw.strip() != str(expected):
                raise PTZTuningConfigurationError(
                    f"{name} debe permanecer en {expected} para el perfil SAFE."
                )
        for name in FIXED_FALSES:
            raw = os.getenv(name)
            if raw is not None and raw.strip().lower() not in {"0", "false", "no", "off"}:
                raise PTZTuningConfigurationError(
                    f"{name} debe permanecer en false para el perfil SAFE."
                )

    def decision_overrides(self) -> dict[str, object]:
        return {
            "kp_pan": self.profile.kp_pan,
            "kp_tilt": self.profile.kp_tilt,
            "max_pan_command": self.profile.max_pan_command,
            "max_tilt_command": self.profile.max_tilt_command,
            "min_track_confidence": 0.50,
            "lost_coast_ms": 300,
            "kd_pan": 0.04,
            "kd_tilt": 0.04,
            "kv_pan": 0.10,
            "kv_tilt": 0.10,
            "lead_time_s": 0.15,
            "filter_min_cutoff": 1.0,
            "filter_beta": 0.05,
            "dead_zone_hysteresis": 0.02,
            "max_command_delta_per_s": 1.50,
            "oscillation_gain": 0.35,
            "dead_zone_x": 0.10,
            "dead_zone_y": 0.10,
        }

    def policy_overrides(self) -> dict[str, object]:
        return {
            "position3d_enabled": False,
            "auto_zoom_enabled": False,
            "min_track_confidence": 0.50,
            "lost_coast_ms": 300,
        }

    def tracker_overrides(self) -> dict[str, object]:
        return {"max_missed_frames": 10}

    def executor_overrides(self) -> dict[str, object]:
        return {
            "physical_max_pan": self.profile.physical_max_pan,
            "physical_max_tilt": self.profile.physical_max_tilt,
            "physical_max_zoom": 0.0,
        }

    def effective_configuration(self) -> dict[str, object]:
        values = asdict(self.profile)
        values["name"] = self.profile.name.value
        values.update({
            "min_track_confidence": 0.50,
            "track_max_missed_frames": 10,
            "lost_coast_ms": 300,
            "position3d_enabled": False,
            "auto_zoom_enabled": False,
            "physical_max_zoom": 0.0,
            "kd_pan": 0.04,
            "kd_tilt": 0.04,
            "kv_pan": 0.10,
            "kv_tilt": 0.10,
            "lead_time_s": 0.15,
            "filter_min_cutoff": 1.0,
            "filter_beta": 0.05,
            "dead_zone_hysteresis": 0.02,
            "max_command_delta_per_s": 1.50,
            "oscillation_gain": 0.35,
            "dead_zone_x": 0.10,
            "dead_zone_y": 0.10,
        })
        return values
