"""Configuración única: entorno del proceso > .env local > .env compartido.

Las credenciales se reutilizan desde la instalación existente sin copiarlas.
Los ajustes del controlador nuevo viven en TOML y se validan al construir sus
dataclasses. Los viejos parámetros SAFE no se aplican al nuevo controlador.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping

from .environment import ConfigurationError, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHARED_ROOT = PROJECT_ROOT.parent / "ptz_test"


@dataclass(frozen=True)
class Settings:
    model_path: Path
    camera_host: str = field(repr=False)
    username: str = field(repr=False)
    password: str = field(repr=False)
    rtsp_channel: str = "101"
    ptz_channel: int = 1
    isapi_port: int = 80
    onvif_port: int = 80
    backend: str = "auto"
    hardware_enabled: bool = False
    device: str = "cpu"
    imgsz: int = 640
    inference_confidence: float = .15
    min_confidence: float = .50
    allowed_classes: tuple[str, ...] = ()
    low_latency: bool = True
    transport: str = "tcp"
    decoder_threads: int = 1
    metrics_enabled: bool = False
    media_directory: Path = Path.home() / "Videos" / "Detectcam"
    sections: Mapping[str, dict[str, Any]] = field(default_factory=dict)

    def public_summary(self) -> dict[str, Any]:
        return {"model": str(self.model_path), "shared_venv": str(PROJECT_ROOT / ".venv"),
                "device": self.device, "imgsz": self.imgsz, "backend": self.backend,
                "hardware_enabled": self.hardware_enabled, "starts_armed": False,
                "metrics_enabled": self.metrics_enabled, "initial_classes": self.allowed_classes,
                "settings_file": str(PROJECT_ROOT / "settings.toml")}


def load_settings(*, simulate: bool = False, root: Path = PROJECT_ROOT,
                  environ: dict[str, str] | None = None) -> Settings:
    """Sólo lee archivos y valida; nunca conecta cámara ni carga un modelo."""
    values = os.environ if environ is None else environ
    for path in (root / ".env", root.parent / "ptz_test" / ".env"):
        if path.is_file():
            load_dotenv(path, environ=values)
    settings_file = root / "settings.toml"
    try:
        sections = tomllib.loads(settings_file.read_text(encoding="utf-8")) if settings_file.exists() else {}
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigurationError("No se pudo leer settings.toml: " + str(error)) from error
    if set(sections) - {"tracking", "control", "runtime", "application"}:
        raise ConfigurationError("Sección desconocida en settings.toml")
    if any(not isinstance(section, dict) for section in sections.values()):
        raise ConfigurationError("Cada sección de settings.toml debe ser una tabla")
    app = sections.get("application", {})
    if set(app) - {"metrics_enabled", "warmup", "initial_classes"}:
        raise ConfigurationError("Opción desconocida en [application]")
    if any(not isinstance(value, bool) for key, value in app.items() if key != "initial_classes"):
        raise ConfigurationError("Los ajustes de [application] deben ser booleanos TOML")
    initial_classes = app.get("initial_classes")
    if initial_classes is not None and (not isinstance(initial_classes, list)
            or any(not isinstance(item, str) or not item.strip() for item in initial_classes)):
        raise ConfigurationError("application.initial_classes debe ser una lista de nombres de clases")
    def get(name: str, default: Any = "") -> str:
        return str(values.get(name, default)).strip()
    def integer(name: str, default: int, low: int, high: int) -> int:
        try:
            result = int(get(name, default))
        except ValueError as error:
            raise ConfigurationError(f"{name} debe ser entero") from error
        if not low <= result <= high:
            raise ConfigurationError(f"{name} fuera de rango {low}..{high}")
        return result
    def confidence(name: str, default: float) -> float:
        try:
            result = float(get(name, default))
        except ValueError as error:
            raise ConfigurationError(f"{name} debe ser numérico") from error
        if not 0 <= result <= 1:
            raise ConfigurationError(f"{name} debe estar entre 0 y 1")
        return result
    def boolean(name: str, default: bool) -> bool:
        raw = get(name, str(default)).lower()
        if raw not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            raise ConfigurationError(f"{name} debe ser true o false")
        return raw in {"true", "1", "yes", "on"}
    host, user = get("DETECTCAM_CAMERA_IP"), get("DETECTCAM_CAMERA_USER")
    password = values.get("DETECTCAM_CAMERA_PASSWORD", "")
    if not host or not user or not password:
        raise ConfigurationError("Faltan CAMERA_IP, CAMERA_USER o CAMERA_PASSWORD en el entorno compartido/local")
    if re.search(r"[\s/@?#]", host):
        raise ConfigurationError("CAMERA_IP debe contener sólo IP/host")
    configured_model = Path(get("DETECTCAM_MODEL_PATH", "model/best.pt")).expanduser()
    model = configured_model if configured_model.is_absolute() else root / configured_model
    if not model.is_file() and not configured_model.is_absolute():
        model = root.parent / "ptz_test" / configured_model
    if not model.is_file():
        raise ConfigurationError("No se encontró el modelo existente; revisa DETECTCAM_MODEL_PATH")
    backend = get("DETECTCAM_PTZ_BACKEND", "auto").lower()
    if backend not in {"auto", "isapi", "onvif"}:
        raise ConfigurationError("Backend debe ser auto, isapi u onvif")
    transport = get("DETECTCAM_RTSP_TRANSPORT", "tcp").lower()
    if transport not in {"tcp", "udp"}:
        raise ConfigurationError("Transporte RTSP debe ser tcp o udp")
    media = get("DETECTCAM_MEDIA_DIR")
    return Settings(
        model_path=model.resolve(), camera_host=host, username=user, password=password,
        rtsp_channel=str(integer("DETECTCAM_RTSP_CHANNEL", 101, 1, 65535)),
        ptz_channel=integer("DETECTCAM_PTZ_CHANNEL", 1, 1, 65535),
        isapi_port=integer("DETECTCAM_ISAPI_PORT", 80, 1, 65535),
        onvif_port=integer("DETECTCAM_ONVIF_PORT", 80, 1, 65535), backend=backend,
        hardware_enabled=False if simulate else boolean("DETECTCAM_PTZ_HARDWARE_ENABLED", False),
        device=get("DETECTCAM_DEVICE", "cpu"), imgsz=integer("DETECTCAM_IMGSZ", 640, 32, 4096),
        inference_confidence=confidence("DETECTCAM_CONFIDENCE", .15),
        min_confidence=confidence("DETECTCAM_TARGET_MIN_CONFIDENCE", .50),
        allowed_classes=tuple(x.strip().casefold() for x in (
            initial_classes if initial_classes is not None else get("DETECTCAM_TARGET_CLASSES").split(",")) if x.strip()),
        low_latency=boolean("DETECTCAM_RTSP_LOW_LATENCY", True), transport=transport,
        decoder_threads=integer("DETECTCAM_RTSP_DECODER_THREADS", 1, 0, 64),
        metrics_enabled=boolean("DETECTCAM_PTZ_METRICS_ENABLED", app.get("metrics_enabled", False)),
        media_directory=Path(media).expanduser() if media else Path.home() / "Videos" / "Detectcam",
        sections=sections,
    )
