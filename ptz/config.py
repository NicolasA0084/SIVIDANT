"""Carga y validación temprana de la configuración de Detectcam.

Este módulo no abre la cámara, no carga el modelo y no crea backends PTZ. Su
objetivo es fallar antes del arranque con mensajes accionables y sin secretos.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Mapping, MutableMapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


class ConfigurationError(ValueError):
    """Indica que la configuración local no permite iniciar Detectcam."""


@dataclass(frozen=True)
class StartupConfiguration:
    """Resultado validado del preflight, sin exponer credenciales."""

    project_root: Path
    env_file: Path
    model_path: Path
    camera_host: str
    hardware_enabled: bool
    ptz_backend: str


_INTEGER_FIELDS = {
    "DETECTCAM_RTSP_CHANNEL": (1, None),
    "DETECTCAM_RTSP_DECODER_THREADS": (0, 64),
    "DETECTCAM_ISAPI_PORT": (1, 65535),
    "DETECTCAM_ONVIF_PORT": (1, 65535),
    "DETECTCAM_PTZ_CHANNEL": (1, None),
    "DETECTCAM_IMGSZ": (1, None),
    "DETECTCAM_TRACK_MAX_MISSED_FRAMES": (0, None),
    "DETECTCAM_PTZ_LOST_COAST_MS": (0, None),
    "DETECTCAM_PTZ_MAX_CONTINUOUS_MS": (1, None),
    "DETECTCAM_PTZ_EXECUTOR_HEARTBEAT_MS": (1, None),
    "DETECTCAM_PTZ_MAX_TARGET_AGE_MS": (1, None),
    "DETECTCAM_PTZ_MAX_FRAME_AGE_MS": (1, None),
    "DETECTCAM_PTZ_MIN_VISIBLE_FRAMES": (1, None),
    "DETECTCAM_PTZ_COMMAND_INTERVAL_MS": (1, None),
    "DETECTCAM_PTZ_RECENTER_STABLE_MS": (0, None),
}

_UNIT_FLOAT_FIELDS = {
    "DETECTCAM_CONFIDENCE",
    "DETECTCAM_TARGET_MIN_CONFIDENCE",
    "DETECTCAM_PTZ_KP_PAN",
    "DETECTCAM_PTZ_KP_TILT",
    "DETECTCAM_PTZ_MAX_PAN_COMMAND",
    "DETECTCAM_PTZ_MAX_TILT_COMMAND",
    "DETECTCAM_PTZ_PHYSICAL_MAX_PAN",
    "DETECTCAM_PTZ_PHYSICAL_MAX_TILT",
    "DETECTCAM_PTZ_PHYSICAL_MAX_ZOOM",
    "DETECTCAM_PTZ_MIN_TRACK_CONFIDENCE",
}

_NONNEGATIVE_FLOAT_FIELDS = {
    "DETECTCAM_TARGET_CONFIDENCE_WEIGHT",
    "DETECTCAM_TARGET_CENTER_WEIGHT",
    "DETECTCAM_TARGET_SIZE_WEIGHT",
}

_BOOLEAN_FIELDS = {
    "DETECTCAM_RTSP_LOW_LATENCY",
    "DETECTCAM_PTZ_HARDWARE_ENABLED",
    "DETECTCAM_PTZ_POSITION3D_ENABLED",
    "DETECTCAM_PTZ_AUTO_ZOOM_ENABLED",
    "DETECTCAM_PTZ_METRICS_ENABLED",
}

_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.I)


def load_dotenv(
    path: str | Path = DEFAULT_ENV_FILE,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Carga un archivo dotenv sencillo sin reemplazar variables existentes."""

    destination = os.environ if environ is None else environ
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise ConfigurationError(
            f"No se encontró la configuración local {env_path}. "
            "Créela con: cp .env.example .env"
        )

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ConfigurationError(f"No se pudo leer {env_path}.") from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ConfigurationError(
                f"Línea {line_number} inválida en {env_path.name}; use NOMBRE=valor."
            )
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ConfigurationError(
                f"Nombre de variable inválido en la línea {line_number} de "
                f"{env_path.name}."
            )
        destination.setdefault(name, _parse_dotenv_value(raw_value, line_number))
    return env_path


def validate_startup_configuration(
    environ: Mapping[str, str] | None = None,
    *,
    project_root: str | Path = PROJECT_ROOT,
    env_file: str | Path = DEFAULT_ENV_FILE,
) -> StartupConfiguration:
    """Valida campos comunes antes de cargar YOLO o abrir conexiones."""

    values = os.environ if environ is None else environ
    root = Path(project_root).expanduser().resolve()
    missing = [
        name for name in (
            "DETECTCAM_CAMERA_IP",
            "DETECTCAM_CAMERA_USER",
            "DETECTCAM_CAMERA_PASSWORD",
        )
        if not values.get(name, "").strip()
    ]
    if missing:
        raise ConfigurationError(
            "Faltan campos obligatorios en .env: " + ", ".join(missing) + "."
        )

    host = values["DETECTCAM_CAMERA_IP"].strip()
    if any(character.isspace() for character in host) or any(
        character in host for character in "/@?#"
    ):
        raise ConfigurationError(
            "DETECTCAM_CAMERA_IP debe ser una IP o un host sin protocolo ni ruta."
        )

    for name, (minimum, maximum) in _INTEGER_FIELDS.items():
        if name in values:
            number = _integer(name, values[name])
            if number < minimum or maximum is not None and number > maximum:
                limit = f"entre {minimum} y {maximum}" if maximum else f">= {minimum}"
                raise ConfigurationError(f"{name} debe estar {limit}.")

    for name in _UNIT_FLOAT_FIELDS:
        if name in values:
            number = _float(name, values[name])
            if not 0.0 <= number <= 1.0:
                raise ConfigurationError(f"{name} debe estar entre 0.0 y 1.0.")

    for name in _NONNEGATIVE_FLOAT_FIELDS:
        if name in values and _float(name, values[name]) < 0.0:
            raise ConfigurationError(f"{name} no puede ser negativo.")

    for name in _BOOLEAN_FIELDS:
        if name in values:
            _boolean(name, values[name])

    transport = values.get("DETECTCAM_RTSP_TRANSPORT", "tcp").strip().lower()
    if transport not in {"tcp", "udp"}:
        raise ConfigurationError("DETECTCAM_RTSP_TRANSPORT debe ser tcp o udp.")

    profile = values.get("DETECTCAM_PTZ_TUNING_PROFILE", "SAFE").strip().upper()
    if profile != "SAFE":
        raise ConfigurationError(
            "DETECTCAM_PTZ_TUNING_PROFILE debe ser SAFE en esta entrega."
        )
    backend = values.get("DETECTCAM_PTZ_BACKEND", "auto").strip().lower()
    if backend not in {"auto", "isapi", "onvif"}:
        raise ConfigurationError(
            "DETECTCAM_PTZ_BACKEND debe ser auto, isapi u onvif."
        )

    model_setting = values.get("DETECTCAM_MODEL_PATH", "model/best.pt").strip()
    if not model_setting:
        raise ConfigurationError("DETECTCAM_MODEL_PATH no puede estar vacío.")
    model_path = Path(model_setting).expanduser()
    if not model_path.is_absolute():
        model_path = root / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise ConfigurationError(
            f"No se encontró el modelo YOLO en {model_path}. "
            "Descargue o copie los pesos y ajuste DETECTCAM_MODEL_PATH."
        )

    hardware_enabled = _boolean(
        "DETECTCAM_PTZ_HARDWARE_ENABLED",
        values.get("DETECTCAM_PTZ_HARDWARE_ENABLED", "false"),
    )
    return StartupConfiguration(
        project_root=root,
        env_file=Path(env_file).expanduser().resolve(),
        model_path=model_path,
        camera_host=host,
        hardware_enabled=hardware_enabled,
        ptz_backend=backend,
    )


def bootstrap_environment(
    path: str | Path = DEFAULT_ENV_FILE,
    *,
    environ: MutableMapping[str, str] | None = None,
    project_root: str | Path = PROJECT_ROOT,
) -> StartupConfiguration:
    """Carga `.env`, normaliza rutas y ejecuta toda la validación temprana."""

    destination = os.environ if environ is None else environ
    env_path = load_dotenv(path, environ=destination)
    configuration = validate_startup_configuration(
        destination, project_root=project_root, env_file=env_path
    )
    # Los módulos existentes siguen consumiendo el entorno. Publicar la ruta ya
    # resuelta elimina la dependencia respecto del cwd sin duplicar ajustes.
    destination["DETECTCAM_MODEL_PATH"] = str(configuration.model_path)
    return configuration


class SecretRedactionFilter(logging.Filter):
    """Elimina credenciales conocidas de mensajes producidos por dependencias."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        secrets = (
            os.getenv("DETECTCAM_CAMERA_PASSWORD", ""),
            os.getenv("DETECTCAM_CAMERA_USER", ""),
        )
        for secret in secrets:
            if secret:
                message = message.replace(secret, "***")
        message = _URL_CREDENTIALS.sub(r"\g<scheme>***@", message)
        record.msg = message
        record.args = ()
        return True


def configure_logging() -> None:
    """Configura un formato común y redacción para todos los handlers raíz."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if not any(isinstance(item, SecretRedactionFilter) for item in handler.filters):
            handler.addFilter(SecretRedactionFilter())


def _parse_dotenv_value(raw_value: str, line_number: int) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise ConfigurationError(
                f"Comillas sin cerrar en la línea {line_number} de .env."
            )
        return value[1:-1]
    # Un comentario requiere espacio antes de # para no truncar contraseñas.
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def _integer(name: str, value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} debe ser un entero.") from error


def _float(name: str, value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{name} debe ser numérico.") from error


def _boolean(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} debe ser true o false.")
