"""Lectura dotenv y logging con redacción; no abre conexiones ni carga pesos."""
from __future__ import annotations
import logging
import os
from pathlib import Path
import re
from typing import MutableMapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/\s@]+@")

class ConfigurationError(ValueError):
    """Ajuste ausente o fuera de rango, sin mostrar secretos."""

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
