"""Arranque local con el entorno compartido; no requiere instalar el paquete."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from detectcam.application.cli import main
    raise SystemExit(main())
