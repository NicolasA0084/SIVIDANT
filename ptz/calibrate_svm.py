"""Calibra el SVM ya entrenado con VALID natural, sin volver a ajustar sus pesos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.frozen import FrozenEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_validation(features_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Selecciona exclusivamente las filas identificadas como valid_natural."""
    configuration = json.loads((features_root / "feature_config.json").read_text(encoding="utf-8"))
    if configuration.get("test_accessed") is not False:
        raise ValueError("La caché debe certificar test_accessed=false.")
    if configuration.get("feature_version") != "HOG_HSV_V1":
        raise ValueError("Se requiere la caché HOG_HSV_V1 del entrenamiento.")
    matrix = np.load(features_root / "features.float32.npy", mmap_mode="r", allow_pickle=False)
    if matrix.ndim != 2 or matrix.shape[1] != configuration["feature_dim"]:
        raise ValueError("La matriz no coincide con feature_config.json.")
    rows, labels = [], []
    with (features_root / "feature_index.csv").open(newline="", encoding="utf-8") as source:
        for record in csv.DictReader(source):
            if record["set_membership"] == "valid_natural":
                rows.append(int(record["feature_row"]))
                labels.append(int(record["assigned_class_id"]))
    if not rows or len(set(rows)) != len(rows) or min(rows) < 0 or max(rows) >= len(matrix):
        raise ValueError("El índice VALID está vacío, duplicado o fuera de la matriz.")
    selected = np.asarray(matrix[rows])
    if not np.isfinite(selected).all():
        raise ValueError("VALID contiene características no finitas.")
    return selected, np.asarray(labels), configuration


def calibrate_model(model, features: np.ndarray, labels: np.ndarray):
    if not callable(getattr(model, "decision_function", None)):
        raise ValueError("El modelo original debe proporcionar decision_function.")
    if features.ndim != 2 or features.shape[1] != model.n_features_in_:
        raise ValueError("La dimensión de VALID no coincide con el modelo.")
    if set(np.unique(labels)) != set(model.classes_):
        raise ValueError("VALID debe incluir todas las clases del modelo.")
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    calibrated.fit(features, labels)
    return calibrated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "model" / "model.joblib")
    parser.add_argument("--features-root", type=Path, default=PROJECT_ROOT / "hog_hsv_features_v1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "model" / "model_calibrated.joblib")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    try:
        report_path = arguments.output.with_suffix(".calibration.json")
        if not arguments.overwrite and (arguments.output.exists() or report_path.exists()):
            raise ValueError("El destino ya existe; use --overwrite para reemplazarlo.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            original = joblib.load(arguments.model)
        features, labels, configuration = load_validation(arguments.features_root)
        calibrated = calibrate_model(original, features, labels)
        report = {
            "method": "sigmoid",
            "base_estimator_frozen": True,
            "source_model_sha256": hashlib.sha256(arguments.model.read_bytes()).hexdigest(),
            "feature_version": configuration["feature_version"],
            "feature_config_sha256": hashlib.sha256(
                (arguments.features_root / "feature_config.json").read_bytes()
            ).hexdigest(),
            "samples": len(labels),
            "split": "valid_natural",
            "class_counts": {str(label): int((labels == label).sum()) for label in original.classes_},
            "validation_used_for_model_selection": True,
            "test_accessed": False,
            "independent_accuracy_evaluated": False,
            "sklearn_version": sklearn_version,
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated, arguments.output, compress=3)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Modelo calibrado guardado: {arguments.output}")
        print(f"Muestras VALID natural: {len(labels)}; pesos originales conservados; TEST sin acceder.")
        return 0
    except (OSError, ValueError, KeyError, InconsistentVersionWarning) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
