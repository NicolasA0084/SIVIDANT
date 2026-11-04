from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "Dependencias SVM opcionales ausentes")
class SVMCalibrationTests(unittest.TestCase):
    def test_cache_selection_excludes_training_and_test_rows(self):
        from ptz.calibrate_svm import load_validation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "feature_config.json").write_text(json.dumps({
                "test_accessed": False, "feature_version": "HOG_HSV_V1", "feature_dim": 2,
            }))
            matrix = np.array([[10, 11], [20, 21], [30, 31], [40, 41]], dtype=np.float32)
            np.save(root / "features.float32.npy", matrix)
            with (root / "feature_index.csv").open("w", newline="") as destination:
                writer = csv.DictWriter(destination, fieldnames=["feature_row", "assigned_class_id", "set_membership"])
                writer.writeheader()
                for row, split in enumerate(("train_bg2x", "valid_natural", "test", "valid_natural")):
                    writer.writerow({"feature_row": row, "assigned_class_id": row % 2, "set_membership": split})
            features, labels, _ = load_validation(root)
            np.testing.assert_array_equal(features, matrix[[1, 3]])
            np.testing.assert_array_equal(labels, [1, 1])

    def test_calibration_preserves_original_scaler_and_svm_coefficients(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from ptz.calibrate_svm import calibrate_model

        rng = np.random.default_rng(51)
        matrix = rng.normal(size=(100, 3))
        labels = (matrix[:, 0] > 0).astype(int)
        model = make_pipeline(StandardScaler(), LinearSVC(random_state=0)).fit(matrix[:60], labels[:60])
        mean_before = model[0].mean_.copy()
        coef_before = model[1].coef_.copy()
        calibrated = calibrate_model(model, matrix[60:], labels[60:])
        np.testing.assert_array_equal(model[0].mean_, mean_before)
        np.testing.assert_array_equal(model[1].coef_, coef_before)
        probabilities = calibrated.predict_proba(matrix[60:])
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue((probabilities >= 0).all() and (probabilities <= 1).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(40))


if __name__ == "__main__":
    unittest.main()
