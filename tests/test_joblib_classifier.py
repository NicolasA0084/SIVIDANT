from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ptz.detector import Detection, InferenceError, ModelLoadError
from ptz.joblib_classifier import (
    ClassifyingDetector,
    JoblibClassifier,
    load_feature_extractor,
    with_optional_classifier,
)


class JoblibClassifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model_path = self.root / "model.joblib"
        self.model_path.touch()
        self.model = Mock(n_features_in_=3, classes_=np.array([0, 1, 2]))
        self.model.predict_proba = None
        self.model.steps = []
        self.model.predict.return_value = np.array([2])
        # Un margen negativo sigue siendo la puntuación mayor; no es una probabilidad.
        self.model.decision_function.return_value = np.array([[-2.0, -3.0, -0.2]])
        self.loader = Mock(return_value=self.model)
        self.classifier = JoblibClassifier(
            self.model_path, labels={"0": "bird", "1": "plane", "2": "drone"},
            model_loader=self.loader,
        )

    def test_predict_preserves_class_mapping_and_unbounded_margin(self):
        for _ in range(2):
            result = self.classifier.predict_features(np.array([1.0, 2.0, 3.0]))[0]
            self.assertEqual((result.class_id, result.label, result.class_name), (2, "2", "drone"))
            self.assertEqual(result.margin, -0.2)
            self.assertIsNone(result.confidence)
        self.loader.assert_called_once_with(str(self.model_path))
        self.assertEqual(self.model.predict.call_args.args[0].shape, (1, 3))
        self.assertFalse(self.classifier.has_probabilities)
        self.assertFalse(self.classifier.describe()["scores_are_probabilities"])

    def test_calibrated_probability_is_selected_by_the_predicted_class(self):
        self.model.predict_proba = Mock(return_value=np.array([[0.1, 0.2, 0.7]]))
        result = self.classifier.predict_features(np.zeros(3))[0]
        self.assertTrue(self.classifier.has_probabilities)
        self.assertTrue(self.classifier.describe()["scores_are_probabilities"])
        self.assertEqual(result.confidence, 0.7)
        self.assertEqual(result.margin, -0.2)
        # El margen y la probabilidad son datos independientes: se conserva la
        # probabilidad de predict aunque el modelo elija por otro criterio.
        self.model.predict_proba.return_value = np.array([[0.6, 0.3, 0.1]])
        self.assertEqual(self.classifier.predict_features(np.zeros(3))[0].confidence, 0.1)

    def test_invalid_probability_shape_range_finiteness_or_sum_is_rejected(self):
        for probabilities in (
            [[0.3, 0.7]], [[0.1, 0.2, 0.3]], [[0.1, np.nan, 0.9]],
            [[-0.1, 0.2, 0.9]], [[0.0, -0.2, 1.2]], [[0.1, 0.2, np.inf]],
        ):
            self.model.predict_proba = Mock(return_value=np.asarray(probabilities))
            with self.subTest(probabilities=probabilities), self.assertRaises(InferenceError):
                self.classifier.predict_features(np.zeros(3))

    def test_hybrid_keeps_yolo_confidence_with_a_calibrated_classifier(self):
        self.model.predict_proba = Mock(return_value=np.array([[0.01, 0.01, 0.98]]))
        detector = Mock()
        detector.detect.return_value = [Detection(10, "old", 0.61, 0, 0, 4, 4)]
        wrapped = ClassifyingDetector(detector, self.classifier, lambda _: np.zeros(3))
        result = wrapped.detect(np.zeros((4, 4, 3), dtype=np.uint8))[0]
        self.assertEqual(result.class_name, "drone")
        self.assertEqual(result.confidence, 0.61)

    def test_invalid_features_never_reach_the_model(self):
        for features in (
            np.zeros(4), np.zeros((1, 1, 3)), np.zeros((0, 3)),
            np.array([np.nan, 0, 0]), np.array([0, np.inf, 0]), ["x", "y", "z"],
        ):
            with self.subTest(features=features), self.assertRaises(InferenceError):
                self.classifier.predict_features(features)
        self.model.predict.assert_not_called()

    def test_binary_margin_follows_predicted_class(self):
        self.model.classes_ = np.array(["bird", "drone"])
        self.model.predict.return_value = np.array(["bird", "drone"])
        self.model.decision_function.return_value = np.array([-2.5, 1.7])
        classifier = JoblibClassifier(self.model_path, model_loader=self.loader)
        results = classifier.predict_features(np.zeros((2, 3)))
        self.assertEqual([result.margin for result in results], [2.5, 1.7])

    def test_incomplete_or_invalid_labels_are_rejected(self):
        for labels in ({"0": "drone"}, {"0": "bird", "1": "plane", "2": ""}, []):
            with self.subTest(labels=labels), self.assertRaisesRegex(ModelLoadError, "mapa JSON"):
                JoblibClassifier(self.model_path, labels=labels, model_loader=self.loader)

    def test_regression_or_untrained_artifact_is_rejected(self):
        with self.assertRaisesRegex(ModelLoadError, "clasificador entrenado"):
            JoblibClassifier(self.model_path, model_loader=lambda _: object())

    def test_non_finite_scores_are_rejected(self):
        self.model.decision_function.return_value = np.array([[0.1, np.nan, 0.2]])
        with self.assertRaises(InferenceError):
            self.classifier.predict_features(np.zeros(3))

    def test_crop_is_clipped_and_box_confidence_is_preserved(self):
        detector = Mock(confidence=0.25)
        detector.detect.return_value = [
            Detection(10, "old", 0.81, -5, 2, 20, 40),
            Detection(10, "old", 0.9, 99, 99, 101, 101),
            Detection(10, "old", 0.9, 5, 5, 2, 2),
        ]
        image = np.arange(10 * 15 * 3, dtype=np.uint8).reshape(10, 15, 3)
        original = image.copy()

        def extract(crop):
            np.testing.assert_array_equal(crop, original[2:10, 0:15])
            crop[:] = 0
            return np.array([1.0, 2.0, 3.0])

        extractor = Mock(side_effect=extract)
        wrapped = ClassifyingDetector(detector, self.classifier, extractor)
        detections = wrapped.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0], Detection(2, "drone", 0.81, 0, 2, 15, 10, -0.2))
        extractor.assert_called_once()
        np.testing.assert_array_equal(image, original)
        self.assertIsNotNone(wrapped.last_inference_ms)
        wrapped.set_confidence(0.1)
        detector.set_confidence.assert_called_once_with(0.1)
        self.assertEqual(wrapped.class_names, {0: "bird", 1: "plane", 2: "drone"})

    def test_no_boxes_skips_feature_extraction_and_classification(self):
        detector = Mock()
        detector.detect.return_value = []
        extractor = Mock()
        wrapped = ClassifyingDetector(detector, self.classifier, extractor)
        self.assertEqual(wrapped.detect(np.zeros((10, 10, 3), dtype=np.uint8)), [])
        extractor.assert_not_called()
        self.model.predict.assert_not_called()

    def test_background_prediction_is_excluded_from_detections_and_tracking_classes(self):
        self.model.classes_ = np.array([0, 1, 2, 3, 4])
        self.model.predict.return_value = np.array([4, 0])
        self.model.decision_function.return_value = np.array([
            [-2.0, -3.0, -1.0, -2.0, 2.0],
            [3.0, -3.0, -1.0, -2.0, -2.0],
        ])
        labels = {
            "0": "drone", "1": "airplane", "2": "helicopter",
            "3": "bird", "4": "background",
        }
        classifier = JoblibClassifier(self.model_path, labels=labels, model_loader=self.loader)
        detector = Mock()
        detector.detect.return_value = [
            Detection(10, "old", 0.9, 0, 0, 2, 2),
            Detection(10, "old", 0.8, 3, 3, 5, 5),
        ]
        wrapped = ClassifyingDetector(detector, classifier, lambda _: np.ones(3))
        detections = wrapped.detect(np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(detections, [Detection(0, "drone", 0.8, 3, 3, 5, 5, 3.0)])
        self.assertEqual(wrapped.class_names, {
            0: "drone", 1: "airplane", 2: "helicopter", 3: "bird",
        })
        self.assertEqual(classifier.class_names[4], "background")

    def test_wrong_image_extractor_shape_is_actionable(self):
        detector = Mock()
        detector.detect.return_value = [Detection(0, "old", 0.8, 0, 0, 4, 4)]
        wrapped = ClassifyingDetector(detector, self.classifier, lambda _: np.zeros(5))
        with self.assertRaisesRegex(InferenceError, "extract_features.*\\(3,\\)"):
            wrapped.detect(np.zeros((10, 10, 3), dtype=np.uint8))
        self.model.predict.assert_not_called()

    def test_optional_classifier_disabled_does_not_load_any_model(self):
        detector = Mock()
        with patch.dict("os.environ", {}, clear=True), \
                patch("ptz.joblib_classifier.JoblibClassifier") as factory:
            self.assertIs(with_optional_classifier(detector), detector)
        factory.assert_not_called()

    def test_feature_module_requires_the_original_function(self):
        source = self.root / "features.py"
        source.write_text("def extract_features(image_bgr):\n    return image_bgr.mean(axis=(0, 1))\n")
        extract = load_feature_extractor(source)
        np.testing.assert_array_equal(extract(np.ones((2, 2, 3))), np.ones(3))
        source.write_text("def another_function(image):\n    return image\n")
        with self.assertRaisesRegex(ModelLoadError, "extract_features"):
            load_feature_extractor(source)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Dependencias Joblib opcionales ausentes")
    def test_real_pipeline_roundtrip_and_environment_integration(self):
        import joblib
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        matrix = np.array([[1, 2, 3], [2, 3, 4], [-1, -2, -3], [-2, -3, -4]])
        model = make_pipeline(StandardScaler(), LinearSVC(random_state=0))
        model.fit(matrix, [7, 7, 9, 9])
        joblib.dump(model, self.model_path)
        labels_path = self.root / "classes.json"
        labels_path.write_text(json.dumps({"7": "bird", "9": "drone"}))
        features_path = self.root / "features.py"
        features_path.write_text("def extract_features(image_bgr):\n    return image_bgr.mean(axis=(0, 1))\n")
        detector = Mock()
        detector.detect.return_value = [Detection(0, "old", 0.7, 0, 0, 2, 2)]
        with patch.dict("os.environ", {
            "DETECTCAM_CLASSIFIER_PATH": str(self.model_path),
            "DETECTCAM_CLASSIFIER_FEATURES_PATH": str(features_path),
            "DETECTCAM_CLASSIFIER_LABELS_PATH": str(labels_path),
        }, clear=True):
            wrapped = with_optional_classifier(detector)
        result = wrapped.detect(np.full((2, 2, 3), -1.0))[0]
        self.assertEqual(model.predict([[-1, -1, -1]]).tolist(), [9])
        self.assertEqual(result.class_name, "drone")
        self.assertAlmostEqual(result.classification_margin, model.decision_function([[-1, -1, -1]])[0])


if __name__ == "__main__":
    unittest.main()
