from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from ptz.detector import Detection, DetectorConfigurationError, InferenceError, ModelLoadError
from ptz.joblib_classifier import Classification
from ptz.svm_detector import SVMDetector


NAMES = {0: "drone", 1: "bird", 4: "background"}


def prediction(class_id=4, confidence=0.99):
    return Classification(class_id, str(class_id), NAMES[class_id], None, confidence)


class FakeCalibratedClassifier:
    has_probabilities = True
    n_features = 3
    class_names = NAMES

    def __init__(self, results=None):
        self.results = results
        self.batches = []

    def predict_features(self, matrix):
        self.batches.append(matrix.copy())
        return ([prediction() for _ in matrix] if self.results is None else list(self.results))


class SVMDetectorTests(unittest.TestCase):
    def test_no_yolo_load_and_single_batch_with_coordinates_restored_to_original_frame(self):
        # 22x14 -> 11x7; las ventanas de cuatro píxeles incluyen x=7 e y=3.
        results = [prediction() for _ in range(8)]
        results[-1] = prediction(0, 0.91)
        classifier = FakeCalibratedClassifier(results)
        extractor = Mock(return_value=np.ones(3))
        with patch("ptz.detector.YOLODetector", side_effect=AssertionError("YOLO no debe cargarse")):
            detector = SVMDetector(classifier, extractor, scan_width=11,
                                   window_sizes=(4,), stride_ratio=0.75)
            detections = detector.detect(np.zeros((14, 22, 3), dtype=np.uint8))
        self.assertEqual(detections, [Detection(0, "drone", 0.91, 14, 6, 22, 14)])
        self.assertEqual(len(classifier.batches), 1)
        self.assertEqual(classifier.batches[0].shape, (8, 3))
        self.assertEqual(extractor.call_count, 8)
        self.assertTrue(all(call.args[0].shape == (4, 4, 3) for call in extractor.call_args_list))
        self.assertIsNotNone(detector.last_inference_ms)
        self.assertEqual(detector.class_names, {0: "drone", 1: "bird"})
        detector.class_names.clear()
        self.assertEqual(detector.class_names, {0: "drone", 1: "bird"})

    def test_small_frames_are_not_upscaled_and_oversized_scales_are_deduplicated(self):
        classifier = FakeCalibratedClassifier()
        original = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        image = original.copy()
        crops = []

        def extract(crop):
            crops.append(crop.copy())
            crop[:] = 0
            return np.zeros(3)

        detector = SVMDetector(classifier, extract, scan_width=320, window_sizes=(8, 16))
        self.assertEqual(detector.detect(image), [])
        self.assertEqual(len(crops), 2)
        np.testing.assert_array_equal(crops[0], original[:, :3])
        np.testing.assert_array_equal(crops[1], original[:, 2:])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(classifier.batches[0].shape, (2, 3))

    def test_background_and_below_threshold_windows_are_discarded(self):
        classifier = FakeCalibratedClassifier([
            prediction(4, 0.99), prediction(0, 0.49), prediction(1, 0.5),
        ])
        detector = SVMDetector(classifier, lambda _: np.zeros(3), window_sizes=(4,), stride_ratio=1.0)
        frame = np.zeros((4, 12, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [Detection(1, "bird", 0.5, 8, 0, 12, 4)])
        detector.set_confidence(0.4)
        self.assertEqual(detector.confidence, 0.4)
        self.assertEqual(detector.detect(frame), [
            Detection(1, "bird", 0.5, 8, 0, 12, 4),
            Detection(0, "drone", 0.49, 4, 0, 8, 4),
        ])

    def test_nms_suppresses_only_same_class_and_keeps_highest_probabilities(self):
        classifier = FakeCalibratedClassifier([
            prediction(0, 0.9), prediction(0, 0.95), prediction(1, 0.85),
            prediction(4, 0.99), prediction(0, 0.7),
        ])
        arguments = dict(window_sizes=(4,), stride_ratio=0.25, nms_iou=0.35)
        frame = np.zeros((4, 8, 3), dtype=np.uint8)
        detector = SVMDetector(classifier, lambda _: np.zeros(3), max_detections=2, **arguments)
        expected = [
            Detection(0, "drone", 0.95, 1, 0, 5, 4),
            Detection(1, "bird", 0.85, 2, 0, 6, 4),
        ]
        self.assertEqual(detector.detect(frame), expected)
        detector = SVMDetector(classifier, lambda _: np.zeros(3), max_detections=10, **arguments)
        self.assertEqual(detector.detect(frame), expected + [Detection(0, "drone", 0.7, 4, 0, 8, 4)])

    def test_multiple_scales_are_predicted_in_one_batch(self):
        classifier = FakeCalibratedClassifier()
        extractor = Mock(return_value=np.zeros(3))
        detector = SVMDetector(classifier, extractor, window_sizes=(4, 8), stride_ratio=1.0)
        self.assertEqual(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])
        self.assertEqual(len(classifier.batches), 1)
        self.assertEqual(classifier.batches[0].shape, (5, 3))
        self.assertEqual([call.args[0].shape for call in extractor.call_args_list],
                         [(4, 4, 3)] * 4 + [(8, 8, 3)])

    def test_raw_model_is_rejected_with_calibration_instructions(self):
        classifier = FakeCalibratedClassifier()
        classifier.has_probabilities = False
        with self.assertRaisesRegex(ModelLoadError, "python -m ptz.calibrate_svm"):
            SVMDetector(classifier, lambda _: np.zeros(3))

    def test_model_requires_a_background_class(self):
        classifier = FakeCalibratedClassifier()
        classifier.class_names = {0: "drone", 1: "bird"}
        with self.assertRaisesRegex(ModelLoadError, "background"):
            SVMDetector(classifier, lambda _: np.zeros(3))

    def test_from_environment_uses_classifier_and_extractor_without_yolo(self):
        classifier = FakeCalibratedClassifier()
        extractor = Mock(return_value=np.zeros(3))
        labels = {str(key): value for key, value in NAMES.items()}
        with patch.dict("os.environ", {
            "DETECTCAM_CLASSIFIER_PATH": "model/calibrated.joblib",
            "DETECTCAM_CLASSIFIER_LABELS_PATH": "model/classes.json",
            "DETECTCAM_CLASSIFIER_FEATURES_PATH": "ptz/hog_hsv_features.py",
            "DETECTCAM_SVM_SCAN_WIDTH": "96",
            "DETECTCAM_SVM_WINDOW_SIZES": "8,16",
            "DETECTCAM_CONFIDENCE": "0.65",
            "DETECTCAM_DEVICE": "vulkan:0",
        }, clear=True), \
                patch("ptz.svm_detector.load_labels", return_value=labels) as read_labels, \
                patch("ptz.svm_detector.JoblibClassifier", return_value=classifier) as factory, \
                patch("ptz.svm_detector.load_feature_extractor", return_value=extractor) as read_extractor:
            detector = SVMDetector.from_environment()
        factory.assert_called_once_with("model/calibrated.joblib", labels=labels)
        read_labels.assert_called_once_with("model/classes.json")
        read_extractor.assert_called_once_with("ptz/hog_hsv_features.py")
        self.assertEqual(detector.confidence, 0.65)
        self.assertEqual(detector.detect(np.zeros((8, 8, 3), dtype=np.uint8)), [])
        self.assertEqual(extractor.call_count, 1)

    def test_invalid_settings_or_confidence_do_not_silently_change_the_configuration(self):
        classifier = FakeCalibratedClassifier()
        for settings in (
            {"confidence": float("nan")}, {"confidence": 1.1}, {"scan_width": 0},
            {"scan_width": True}, {"window_sizes": ()}, {"window_sizes": (0, 32)},
            {"window_sizes": "32,64"}, {"stride_ratio": 0.0}, {"stride_ratio": 1.1},
            {"nms_iou": -0.1}, {"max_detections": 0},
        ):
            with self.subTest(settings=settings), self.assertRaises(DetectorConfigurationError):
                SVMDetector(classifier, lambda _: np.zeros(3), **settings)
        detector = SVMDetector(classifier, lambda _: np.zeros(3))
        for confidence in (float("inf"), -0.1, "invalid"):
            with self.subTest(confidence=confidence), self.assertRaises(DetectorConfigurationError):
                detector.set_confidence(confidence)
            self.assertEqual(detector.confidence, 0.5)

    def test_invalid_frame_or_feature_dimensions_are_rejected(self):
        classifier = FakeCalibratedClassifier()
        detector = SVMDetector(classifier, lambda _: np.zeros(3))
        for frame in (None, np.zeros((0, 3, 3), dtype=np.uint8), np.zeros((8, 8)),
                      np.zeros((8, 8, 4), dtype=np.uint8), np.zeros((8, 8, 3), dtype=np.float32)):
            with self.subTest(frame_shape=getattr(frame, "shape", None)), \
                    self.assertRaises(DetectorConfigurationError):
                detector.detect(frame)
        detector = SVMDetector(classifier, lambda _: np.zeros(5))
        with self.assertRaisesRegex(InferenceError, "extractor SVM.*\\(3,\\)"):
            detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(classifier.batches, [])

    def test_missing_calibrated_probability_never_becomes_confidence_one(self):
        classifier = FakeCalibratedClassifier([prediction(0, None)])
        detector = SVMDetector(classifier, lambda _: np.zeros(3), window_sizes=(8,))
        with self.assertRaisesRegex(InferenceError, "probabilidad válida"):
            detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
