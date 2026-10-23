import unittest
import cv2
import numpy as np
from detectcam.adapters.camera_motion import CameraMotionEstimator


class BackgroundMotionTests(unittest.TestCase):
    def test_known_shift_is_measured_without_model_or_camera(self):
        rng = np.random.default_rng(712)
        gray = rng.integers(0, 256, (240, 320), dtype=np.uint8)
        first = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        second = cv2.warpAffine(first, np.float32([[1, 0, 7], [0, 1, -4]]), (320, 240))
        estimator = CameraMotionEstimator()
        self.assertFalse(estimator.update(first, captured_at=10).reliable)
        result = estimator.update(second, captured_at=10.05)
        self.assertTrue(result.reliable)
        self.assertAlmostEqual(result.dx, 7, delta=.7)
        self.assertAlmostEqual(result.dy, -4, delta=.7)

    def test_textureless_sky_and_expired_frames_do_not_invent_motion(self):
        image = np.full((240, 320, 3), 200, np.uint8)
        estimator = CameraMotionEstimator()
        estimator.update(image, captured_at=10)
        self.assertFalse(estimator.update(image, captured_at=10.05).reliable)
        self.assertFalse(estimator.update(image, captured_at=11).reliable)
