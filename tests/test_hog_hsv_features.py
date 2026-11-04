from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE = PROJECT_ROOT / "extract_hog_hsv_features_v1.py"


@unittest.skipUnless(importlib.util.find_spec("skimage"), "Dependencias HOG/HSV opcionales ausentes")
class HOGHSVFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import ptz.hog_hsv_features as features
        from skimage.feature import hog

        cls.features = features
        cls.cv2 = cv2
        cls.reference = None
        if REFERENCE_SOURCE.is_file():
            # Aísla sólo las funciones matemáticas originales: no importa pandas,
            # no ejecuta main y no toca los datos ni los archivos de la caché.
            tree = ast.parse(REFERENCE_SOURCE.read_text(encoding="utf-8"))
            constants = {
                "FEATURE_DIM", "CANVAS_SIZE", "HOG_ORIENTATIONS",
                "HOG_PIXELS_PER_CELL", "HOG_CELLS_PER_BLOCK", "HOG_BLOCK_NORM",
                "HSV_BINS",
            }
            namespace = {"np": np, "cv2": cv2, "hog": hog}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in constants:
                            namespace[target.id] = ast.literal_eval(node.value)
            function_names = {"resize_preserve_aspect", "hsv_histogram", "extract_feature"}
            definitions = [node for node in tree.body
                           if isinstance(node, ast.FunctionDef) and node.name in function_names]
            if len(definitions) != len(function_names):
                raise AssertionError("El extractor original no contiene las tres funciones esperadas")
            isolated = ast.Module(body=definitions, type_ignores=[])
            exec(compile(isolated, str(REFERENCE_SOURCE), "exec"), namespace)
            cls.reference = namespace

    def require_reference(self):
        if self.reference is None:
            self.skipTest("Extractor de entrenamiento de referencia no disponible")
        return self.reference

    def test_exact_parity_with_training_extractor_for_varied_crop_shapes(self):
        reference = self.require_reference()
        rng = np.random.default_rng(281)
        for height, width in (
            (64, 64), (32, 91), (91, 32), (9, 19), (129, 53),
            (65, 64), (1, 1), (1, 17), (23, 1), (2, 201),
        ):
            image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
            with self.subTest(shape=image.shape):
                expected = reference["extract_feature"](image)
                actual = self.features.extract_features(image)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.dtype("float32"))
                self.assertEqual(actual.shape, (1812,))
                self.assertTrue(np.isfinite(actual).all())
        image = rng.integers(0, 256, size=(79, 103, 3), dtype=np.uint8)[::2, ::-1]
        np.testing.assert_array_equal(
            self.features.extract_features(image), reference["extract_feature"](image),
        )

    def test_resize_matches_original_interpolation_and_one_pixel_padding(self):
        reference = self.require_reference()
        rng = np.random.default_rng(99)
        for shape in ((15, 31, 3), (128, 95, 3), (1, 77, 3), (81, 1, 3)):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            with self.subTest(shape=shape):
                expected_content, expected_canvas = reference["resize_preserve_aspect"](image)
                content, canvas = self.features.resize_preserve_aspect(image)
                np.testing.assert_array_equal(content, expected_content)
                np.testing.assert_array_equal(canvas, expected_canvas)
                self.assertEqual(canvas.shape, (64, 64, 3))

    def test_hsv_histograms_match_training_function(self):
        reference = self.require_reference()
        image = np.random.default_rng(13).integers(0, 256, size=(17, 51, 3), dtype=np.uint8)
        actual = self.features.hsv_histogram(image)
        np.testing.assert_array_equal(actual, reference["hsv_histogram"](image))
        self.assertEqual(actual.dtype, np.dtype("float32"))
        np.testing.assert_allclose(actual.reshape(3, 16).sum(axis=1), np.ones(3), atol=1e-6)

    def test_hsv_uses_resized_content_before_padding(self):
        image = np.empty((11, 31, 3), dtype=np.uint8)
        image[:3] = (0, 0, 255)
        image[3:] = (0, 255, 0)
        content, canvas = self.features.resize_preserve_aspect(image)
        actual = self.features.extract_features(image)
        expected_hsv = self.features.hsv_histogram(content)
        padded_hsv = self.features.hsv_histogram(canvas)
        self.assertFalse(np.array_equal(expected_hsv, padded_hsv))
        np.testing.assert_array_equal(actual[1764:], expected_hsv)

    def test_constant_red_has_zero_hog_and_h_s_v_histogram_order(self):
        image = np.empty((13, 29, 3), dtype=np.uint8)
        image[:] = (0, 0, 255)
        expected = np.zeros(1812, dtype=np.float32)
        expected[1764] = 1.0  # Hue rojo: primer bin.
        expected[1764 + 16 + 15] = 1.0  # Saturación máxima.
        expected[1764 + 32 + 15] = 1.0  # Valor máximo.
        np.testing.assert_array_equal(self.features.extract_features(image), expected)

    def test_invalid_images_raise_value_error(self):
        images = (
            None, [[[0, 0, 0]]],
            np.empty((0, 64, 3), dtype=np.uint8),
            np.empty((64, 0, 3), dtype=np.uint8),
            np.zeros((64, 64), dtype=np.uint8),
            np.zeros((64, 64, 1), dtype=np.uint8),
            np.zeros((64, 64, 4), dtype=np.uint8),
            np.zeros((1, 64, 64, 3), dtype=np.uint8),
            np.zeros((64, 64, 3), dtype=np.float32),
            np.zeros((64, 64, 3), dtype=np.uint16),
        )
        for image in images:
            with self.subTest(shape=getattr(image, "shape", None), dtype=getattr(image, "dtype", None)):
                with self.assertRaises(ValueError):
                    self.features.extract_features(image)

    def test_feature_size_agrees_with_original_cache_configuration(self):
        path = PROJECT_ROOT / "hog_hsv_features_v1" / "feature_config.json"
        if not path.is_file():
            self.skipTest("Configuración de caché de referencia no disponible")
        config = json.loads(path.read_text(encoding="utf-8"))
        feature = self.features.extract_features(np.zeros((64, 64, 3), dtype=np.uint8))
        self.assertEqual(feature.shape, (config["feature_dim"],))
        self.assertEqual(config["hog"]["dimension"] + config["hsv"]["dimension"], feature.size)


if __name__ == "__main__":
    unittest.main()
