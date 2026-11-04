from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from ptz.detector import ModelLoadError, YOLODetector


class DetectorArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_ncnn_directory_loads_once_and_passes_vulkan_to_inference(self) -> None:
        model_path = self.root / "best_ncnn_model"
        model_path.mkdir()
        (model_path / "model.ncnn.param").write_bytes(b"parameters")
        (model_path / "model.ncnn.bin").write_bytes(b"weights")
        model = Mock(names={0: "drone"})
        model.predict.return_value = []
        factory = Mock(return_value=model)
        with patch.dict("os.environ", {
            "DETECTCAM_MODEL_PATH": str(model_path),
            "DETECTCAM_DEVICE": "vulkan:0",
            "DETECTCAM_IMGSZ": "416",
        }):
            detector = YOLODetector.from_environment(model_factory=factory)
        frame = np.zeros((416, 416, 3), dtype=np.uint8)
        self.assertEqual(detector.detect(frame), [])
        self.assertEqual(detector.detect(frame), [])
        factory.assert_called_once_with(str(model_path))
        self.assertEqual(detector.class_names, {0: "drone"})
        self.assertEqual(model.predict.call_count, 2)
        self.assertIs(model.predict.call_args.kwargs["source"], frame)
        self.assertEqual(model.predict.call_args.kwargs["device"], "vulkan:0")
        self.assertEqual(model.predict.call_args.kwargs["imgsz"], 416)

    def test_incomplete_ncnn_export_does_not_call_model_loader(self) -> None:
        model_path = self.root / "best_ncnn_model"
        model_path.mkdir()
        (model_path / "model.ncnn.param").write_bytes(b"parameters")
        factory = Mock()
        with self.assertRaisesRegex(ModelLoadError, "falta .*model.ncnn.bin"):
            YOLODetector(model_path, model_factory=factory)
        factory.assert_not_called()

    def test_existing_weight_files_keep_their_original_path(self) -> None:
        for extension in ("pt", "onnx"):
            model_path = self.root / f"best.{extension}"
            model_path.write_bytes(b"weights")
            factory = Mock(return_value=Mock(names={0: "drone"}))
            with self.subTest(extension=extension):
                detector = YOLODetector(model_path, model_factory=factory)
                self.assertEqual(detector.model_path, model_path)
                factory.assert_called_once_with(str(model_path))


if __name__ == "__main__":
    unittest.main()
