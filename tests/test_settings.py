from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from detectcam.config.settings import load_settings
from detectcam.config.environment import ConfigurationError


class SettingsTests(unittest.TestCase):
    def fixture(self, base):
        root = Path(base) / "ptz-propuesta"
        shared = Path(base) / "ptz_test"
        root.mkdir(); shared.mkdir()
        (shared / "model").mkdir()
        (shared / "model/best.pt").write_bytes(b"test model path only")
        (shared / ".env").write_text(
            "DETECTCAM_CAMERA_IP=camera.local\nDETECTCAM_CAMERA_USER=operator\n"
            "DETECTCAM_CAMERA_PASSWORD=secret\nDETECTCAM_DEVICE=cpu\n"
            "DETECTCAM_PTZ_HARDWARE_ENABLED=true\n")
        return root

    def test_shared_weights_and_local_then_process_precedence(self):
        with TemporaryDirectory() as temp:
            root = self.fixture(temp)
            (root / ".env").write_text("DETECTCAM_DEVICE=0\n")
            settings = load_settings(root=root, environ={})
            self.assertEqual(settings.device, "0")
            self.assertTrue(settings.model_path.is_file())
            self.assertTrue(settings.hardware_enabled)
            settings = load_settings(root=root, environ={"DETECTCAM_DEVICE": "1"}, simulate=True)
            self.assertEqual(settings.device, "1")
            self.assertFalse(settings.hardware_enabled)
            self.assertNotIn("secret", repr(settings))
            self.assertNotIn("secret", str(settings.public_summary()))

    def test_invalid_limits_fail_before_connecting(self):
        with TemporaryDirectory() as temp:
            root = self.fixture(temp)
            for values in ({"DETECTCAM_PTZ_CHANNEL": "0"}, {"DETECTCAM_CONFIDENCE": "nan"},
                           {"DETECTCAM_RTSP_TRANSPORT": "unknown"}, {"DETECTCAM_PTZ_BACKEND": "bad"}):
                with self.assertRaises(ConfigurationError):
                    load_settings(root=root, environ=values)
