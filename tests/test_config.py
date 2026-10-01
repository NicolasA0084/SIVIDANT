from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ptz.config import (
    ConfigurationError,
    bootstrap_environment,
    load_dotenv,
    validate_startup_configuration,
)


class ConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "model").mkdir()
        (self.root / "model" / "best.pt").write_bytes(b"test weights placeholder")
        self.env = {
            "DETECTCAM_CAMERA_IP": "192.0.2.10",
            "DETECTCAM_CAMERA_USER": "camera-user",
            "DETECTCAM_CAMERA_PASSWORD": "camera-password",
            "DETECTCAM_MODEL_PATH": "model/best.pt",
            "DETECTCAM_PTZ_HARDWARE_ENABLED": "false",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_environment_has_priority_over_dotenv(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text("DETECTCAM_CAMERA_USER=from-file\n", encoding="utf-8")
        values = {"DETECTCAM_CAMERA_USER": "from-environment"}
        load_dotenv(env_file, environ=values)
        self.assertEqual(values["DETECTCAM_CAMERA_USER"], "from-environment")

    def test_relative_model_path_uses_project_root_not_cwd(self) -> None:
        configuration = validate_startup_configuration(
            self.env, project_root=self.root, env_file=self.root / ".env"
        )
        self.assertEqual(configuration.model_path, self.root / "model" / "best.pt")

    def test_missing_required_fields_are_reported_together(self) -> None:
        with self.assertRaisesRegex(
            ConfigurationError,
            "DETECTCAM_CAMERA_USER.*DETECTCAM_CAMERA_PASSWORD",
        ):
            validate_startup_configuration(
                {"DETECTCAM_CAMERA_IP": "192.0.2.10"}, project_root=self.root
            )

    def test_invalid_boolean_is_rejected(self) -> None:
        values = dict(self.env, DETECTCAM_PTZ_HARDWARE_ENABLED="quizas")
        with self.assertRaisesRegex(ConfigurationError, "debe ser true o false"):
            validate_startup_configuration(values, project_root=self.root)

    def test_invalid_rtsp_latency_options_are_rejected(self) -> None:
        for name, value in (
            ("DETECTCAM_RTSP_TRANSPORT", "http"),
            ("DETECTCAM_RTSP_DECODER_THREADS", "-1"),
            ("DETECTCAM_RTSP_LOW_LATENCY", "quizas"),
        ):
            values = dict(self.env, **{name: value})
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                validate_startup_configuration(values, project_root=self.root)

    def test_bootstrap_publishes_absolute_model_path(self) -> None:
        env_file = self.root / ".env"
        env_file.write_text(
            "DETECTCAM_CAMERA_IP=192.0.2.10\n"
            "DETECTCAM_CAMERA_USER=user\n"
            "DETECTCAM_CAMERA_PASSWORD=password\n"
            "DETECTCAM_MODEL_PATH=model/best.pt\n",
            encoding="utf-8",
        )
        values: dict[str, str] = {}
        bootstrap_environment(env_file, environ=values, project_root=self.root)
        self.assertEqual(values["DETECTCAM_MODEL_PATH"], str(self.root / "model" / "best.pt"))


if __name__ == "__main__":
    unittest.main()
