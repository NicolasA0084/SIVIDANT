from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from time import monotonic
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from detectcam.adapters.media import MediaStore
from detectcam.adapters.stream import RTSPStream


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = MediaStore(Path(self.temp.name) / "archivos")
        self.addCleanup(self.store.close)
        self.source = RTSPStream(username="test", password="test")
        self.image = np.full((48, 64, 3), (30, 90, 180), dtype=np.uint8)
        with self.source._state_lock:
            self.source._latest_frame = self.image
            self.source._last_frame_time = monotonic()
            self.source._frame_id = 1
            self.source._frame_rate = 20.0
            self.source._connected = True

    def wait_for(self, predicate):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            if predicate():
                return
            threading.Event().wait(0.005)
        self.fail("La operación de medios no terminó a tiempo")

    def inspect_video(self, path):
        reader = cv2.VideoCapture(str(path))
        try:
            self.assertTrue(reader.isOpened(), str(path))
            self.assertEqual(reader.get(cv2.CAP_PROP_FPS), 20.0)
            count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, frame = reader.read()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, self.image.shape)
            return count
        finally:
            reader.release()

    def test_capture_creates_jpeg_without_mutating_or_overwriting_images(self):
        first = self.store.capture(self.image)
        second = self.store.capture(self.image)
        self.image[:] = 0
        self.store.close()
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, "capturas")
        self.assertEqual(first.suffix, ".jpg")
        for path in (first, second):
            loaded = cv2.imread(str(path))
            self.assertEqual(loaded.shape, (48, 64, 3))
            self.assertGreater(float(loaded.mean()), 50)

    def test_recording_produces_playable_mp4_at_camera_fps_and_real_duration(self):
        self.assertTrue(self.store.start_recording(self.source))
        self.assertFalse(self.store.start_recording(self.source))
        self.wait_for(lambda: self.store.status.started_at is not None)
        started = monotonic()
        threading.Event().wait(0.25)
        self.store.close()
        elapsed = monotonic() - started
        path = self.store.status.path
        self.assertEqual(path.parent.name, "grabaciones")
        self.assertEqual(path.suffix, ".mp4")
        self.assertFalse(self.store.status.recording)
        count = self.inspect_video(path)
        self.assertGreaterEqual(count, 4)
        self.assertLess(abs(count / 20 - elapsed), 0.15)

    def test_loss_of_signal_finalizes_the_video(self):
        self.store.start_recording(self.source)
        self.wait_for(lambda: self.store.status.started_at is not None)
        threading.Event().wait(0.08)
        with self.assertLogs("detectcam.adapters.media", level="INFO") as logs:
            with self.source._state_lock:
                self.source._connected = False
            self.wait_for(lambda: not self.store.status.recording)
        self.assertIn("pérdida de señal", "\n".join(logs.output))
        self.assertGreater(self.inspect_video(self.store.status.path), 0)

    def test_cannot_start_without_signal_or_change_directory_during_recording(self):
        with self.source._state_lock:
            self.source._connected = False
        self.assertFalse(self.store.start_recording(self.source))
        self.assertFalse(self.store.directory.exists())
        with self.source._state_lock:
            self.source._connected = True
        self.assertTrue(self.store.start_recording(self.source))
        with self.assertRaises(ValueError):
            self.store.set_directory(Path(self.temp.name) / "otra")
        self.store.close()
        self.store.set_directory(Path(self.temp.name) / "otra")
        self.assertEqual(self.store.directory.name, "otra")

    def test_encoder_failure_is_reported_and_leaves_no_active_recording(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch("detectcam.adapters.media.cv2.VideoWriter", return_value=writer), \
             self.assertLogs("detectcam.adapters.media", level="ERROR"):
            self.store.start_recording(self.source)
            self.wait_for(lambda: not self.store.status.recording)
        self.assertIn("Error", self.store.status.message)
        self.assertIsNone(self.store.status.path)
        writer.release.assert_called_once()

    def test_image_write_failure_is_logged(self):
        with patch("detectcam.adapters.media.cv2.imwrite", return_value=False), \
             self.assertLogs("detectcam.adapters.media", level="ERROR"):
            self.store.capture(self.image)
            self.store.close()
        self.assertIn("Error", self.store.status.message)


if __name__ == "__main__":
    unittest.main()
