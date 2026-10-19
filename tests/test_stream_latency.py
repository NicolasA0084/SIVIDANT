from __future__ import annotations

import os
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ptz.detector import Detection
from ptz.preview import PreviewAnnotations, render_preview
from ptz.stream import LOW_LATENCY_OPTIONS, RTSPStream, StreamConfigurationError
from ptz.target_tracker import TrackState


class StreamLatencyTests(unittest.TestCase):
    def stream(self, **kwargs):
        return RTSPStream(username="test-user", password="test-password", **kwargs)

    def test_low_latency_applies_at_open_and_keeps_timeout(self):
        stream = self.stream()
        cap = Mock()
        cap.isOpened.return_value = True
        cap.set.return_value = False
        with patch.dict(os.environ, {}, clear=True), patch("ptz.stream.cv2.VideoCapture", return_value=cap) as factory:
            self.assertIs(stream._open_capture(), cap)
            self.assertEqual(os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "rtsp_transport;tcp|" + LOW_LATENCY_OPTIONS)
            params = factory.call_args.args[2]
            options = dict(zip(params[::2], params[1::2]))
            self.assertEqual(options[cv2.CAP_PROP_N_THREADS], 1)
            self.assertGreater(options[cv2.CAP_PROP_READ_TIMEOUT_MSEC], 0)
            self.assertFalse(stream.is_connected())
        cap.release()

    def test_explicit_external_ffmpeg_options_are_preserved(self):
        with patch.dict(os.environ, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "rtsp_transport;udp"}, clear=True), patch("ptz.stream.cv2.VideoCapture") as factory:
            cap = factory.return_value
            cap.isOpened.return_value = True
            self.stream()._open_capture()
            self.assertEqual(os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "rtsp_transport;udp")

    def test_standard_mode_does_not_add_probe_flags(self):
        with patch.dict(os.environ, {}, clear=True), patch("ptz.stream.cv2.VideoCapture") as factory:
            factory.return_value.isOpened.return_value = True
            self.stream(low_latency=False, decoder_threads=0)._open_capture()
            self.assertEqual(os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"], "rtsp_transport;tcp")

    def test_packet_timestamp_and_id_belong_to_same_frame(self):
        stream = self.stream()
        with stream._state_lock:
            stream._latest_frame = np.full((4, 6, 3), 7, dtype=np.uint8)
            stream._frame_id = 5
            stream._last_frame_time = 10.0
        packet = stream.read_packet()
        with stream._state_lock:
            stream._latest_frame = np.full((4, 6, 3), 9, dtype=np.uint8)
            stream._frame_id = 6
            stream._last_frame_time = 11.0
        self.assertEqual((packet.frame_id, packet.received_at), (5, 10))
        self.assertEqual(int(packet.frame[0, 0, 0]), 7)
        self.assertIsNone(stream.read_packet(after_id=6))
        latest = stream.read_packet(after_id=5)
        self.assertEqual((latest.frame_id, latest.received_at), (6, 11))

    def test_bad_stream_options_fail_before_connecting(self):
        for kwargs in ({"transport": "http"}, {"decoder_threads": -1},
                       {"decoder_threads": 1.5}, {"low_latency": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(StreamConfigurationError):
                self.stream(**kwargs)

    def test_offline_camera_and_lost_signal_reconnect_until_frames_arrive(self):
        stream = self.stream(max_read_failures=1, reconnect_delay=0.001)
        offline, interrupted, restored = Mock(), Mock(), Mock()
        offline.isOpened.return_value = False
        for capture in (interrupted, restored):
            capture.isOpened.return_value = True
            capture.set.return_value = False
        image = np.full((4, 6, 3), 7, dtype=np.uint8)
        interrupted.read.side_effect = [(True, image), (False, None)]
        reconnecting, allow_connect = threading.Event(), threading.Event()
        received, allow_read = threading.Event(), threading.Event()
        attempts = iter([offline, RuntimeError("offline"), interrupted, restored])

        def open_capture(*_args):
            result = next(attempts)
            if isinstance(result, Exception):
                raise result
            if result is restored:
                reconnecting.set()
                allow_connect.wait(2)
            return result

        def read_after_frame():
            received.set()
            allow_read.wait(2)
            return False, None

        def read_frame():
            if restored.read.call_count == 1:
                return True, image
            return read_after_frame()

        restored.read.side_effect = read_frame
        with patch.dict(os.environ, {}, clear=True), \
             patch("ptz.stream.cv2.VideoCapture", side_effect=open_capture) as factory, \
             self.assertLogs("ptz.stream", level="INFO") as logs:
            try:
                stream.start()
                self.assertTrue(reconnecting.wait(2))
                self.assertFalse(stream.is_connected())
                self.assertIsNone(stream.read_packet())
                self.assertIsNone(stream.last_frame_age)
                allow_connect.set()
                self.assertTrue(received.wait(2))
                self.assertTrue(stream.is_connected())
                self.assertEqual(stream.read_packet().frame_id, 2)
                self.assertEqual(factory.call_count, 4)
            finally:
                stream._stop_event.set()
                allow_connect.set()
                allow_read.set()
                stream.stop()
        text = "\n".join(logs.output)
        self.assertIn("Cámara sin señal RTSP", text)
        self.assertIn("Cámara sin señal de video", text)
        self.assertEqual(text.count("Señal de cámara recibida"), 2)
        self.assertNotIn("test-password", text)
        for capture in (offline, interrupted, restored):
            capture.release.assert_called_once()


class PreviewAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((160, 240, 3), dtype=np.uint8)
        self.annotations = PreviewAnnotations(
            self.frame.shape, 10.0,
            (Detection(0, "drone", 0.9, 20, 30, 60, 60),),
            None, TrackState.NO_TARGET, (0.1, 0.1),
        )

    def test_recent_boxes_draw_without_mutating_capture(self):
        display = render_preview(self.frame, self.annotations, timestamp=10.1)
        self.assertTrue(np.any(display[30:60, 20:60]))
        self.assertFalse(np.any(self.frame))

    def test_expired_or_future_boxes_are_not_drawn(self):
        for now in (10.3, 9.9):
            display = render_preview(self.frame, self.annotations, timestamp=now)
            self.assertFalse(np.any(display[30:60, 20:60]))

    def test_different_resolution_rejects_old_boxes(self):
        frame = np.zeros((320, 480, 3), dtype=np.uint8)
        display = render_preview(frame, self.annotations, timestamp=10.1)
        self.assertFalse(np.any(display[30:60, 20:60]))


if __name__ == "__main__":
    unittest.main()
