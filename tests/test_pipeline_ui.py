from __future__ import annotations

import threading
from time import monotonic
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ptz.main import main, run_detection_loop
from ptz.operator_control import OperatorControl
from ptz.ptz_decision import PTZDecisionEngine
from ptz.ptz_execution_policy import PTZExecutionPolicy
from ptz.ptz_metrics import PTZMetrics
from ptz.stream import ReceivedFrame, RTSPStream
from ptz.target_selector import TargetSelector
from ptz.target_tracker import TargetTracker
from ptz.ui import FrameMailbox


class PipelineTests(unittest.TestCase):
    def test_video_is_clean_and_telemetry_is_separate(self):
        stop = threading.Event()
        image = np.full((360, 640, 3), 70, dtype=np.uint8)
        stream = Mock()
        stream.is_connected.return_value = True
        stream.read_packet.return_value = ReceivedFrame(1, image, monotonic() - 0.02)
        stream.last_frame_age = 0.02
        detector = Mock()
        detector.detect.return_value = []
        detector.last_inference_ms = 12.0
        mailbox = FrameMailbox()
        publish = mailbox.publish

        def finish(snapshot):
            publish(snapshot)
            stop.set()

        mailbox.publish = finish
        metrics = PTZMetrics(enabled=False)
        run_detection_loop(
            stream, detector, TargetSelector(), TargetTracker(),
            PTZDecisionEngine(), metrics, OperatorControl(None, lambda: 0.02),
            mailbox, stop,
        )
        snapshot = mailbox.take()
        self.assertTrue(snapshot.connected)
        self.assertEqual(len(snapshot.telemetry), 12)
        self.assertGreaterEqual(len(snapshot.ptz), 14)
        # Las zonas antes cubiertas por los paneles permanecen intactas.
        np.testing.assert_array_equal(snapshot.frame[10:100, 10:100], image[10:100, 10:100])
        np.testing.assert_array_equal(snapshot.frame[-60:, :100], image[-60:, :100])
        self.assertIsNotNone(snapshot.captured_at)
        metrics.close()


    def test_preview_keeps_up_while_detector_is_blocked(self):
        import tkinter as tk
        from ptz.ui import DetectcamDashboard, UILogHandler

        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Sin pantalla gráfica: {error}")
        stream = RTSPStream(username="test", password="test")
        with stream._state_lock:
            stream._latest_frame = np.zeros((160, 240, 3), dtype=np.uint8)
            stream._frame_id = 1
            stream._last_frame_time = monotonic()
            stream._connected = True
        mailbox = FrameMailbox()
        control = OperatorControl(None, lambda: stream.last_frame_age)
        app = DetectcamDashboard(control, mailbox, UILogHandler(), root=root, preview_source=stream)
        entered, release, stop = threading.Event(), threading.Event(), threading.Event()
        detector = Mock()
        detector.last_inference_ms = 400.0

        def blocked_detect(_frame):
            entered.set()
            release.wait(2)
            return []

        detector.detect.side_effect = blocked_detect
        metrics = PTZMetrics(enabled=False)
        worker = threading.Thread(target=run_detection_loop, args=(
            stream, detector, TargetSelector(), TargetTracker(),
            PTZDecisionEngine(), metrics, control, mailbox, stop,
        ), daemon=True)
        try:
            worker.start()
            self.assertTrue(entered.wait(1))
            received_at = monotonic()
            with stream._state_lock:
                stream._latest_frame = np.full((160, 240, 3), 90, dtype=np.uint8)
                stream._frame_id = 2
                stream._last_frame_time = received_at
            root.after(120, root.quit)
            app.run()
            self.assertEqual(app._preview_frame_id, 2)
            self.assertEqual(app._preview_received_at, received_at)
            self.assertEqual(app._image.getpixel((0, 0)), (90, 90, 90))
            self.assertIsNone(app._snapshot)
            self.assertTrue(worker.is_alive())
        finally:
            stop.set()
            release.set()
            worker.join(timeout=2)
            app.close()
            metrics.close()
        self.assertFalse(worker.is_alive())

    def test_headless_startup_closes_prepared_resources(self):
        import tkinter as tk

        stream = Mock()
        detector = Mock()
        detector.class_names = {0: "drone"}
        metrics = Mock()
        metrics.enabled = False
        with patch("ptz.main.RTSPStream.from_environment", return_value=stream), \
             patch("ptz.main.YOLODetector.from_environment", return_value=detector), \
             patch("ptz.main.PTZMetrics.from_environment", return_value=metrics), \
             patch("ptz.main.create_hardware_executor") as factory, \
             patch("ptz.ui.DetectcamDashboard", side_effect=tk.TclError("no display")):
            self.assertEqual(main(), 1)
            factory.assert_not_called()
        stream.stop.assert_called_once()
        metrics.close.assert_called_once()

    def test_dashboard_runs_while_camera_hardware_connection_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        stream = Mock()
        stream.is_connected.return_value = False
        stream.read_packet.return_value = None
        stream.last_frame_age = None
        detector = Mock()
        detector.class_names = {0: "drone"}
        detector.confidence = 0.25
        metrics = PTZMetrics(enabled=False)
        dashboard = Mock(closed=False)

        def connect(*_args):
            entered.set()
            release.wait(2)
            raise RuntimeError("camera offline")

        def run_dashboard():
            try:
                self.assertTrue(entered.wait(2))
                stream.start.assert_called_once()
                control = dashboard_factory.call_args.args[0]
                self.assertEqual(control.status.state, "CONNECTING")
                self.assertFalse(control.status.hardware)
                self.assertIsNone(control.executor)
            finally:
                release.set()

        dashboard.run.side_effect = run_dashboard
        with patch("ptz.main.RTSPStream.from_environment", return_value=stream), \
             patch("ptz.main.YOLODetector.from_environment", return_value=detector), \
             patch("ptz.main.PTZMetrics.from_environment", return_value=metrics), \
             patch("ptz.main.PTZExecutionPolicy.from_environment", return_value=PTZExecutionPolicy(hardware_enabled=True)), \
             patch("ptz.main.create_hardware_executor", side_effect=connect) as hardware_factory, \
             patch("ptz.ui.DetectcamDashboard", return_value=dashboard) as dashboard_factory:
            self.assertEqual(main(), 0)
            hardware_factory.assert_called_once()
        stream.stop.assert_called_once()
        dashboard.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
