"""Integration contracts using synthetic frames and local-only doubles.

No test opens RTSP, constructs a real detector, or creates a network transport.
The arm/cancel race is synchronized explicitly rather than relying on a loop.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from detectcam.adapters.detector import Detection as DisplayDetection
from detectcam.adapters.stream import ReceivedFrame
from detectcam.application.operator import ARM_CONFIRMATION, OperatorControl
from detectcam.application.pipeline import VisionPipeline
from detectcam.application.tracking_settings import TrackingOptions
from detectcam.config.environment import ConfigurationError
from detectcam.config.settings import load_settings
from detectcam.core.control import Controller
from detectcam.core.tracking import Tracker
from detectcam.presentation.dashboard import DetectcamDashboard, FrameMailbox
from detectcam.runtime.models import MotionCapabilities, MotionSnapshot
from detectcam.runtime.motion import MotionRuntime


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class LocalRuntime:
    """Runtime-shaped recorder; transport behavior has its own test suite."""
    def __init__(self, clock):
        self.clock = clock
        self._snapshot = MotionSnapshot(connected=True, state="DISARMED", stop_confirmed=True)
        self.submissions = []
        self.stops = []
        self.arm_entered = self.arm_release = None

    @property
    def snapshot(self):
        return self._snapshot

    def arm(self):
        if self.arm_entered is not None:
            self.arm_entered.set()
            if not self.arm_release.wait(1.0):
                raise RuntimeError("test arm gate was not released")
        self._snapshot = replace(self._snapshot, armed=True, state="ARMED_IDLE")
        return True

    def heartbeat(self):
        pass

    def stop(self, reason="stop"):
        self.stops.append(reason)
        self._snapshot = replace(self._snapshot, pan=0., tilt=0., zoom=0., reason=reason)

    def disarm(self):
        self.stop("disarm")
        self._snapshot = replace(self._snapshot, armed=False, state="DISARMED")

    def emergency_stop(self):
        self.disarm()

    def recover(self):
        self.disarm()

    def submit_velocity(self, pan, tilt, zoom=0., **fields):
        if not self._snapshot.armed or fields["expires_at"] <= self.clock():
            return None
        self.submissions.append(dict(pan=pan, tilt=tilt, zoom=zoom, **fields))
        self._snapshot = replace(self._snapshot, pan=pan, tilt=tilt, zoom=zoom)
        return len(self.submissions)

    def close(self):
        self.disarm()


class LocalBackend:
    def connect(self):
        return MotionCapabilities(pan_tilt=True, zoom=True, backend="local-test")

    def stop(self):
        pass

    def close(self):
        pass


class SyntheticStream:
    def __init__(self, packets):
        self.packets = list(packets)

    def read_packet(self, *, copy=False, after_id=-1):
        return next((packet for packet in self.packets if packet.frame_id > after_id), None)

    def is_connected(self):
        return True


class SyntheticDetector:
    confidence = .15
    last_inference_ms = 0.

    def __init__(self, clock, *, delay=.05, after_detect=None):
        self.clock, self.delay, self.after_detect = clock, delay, after_detect

    def set_confidence(self, confidence):
        self.confidence = confidence

    def detect(self, frame):
        self.clock.advance(self.delay)
        self.last_inference_ms = self.delay*1000
        if self.after_detect is not None:
            self.after_detect()
        return [DisplayDetection(0, "Drone", .9, 72, 42, 88, 58)]


class ApplicationTests(unittest.TestCase):
    def make_control(self, *, automatic=False):
        clock = Clock()
        runtime = LocalRuntime(clock)
        options = TrackingOptions({0: "Drone", 1: "Bird"}, min_confidence=.5)
        control = OperatorControl(runtime, lambda: 0., controller=Controller(),
                                  tracking_options=options, clock=clock)
        self.addCleanup(control.close)
        control.heartbeat()
        control.preview_seen(clock())
        if automatic:
            control.request("mode", "automatic")
        control.step()
        control.request("arm", ARM_CONFIRMATION)
        control.step()
        self.assertTrue(runtime.snapshot.armed)
        return clock, runtime, options, control

    def make_pipeline(self, *, delay=.05, after_detect=None):
        clock, runtime, options, control = self.make_control(automatic=True)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        stream = SyntheticStream([ReceivedFrame(1, frame, clock()),
                                  ReceivedFrame(2, frame, clock()+.05)])
        detector = SyntheticDetector(clock, delay=delay, after_detect=after_detect)
        telemetry = SimpleNamespace(emit=Mock(), snapshot={
            "events": {}, "dropped": 0, "backend_p50_ms": None, "backend_p95_ms": None})
        mailbox = FrameMailbox()
        pipeline = VisionPipeline(stream, detector, Tracker(), control, mailbox,
                                  options, telemetry, warmup=False, clock=clock)
        return clock, runtime, options, control, pipeline, mailbox

    def test_first_real_target_reaches_preview_and_control_with_source_timestamp(self):
        clock, runtime, _options, control, pipeline, mailbox = self.make_pipeline(delay=.2)
        self.assertTrue(pipeline.step())
        view = mailbox.take()
        self.assertIsNotNone(view.annotations.target)
        self.assertEqual(view.captured_at, 10.)
        self.assertEqual(view.annotations.target.detection.class_name, "Drone")
        control.heartbeat()
        control.step()
        self.assertTrue(runtime.submissions)
        self.assertEqual(runtime.submissions[-1]["frame_id"], 1)
        self.assertLessEqual(runtime.submissions[-1]["expires_at"], 10.5)
        self.assertGreater(runtime.submissions[-1]["expires_at"], clock())

    def test_slow_inference_cannot_rejuvenate_target_evidence(self):
        _clock, runtime, _options, control, pipeline, _mailbox = self.make_pipeline(delay=.7)
        pipeline.step()
        control.heartbeat()
        control.step()
        self.assertEqual(runtime.submissions, [])

    def test_changed_settings_reject_inference_that_started_on_previous_revision(self):
        _clock, runtime, options, control, pipeline, _mailbox = self.make_pipeline()
        pipeline.detector.after_detect = lambda: options.update(
            min_confidence=.6, allowed_classes=["Drone"])
        pipeline.step()
        control.heartbeat()
        control.step()
        self.assertEqual(runtime.submissions, [])
        pipeline.detector.after_detect = None
        pipeline.step()
        control.heartbeat()
        control.step()
        self.assertEqual(runtime.submissions[-1]["settings_revision"], options.snapshot.revision)
        self.assertEqual(runtime.submissions[-1]["frame_id"], 2)

    def test_invalid_stream_or_display_age_never_allows_manual_motion(self):
        clock, runtime, _options, control = self.make_control()
        for video_age, preview_at in ((-.1, clock()), (float("nan"), clock()),
                                      (0., clock()+1.), (0., float("nan")), (0., None)):
            with self.subTest(video_age=video_age, preview_at=preview_at):
                runtime.submissions.clear()
                control._video_age = lambda age=video_age: age
                control.preview_seen(preview_at)
                control.set_manual(.2, 0.)
                control.heartbeat()
                control.step()
                self.assertEqual(runtime.submissions, [])

    def test_live_ui_keeps_idle_runtime_armed_without_extending_motion_evidence(self):
        clock = Clock()
        runtime = MotionRuntime(LocalBackend(), clock=clock, heartbeat_timeout_s=.4)
        runtime._connect()  # LocalBackend performs no I/O and starts no threads.
        control = OperatorControl(runtime, lambda: 0., clock=clock)
        self.addCleanup(control.close)
        control.heartbeat()
        control.preview_seen(clock())
        control.step()
        control.request("arm", ARM_CONFIRMATION)
        control.step()
        for _ in range(4):
            clock.advance(.25)
            control.heartbeat()
            control.preview_seen(clock())
            control.step()
            runtime.check_deadlines()
            self.assertTrue(runtime.snapshot.armed)
        clock.advance(.6)
        control.step()
        runtime.check_deadlines()
        self.assertFalse(runtime.snapshot.armed)

    def test_cancellation_arriving_during_arm_wins_after_both_calls_return(self):
        for action, value in (("emergency", ""), ("disarm", ""),
                              ("mode", "automatic"), ("recover", "")):
            with self.subTest(action=action):
                clock = Clock()
                runtime = LocalRuntime(clock)
                control = OperatorControl(runtime, lambda: 0., clock=clock)
                self.addCleanup(control.close)
                runtime.arm_entered, runtime.arm_release = threading.Event(), threading.Event()
                control.heartbeat()
                control.request("arm", ARM_CONFIRMATION)
                worker = threading.Thread(target=control.step, daemon=True)
                worker.start()
                self.assertTrue(runtime.arm_entered.wait(1.))
                attempted, completed = threading.Event(), threading.Event()

                def cancel():
                    attempted.set()
                    control.request(action, value)
                    completed.set()

                stopper = threading.Thread(target=cancel, daemon=True)
                stopper.start()
                self.assertTrue(attempted.wait(1.))
                # Correct code can serialize arm and cancellation under a short
                # lock; the old race completes cancellation before arm returns.
                completed.wait(.03)
                runtime.arm_release.set()
                worker.join(1.)
                stopper.join(1.)
                self.assertFalse(worker.is_alive())
                self.assertFalse(stopper.is_alive())
                control.step()
                self.assertFalse(runtime.snapshot.armed)
                self.assertEqual(runtime.submissions, [])

    def test_dashboard_reads_camera_while_detector_is_blocked_without_tk_display(self):
        clock, _runtime, _options, control, pipeline, mailbox = self.make_pipeline()
        entered, release = threading.Event(), threading.Event()

        def blocked(_frame):
            entered.set()
            if not release.wait(1.):
                raise RuntimeError("test detector gate was not released")
            return []

        pipeline.detector.detect = blocked
        worker = threading.Thread(target=pipeline.step, daemon=True)
        worker.start()
        self.assertTrue(entered.wait(1.))
        app = object.__new__(DetectcamDashboard)
        app.preview_source = pipeline.stream
        app.control = control
        app._preview_frame_id = -1
        app._preview_frame = app._preview_received_at = app._preview_shape = None
        app._snapshot = app._image = None
        app._refresh_preview = Mock()
        app._render_video = Mock()
        app.connection, app.video_info = Mock(), Mock()
        try:
            app._update_preview(None, clock())
            self.assertIsNotNone(app._preview_frame)
            self.assertEqual(app._preview_received_at, 10.)
            app._refresh_preview.assert_called_once()
            self.assertIsNone(mailbox.take())
        finally:
            release.set()
            worker.join(1.)
        self.assertFalse(worker.is_alive())


class ApplicationConfigurationTests(unittest.TestCase):
    def fixture(self, temporary):
        root = Path(temporary) / "ptz-propuesta"
        shared = Path(temporary) / "ptz_test"
        root.mkdir()
        (shared / "model").mkdir(parents=True)
        (shared / "model/best.pt").touch()
        (shared / ".env").write_text(
            "DETECTCAM_CAMERA_IP=camera.invalid\nDETECTCAM_CAMERA_USER=test\n"
            "DETECTCAM_CAMERA_PASSWORD=synthetic-secret\n", encoding="utf-8")
        return root

    def test_toml_section_and_boolean_types_are_rejected_before_runtime(self):
        for content in ('application = 1\n', '[application]\nmetrics_enabled = "false"\n',
                        '[application]\nmetrics_enabled = 1\n'):
            with self.subTest(content=content), TemporaryDirectory() as temporary:
                root = self.fixture(temporary)
                (root / "settings.toml").write_text(content, encoding="utf-8")
                with self.assertRaises(ConfigurationError):
                    load_settings(root=root, environ={})

    def test_rtsp_channel_is_checked_by_preflight(self):
        for channel in ("0", "abc", "-1"):
            with self.subTest(channel=channel), TemporaryDirectory() as temporary:
                root = self.fixture(temporary)
                with self.assertRaises(ConfigurationError):
                    load_settings(root=root, environ={"DETECTCAM_RTSP_CHANNEL": channel})
