from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import timedelta
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from detectcam.runtime import (
    ISAPIBackend, ONVIFBackend, MotionCapabilities, MotionLimits, MotionRuntime, NetworkConfig,
    TransportError,
)


def eventually(predicate, timeout=1.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.003)
    raise AssertionError("condition did not become true before timeout")


class FakeBackend:
    def __init__(self):
        self.moves = []
        self.stops = 0
        self.connections = 0
        self.move_entered = threading.Event()
        self.release_move = threading.Event()
        self.stop_seen = threading.Event()
        self.block_move = False
        self.fail_move = False
        self.fail_stop = False

    def connect(self):
        self.connections += 1
        return MotionCapabilities(True, True, False, "fake")

    def move_velocity(self, pan, tilt, zoom=0.0, *, duration_s=0.35):
        self.moves.append((pan, tilt, zoom))
        self.move_entered.set()
        if self.block_move:
            self.release_move.wait(3)
        if self.fail_move:
            raise RuntimeError("simulated movement error")

    def stop(self):
        self.stops += 1
        self.stop_seen.set()
        if self.fail_stop:
            raise RuntimeError("simulated STOP error")

    def close(self):
        pass


class MotionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.runtime = MotionRuntime(self.backend, heartbeat_timeout_s=2.0)
        self.runtime.start()
        eventually(lambda: self.runtime.snapshot.connected)
        self.assertFalse(self.runtime.snapshot.armed)
        self.assertTrue(self.runtime.arm())
        self.backend.stop_seen.clear()

    def tearDown(self):
        self.backend.release_move.set()
        self.runtime.close()

    def submit(self, pan=0.5, tilt=0.0, zoom=0.0, ttl=0.3):
        return self.runtime.submit_velocity(pan, tilt, zoom, expires_at=time.monotonic() + ttl)

    def test_snapshots_and_emergency_never_wait_for_blocked_movement(self):
        self.backend.block_move = True
        self.submit()
        self.assertTrue(self.backend.move_entered.wait(1))
        started = time.monotonic()
        snapshot = self.runtime.snapshot
        with self.assertRaises(FrozenInstanceError):
            snapshot.pan = 0.0
        self.runtime.emergency_stop()
        self.assertLess(time.monotonic() - started, 0.1)
        self.assertFalse(self.runtime.snapshot.armed)
        self.assertTrue(self.backend.stop_seen.wait(1))
        self.assertTrue(self.runtime.snapshot.inflight)
        self.assertFalse(self.runtime.snapshot.stop_confirmed)
        urgent_stops = self.backend.stops
        self.backend.release_move.set()
        eventually(lambda: self.runtime.snapshot.stop_confirmed)
        self.assertGreater(self.backend.stops, urgent_stops)
        self.assertFalse(self.runtime.snapshot.motion_confirmed)

    def test_deadline_watchdog_requests_stop_while_network_is_blocked(self):
        self.backend.block_move = True
        self.submit(ttl=0.06)
        self.assertTrue(self.backend.move_entered.wait(1))
        self.assertTrue(self.backend.stop_seen.wait(0.5))
        self.assertTrue(self.runtime.snapshot.armed)
        self.assertEqual(self.runtime.snapshot.reason, "command evidence expired")
        self.assertTrue(self.runtime.snapshot.inflight)

    def test_pending_motion_is_latest_only(self):
        self.backend.block_move = True
        self.submit(pan=0.1)
        self.assertTrue(self.backend.move_entered.wait(1))
        self.submit(pan=0.2)
        self.submit(pan=0.3)
        self.submit(pan=0.4)
        self.backend.release_move.set()
        eventually(lambda: len(self.backend.moves) >= 2)
        self.assertEqual(self.backend.moves[:2], [(0.1, 0.0, 0.0), (0.4, 0.0, 0.0)])

    def test_zero_velocity_bypasses_interval_and_cancels_pending_move(self):
        self.runtime._interval = 0.4
        self.submit(pan=0.2)
        self.assertTrue(self.backend.move_entered.wait(1))
        eventually(lambda: not self.runtime.snapshot.inflight)
        self.submit(pan=-0.4)
        self.submit(pan=0.0)
        self.assertTrue(self.backend.stop_seen.wait(0.2))
        eventually(lambda: self.runtime.snapshot.stop_confirmed)
        self.assertTrue(self.runtime.snapshot.armed)
        self.assertEqual(self.backend.moves, [(0.2, 0.0, 0.0)])

    def test_centered_stop_is_coalesced(self):
        stops = self.backend.stops
        for _ in range(30):
            self.runtime.stop("target centered")
        self.assertEqual(self.backend.stops, stops)
        self.assertTrue(self.runtime.snapshot.armed)

    def test_limits_are_configurable_and_zoom_is_preserved(self):
        self.runtime._limits = MotionLimits(0.8, 0.9, 0.7)
        self.submit(1.0, -1.0, 1.0)
        self.assertTrue(self.backend.move_entered.wait(1))
        self.assertEqual(self.backend.moves[0], (0.8, -0.9, 0.7))

    def test_invalid_and_expired_commands_never_reach_transport(self):
        self.assertIsNone(self.runtime.submit_velocity(0.5, 0.0, expires_at=time.monotonic() - 0.1))
        for velocity in (float("nan"), float("inf"), 1.1):
            with self.assertRaises(ValueError):
                self.submit(velocity)
        self.assertEqual(self.backend.moves, [])

    def test_fault_requires_explicit_recovery_and_never_rearms(self):
        self.backend.fail_move = True
        self.submit()
        eventually(lambda: self.runtime.snapshot.state == "FAULT")
        eventually(lambda: self.runtime.snapshot.stop_confirmed)
        self.assertFalse(self.runtime.arm())
        self.assertEqual(self.backend.connections, 1)
        self.backend.fail_move = False
        self.runtime.recover()
        eventually(lambda: self.runtime.snapshot.connected)
        self.assertEqual(self.backend.connections, 2)
        self.assertFalse(self.runtime.snapshot.armed)
        self.assertEqual(len(self.backend.moves), 1)

    def test_unconfirmed_stop_is_not_reported_as_success(self):
        self.backend.fail_stop = True
        self.runtime.emergency_stop()
        eventually(lambda: self.runtime.snapshot.state == "FAULT")
        self.assertFalse(self.runtime.snapshot.stop_confirmed)
        self.assertFalse(self.runtime.snapshot.connected)
        self.assertFalse(self.runtime.snapshot.armed)

    def test_events_include_frame_correlation_and_deadline(self):
        events = []
        self.runtime._callback = events.append
        command_id = self.runtime.submit_velocity(0.2, 0, expires_at=time.monotonic() + 0.25,
                                                 frame_id=42, settings_revision=3)
        eventually(lambda: any(e["event"] == "completed" for e in events))
        accepted = next(e for e in events if e["event"] == "accepted")
        completed = next(e for e in events if e["event"] == "completed")
        self.assertEqual((accepted["frame_id"], completed["command_id"]), (42, command_id))
        self.assertGreater(accepted["deadline"], accepted["accepted_at"])

    def test_close_has_a_caller_budget_and_finishes_stop_after_inflight_returns(self):
        self.backend.block_move = True
        self.submit()
        self.assertTrue(self.backend.move_entered.wait(1))
        started = time.monotonic()
        self.runtime.close(timeout=0.02)
        self.assertLess(time.monotonic() - started, 0.15)
        self.assertFalse(self.runtime.snapshot.armed)
        self.assertTrue(self.backend.stop_seen.wait(1))
        self.assertFalse(self.runtime.snapshot.stop_confirmed)
        self.backend.release_move.set()
        eventually(lambda: self.runtime.snapshot.state == "CLOSED")
        self.assertTrue(self.runtime.snapshot.stop_confirmed)

    def test_heartbeat_does_not_extend_motion_evidence(self):
        self.submit(ttl=0.06)
        self.assertTrue(self.backend.move_entered.wait(1))
        end = time.monotonic() + 0.10
        while time.monotonic() < end:
            self.runtime.heartbeat()
            threading.Event().wait(0.005)
        eventually(lambda: self.runtime.snapshot.stop_confirmed)
        self.assertTrue(self.runtime.snapshot.armed)
        self.assertEqual(self.runtime.snapshot.reason, "command evidence expired")

    def test_initial_connection_retries_without_retaining_arm_or_velocity(self):
        backend = FakeBackend()
        real_connect = backend.connect
        backend.connect = Mock(side_effect=[TransportError("offline"), real_connect()])
        runtime = MotionRuntime(backend, reconnect_delay_s=0.01)
        try:
            self.assertFalse(runtime.arm())
            self.assertIsNone(runtime.submit_velocity(0.3, 0, expires_at=time.monotonic() + 1))
            runtime.start()
            eventually(lambda: runtime.snapshot.connected)
            self.assertFalse(runtime.snapshot.armed)
            self.assertEqual(backend.moves, [])
            self.assertEqual(backend.connect.call_count, 2)
        finally:
            runtime.close()

    def test_slow_telemetry_callback_cannot_delay_urgent_or_postflight_stop(self):
        callback_entered, release_callback = threading.Event(), threading.Event()

        def blocked_callback(_event):
            callback_entered.set()
            release_callback.wait(2)

        self.runtime._callback = blocked_callback
        self.backend.block_move = True
        try:
            self.submit()
            self.assertTrue(callback_entered.wait(1))
            self.assertTrue(self.backend.move_entered.wait(1))
            self.runtime.emergency_stop()
            self.assertTrue(self.backend.stop_seen.wait(0.3))
            urgent_stops = self.backend.stops
            self.backend.release_move.set()
            eventually(lambda: self.runtime.snapshot.stop_confirmed)
            self.assertGreater(self.backend.stops, urgent_stops)
        finally:
            release_callback.set()


CAPABILITIES = b"""<PTZChannel xmlns="http://www.isapi.org/ver20/XMLSchema">
<ContinuousPanTiltSpace><XRange><Min>-80</Min><Max>120</Max></XRange>
<YRange><Min>-50</Min><Max>70</Max></YRange></ContinuousPanTiltSpace>
<ContinuousZoomSpace><ZRange><Min>-30</Min><Max>40</Max></ZRange></ContinuousZoomSpace>
</PTZChannel>"""


class ISAPITransportTests(unittest.TestCase):
    def setUp(self):
        self.motion, self.urgent = Mock(), Mock()
        self.motion.request.return_value = Mock(status_code=200, content=CAPABILITIES)
        self.urgent.request.return_value = Mock(status_code=200, content=b"")
        self.config = NetworkConfig("camera.local", "username", "secret-password", backend="isapi")
        sessions = iter((self.motion, self.urgent))
        self.backend = ISAPIBackend(self.config, session_factory=lambda: next(sessions))
        self.caps = self.backend.connect()

    def tearDown(self):
        self.backend.close()

    def test_ranges_and_screen_tilt_are_mapped_from_capabilities(self):
        self.assertTrue(self.caps.zoom)
        self.assertFalse(self.caps.server_motion_timeout)
        self.motion.request.return_value = Mock(status_code=200, content=b"")
        self.backend.move_velocity(0.5, 0.5, -0.5)
        xml = ET.fromstring(self.motion.request.call_args.kwargs["data"])
        values = {node.tag.rsplit("}", 1)[-1]: node.text for node in xml}
        self.assertEqual(values, {"pan": "60", "tilt": "-25", "zoom": "-15"})

    def test_stop_uses_a_different_session_and_bounded_timeouts(self):
        self.backend.stop()
        self.assertEqual(self.motion.request.call_count, 1)
        args, kwargs = self.urgent.request.call_args
        self.assertEqual(args[0], "PUT")
        self.assertEqual(kwargs["timeout"], (0.35, 0.4))
        self.assertFalse(kwargs["allow_redirects"])
        self.assertFalse(self.urgent.trust_env)

    def test_protocol_failure_does_not_expose_camera_credentials(self):
        self.motion.request.side_effect = RuntimeError("secret-password rtsp://username@camera")
        with self.assertRaises(TransportError) as caught:
            self.backend.move_velocity(0.5, 0.0)
        self.assertNotIn("secret-password", str(caught.exception))
        self.assertNotIn("username", repr(self.config))
        self.assertNotIn("secret-password", repr(self.config))

    def test_empty_or_malformed_capabilities_fail_before_motion(self):
        for content in (b"", b"not xml", b"<capabilities/>"):
            self.motion.request.return_value = Mock(status_code=200, content=content)
            with self.assertRaises(TransportError):
                self.backend.connect()

    def test_nonfinite_velocity_ranges_are_rejected(self):
        for value in (b"inf", b"nan", b"-inf"):
            self.motion.request.return_value = Mock(status_code=200, content=CAPABILITIES.replace(b"120", value))
            with self.assertRaises(TransportError):
                self.backend.connect()


class ONVIFTransportTests(unittest.TestCase):
    def connect_fake(self, minimum=0.05):
        import onvif

        motion, urgent = Mock(), Mock()
        profile = SimpleNamespace(token="profile", PTZConfiguration=SimpleNamespace(token="ptz-config"))
        range_x = SimpleNamespace(Min=-2, Max=4)
        range_y = SimpleNamespace(Min=-3, Max=5)
        range_z = SimpleNamespace(Min=-1, Max=2)
        motion.GetConfigurationOptions.return_value = SimpleNamespace(
            Spaces=SimpleNamespace(
                ContinuousPanTiltVelocitySpace=[SimpleNamespace(XRange=range_x, YRange=range_y, URI="xy-space")],
                ContinuousZoomVelocitySpace=[SimpleNamespace(XRange=range_z, URI="z-space")],
            ), PTZTimeout=SimpleNamespace(Min=timedelta(seconds=minimum), Max=timedelta(seconds=5)),
        )
        motion.create_type.return_value = SimpleNamespace()
        camera_motion, camera_urgent = Mock(), Mock()
        camera_motion.create_ptz_service.return_value = motion
        camera_urgent.create_ptz_service.return_value = urgent
        camera_motion.create_media_service.return_value.GetProfiles.return_value = [profile]
        backend = ONVIFBackend(NetworkConfig("camera", "user", "password", onvif_port=8080))
        with patch.object(onvif, "ONVIFCamera", side_effect=[camera_motion, camera_urgent]) as factory:
            caps = backend.connect()
        return backend, motion, urgent, caps, factory

    def test_discovered_spaces_stop_connection_port_and_server_timeout(self):
        backend, motion, urgent, caps, factory = self.connect_fake()
        try:
            self.assertTrue(caps.zoom)
            self.assertTrue(caps.server_motion_timeout)
            self.assertEqual(caps.server_timeout_min_s, 0.05)
            self.assertEqual(caps.server_timeout_max_s, 5.0)
            self.assertEqual(factory.call_args.args[1], 8080)
            backend.move_velocity(0.5, 0.5, 0.5, duration_s=0.2)
            request = motion.ContinuousMove.call_args.args[0]
            self.assertEqual(request.Velocity, {
                "PanTilt": {"x": 2.0, "y": -1.5, "space": "xy-space"},
                "Zoom": {"x": 1.0, "space": "z-space"},
            })
            self.assertEqual(request.Timeout, timedelta(seconds=0.2))
            backend.stop()
            urgent.Stop.assert_called_once_with({"ProfileToken": "profile", "PanTilt": True, "Zoom": True})
            motion.Stop.assert_not_called()
        finally:
            backend.close()

    def test_large_server_minimum_is_exposed_and_not_claimed_as_source_deadline(self):
        backend, motion, _urgent, caps, _factory = self.connect_fake(minimum=1.0)
        try:
            self.assertTrue(caps.server_motion_timeout)
            self.assertEqual(caps.server_timeout_min_s, 1.0)
            backend.move_velocity(0.2, 0.0, duration_s=0.12)
            self.assertEqual(motion.ContinuousMove.call_args.args[0].Timeout, timedelta(seconds=1))
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()
