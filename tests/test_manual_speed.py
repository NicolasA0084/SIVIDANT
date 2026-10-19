from __future__ import annotations

import unittest

from ptz.ptz_backend import PTZCapabilities
from ptz.ptz_decision import DecisionAction, PTZDecision
from ptz.ptz_execution_policy import (
    PTZExecutionContext,
    PTZExecutionPolicy,
    PTZExecutionPolicyConfigurationError,
)
from ptz.ptz_executor import PTZCommandExecutor
from ptz.target_tracker import TrackState


class FakeBackend:
    backend_capabilities = PTZCapabilities(continuous_pan_tilt=True)

    def __init__(self):
        self.moves = []
        self.stops = 0

    def is_connected(self):
        return True

    def move_velocity(self, pan, tilt, zoom):
        self.moves.append((pan, tilt, zoom))

    def stop(self):
        self.stops += 1

    def close(self):
        pass


class ManualSpeedTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.executor = PTZCommandExecutor(
            self.backend,
            PTZExecutionPolicy(hardware_enabled=True, command_interval_ms=0),
            clock=lambda: 10.0,
            start_watchdog=False,
        )
        self.addCleanup(self.executor.close)
        self.executor.arm("MOVER")

    def move_manual(self, pan, tilt, **ages):
        return self.executor.execute_manual(
            pan, tilt,
            frame_age_ms=ages.get("frame_age_ms", 0.0),
            input_age_ms=ages.get("input_age_ms", 0.0),
        )

    def move_automatic(self, confidence=0.9):
        return self.executor.execute(
            PTZDecision(
                action=DecisionAction.MOVE,
                pan=0.75, tilt=-0.75, zoom=0.0,
                target_error_x=0.5, target_error_y=-0.5,
                inside_dead_zone=False, safe_to_move=True, reason="test target",
            ),
            PTZExecutionContext(
                tracker_state=TrackState.TRACKED,
                target_age_ms=0.0, frame_age_ms=0.0,
                confidence=confidence, visible_frames=3,
            ),
        )

    def test_manual_range_reaches_75_and_caps_both_axes(self):
        for requested, expected in (
            ((0.1, -0.2), (0.1, -0.2, 0.0)),
            ((0.75, -0.75), (0.75, -0.75, 0.0)),
            ((-1.0, 1.0), (-0.75, 0.75, 0.0)),
        ):
            with self.subTest(requested=requested):
                self.assertTrue(self.move_manual(*requested).executed)
                self.assertEqual(self.backend.moves[-1], expected)
                self.assertEqual(self.executor.last_physical_velocity, expected)

    def test_automatic_motion_retains_physical_limits_after_manual_motion(self):
        self.move_manual(0.75, -0.75)
        self.assertTrue(self.move_automatic().executed)
        self.assertEqual(self.backend.moves[-1], (0.3, -0.3, 0.0))

    def test_manual_zoom_requires_capability_is_limited_and_stops(self):
        result = self.executor.execute_manual(0, 0, zoom=0.9, frame_age_ms=0, input_age_ms=0)
        self.assertFalse(result.executed)
        self.assertIn("lacks continuous zoom", result.reason)
        self.backend.backend_capabilities = PTZCapabilities(continuous_pan_tilt=True, continuous_zoom=True)
        self.assertTrue(self.executor.execute_manual(0, 0, zoom=0.9, frame_age_ms=0, input_age_ms=0).executed)
        self.assertEqual(self.backend.moves[-1], (0, 0, 0.3))
        self.assertTrue(self.executor.execute_manual(0.2, -0.1, zoom=-0.8, frame_age_ms=0, input_age_ms=0).executed)
        self.assertEqual(self.backend.moves[-1], (0.2, -0.1, -0.3))
        self.executor.execute_manual(0, 0, zoom=-0.2, frame_age_ms=0, input_age_ms=251)
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
        self.assertEqual(self.backend.stops, 1)

    def test_automatic_zoom_stays_blocked_after_manual_zoom(self):
        self.backend.backend_capabilities = PTZCapabilities(continuous_pan_tilt=True, continuous_zoom=True)
        self.executor.execute_manual(0, 0, zoom=0.2, frame_age_ms=0, input_age_ms=0)
        self.assertTrue(self.move_automatic().executed)
        self.assertEqual(self.backend.moves[-1][2], 0)

    def test_manual_high_speed_stops_on_stale_input_and_deadman(self):
        self.move_manual(0.75, -0.75)
        self.assertFalse(self.move_manual(0.75, -0.75, input_age_ms=251.0).executed)
        self.assertEqual(self.executor.last_physical_velocity, (0.0, 0.0, 0.0))
        self.move_manual(0.75, -0.75)
        self.executor.check_timeouts(timestamp=10.41)
        self.assertEqual(self.backend.stops, 2)
        self.assertEqual(self.executor.deadman_activations, 1)
        self.assertEqual(self.executor.last_physical_velocity, (0.0, 0.0, 0.0))

    def test_live_confidence_changes_gate_automatic_motion(self):
        self.assertFalse(self.move_automatic(confidence=0.4).executed)
        self.executor.set_min_track_confidence(0.4)
        self.assertTrue(self.move_automatic(confidence=0.4).executed)
        self.executor.set_min_track_confidence(0.6)
        self.assertFalse(self.move_automatic(confidence=0.4).executed)
        self.assertEqual(self.backend.stops, 1)
        self.assertTrue(self.executor.armed)

    def test_invalid_confidence_updates_preserve_previous_threshold(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), "bad", None):
            with self.subTest(value=value):
                with self.assertRaises(PTZExecutionPolicyConfigurationError):
                    self.executor.set_min_track_confidence(value)
                self.assertFalse(self.move_automatic(confidence=0.4).executed)
                self.assertTrue(self.move_automatic(confidence=0.5).executed)


if __name__ == "__main__":
    unittest.main()
