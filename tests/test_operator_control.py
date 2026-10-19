from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ptz.operator_control import OperatorControl
from ptz.ptz_backend import PTZCapabilities
from ptz.ptz_decision import DecisionAction, PTZDecision
from ptz.ptz_execution_policy import PTZExecutionContext, PTZExecutionPolicy
from ptz.ptz_executor import PTZArmError, PTZCommandExecutor, PTZExecutorState
from ptz.target_tracker import TrackState


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeBackend:
    backend_capabilities = PTZCapabilities(continuous_pan_tilt=True)

    def __init__(self):
        self.connected = True
        self.moves = []
        self.stops = 0
        self.closed = False
        self.fail_move = False
        self.fail_stop = False

    def is_connected(self):
        return self.connected

    def move_velocity(self, pan, tilt, zoom=0.0):
        if self.fail_move:
            raise RuntimeError("simulated movement failure")
        self.moves.append((pan, tilt, zoom))

    def stop(self):
        if self.fail_stop:
            raise RuntimeError("simulated stop failure")
        self.stops += 1

    def close(self):
        self.closed = True


class ManualExecutorTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = FakeBackend()
        self.executor = PTZCommandExecutor(
            self.backend, PTZExecutionPolicy(hardware_enabled=True),
            clock=self.clock, start_watchdog=False,
        )

    def tearDown(self):
        self.executor.close()

    def move(self, pan=0.2, tilt=-0.2, frame_age_ms=10.0, input_age_ms=0.0):
        return self.executor.execute_manual(
            pan, tilt, frame_age_ms=frame_age_ms, input_age_ms=input_age_ms,
        )

    def test_hardware_and_arm_gates(self):
        self.assertFalse(self.move().executed)
        self.assertEqual(self.backend.moves, [])
        disabled = PTZCommandExecutor(
            self.backend, PTZExecutionPolicy(hardware_enabled=False), start_watchdog=False,
        )
        with self.assertRaises(PTZArmError):
            disabled.arm("MOVER")
        self.assertFalse(disabled.execute_manual(0.1, 0.0, frame_age_ms=0, input_age_ms=0).executed)
        disabled.close()

    def test_explicit_confirmation_and_physical_clamp(self):
        with self.assertRaises(PTZArmError):
            self.executor.arm("yes")
        self.executor.arm("MOVER")
        self.assertTrue(self.move(1.0, -1.0).executed)
        self.assertEqual(self.backend.moves[-1], (0.75, -0.75, 0.0))

    def test_release_bypasses_cooldown(self):
        self.executor.arm("MOVER")
        self.move()
        self.clock.advance(0.01)
        self.move(0, 0)
        self.assertEqual(self.backend.stops, 1)
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
        self.assertTrue(self.executor.armed)

    def test_stale_video_or_input_stops_motion(self):
        self.executor.arm("MOVER")
        for changes in ({"frame_age_ms": 501}, {"frame_age_ms": None},
                        {"input_age_ms": 251}, {"input_age_ms": float("nan")},
                        {"pan": float("nan")}, {"tilt": float("inf")}):
            self.move()
            before = self.backend.stops
            self.assertFalse(self.move(**changes).executed)
            self.assertGreater(self.backend.stops, before)
            self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))

    def test_manual_commands_respect_cooldown(self):
        self.executor.arm("MOVER")
        self.move()
        self.clock.advance(0.05)
        self.assertFalse(self.move().executed)
        self.assertEqual(len(self.backend.moves), 1)
        self.clock.advance(0.16)
        self.assertTrue(self.move().executed)

    def test_deadman_stops_without_new_input(self):
        self.executor.arm("MOVER")
        self.move()
        self.clock.advance(0.41)
        self.executor.check_timeouts()
        self.assertEqual(self.backend.stops, 1)
        self.assertEqual(self.executor.deadman_activations, 1)

    def test_disconnect_enters_fault(self):
        self.executor.arm("MOVER")
        self.move()
        self.backend.connected = False
        self.move()
        self.assertFalse(self.executor.armed)
        self.assertEqual(self.executor.state, PTZExecutorState.FAULT)

    def test_movement_and_stop_failures_keep_fault(self):
        self.executor.arm("MOVER")
        self.backend.fail_move = True
        self.assertFalse(self.move().executed)
        self.assertEqual(self.executor.state, PTZExecutorState.FAULT)
        self.backend.fail_stop = True
        self.executor.emergency_stop()
        self.assertEqual(self.executor.state, PTZExecutorState.FAULT)
        self.assertFalse(self.executor.armed)


class OperatorControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.backend = FakeBackend()
        self.executor = PTZCommandExecutor(
            self.backend, PTZExecutionPolicy(hardware_enabled=True),
            clock=self.clock, start_watchdog=False,
        )
        self.video_age = 0.01
        self.control = OperatorControl(self.executor, lambda: self.video_age, clock=self.clock)
        self.control.heartbeat()
        self.control.preview_seen(self.clock())
        self.control.request("arm", "MOVER")
        self.control.step()

    def tearDown(self):
        self.control.close()

    def auto_sample(self):
        self.control.submit_automatic(
            PTZDecision(DecisionAction.MOVE, 0.3, 0, 0, 0.5, 0, False, True, "test"),
            PTZExecutionContext(TrackState.TRACKED, 0, 0, 0.9, 10),
        )

    def test_manual_wins_over_automatic_decisions(self):
        self.auto_sample()
        self.control.set_manual(-0.1, 0)
        self.control.step()
        self.assertEqual(self.backend.moves, [(-0.1, 0, 0)])
        self.control.set_manual(0, 0)
        self.control.step()
        self.assertEqual(self.backend.stops, 1)

    def test_zoom_uses_manual_path_and_emergency_clears_it(self):
        self.backend.backend_capabilities = PTZCapabilities(continuous_pan_tilt=True, continuous_zoom=True)
        self.control.set_manual(0, 0, zoom=0.2)
        self.control.step()
        self.assertEqual(self.backend.moves[-1], (0, 0, 0.2))
        self.assertEqual(self.control.status.zoom, 0.2)
        self.assertTrue(self.control.status.zoom_supported)
        self.control.request("emergency")
        self.control.step()
        self.assertFalse(self.control.status.armed)
        self.assertEqual(self.control.status.zoom, 0)
        self.assertEqual(self.backend.stops, 1)

    def test_mode_switch_stops_and_requires_rearming(self):
        self.control.set_manual(0.1, 0)
        self.control.step()
        self.control.request("mode", "automatic")
        self.control.step()
        self.assertFalse(self.control.status.armed)
        self.assertEqual(self.control.status.mode, "automatic")
        self.assertEqual(self.backend.stops, 1)
        self.auto_sample()
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.control.request("arm", "MOVER")
        self.control.step()
        self.assertEqual(self.backend.moves[-1], (0.3, 0, 0))

    def test_automatic_evidence_ages_without_vision_updates(self):
        self.control.request("mode", "automatic")
        self.control.step()
        self.control.request("arm", "MOVER")
        self.auto_sample()
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.clock.advance(0.35)
        self.control.heartbeat()
        self.control.step()
        self.assertEqual(self.backend.stops, 1)
        self.assertIn("stale", self.control.status.reason)

    def test_emergency_discards_pending_arm(self):
        self.control.request("arm", "MOVER")
        self.control.request("emergency")
        self.control.step()
        self.assertFalse(self.control.status.armed)
        self.assertEqual(self.backend.moves, [])
        self.assertEqual(self.backend.stops, 1)

    def test_missing_ui_heartbeat_disarms(self):
        self.control.set_manual(0.1, 0)
        self.control.step()
        self.clock.advance(0.6)
        self.control.step()
        self.assertFalse(self.control.status.armed)
        self.assertEqual(self.backend.stops, 1)

    def test_stale_display_blocks_manual_even_with_fresh_stream(self):
        self.clock.advance(0.6)
        self.control.heartbeat()
        self.control.set_manual(0.1, 0)
        self.control.step()
        self.assertEqual(self.backend.moves, [])
        self.assertIn("stale", self.control.status.reason)

    def test_duplicate_arm_does_not_hide_an_active_movement(self):
        self.control.set_manual(0.1, 0)
        self.control.step()
        self.control.request("arm", "MOVER")
        self.control.step()
        self.assertTrue(self.control.status.armed)
        self.assertEqual(self.backend.stops, 1)
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))

    def test_emergency_arriving_during_arm_prevents_auto_movement(self):
        self.control.request("mode", "automatic")
        self.control.step()
        self.auto_sample()
        original_arm = self.executor.arm

        def interrupted_arm(confirmation):
            self.control.request("emergency")
            return original_arm(confirmation)

        self.control.request("arm", "MOVER")
        with patch.object(self.executor, "arm", side_effect=interrupted_arm):
            self.control.step()
        self.assertEqual(self.backend.moves, [])
        self.control.step()
        self.assertFalse(self.control.status.armed)

    def test_simulation_never_constructs_hardware(self):
        control = OperatorControl(None, lambda: 0)
        control.request("arm", "MOVER")
        control.set_manual(0.2, 0)
        control.step()
        self.assertFalse(control.status.hardware)
        self.assertFalse(control.status.armed)
        control.close()


class CameraConnectionTests(unittest.TestCase):
    def test_offline_camera_retries_and_connects_disarmed(self):
        clock = Clock()
        backend = FakeBackend()
        executor = PTZCommandExecutor(
            backend, PTZExecutionPolicy(hardware_enabled=True),
            clock=clock, start_watchdog=False,
        )
        factory = Mock(side_effect=[RuntimeError("offline"), RuntimeError("offline"), executor])
        control = OperatorControl(None, lambda: None, clock=clock, executor_factory=factory)
        try:
            self.assertEqual(control.status.state, "CONNECTING")
            with self.assertLogs("ptz.operator_control", level="INFO") as logs:
                control.step()
                self.assertFalse(control.status.hardware)
                self.assertEqual(control.status.state, "CONNECTING")
                control.step()
                factory.assert_called_once()
                clock.advance(2)
                control.step()
                self.assertEqual(factory.call_count, 2)
                clock.advance(2)
                control.request("arm", "MOVER")
                control.set_manual(0.2, 0)
                control.step()
            self.assertEqual(factory.call_count, 3)
            self.assertTrue(control.status.hardware)
            self.assertFalse(control.status.armed)
            self.assertEqual(control.status.state, "DISARMED")
            self.assertEqual(backend.moves, [])
            self.assertIn("se reintentará en 2.0 s", "\n".join(logs.output))
            self.assertIn("Control PTZ conectado", "\n".join(logs.output))
        finally:
            control.close()
        self.assertTrue(backend.closed)

    def test_closing_during_connection_closes_new_executor(self):
        backend = FakeBackend()
        executor = PTZCommandExecutor(
            backend, PTZExecutionPolicy(hardware_enabled=True), start_watchdog=False,
        )

        def connect():
            control.close()
            return executor

        control = OperatorControl(None, lambda: None, executor_factory=connect)
        control.step()
        self.assertIsNone(control.executor)
        self.assertTrue(backend.closed)
        self.assertFalse(executor.armed)
        self.assertEqual(backend.moves, [])


if __name__ == "__main__":
    unittest.main()
