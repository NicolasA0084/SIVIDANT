"""Control invariants tested with synthetic observations and monotonic times."""
from dataclasses import replace
import unittest

from detectcam.core.control import ControlConfig, Controller
from detectcam.core.models import Detection, TrackPhase
from detectcam.core.tracking import Tracker


def observed(x=550., y=240., frame_id=1, time=10.):
    detection = Detection(0, "drone", .9, x - 10., y - 10., x + 10., y + 10.)
    return Tracker().update([detection], frame_id, time, (480, 640, 3))


class ControlCoreTests(unittest.TestCase):
    def test_first_detection_produces_strong_capped_acquisition(self):
        command = Controller().command(observed(), 10.01, .01)
        self.assertAlmostEqual(command.pan, .65)
        self.assertEqual(command.tilt, 0.)
        self.assertEqual(command.zoom, 0.)
        self.assertTrue(command.moving)
        self.assertEqual(command.source_frame_id, 1)
        self.assertLessEqual(command.expires_at, 10.50)

    def test_centered_observation_stops_immediately_without_cooldown(self):
        controller = Controller()
        self.assertTrue(controller.command(observed(), 10., 0.).moving)
        centered = controller.command(observed(x=320., frame_id=2, time=10.01), 10.01, 0.)
        self.assertFalse(centered.moving)
        self.assertEqual(centered.reason, "target centered")

    def test_small_actionable_error_uses_minimum_command_and_hysteresis(self):
        controller = Controller()
        still = controller.command(observed(x=330.), 10., 0.)
        self.assertFalse(still.moving)  # normalized .031 < .04 exit hysteresis
        moved = controller.command(observed(x=334., frame_id=2, time=10.01), 10.01, 0.)
        self.assertAlmostEqual(moved.pan, .07)
        held_motion = controller.command(observed(x=330., frame_id=3, time=10.02), 10.02, 0.)
        self.assertGreater(held_motion.pan, 0.)
        stopped = controller.command(observed(x=327., frame_id=4, time=10.03), 10.03, 0.)
        self.assertFalse(stopped.moving)

    def test_no_blind_feedforward_from_apparent_velocity(self):
        snapshot = replace(observed(x=350.), velocity_x=10000., velocity_y=-10000.,
                           object_velocity_x=10000., camera_compensated=False,
                           consecutive_hits=10)
        expected = Controller().command(observed(x=350.), 10.01, .01)
        actual = Controller().command(snapshot, 10.01, .01)
        self.assertAlmostEqual(actual.pan, expected.pan)
        self.assertAlmostEqual(actual.tilt, expected.tilt)

    def test_compensated_prediction_is_small_and_bounded(self):
        snapshot = replace(observed(x=350.), object_velocity_x=100000.,
                           camera_compensated=True, consecutive_hits=10)
        command = Controller().command(snapshot, 10., 0.)
        self.assertLessEqual(command.error_x, (350. - 320.) / 320. + .08)
        self.assertLessEqual(abs(command.pan), .65)

    def test_old_tracking_snapshot_ages_into_recovery_and_cannot_refresh(self):
        snapshot = observed()
        controller = Controller()
        controller.command(snapshot, 10., 0.)
        recovery = controller.command(snapshot, 10.31, .01)
        self.assertEqual(recovery.phase, TrackPhase.RECOVERING.value)
        self.assertLessEqual(abs(recovery.pan), .35)
        expired = controller.command(snapshot, 10.51, .01)
        self.assertFalse(expired.moving)
        self.assertLessEqual(expired.expires_at, 10.51)

    def test_fresh_empty_frames_do_not_extend_last_seen_recovery_deadline(self):
        controller = Controller()
        original = observed()
        controller.command(original, 10., 0.)
        for index in range(1, 10):
            now = 10. + index * .1
            missing = replace(original, phase=TrackPhase.RECOVERING, frame_id=index + 1,
                              captured_at=now, camera_compensated=True)
            result = controller.command(missing, now, 0.)
        self.assertFalse(result.moving)
        self.assertIn("expired", result.reason)

    def test_blind_recovery_spends_finite_command_budget_across_new_empty_frames(self):
        controller = Controller(ControlConfig(recovery_budget=.025))
        original = observed()
        controller.command(original, 10., 0.)
        integral = 0.
        previous = None
        for index in range(1, 70):
            now = 10. + index * .01
            if previous is not None:
                integral += abs(previous.pan) * max(0., min(now, previous.expires_at) - previous.created_at)
            missing = replace(original, phase=TrackPhase.RECOVERING, frame_id=index + 1,
                              captured_at=now, camera_compensated=False)
            previous = controller.command(missing, now, 0.)
        self.assertLessEqual(integral, .025 + 1e-9)
        self.assertFalse(previous.moving)

    def test_visual_recovery_stops_at_last_point_and_does_not_reverse_search(self):
        controller = Controller()
        original = observed()
        controller.command(original, 10., 0.)
        missing = replace(original, phase=TrackPhase.RECOVERING, frame_id=2,
                          captured_at=10.1, camera_compensated=True, center=(400., 240.))
        self.assertGreater(controller.command(missing, 10.1, 0.).pan, 0.)
        crossed = replace(missing, frame_id=3, captured_at=10.2, center=(300., 240.))
        self.assertFalse(controller.command(crossed, 10.2, 0.).moving)
        drifted = replace(missing, frame_id=4, captured_at=10.3)
        self.assertFalse(controller.command(drifted, 10.3, 0.).moving)

    def test_new_actual_detection_renews_recovery_budget(self):
        controller = Controller(ControlConfig(recovery_budget=.001))
        original = observed()
        first_loss = replace(original, phase=TrackPhase.RECOVERING, frame_id=2, captured_at=10.05)
        controller.command(first_loss, 10.05, 0.)
        self.assertFalse(controller.command(first_loss, 10.1, 0.).moving)
        fresh = observed(frame_id=3, time=10.2)
        self.assertTrue(controller.command(fresh, 10.2, 0.).moving)
        lost_again = replace(fresh, phase=TrackPhase.RECOVERING, frame_id=4, captured_at=10.25)
        self.assertTrue(controller.command(lost_again, 10.25, 0.).moving)

    def test_older_snapshot_cannot_resurrect_a_replaced_identity(self):
        controller = Controller()
        old = observed(frame_id=1, time=10.)
        new = replace(observed(x=100., frame_id=3, time=10.1), track_id=2)
        controller.command(old, 10., 0.)
        self.assertLess(controller.command(new, 10.1, 0.).pan, 0.)
        command = controller.command(old, 10.11, 0.)
        self.assertFalse(command.moving)
        self.assertIn("out-of-order", command.reason)

    def test_same_source_id_cannot_be_given_a_new_acquisition_time(self):
        controller = Controller()
        original = observed()
        controller.command(original, 10., 0.)
        forged = replace(original, captured_at=10.4, last_seen_at=10.4)
        result = controller.command(forged, 10.4, 0.)
        self.assertFalse(result.moving)
        self.assertIn("timestamp changed", result.reason)

    def test_absent_stale_future_or_unconfirmed_evidence_holds(self):
        cases = [
            (observed(), 10., None),
            (observed(), 10.01, .51),
            (observed(), 10.51, .01),
            (observed(time=10.1), 10., 0.),
            (replace(observed(), confirmed=False, phase=TrackPhase.ACQUIRING), 10., 0.),
        ]
        for snapshot, now, video_age in cases:
            with self.subTest(now=now, video_age=video_age, phase=snapshot.phase):
                self.assertFalse(Controller().command(snapshot, now, video_age).moving)

    def test_configuration_rejects_nonfinite_and_impossible_limits(self):
        for kwargs in ({"max_pan": float("nan")}, {"max_frame_age_s": 0.},
                       {"min_command": .7}, {"recovery_budget": -1.}):
            with self.assertRaises(ValueError):
                ControlConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
