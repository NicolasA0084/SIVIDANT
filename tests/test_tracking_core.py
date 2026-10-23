"""Deterministic perception tests; no model, network, or camera is loaded."""
import unittest

from detectcam.core.models import Detection, FrameObservation, TrackPhase
from detectcam.core.tracking import Tracker, TrackerConfig


SHAPE = (480, 640, 3)


def box(x=450., y=220., confidence=.9, class_id=0, name="drone"):
    return Detection(class_id, name, confidence, x - 20., y - 15., x + 20., y + 15.)


class TrackingCoreTests(unittest.TestCase):
    def test_first_valid_detection_acquires_without_three_frame_delay(self):
        tracker = Tracker()
        result = tracker.update([box()], 1, 10., SHAPE)
        self.assertTrue(result.confirmed)
        self.assertEqual(result.phase, TrackPhase.TRACKING)
        self.assertEqual(result.last_seen_at, 10.)
        self.assertEqual(result.captured_at, 10.)

    def test_optional_confirmation_requires_consecutive_observations(self):
        tracker = Tracker(TrackerConfig(acquisition_mode="confirm", confirmation_hits=2))
        first = tracker.update([box()], 1, 10., SHAPE)
        self.assertFalse(first.confirmed)
        tracker.update([], 2, 10.05, SHAPE)
        third = tracker.update([box()], 3, 10.10, SHAPE)
        self.assertFalse(third.confirmed)
        fourth = tracker.update([box()], 4, 10.15, SHAPE)
        self.assertTrue(fourth.confirmed)
        self.assertEqual(first.track_id, fourth.track_id)

    def test_velocity_uses_actual_acquisition_interval(self):
        tracker = Tracker()
        tracker.update([box(x=200.)], 1, 10., SHAPE)
        result = tracker.update([box(x=220.)], 2, 10.10, SHAPE)
        self.assertAlmostEqual(result.velocity_x, 200.)
        self.assertAlmostEqual(result.velocity_y, 0.)
        self.assertEqual(result.last_seen_at, 10.10)

    def test_prediction_associates_target_after_velocity_established(self):
        tracker = Tracker(TrackerConfig(association_distance=.04, min_iou=.10))
        first = tracker.update([box(x=200.)], 1, 10., SHAPE)
        tracker.update([box(x=230.)], 2, 10.10, SHAPE)
        result = tracker.update([box(x=275.)], 3, 10.25, SHAPE)
        self.assertEqual(result.track_id, first.track_id)

    def test_any_eligible_replacement_supersedes_unmatched_target_immediately(self):
        tracker = Tracker()
        first = tracker.update([box(x=100.)], 1, 10., SHAPE)
        result = tracker.update([box(x=570., class_id=1, name="bird")], 2, 10.05, SHAPE)
        self.assertNotEqual(first.track_id, result.track_id)
        self.assertEqual(result.target.class_name, "bird")
        self.assertTrue(result.confirmed)

    def test_loss_expires_by_elapsed_time_regardless_of_frame_count(self):
        for count in (2, 20):
            tracker = Tracker()
            tracker.update([box()], 0, 10., SHAPE)
            for index in range(1, count + 1):
                result = tracker.update([], index, 10. + .9 * index / count, SHAPE)
            self.assertEqual(result.phase, TrackPhase.EXPIRED)
            self.assertEqual(result.last_seen_at, 10.)
            self.assertAlmostEqual(result.recovery_deadline, 10.85)

    def test_background_shift_transports_last_observed_point_during_loss(self):
        tracker = Tracker()
        tracker.update([box(x=450.)], 1, 10., SHAPE)
        lost = tracker.update([], 2, 10.05, SHAPE, camera_delta=(-40., 3.))
        self.assertEqual(lost.center, (410., 223.))
        self.assertTrue(lost.camera_compensated)
        next_lost = tracker.update([], 3, 10.10, SHAPE, camera_delta=(-30., 2.))
        self.assertEqual(next_lost.center, (380., 225.))
        predicted = tracker.predict(10.3)
        self.assertEqual(predicted.center, next_lost.center)
        self.assertEqual(predicted.captured_at, 10.10)

    def test_compensation_removes_camera_motion_from_object_velocity(self):
        tracker = Tracker()
        tracker.update([box(x=400.)], 1, 10., SHAPE)
        result = tracker.update([box(x=370.)], 2, 10.1, SHAPE, camera_delta=(-30., 0.))
        self.assertAlmostEqual(result.velocity_x, -300.)
        self.assertAlmostEqual(result.object_velocity_x, 0.)
        self.assertTrue(result.camera_compensated)

    def test_missing_background_evidence_breaks_recovery_compensation(self):
        tracker = Tracker()
        tracker.update([box()], 1, 10., SHAPE)
        tracker.update([], 2, 10.05, SHAPE, camera_delta=(-10., 0.))
        result = tracker.update([], 3, 10.1, SHAPE, camera_delta=None)
        self.assertFalse(result.camera_compensated)
        self.assertEqual(result.center, (440., 220.))

    def test_old_frames_do_not_create_evidence_or_renew_recovery(self):
        tracker = Tracker()
        current = tracker.update([box()], 2, 10., SHAPE)
        self.assertIs(tracker.update([box(x=500.)], 2, 10.1, SHAPE), current)
        self.assertIs(tracker.update([box(x=500.)], 1, 10.1, SHAPE), current)
        self.assertIs(tracker.update([box(x=500.)], 3, 9.9, SHAPE), current)
        expired = tracker.predict(11.)
        self.assertEqual(expired.phase, TrackPhase.EXPIRED)
        self.assertEqual(expired.frame_id, 2)
        self.assertEqual(expired.captured_at, 10.)

    def test_filter_changes_drop_previous_target_and_do_not_accept_invalid_boxes(self):
        tracker = Tracker()
        first = tracker.update([box()], 1, 10., SHAPE)
        empty = tracker.update([box(), box(confidence=float("nan"))], 2, 10.1, SHAPE,
                               allowed_classes=[])
        self.assertIsNone(empty.detection)
        next_target = tracker.update([box(confidence=.8)], 3, 10.2, SHAPE,
                                     min_confidence=.75, allowed_classes=["DRONE"])
        self.assertNotEqual(next_target.track_id, first.track_id)

    def test_frame_observation_api_and_configuration_validation(self):
        tracker = Tracker()
        result = tracker.observe(FrameObservation(1, 10., SHAPE, (box(),)))
        self.assertEqual(result.target.center, result.center)
        for overrides in ({"recovery_s": float("nan")}, {"forget_s": .1},
                          {"acquisition_mode": "unknown"}, {"confirmation_hits": 1.5}):
            with self.assertRaises(ValueError):
                TrackerConfig(**overrides)


if __name__ == "__main__":
    unittest.main()
