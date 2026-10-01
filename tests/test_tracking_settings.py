from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import threading
from time import monotonic
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from ptz.detector import Detection, YOLODetector
from ptz.main import run_detection_loop
from ptz.operator_control import OperatorControl
from ptz.ptz_backend import PTZCapabilities
from ptz.ptz_decision import DecisionAction, PTZDecision, PTZDecisionEngine
from ptz.ptz_execution_policy import PTZExecutionContext, PTZExecutionPolicy
from ptz.ptz_executor import PTZCommandExecutor
from ptz.ptz_metrics import PTZMetrics
from ptz.stream import ReceivedFrame
from ptz.target_selector import TargetSelector
from ptz.target_tracker import TargetTracker, TrackState
from ptz.tracking_settings import TrackingOptions


CLASSES = {0: "Drone", 1: "Bird", 2: "Person"}
FRAME = np.zeros((360, 640, 3), dtype=np.uint8)


def detection(name="Drone", confidence=0.9, x=460):
    class_id = next(key for key, value in CLASSES.items() if value == name)
    return Detection(class_id, name, confidence, x, 140, x + 60, 200)


class FakeModel:
    names = CLASSES

    def __init__(self, batches):
        self.batches = iter(batches)
        self.confidences = []

    def predict(self, *, source, conf, **_kwargs):
        self.confidences.append(conf)
        boxes = [SimpleNamespace(
            cls=np.array([item.class_id]), conf=np.array([item.confidence]),
            xyxy=np.array([[item.x1, item.y1, item.x2, item.y2]]),
        ) for item in next(self.batches) if item.confidence >= conf]
        return [SimpleNamespace(boxes=boxes)]


class TrackingOptionsTests(unittest.TestCase):
    def test_initial_selection_and_copied_class_names(self):
        names = dict(CLASSES)
        options = TrackingOptions(names)
        names.clear()
        options.class_names.clear()
        self.assertEqual(options.class_names, CLASSES)
        self.assertEqual(options.snapshot.allowed_classes, {"drone", "bird", "person"})
        self.assertEqual(options.snapshot.min_confidence, 0.5)

    def test_snapshots_are_immutable_and_revisions_change_only_with_settings(self):
        options = TrackingOptions(CLASSES)
        original = options.snapshot
        updated = options.update(min_confidence=0.3, allowed_classes=[" DRONE ", "Bird"])
        self.assertEqual(updated.allowed_classes, {"drone", "bird"})
        self.assertEqual(updated.revision, original.revision + 1)
        self.assertEqual(original.min_confidence, 0.5)
        self.assertEqual(original.allowed_classes, {"drone", "bird", "person"})
        self.assertIs(options.update(min_confidence=0.3, allowed_classes=["bird", "drone"]), updated)
        with self.assertRaises(FrozenInstanceError):
            updated.min_confidence = 0.7

    def test_invalid_confidence_or_classes_leave_previous_snapshot_intact(self):
        options = TrackingOptions(CLASSES)
        original = options.snapshot
        for value in (-0.01, 1.01, float("nan"), float("inf"), "invalid"):
            with self.subTest(confidence=value):
                with self.assertRaises(ValueError):
                    options.update(min_confidence=value, allowed_classes=["drone"])
                self.assertIs(options.snapshot, original)
        with self.assertRaises(ValueError):
            options.update(min_confidence=0.5, allowed_classes=["unknown"])
        self.assertIs(options.snapshot, original)

    def test_no_selection_and_confidence_endpoints_are_valid(self):
        options = TrackingOptions(CLASSES, min_confidence=0.0, allowed_classes=[])
        self.assertEqual(options.snapshot.allowed_classes, frozenset())
        result = options.update(min_confidence=1.0, allowed_classes=[])
        self.assertEqual(result.min_confidence, 1.0)
        self.assertFalse(result.allowed_classes)


class TrackingPipelineTests(unittest.TestCase):
    def run_frames(self, options, batches, *, after_frame=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        model_path = Path(temporary.name) / "fake.pt"
        model_path.touch()
        model = FakeModel(batches)
        detector = YOLODetector(model_path, model_factory=lambda _path: model)
        stream = Mock()
        stream.is_connected.return_value = True
        stream.last_frame_age = 0.0
        stream.read_packet.side_effect = lambda **kwargs: ReceivedFrame(
            kwargs["after_id"] + 1, FRAME, monotonic(),
        )
        selector = TargetSelector()
        tracker = TargetTracker(allow_class_change=True)
        engine = PTZDecisionEngine(min_visible_frames=1, command_interval_ms=0)
        metrics = PTZMetrics(enabled=False)
        self.addCleanup(metrics.close)
        control = OperatorControl(None, lambda: 0.0, tracking_options=options)
        self.addCleanup(control.close)
        snapshots = []
        stop = threading.Event()

        def publish(snapshot):
            snapshots.append(snapshot)
            if after_frame is not None:
                after_frame(len(snapshots))
            if len(snapshots) == len(batches):
                stop.set()

        mailbox = Mock()
        mailbox.publish.side_effect = publish
        with patch.object(control, "submit_automatic", wraps=control.submit_automatic) as submit:
            run_detection_loop(
                stream, detector, selector, tracker, engine, metrics, control,
                mailbox, stop, tracking_options=options,
            )
        return SimpleNamespace(
            snapshots=snapshots, model=model, detector=detector,
            selector=selector, engine=engine, submissions=submit.call_args_list,
        )

    def test_confidence_below_detector_default_reaches_selection_and_motion(self):
        options = TrackingOptions(CLASSES, min_confidence=0.1, allowed_classes=["drone"])
        result = self.run_frames(options, [[detection(confidence=0.18)]])
        self.assertEqual(result.model.confidences, [0.1])
        self.assertEqual(result.selector.min_confidence, 0.1)
        self.assertEqual(result.engine.min_track_confidence, 0.1)
        target = result.snapshots[0].annotations.target
        self.assertIsNotNone(target)
        self.assertEqual(target.detection.confidence, 0.18)
        decision, context = result.submissions[0].args
        self.assertTrue(decision.safe_to_move)
        self.assertEqual(decision.action, DecisionAction.MOVE)
        self.assertEqual(context.confidence, 0.18)
        self.assertEqual(result.submissions[0].kwargs["settings_revision"], 0)

    def test_class_filter_applies_to_acquisition_and_existing_track_association(self):
        options = TrackingOptions(CLASSES, allowed_classes=["drone"])
        result = self.run_frames(options, [
            [detection("Bird", 0.99), detection("Drone", 0.6)],
            [detection("Bird", 0.99), detection("Drone", 0.7, x=465)],
        ])
        first, second = [snapshot.annotations.target for snapshot in result.snapshots]
        self.assertEqual(first.detection.class_name, "Drone")
        self.assertEqual(second.detection.class_name, "Drone")
        self.assertEqual(second.detection.confidence, 0.7)
        self.assertEqual(first.track_id, second.track_id)
        self.assertEqual(second.visible_frames, 2)

    def test_multiple_selected_classes_compete_without_unselected_classes(self):
        options = TrackingOptions(CLASSES, allowed_classes=["drone", "bird"])
        result = self.run_frames(options, [[
            detection("Drone", 0.6), detection("Bird", 0.8), detection("Person", 0.99),
        ]])
        self.assertEqual(result.snapshots[0].annotations.target.detection.class_name, "Bird")

    def test_low_confidence_association_and_subsequent_loss_both_hold(self):
        options = TrackingOptions(CLASSES, min_confidence=0.8, allowed_classes=["drone"])
        result = self.run_frames(options, [[detection()], [detection(confidence=0.3)], []])
        self.assertTrue(result.submissions[0].args[0].safe_to_move)
        associated = result.snapshots[1].annotations.target
        self.assertEqual(associated.detection.confidence, 0.3)
        self.assertEqual(associated.visible_frames, 2)
        for submission, expected_state in zip(result.submissions[1:], (
            TrackState.TRACKED, TrackState.TEMPORARILY_LOST,
        )):
            decision, context = submission.args
            self.assertEqual(context.tracker_state, expected_state)
            self.assertEqual(decision.action, DecisionAction.HOLD)
            self.assertFalse(decision.safe_to_move)
            self.assertIn("confidence", decision.reason)

    def test_raising_confidence_resets_current_target_before_association(self):
        options = TrackingOptions(CLASSES, allowed_classes=["drone"])

        def update(count):
            if count == 1:
                options.update(min_confidence=0.8, allowed_classes=["drone"])

        result = self.run_frames(options, [[detection(confidence=0.6)]] * 2, after_frame=update)
        self.assertIsNotNone(result.snapshots[0].annotations.target)
        self.assertIsNone(result.snapshots[1].annotations.target)
        self.assertEqual(result.submissions[1].args[0].action, DecisionAction.HOLD)
        self.assertEqual(result.submissions[1].kwargs["settings_revision"], 1)

    def test_class_change_starts_new_track_and_no_selection_clears_it(self):
        options = TrackingOptions(CLASSES, allowed_classes=["drone"])

        def update(count):
            if count == 1:
                options.update(min_confidence=0.5, allowed_classes=["bird"])
            elif count == 2:
                options.update(min_confidence=0.5, allowed_classes=[])

        result = self.run_frames(options, [[detection(), detection("Bird", 0.8)]] * 3,
                                 after_frame=update)
        first, second, third = [snapshot.annotations.target for snapshot in result.snapshots]
        self.assertEqual(first.detection.class_name, "Drone")
        self.assertEqual(second.detection.class_name, "Bird")
        self.assertNotEqual(first.track_id, second.track_id)
        self.assertEqual(second.visible_frames, 1)
        self.assertIsNone(third)
        self.assertFalse(result.submissions[-1].args[0].safe_to_move)


class Clock:
    def __call__(self):
        return 10.0


class FakeBackend:
    backend_capabilities = PTZCapabilities(continuous_pan_tilt=True)

    def __init__(self):
        self.moves = []
        self.stops = 0

    def is_connected(self):
        return True

    def move_velocity(self, pan, tilt, zoom=0.0):
        self.moves.append((pan, tilt, zoom))

    def stop(self):
        self.stops += 1

    def close(self):
        pass


class TrackingControlTests(unittest.TestCase):
    def setUp(self):
        self.options = TrackingOptions(CLASSES, allowed_classes=["drone"])
        self.clock = Clock()
        self.backend = FakeBackend()
        self.executor = PTZCommandExecutor(
            self.backend, PTZExecutionPolicy(hardware_enabled=True, command_interval_ms=0),
            clock=self.clock, start_watchdog=False,
        )
        self.control = OperatorControl(
            self.executor, lambda: 0.0, clock=self.clock, tracking_options=self.options,
        )
        self.addCleanup(self.control.close)
        self.control.heartbeat()
        self.control.preview_seen(self.clock())
        self.control.request("mode", "automatic")
        self.control.step()
        self.control.request("arm", "MOVER")

    def submit(self, confidence=0.9, *, revision=None):
        self.control.submit_automatic(
            PTZDecision(DecisionAction.MOVE, 0.3, 0, 0, 0.5, 0, False, True, "test"),
            PTZExecutionContext(TrackState.TRACKED, 0, 0, confidence, 10),
            settings_revision=self.options.snapshot.revision if revision is None else revision,
        )

    def test_new_settings_stop_motion_and_reject_late_previous_revision(self):
        self.submit()
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        previous_revision = self.options.snapshot.revision
        self.options.update(min_confidence=0.5, allowed_classes=["bird"])
        self.control.step()
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
        self.assertTrue(self.control.status.armed)
        self.assertGreater(self.backend.stops, 0)
        self.submit(revision=previous_revision)
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.submit()
        self.control.step()
        self.assertEqual(len(self.backend.moves), 2)

    def test_live_confidence_updates_reach_physical_execution_policy(self):
        self.submit(confidence=0.3)
        self.control.step()
        self.assertEqual(self.backend.moves, [])
        self.options.update(min_confidence=0.2, allowed_classes=["drone"])
        self.submit(confidence=0.3)
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.options.update(min_confidence=0.8, allowed_classes=["drone"])
        self.submit(confidence=0.7)
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
        self.assertIn("confidence", self.control.status.reason)
        self.assertTrue(self.control.status.armed)

    def test_empty_selection_blocks_even_new_revision_automatic_decisions(self):
        self.submit()
        self.control.step()
        self.options.update(min_confidence=0.0, allowed_classes=[])
        self.submit(confidence=1.0)
        self.control.step()
        self.assertEqual(len(self.backend.moves), 1)
        self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
        self.assertTrue(self.control.status.armed)

    def test_settings_stop_armed_motion_while_detector_is_pending(self):
        self.submit()
        self.control.step()
        entered, release, published, stop = [threading.Event() for _ in range(4)]
        detector = Mock(confidence=0.25, last_inference_ms=0.0)

        def detect(_frame):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("The test did not release inference")
            return [detection()]

        detector.detect.side_effect = detect
        stream = Mock(last_frame_age=0.0)
        stream.is_connected.return_value = True
        stream.read_packet.side_effect = lambda **_kwargs: ReceivedFrame(1, FRAME, monotonic())
        mailbox = Mock()

        def publish(_snapshot):
            published.set()
            stop.set()

        mailbox.publish.side_effect = publish
        metrics = PTZMetrics(enabled=False)
        self.addCleanup(metrics.close)
        failures = []

        def run():
            try:
                run_detection_loop(
                    stream, detector, TargetSelector(), TargetTracker(),
                    PTZDecisionEngine(min_visible_frames=1), metrics,
                    self.control, mailbox, stop, tracking_options=self.options,
                )
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            self.options.update(min_confidence=0.5, allowed_classes=["bird"])
            self.control.step()
            self.assertTrue(worker.is_alive())
            self.assertEqual(self.executor.last_physical_velocity, (0, 0, 0))
            self.assertTrue(self.control.status.armed)
            release.set()
            self.assertTrue(published.wait(2))
            self.control.step()
            self.assertEqual(len(self.backend.moves), 1)
            self.assertTrue(self.control.status.armed)
        finally:
            stop.set()
            release.set()
            worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
