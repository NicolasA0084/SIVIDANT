import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from detectcam.application.cli import main
from detectcam.application.replay import replay
from detectcam.application.telemetry import Telemetry
from detectcam.config.settings import PROJECT_ROOT


class ReplayTests(unittest.TestCase):
    def test_loss_replacement_and_expiry_from_documented_replay(self):
        rows = replay(PROJECT_ROOT / "docs/replay-ejemplo.json")
        self.assertGreater(rows[0]["pan"], .3)
        self.assertEqual(rows[2]["phase"], "RECOVERING")
        self.assertGreater(rows[2]["pan"], 0)
        self.assertNotEqual(rows[0]["track_id"], rows[4]["track_id"])
        self.assertLess(rows[4]["pan"], 0)
        self.assertEqual(rows[-1]["pan"], 0)
        self.assertEqual(rows[-1]["phase"], "HOLD")

    def test_replay_cli_does_not_need_camera_configuration(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--replay", str(PROJECT_ROOT / "docs/replay-ejemplo.json")])
        self.assertEqual(result, 0)
        self.assertEqual(len(json.loads(output.getvalue())), 7)


class TelemetryTests(unittest.TestCase):
    def test_correlated_events_are_persisted_and_latencies_use_window(self):
        with TemporaryDirectory() as temporary:
            sink = Telemetry(Path(temporary), enabled=True, metadata={"schema": 2})
            sink.emit("observation", frame_id=7, captured_at=10.)
            sink.emit("completed", command_id=2, frame_id=7, backend_latency_ms=12.)
            sink.emit("completed", command_id=3, frame_id=8, backend_latency_ms=20.)
            sink.close()
            rows = [json.loads(line) for line in (sink.path / "events.jsonl").read_text().splitlines()]
            self.assertEqual(rows[1]["command_id"], 2)
            self.assertEqual(rows[1]["frame_id"], 7)
            self.assertEqual(sink.snapshot["events"]["completed"], 2)
            self.assertEqual(sink.snapshot["backend_p50_ms"], 12.)
            self.assertTrue((sink.path / "events.csv").is_file())
            self.assertTrue((sink.path / "summary.json").is_file())

    def test_disabled_sink_creates_no_files(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary) / "not-created"
            sink = Telemetry(directory)
            sink.emit("anything")
            sink.close()
            self.assertFalse(directory.exists())
