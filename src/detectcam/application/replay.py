"""Replay determinista de detecciones sin modelo, cámara ni conexiones."""
from dataclasses import asdict
import json
from pathlib import Path

from detectcam.core import Controller, ControlConfig, Detection, Tracker, TrackerConfig


def replay(path: Path, *, tracker_config=None, control_config=None):
    document = json.loads(path.read_text(encoding="utf-8"))
    tracker = Tracker(tracker_config or TrackerConfig())
    controller = Controller(control_config or ControlConfig())
    results = []
    for frame in document["frames"]:
        captured = float(frame["time_s"])
        now = float(frame.get("processed_at_s", captured + .02))
        detections = tuple(Detection(**item) for item in frame.get("detections", []))
        delta = frame.get("camera_delta")
        snapshot = tracker.update(detections, frame_id=frame["frame_id"], captured_at=captured,
            frame_shape=(frame.get("height", 1080), frame.get("width", 1920), 3),
            min_confidence=document.get("min_confidence", .5),
            allowed_classes=document.get("allowed_classes"),
            camera_delta=tuple(delta) if delta is not None else None)
        intent = controller.command(snapshot, now, video_age_s=frame.get("video_age_s", .02))
        results.append({"frame_id": frame["frame_id"], "track_id": snapshot.track_id,
                        "track_phase": snapshot.phase.value, **asdict(intent)})
    return results
