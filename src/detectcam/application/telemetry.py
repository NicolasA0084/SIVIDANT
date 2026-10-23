"""Registro asíncrono acotado: el disco nunca está en el camino de control.

Cada fila describe un evento, no una mezcla de una intención nueva y el estado
de una orden anterior. ``frame_id`` y ``command_id`` permiten correlacionarlos.
Las ventanas de estadísticas y la cola tienen límites para sesiones largas.
"""
from __future__ import annotations

from collections import deque
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from time import monotonic
from typing import Any

LOGGER = logging.getLogger(__name__)


class Telemetry:
    def __init__(self, directory: Path, *, enabled: bool = False,
                 metadata: dict[str, Any] | None = None, capacity: int = 2048) -> None:
        self.enabled = enabled
        self._queue: Queue[dict[str, Any]] = Queue(maxsize=capacity)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=4096)
        self._counts: dict[str, int] = {}
        self.dropped = 0
        self._thread: threading.Thread | None = None
        self.path: Path | None = None
        self._metadata = dict(metadata or {})
        if enabled:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            self.path = directory / stamp
            self._thread = threading.Thread(target=self._write, name="detectcam-metrics", daemon=True)
            self._thread.start()

    def emit(self, event: str, **fields: Any) -> None:
        """Se omite una muestra al saturarse; nunca se detiene visión ni STOP."""
        if not self.enabled or self._stop.is_set():
            return
        record = {"event": event, "time_s": monotonic(), **fields}
        try:
            self._queue.put_nowait(record)
        except Full:
            with self._lock:
                self.dropped += 1

    @property
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = sorted(self._latencies)
            result = {"events": dict(self._counts), "dropped": self.dropped}
        for name, quantile in (("backend_p50_ms", .5), ("backend_p95_ms", .95)):
            result[name] = values[min(len(values)-1, int((len(values)-1)*quantile))] if values else None
        return result

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _write(self) -> None:
        assert self.path is not None
        try:
            self.path.mkdir(parents=True)
            (self.path / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            # CSV has a stable envelope; the payload retains every event's fields.
            with (self.path / "events.jsonl").open("w", encoding="utf-8") as js, \
                    (self.path / "events.csv").open("w", newline="", encoding="utf-8") as cs:
                writer = csv.writer(cs)
                writer.writerow(("time_s", "event", "frame_id", "command_id", "payload_json"))
                pending_flush = 0
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        item = self._queue.get(timeout=.1)
                    except Empty:
                        js.flush()
                        cs.flush()
                        continue
                    payload = json.dumps(item, ensure_ascii=False, default=str, allow_nan=False)
                    js.write(payload + "\n")
                    writer.writerow((item["time_s"], item["event"], item.get("frame_id", ""),
                                     item.get("command_id", ""), payload))
                    with self._lock:
                        event = item["event"]
                        self._counts[event] = self._counts.get(event, 0) + 1
                        latency = item.get("backend_latency_ms")
                        if isinstance(latency, (float, int)) and latency >= 0:
                            self._latencies.append(float(latency))
                    pending_flush += 1
                    if pending_flush >= 30:
                        js.flush()
                        cs.flush()
                        pending_flush = 0
            (self.path / "summary.json").write_text(
                json.dumps(self.snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            LOGGER.exception("No se pudo guardar la telemetría; el control sigue independiente")
