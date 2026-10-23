"""Visión: único propietario del tracker, sin acceso al transporte PTZ.

El receptor comparte el último ndarray como sólo lectura. Las detecciones se
fechan con received_at del cuadro original, incluso si YOLO tarda. El operador
recibe valores inmutables y mantiene su propio reloj de control a 50 Hz.
"""
from __future__ import annotations

from collections import deque
import logging
import threading
from time import monotonic

import numpy as np

from detectcam.adapters.camera_motion import CameraMotionEstimator
from detectcam.adapters.detector import Detection as DisplayDetection, InferenceError
from detectcam.core.models import Detection
from detectcam.presentation.dashboard import DashboardFrame
from detectcam.presentation.preview import PreviewAnnotations
from detectcam.presentation.view_models import TrackedTarget, TrackState

LOGGER = logging.getLogger(__name__)


class VisionPipeline:
    def __init__(self, stream, detector, tracker, control, mailbox, tracking_options,
                 telemetry, *, dead_zone=.025, max_frame_age_s=.5, warmup=True, clock=monotonic):
        self.stream, self.detector, self.tracker = stream, detector, tracker
        self.control, self.mailbox = control, mailbox
        self.options, self.telemetry = tracking_options, telemetry
        self.dead_zone = dead_zone
        self.max_frame_age_s = max_frame_age_s
        self.warmup = warmup
        self.clock = clock
        self._previous_id = -1
        self._revision = None
        self._ticks = deque(maxlen=60)
        self._motion = CameraMotionEstimator()
        self._last_publish = float("-inf")
        self._last_annotations = None
        self.failed = False

    def step(self) -> bool:
        settings = self.options.snapshot
        if settings.revision != self._revision:
            self.tracker.reset()
            self._motion.reset()
            self._revision = settings.revision
            # Mantener detecciones de la confianza elegida en UI, sin que el
            # umbral de inferencia anterior oculte las más débiles.
            self.detector.set_confidence(min(self.detector.confidence, settings.min_confidence))
        packet = self.stream.read_packet(copy=False, after_id=self._previous_id)
        now = self.clock()
        if packet is None:
            snapshot = self.tracker.predict(now)
            self.control.submit_snapshot(snapshot, settings_revision=settings.revision)
            if now-self._last_publish >= .1:
                self._publish(None, snapshot, (), settings.revision)
            return False
        self._previous_id = packet.frame_id
        if now-packet.received_at > self.max_frame_age_s:
            snapshot = self.tracker.predict(now)
            self.control.submit_snapshot(snapshot, settings_revision=settings.revision)
            self.telemetry.emit("frame_rejected", frame_id=packet.frame_id, reason="stale before inference")
            return False
        try:
            detections = self.detector.detect(packet.frame)
        except InferenceError:
            LOGGER.exception("Inferencia omitida; recuperación temporal permanece acotada")
            detections = []
        self._ticks.append(self.clock())
        # Optical flow uses the captured pixels, not positions inferred from
        # commanded velocity: unsupported movement never becomes fake odometry.
        background = self._motion.update(packet.frame, captured_at=packet.received_at, boxes=detections)
        snapshot = self.tracker.update(
            tuple(Detection(d.class_id, d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2) for d in detections),
            frame_id=packet.frame_id, captured_at=packet.received_at, frame_shape=packet.frame.shape,
            min_confidence=settings.min_confidence, allowed_classes=settings.allowed_classes,
            camera_delta=(background.dx, background.dy) if background.reliable else None,
        )
        self.control.submit_snapshot(snapshot, settings_revision=settings.revision)
        self.telemetry.emit("observation", frame_id=packet.frame_id, captured_at=packet.received_at,
            processed_at=self.clock(), inference_ms=self.detector.last_inference_ms,
            frame_age_ms=(self.clock()-packet.received_at)*1000,
            track_id=snapshot.track_id, phase=snapshot.phase.value,
            confidence=None if snapshot.detection is None else snapshot.detection.confidence,
            background_reliable=background.reliable, settings_revision=settings.revision)
        self._publish(packet, snapshot, detections, settings.revision)
        return True

    def _publish(self, packet, snapshot, detections, revision):
        now = self.clock()
        self._last_publish = now
        phase = snapshot.phase.value
        actual = snapshot.detection
        target = None
        state = TrackState.NO_TARGET
        if actual is not None and snapshot.center is not None and phase not in {"IDLE", "EXPIRED"}:
            cx, cy = snapshot.center
            half_w, half_h = actual.width/2, actual.height/2
            box = DisplayDetection(actual.class_id, actual.class_name, actual.confidence,
                round(cx-half_w), round(cy-half_h), round(cx+half_w), round(cy+half_h))
            state = TrackState.TEMPORARILY_LOST if phase == "RECOVERING" else TrackState.TRACKED
            target = TrackedTarget(snapshot.track_id or 0, box, 0, snapshot.consecutive_hits,
                1 if state == TrackState.TEMPORARILY_LOST else 0, snapshot.last_seen_at or 0.,
                previous_center=None, velocity_x=snapshot.velocity_x, velocity_y=snapshot.velocity_y)
        fps = 0.
        if len(self._ticks) >= 2 and self._ticks[-1] > self._ticks[0]:
            fps = (len(self._ticks)-1)/(self._ticks[-1]-self._ticks[0])
        frame_age = None if snapshot.frame_id < 0 else max(0., now-snapshot.captured_at)
        seen_age = None if snapshot.last_seen_at is None else max(0., now-snapshot.last_seen_at)
        status, intent = self.control.status, self.control.last_intent
        telemetry = (
            f"Video: {'CONNECTED' if self.stream.is_connected() else 'DISCONNECTED'}",
            f"Cuadro: {snapshot.frame_id} · revisión: {revision}",
            f"Inferencia completa: {self.detector.last_inference_ms or 0.:.1f} ms",
            f"FPS detección: {fps:.1f}", f"Detecciones: {len(detections)}",
            f"Seguimiento: {snapshot.track_id} · {phase}",
            f"Objetivo: {'NONE' if actual is None else actual.class_name}",
            f"Confianza: {'—' if actual is None else f'{actual.confidence:.2f}'}",
            f"Velocidad aparente: ({snapshot.velocity_x:+.1f}, {snapshot.velocity_y:+.1f}) px/s",
            f"Compensación de fondo: {'válida' if snapshot.camera_compensated else 'no disponible'}",
            f"Edad cuadro procesado: {'—' if frame_age is None else f'{frame_age*1000:.0f} ms'}",
            f"Desde última detección: {'—' if seen_age is None else f'{seen_age*1000:.0f} ms'}",
        )
        runtime = self.control.executor.snapshot if self.control.executor else None
        metrics = self.telemetry.snapshot
        ptz = (
            f"Modo: {status.mode} · {status.state} · armado: {status.armed}",
            f"Fase de control: {status.phase}",
            f"Motivo: {status.reason}",
            f"Intención: {'esperando' if intent is None else intent.reason}",
            f"Error X/Y: {'—' if intent is None else f'{intent.error_x:+.3f} / {intent.error_y:+.3f}'}",
            f"Orden pan/tilt: {'—' if intent is None else f'{intent.pan:+.3f} / {intent.tilt:+.3f}'}",
            f"Orden despachada P/T/Z: {status.pan:+.3f} / {status.tilt:+.3f} / {status.zoom:+.3f}",
            f"Comando aceptado: {status.command_id} · último ACK: {getattr(runtime, 'acknowledged_command_id', '—')}",
            f"Cuadro origen: {getattr(runtime, 'frame_id', '—')}",
            f"Backend: {getattr(runtime, 'backend', 'simulación')}",
            f"Latencia petición: {status.backend_latency_ms if status.backend_latency_ms is not None else '—'} ms",
            f"Parada pendiente/confirmada: {getattr(runtime, 'stop_pending', False)} / {getattr(runtime, 'stop_confirmed', False)}",
            "Confirmación de protocolo; posición mecánica no medida",
            f"Eventos: {metrics['events']} · muestras omitidas: {metrics['dropped']}",
            f"Latencia backend p50/p95: {metrics['backend_p50_ms']} / {metrics['backend_p95_ms']} ms",
        )
        if packet is not None:
            self._last_annotations = PreviewAnnotations(
                packet.frame.shape, packet.received_at, tuple(detections), target, state,
                (self.dead_zone, self.dead_zone))
        annotations = self._last_annotations
        self.mailbox.publish(DashboardFrame(
            None if packet is None else packet.frame, telemetry, ptz, self.stream.is_connected(),
            fps=fps, detections=len(detections),
            captured_at=None if packet is None else packet.received_at, annotations=annotations))

    def run(self, stop_event: threading.Event) -> None:
        try:
            if self.warmup:
                LOGGER.info("Calentando inferencia con imagen sintética antes de consumir video")
                self.detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
            while not stop_event.is_set():
                if not self.step():
                    stop_event.wait(.02)
        except Exception:
            LOGGER.exception("Se detuvo el pipeline de visión")
            self.failed = True
            stop_event.set()
            self.control.request("emergency")
