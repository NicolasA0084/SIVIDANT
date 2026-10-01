"""Orquesta video, detección, tracking y control PTZ seguro.

El proceso arranca siempre desarmado. La política y el ejecutor son las únicas
capas autorizadas para convertir una decisión visual en movimiento físico.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from math import isfinite
from time import monotonic, perf_counter
import threading
from typing import Callable, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from ptz.detector import Detection, DetectorError, InferenceError, YOLODetector
from ptz.preview import PreviewAnnotations
from ptz.stream import RTSPStream, StreamConfigurationError
from ptz.ptz_decision import (
    PTZDecision,
    PTZDecisionConfigurationError,
    PTZDecisionEngine,
)
from ptz.ptz_execution_policy import (
    PTZExecutionContext,
    PTZExecutionPolicy,
    PTZExecutionPolicyConfigurationError,
)
from ptz.ptz_metrics import (
    PTZMetricSample,
    PTZMetrics,
    PTZMetricsConfigurationError,
    PTZMetricsSnapshot,
)
from ptz.ptz_tuning import (
    PTZTuningConfiguration,
    PTZTuningConfigurationError,
)
from ptz.target_selector import TargetSelector, TargetSelectorConfigurationError
from ptz.tracking_settings import TrackingOptions
from ptz.target_tracker import (
    TargetTracker,
    TargetTrackerConfigurationError,
    TrackedTarget,
    TrackState,
)

LOGGER = logging.getLogger(__name__)
IDLE_DELAY = 0.02


class PTZRuntime(Protocol):
    """Superficie mínima usada por main sin importar el ejecutor al arrancar."""

    @property
    def state(self) -> object: ...
    @property
    def armed(self) -> bool: ...
    @property
    def last_physical_velocity(self) -> tuple[float, float, float]: ...
    @property
    def heartbeat_age_ms(self) -> float | None: ...
    @property
    def deadman_active(self) -> bool: ...
    @property
    def last_backend_latency_ms(self) -> float | None: ...
    @property
    def last_command_interval_ms(self) -> float | None: ...
    @property
    def deadman_activations(self) -> int: ...
    @property
    def watchdog_activations(self) -> int: ...
    @property
    def stop_commands(self) -> int: ...
    def execute_manual(self, pan: float, tilt: float, *, frame_age_ms: float | None,
                       input_age_ms: float | None) -> object: ...
    def arm(self, confirmation: str) -> object: ...
    def disarm(self) -> object: ...
    def emergency_stop(self) -> object: ...
    def execute(self, decision: PTZDecision, context: PTZExecutionContext,
                *, timestamp: float | None = None) -> object: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class PTZRuntimeStatus:
    hardware: str
    executor_state: str
    armed: bool
    physical_pan: float
    physical_tilt: float
    heartbeat_age_ms: float | None
    deadman_active: bool
    backend_latency_ms: float | None
    command_interval_ms: float | None
    deadman_activations: int
    watchdog_activations: int
    stop_commands: int
    execution_reason: str


class DetectionRate:
    """Calcula FPS recientes a partir de ciclos de inferencia terminados."""

    def __init__(self, samples: int = 60) -> None:
        self._timestamps: deque[float] = deque(maxlen=samples)

    def tick(self) -> None:
        self._timestamps.append(perf_counter())

    @property
    def fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def telemetry_lines(
    frame_shape: tuple[int, ...], *, connected: bool, inference_ms: float | None,
    detection_fps: float, detection_count: int, frame_age: float | None,
    candidate_count: int, target: TrackedTarget | None,
    track_state: TrackState, association_score: float | None,
    max_missed_frames: int,
) -> tuple[str, ...]:
    """Datos de video para el panel; no dibuja sobre la imagen."""
    height, width = frame_shape[:2]
    age_text = "N/A" if frame_age is None else f"{frame_age * 1000:.0f} ms"
    inference_text = "N/A" if inference_ms is None else f"{inference_ms:.1f} ms"
    detection = None if target is None else target.detection
    target_name = "NONE" if detection is None else detection.class_name
    target_confidence = "N/A" if detection is None else f"{detection.confidence:.2f}"
    track_id = "N/A" if target is None else str(target.track_id)
    track_age = "N/A" if target is None else str(target.age_frames)
    missed = "N/A" if target is None else (
        f"{target.missed_frames}/{max_missed_frames}"
    )
    velocity = "N/A" if target is None else (
        f"({target.velocity_x:+.1f}, {target.velocity_y:+.1f}) px/f"
    )
    score_text = "N/A" if association_score is None else f"{association_score:.3f}"
    lines = (
        f"Video: {'CONNECTED' if connected else 'DISCONNECTED'}",
        f"Resolución: {width} x {height}",
        f"Inferencia: {inference_text}",
        f"FPS detección: {detection_fps:.1f}",
        f"Detecciones: {detection_count}",
        f"Candidatos: {candidate_count}",
        f"Seguimiento: {track_id} [{track_state.value}]",
        f"Objetivo: {target_name} ({target_confidence})",
        f"Edad / pérdidas: {track_age} / {missed}",
        f"Velocidad: {velocity}",
        f"Asociación: {score_text}",
        f"Edad local del cuadro: {age_text}",
    )
    return lines


def ptz_telemetry_lines(
    decision: PTZDecision,
    hardware_status: str = "DISABLED",
    runtime_status: PTZRuntimeStatus | None = None,
    metrics_snapshot: PTZMetricsSnapshot | None = None,
) -> tuple[str, ...]:
    """Telemetría PTZ para el registro, separada del video."""
    frame_age_text = (
        "N/A" if decision.frame_age_ms is None
        else f"{decision.frame_age_ms:.0f} ms"
    )
    status = runtime_status or PTZRuntimeStatus(
        hardware=hardware_status, executor_state="N/A", armed=False,
        physical_pan=0.0, physical_tilt=0.0, heartbeat_age_ms=None,
        deadman_active=False, backend_latency_ms=None,
        command_interval_ms=None, deadman_activations=0,
        watchdog_activations=0, stop_commands=0,
        execution_reason="simulation only",
    )
    metrics = metrics_snapshot or PTZMetricsSnapshot(
        recenter_active=False, pan_sign_flips=0, tilt_sign_flips=0,
        pan_saturations=0, tilt_saturations=0, dead_zone_chatter_events=0,
    )
    heartbeat = (
        "N/A" if status.heartbeat_age_ms is None
        else f"{status.heartbeat_age_ms:.0f} ms"
    )
    lines = (
        f"Decisión automática: {decision.action.value}",
        f"Hardware PTZ: {status.hardware} | Estado: {status.executor_state} | "
        f"Armado: {'YES' if status.armed else 'NO'}",
        f"Movimiento válido: {'YES' if decision.safe_to_move else 'NO'} | Motivo: "
        f"{decision.reason}",
        f"Error X/Y: {decision.raw_error_x:+.3f} / {decision.raw_error_y:+.3f}",
        f"Filtrado X/Y: {decision.filtered_error_x:+.3f} / {decision.filtered_error_y:+.3f}",
        f"Predicción X/Y: {decision.predicted_error_x:+.3f} / {decision.predicted_error_y:+.3f}",
        f"Pan/tilt automático: {decision.pan:+.3f} / {decision.tilt:+.3f}",
        f"Pan/tilt físico: {status.physical_pan:+.3f} / "
        f"{status.physical_tilt:+.3f}",
        f"Centrado: {'ACTIVE' if metrics.recenter_active else 'IDLE'} | "
        f"Cambios signo P/T: {metrics.pan_sign_flips}/{metrics.tilt_sign_flips}",
        f"Saturación P/T: {metrics.pan_saturations}/{metrics.tilt_saturations} | "
        f"Cruces zona muerta: {metrics.dead_zone_chatter_events}",
        f"Dead-man: {'ACTIVE' if status.deadman_active else 'IDLE'} | "
        f"Edad heartbeat: {heartbeat}",
        f"Oscilación: "
        f"{decision.oscillation_score_pan:.2f} / {decision.oscillation_score_tilt:.2f}",
        f"Edad local del cuadro: {frame_age_text} | Candidato Position3D: "
        f"{'YES' if decision.position3d_candidate else 'NO'}",
        f"Motivo del ejecutor: {status.execution_reason}",
    )
    return lines


def update_tracked_target(
    tracker: TargetTracker,
    selector: TargetSelector,
    detections: list[Detection],
    frame_shape: tuple[int, ...],
) -> tuple[TrackedTarget | None, int]:
    """Asocia el track vigente o pide un candidato sólo cuando es necesario."""
    candidate_count = 0
    if selector.allowed_classes is not None:
        detections = [detection for detection in detections
                      if detection.class_name.casefold() in selector.allowed_classes]
        current = tracker.current_target
        if current is not None and current.detection.class_name.casefold() not in selector.allowed_classes:
            tracker.reset()
    if tracker.has_target:
        target = tracker.update(detections, None, frame_shape)
        if target is not None:
            return target, candidate_count

    candidate = selector.select(detections, frame_shape)
    candidate_count = selector.last_candidate_count
    target = tracker.update(detections, candidate, frame_shape)
    return target, candidate_count


def create_hardware_executor(
    policy: PTZExecutionPolicy,
    tuning: PTZTuningConfiguration | None = None,
) -> PTZRuntime:
    """Conecta el backend configurado sólo si hardware está habilitado.

    En modo auto se intenta ISAPI y se usa ONVIF como fallback únicamente
    durante la conexión inicial. Esta función nunca arma el ejecutor.
    """
    from ptz.ptz_backend import create_ptz_controller
    from ptz.ptz_executor import PTZCommandExecutor

    backend = create_ptz_controller()
    try:
        backend.connect()
        executor = PTZCommandExecutor.from_environment(
            backend,
            policy=policy,
            **({} if tuning is None else tuning.executor_overrides()),
        )
    except Exception:
        backend.close()
        raise
    LOGGER.info(
        "Hardware PTZ habilitado; backend %s conectado y DISARMED",
        type(backend).__name__,
    )
    return executor


def prepare_hardware_executor(
    policy: PTZExecutionPolicy,
    factory: Callable[[PTZExecutionPolicy], PTZRuntime] = create_hardware_executor,
) -> PTZRuntime | None:
    """Hard gate comprobable: con hardware=false no invoca la fábrica."""
    if not policy.hardware_enabled:
        LOGGER.info("PTZ hardware deshabilitado; main permanece en simulación")
        return None
    return factory(policy)


def execution_context(
    target: TrackedTarget | None,
    tracker_state: TrackState,
    *,
    timestamp: float,
    frame_age_s: float | None,
) -> PTZExecutionContext:
    """Evidencia temporal de una decisión, sin ejecutar hardware."""
    target_age_ms: float | None = None
    if target is not None:
        age = (timestamp - target.last_seen_timestamp) * 1000.0
        if isfinite(age) and age >= 0.0:
            target_age_ms = age
    frame_age_ms = (
        None if frame_age_s is None else max(0.0, frame_age_s * 1000.0)
    )
    return PTZExecutionContext(
        tracker_state=tracker_state,
        target_age_ms=target_age_ms,
        frame_age_ms=frame_age_ms,
        confidence=None if target is None else target.detection.confidence,
        visible_frames=0 if target is None else target.visible_frames,
    )


def runtime_status(
    executor: PTZRuntime | None, execution_reason: str,
) -> PTZRuntimeStatus:
    """Normaliza el estado opcional del hardware para telemetría y métricas."""

    if executor is None:
        return PTZRuntimeStatus(
            hardware="DISABLED", executor_state="N/A", armed=False,
            physical_pan=0.0, physical_tilt=0.0, heartbeat_age_ms=None,
            deadman_active=False, backend_latency_ms=None,
            command_interval_ms=None, deadman_activations=0,
            watchdog_activations=0, stop_commands=0,
            execution_reason=execution_reason,
        )
    physical_pan, physical_tilt, _ = executor.last_physical_velocity
    state = getattr(executor.state, "value", str(executor.state))
    return PTZRuntimeStatus(
        hardware="ENABLED", executor_state=str(state), armed=executor.armed,
        physical_pan=physical_pan, physical_tilt=physical_tilt,
        heartbeat_age_ms=executor.heartbeat_age_ms,
        deadman_active=executor.deadman_active,
        backend_latency_ms=executor.last_backend_latency_ms,
        command_interval_ms=executor.last_command_interval_ms,
        deadman_activations=executor.deadman_activations,
        watchdog_activations=executor.watchdog_activations,
        stop_commands=executor.stop_commands,
        execution_reason=execution_reason,
    )


def record_ptz_metrics(
    metrics: PTZMetrics,
    *,
    timestamp: float,
    target: TrackedTarget | None,
    tracker_state: TrackState,
    decision: PTZDecision,
    status: PTZRuntimeStatus,
    frame_age_s: float | None,
    inference_ms: float | None,
    decision_processing_ms: float,
    executor_processing_ms: float,
) -> None:
    """Convierte el estado del pipeline en una muestra; no hace nada si está off."""
    if not metrics.enabled:
        return
    target_age_ms: float | None = None
    if target is not None:
        age = (timestamp - target.last_seen_timestamp) * 1000.0
        if isfinite(age) and age >= 0.0:
            target_age_ms = age
    metrics.record(PTZMetricSample(
        time_s=timestamp,
        track_id=None if target is None else target.track_id,
        tracker_state=tracker_state.value,
        confidence=None if target is None else target.detection.confidence,
        raw_ex=decision.raw_error_x,
        raw_ey=decision.raw_error_y,
        filtered_ex=decision.filtered_error_x,
        filtered_ey=decision.filtered_error_y,
        predicted_ex=decision.predicted_error_x,
        predicted_ey=decision.predicted_error_y,
        pan_cmd=decision.pan,
        tilt_cmd=decision.tilt,
        pan_physical=status.physical_pan,
        tilt_physical=status.physical_tilt,
        decision=decision.action.value,
        executor_state=status.executor_state,
        frame_age_ms=(
            None if frame_age_s is None else max(0.0, frame_age_s * 1000.0)
        ),
        target_age_ms=target_age_ms,
        inference_ms=inference_ms,
        decision_processing_ms=decision_processing_ms,
        executor_processing_ms=executor_processing_ms,
        backend_latency_ms=status.backend_latency_ms,
        command_interval_ms=(
            status.command_interval_ms
            if status.execution_reason == "continuous movement executed" else None
        ),
        deadman_activations=status.deadman_activations,
        watchdog_activations=status.watchdog_activations,
        osc_pan=decision.oscillation_score_pan,
        osc_tilt=decision.oscillation_score_tilt,
        inside_dead_zone=decision.inside_dead_zone,
        safe_to_move=decision.safe_to_move,
        reason=status.execution_reason,
    ))
    metrics.sync_executor_counters(
        stop_commands=status.stop_commands,
        deadman_activations=status.deadman_activations,
        watchdog_activations=status.watchdog_activations,
    )



def run_detection_loop(
    stream: RTSPStream, detector: YOLODetector, selector: TargetSelector,
    tracker: TargetTracker, decision_engine: PTZDecisionEngine,
    metrics: PTZMetrics, control, mailbox, stop_event: threading.Event,
    tracking_options: TrackingOptions | None = None,
) -> None:
    """Procesa visión en segundo plano; nunca toca widgets ni ordena movimientos."""
    from ptz.ui import DashboardFrame

    rate = DetectionRate()
    previous_frame_id = -1
    previous_connection = None
    last_empty_update = 0.0
    applied_revision = None
    detector_confidence = detector.confidence if tracking_options is not None else None
    while not stop_event.is_set():
        settings = None if tracking_options is None else tracking_options.snapshot
        revision = None if settings is None else settings.revision
        if settings is not None and revision != applied_revision:
            selector.configure_tracking(
                min_confidence=settings.min_confidence, allowed_classes=settings.allowed_classes,
            )
            detector.set_confidence(min(detector_confidence, settings.min_confidence))
            decision_engine.set_min_track_confidence(settings.min_confidence)
            tracker.reset()
            decision_engine.reset()
            applied_revision = revision
            LOGGER.info("Seguimiento actualizado: confianza=%.0f%%, clases=%s",
                        settings.min_confidence * 100, ", ".join(sorted(settings.allowed_classes)) or "ninguna")
        connected = stream.is_connected()
        if connected != previous_connection:
            LOGGER.info("Stream %s", "conectado" if connected else "desconectado")
            previous_connection = connected
        read_started = monotonic()
        packet = stream.read_packet(copy=False, after_id=previous_frame_id)
        age_at_read = stream.last_frame_age
        if packet is None:
            # Entre frames conservamos la última imagen, sin parpadeos ni HOLD
            # espurios. El controlador envejece por su cuenta la evidencia.
            if (not connected or age_at_read is None or age_at_read > 0.5) and read_started - last_empty_update > 0.1:
                last_empty_update = read_started
                state = TrackState.TEMPORARILY_LOST if tracker.current_target else TrackState.NO_TARGET
                decision = decision_engine.decide(
                    tracker.current_target, (360, 640, 3),
                    tracker_state=state, frame_age_s=None, timestamp=read_started,
                )
                control.submit_automatic(decision, execution_context(
                    tracker.current_target, state, timestamp=read_started, frame_age_s=None,
                ), settings_revision=revision)
                status = runtime_status(control.executor, control.status.reason)
                mailbox.publish(DashboardFrame(
                    None,
                    telemetry_lines(
                        (360, 640, 3), connected=False, inference_ms=None,
                        detection_fps=rate.fps, detection_count=0, frame_age=age_at_read,
                        candidate_count=0, target=tracker.current_target, track_state=state,
                        association_score=None, max_missed_frames=tracker.max_missed_frames,
                    ),
                    ptz_telemetry_lines(decision, runtime_status=status, metrics_snapshot=metrics.snapshot),
                    connected=False,
                ))
            stop_event.wait(IDLE_DELAY)
            continue

        previous_frame_id = packet.frame_id
        frame = packet.frame
        try:
            detections = detector.detect(frame)
        except InferenceError as error:
            LOGGER.warning("Se omitió un frame por error de inferencia: %s", error)
            detections = []
        else:
            rate.tick()
        if stop_event.is_set():
            break
        target, candidate_count = update_tracked_target(
            tracker, selector, detections, frame.shape,
        )
        now = monotonic()
        # Edad del cuadro procesado: incluye el tiempo de inferencia aunque
        # RTSP haya recibido cuadros más recientes mientras YOLO trabajaba.
        frame_age = max(0.0, now - packet.received_at)
        decision_started = perf_counter()
        decision = decision_engine.decide(
            target, frame.shape, tracker_state=tracker.state,
            frame_age_s=frame_age, timestamp=now,
        )
        decision_processing_ms = (perf_counter() - decision_started) * 1000.0
        control.submit_automatic(decision, execution_context(
            target, tracker.state, timestamp=now, frame_age_s=frame_age,
        ), settings_revision=revision)
        status = runtime_status(control.executor, control.status.reason)
        record_ptz_metrics(
            metrics, timestamp=now, target=target, tracker_state=tracker.state,
            decision=decision, status=status, frame_age_s=frame_age,
            inference_ms=detector.last_inference_ms,
            decision_processing_ms=decision_processing_ms, executor_processing_ms=0.0,
        )
        association = tracker.last_association
        mailbox.publish(DashboardFrame(
            frame,
            telemetry_lines(
                frame.shape, connected=connected, inference_ms=detector.last_inference_ms,
                detection_fps=rate.fps, detection_count=len(detections), frame_age=frame_age,
                candidate_count=candidate_count, target=target, track_state=tracker.state,
                association_score=None if association is None else association.score,
                max_missed_frames=tracker.max_missed_frames,
            ),
            ptz_telemetry_lines(decision, runtime_status=status, metrics_snapshot=metrics.snapshot),
            connected=connected, fps=rate.fps, detections=len(detections),
            captured_at=packet.received_at,
            annotations=PreviewAnnotations(
                frame.shape, packet.received_at, tuple(detections),
                target, tracker.state, decision_engine.dead_zone,
            ),
        ))

def main() -> int:
    """Ejecuta Tk en el hilo principal y visión/control en segundo plano."""
    try:
        import tkinter as tk
        from ptz.ui import DetectcamDashboard, FrameMailbox, UILogHandler
    except ImportError:
        LOGGER.error("La interfaz requiere python3-tk y Pillow. Consulte README.md.")
        return 1

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    LOGGER.info("Iniciando Detectcam: RTSP + detección YOLO")
    try:
        tuning = PTZTuningConfiguration.from_environment()
        stream = RTSPStream.from_environment()
        detector = YOLODetector.from_environment()
        selector = TargetSelector.from_environment(
            default_classes=detector.class_names.values()
        )
        tracker = TargetTracker.from_environment(**tuning.tracker_overrides())
        decision_engine = PTZDecisionEngine.from_environment(
            **tuning.decision_overrides()
        )
        execution_policy = PTZExecutionPolicy.from_environment(
            **tuning.policy_overrides()
        )
        tracking_options = TrackingOptions(
            detector.class_names,
            min_confidence=max(selector.min_confidence, decision_engine.min_track_confidence),
            allowed_classes=selector.allowed_classes,
        )
        effective_configuration = tuning.effective_configuration()
        metrics = PTZMetrics.from_environment(
            profile=tuning.profile.name.value,
            effective_configuration=effective_configuration,
            physical_max_pan=tuning.profile.physical_max_pan,
            physical_max_tilt=tuning.profile.physical_max_tilt,
        )
        LOGGER.info(
            "Perfil PTZ %s seleccionado: Kp=%.2f max_command=%.2f "
            "physical_max=%.2f",
            tuning.profile.name.value,
            tuning.profile.kp_pan,
            tuning.profile.max_pan_command,
            tuning.profile.physical_max_pan,
        )
    except (StreamConfigurationError, DetectorError,
            TargetSelectorConfigurationError,
            TargetTrackerConfigurationError,
            PTZDecisionConfigurationError,
            PTZExecutionPolicyConfigurationError,
            PTZMetricsConfigurationError,
            PTZTuningConfigurationError, ValueError) as error:
        LOGGER.error("Configuración inicial inválida: %s", error)
        return 1

    try:
        executor = prepare_hardware_executor(
            execution_policy,
            factory=lambda policy: create_hardware_executor(policy, tuning),
        )
    except Exception as error:
        LOGGER.error("No se pudo preparar el backend PTZ: %s", error)
        metrics.close()
        return 1


    from ptz.operator_control import OperatorControl

    def video_age() -> float | None:
        return stream.last_frame_age if stream.is_connected() else None

    control = OperatorControl(
        executor, video_age, on_emergency=metrics.note_emergency_stop,
        tracking_options=tracking_options,
    )
    mailbox = FrameMailbox()
    stop_event = threading.Event()
    pipeline_failed = threading.Event()
    dashboard = None
    worker = None
    log_handler = UILogHandler()
    logging.getLogger().addHandler(log_handler)

    def process_video() -> None:
        try:
            run_detection_loop(
                stream, detector, selector, tracker, decision_engine,
                metrics, control, mailbox, stop_event,
                tracking_options=tracking_options,
            )
        except Exception:
            LOGGER.exception("Se detuvo el procesamiento de video")
            pipeline_failed.set()
            stop_event.set()
            control.request("emergency")
            from ptz.ui import DashboardFrame
            mailbox.publish(DashboardFrame(
                None, ("Procesamiento detenido. Consulta la pestaña Eventos.",), (), False,
            ))

    try:
        dashboard = DetectcamDashboard(
            control, mailbox, log_handler, preview_source=stream,
            tracking_options=tracking_options,
        )
        control.start()
        stream.start()
        worker = threading.Thread(target=process_video, name="detectcam-vision", daemon=True)
        worker.start()
        LOGGER.info("Interfaz lista: video, registro plegable y control PTZ")
        dashboard.run()
    except KeyboardInterrupt:
        LOGGER.info("Interrupción recibida")
    except (tk.TclError, cv2.error) as error:
        LOGGER.error("No se pudo abrir la interfaz: %s", error)
        return 1
    finally:
        stop_event.set()
        # La parada física es independiente del cierre del stream y de YOLO.
        control.close()
        stream.stop()
        if worker is not None:
            worker.join()
        if dashboard is not None and not dashboard.closed:
            dashboard.close()
        if executor is not None:
            metrics.sync_executor_counters(
                stop_commands=executor.stop_commands,
                deadman_activations=executor.deadman_activations,
                watchdog_activations=executor.watchdog_activations,
            )
        metrics.close()
        logging.getLogger().removeHandler(log_handler)
        log_handler.close()
        if metrics.enabled:
            LOGGER.info("\n%s", metrics.summary_text())
        LOGGER.info("Detectcam terminado")
    return 1 if pipeline_failed.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
