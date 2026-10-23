"""Composición de la aplicación y herramientas offline; sin lógica de control."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata
import json
import logging
from pathlib import Path
import threading
from time import monotonic, sleep
import tomllib

from detectcam.config.environment import ConfigurationError, configure_logging
from detectcam.config.settings import PROJECT_ROOT, Settings, load_settings
from detectcam.core import Controller, ControlConfig, Tracker, TrackerConfig
from detectcam.runtime import MotionLimits, MotionRuntime, NetworkConfig, create_backend

LOGGER = logging.getLogger(__name__)


def components(settings):
    """Valida todos los parámetros antes de abrir hilos o conexiones."""
    tracker = TrackerConfig(**settings.sections.get("tracking", {}))
    control = ControlConfig(**settings.sections.get("control", {}))
    runtime = dict(settings.sections.get("runtime", {}))
    network_options = {key: runtime.pop(key) for key in (
        "connect_timeout_s", "read_timeout_s", "server_motion_timeout_s") if key in runtime}
    limits = MotionLimits(runtime.pop("manual_max_pan", .75), runtime.pop("manual_max_tilt", .75),
                          runtime.pop("manual_max_zoom", .30))
    allowed = {"command_interval_s", "heartbeat_timeout_s", "max_command_ttl_s", "reconnect_delay_s"}
    if set(runtime)-allowed:
        raise ConfigurationError("Opción desconocida en [runtime]: " + ", ".join(sorted(set(runtime)-allowed)))
    if control.max_pan > limits.max_pan or control.max_tilt > limits.max_tilt:
        raise ConfigurationError("El límite de transporte debe cubrir el límite automático")
    if tracker.recovery_s != control.recovery_s:
        raise ConfigurationError("tracking.recovery_s y control.recovery_s deben coincidir")
    network = NetworkConfig(settings.camera_host, settings.username, settings.password,
        backend=settings.backend, port=settings.isapi_port, channel=settings.ptz_channel,
        onvif_port=settings.onvif_port, **network_options)
    # Construir runtime no abre conexiones; valida también sus presupuestos.
    class ValidationBackend:
        pass
    MotionRuntime(ValidationBackend(), limits=limits, **runtime)
    return tracker, control, runtime, limits, network


def _versions():
    result = {}
    for name in ("numpy", "opencv-python", "torch", "torchvision", "ultralytics", "requests", "Pillow", "onvif-zeep"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "MISSING"
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Detectcam 2: seguimiento rápido y recuperación temporal")
    parser.add_argument("--check", action="store_true", help="Validar configuración sin cargar YOLO ni conectar")
    parser.add_argument("--simulate", action="store_true", help="Video/YOLO reales, sin crear transporte PTZ")
    parser.add_argument("--demo", action="store_true", help="Cámara, detecciones y motores sintéticos sin red")
    parser.add_argument("--headless-seconds", type=float, default=0., help="Sólo --demo: probar sin interfaz")
    parser.add_argument("--metrics", action="store_true", help="Guardar eventos correlacionados en runs/")
    parser.add_argument("--replay", type=Path, help="JSON de detecciones; imprime intenciones sin cámara")
    args = parser.parse_args(argv)
    configure_logging()
    try:
        if args.replay:
            from .replay import replay
            sections = tomllib.loads((PROJECT_ROOT / "settings.toml").read_text(encoding="utf-8"))
            output = replay(args.replay, tracker_config=TrackerConfig(**sections.get("tracking", {})),
                            control_config=ControlConfig(**sections.get("control", {})))
            print(json.dumps(output, ensure_ascii=False, indent=2))
            return 0
        if args.headless_seconds and (not args.demo or not 0 < args.headless_seconds <= 300):
            raise ConfigurationError("--headless-seconds requiere --demo y duración 0..300 s")
        if args.demo:
            sections = tomllib.loads((PROJECT_ROOT / "settings.toml").read_text(encoding="utf-8"))
            settings = Settings(PROJECT_ROOT / "model/best.pt", "demo.local", "demo", "demo",
                                allowed_classes=("drone",), sections=sections)
        else:
            settings = load_settings(simulate=args.simulate)
        configs = components(settings)
        if args.check:
            print(json.dumps({**settings.public_summary(), "dependencies": _versions(),
                              "tracking": asdict(configs[0]), "control": asdict(configs[1])},
                             ensure_ascii=False, indent=2))
            return 0
        return run(settings, configs, demo=args.demo, duration=args.headless_seconds,
                   metrics_enabled=args.metrics or settings.metrics_enabled)
    except (ConfigurationError, ValueError, TypeError, OSError, RuntimeError) as error:
        LOGGER.error("No se pudo iniciar: %s", error)
        return 2


def run(settings, configs, *, demo=False, duration=0., metrics_enabled=False):
    from detectcam.adapters.detector import YOLODetector
    from detectcam.adapters.media import MediaStore
    from detectcam.adapters.stream import RTSPStream
    from detectcam.application.operator import OperatorControl
    from detectcam.application.pipeline import VisionPipeline
    from detectcam.application.telemetry import Telemetry
    from detectcam.application.tracking_settings import TrackingOptions
    from detectcam.presentation.dashboard import DetectcamDashboard, FrameMailbox, UILogHandler

    tracker_cfg, control_cfg, runtime_cfg, limits, network = configs
    if demo:
        from detectcam.adapters.demo import DemoCamera, DemoDetector
        stream, detector = DemoCamera(), DemoDetector()
    else:
        # Las credenciales se inyectan una sola vez; los adaptadores no vuelven a
        # interpretar ajustes globales ni importan al otro proyecto.
        stream = RTSPStream(username=settings.username, password=settings.password,
            camera_ip=settings.camera_host, channel=settings.rtsp_channel,
            low_latency=settings.low_latency, transport=settings.transport,
            decoder_threads=settings.decoder_threads, read_timeout=1., max_read_failures=3)
        detector = YOLODetector(model_path=settings.model_path, device=settings.device,
                               imgsz=settings.imgsz, confidence=min(settings.inference_confidence, settings.min_confidence))
    telemetry = Telemetry(PROJECT_ROOT / "runs", enabled=metrics_enabled,
        metadata={"schema": 2, "application": "detectcam-propuesta 2.0.0",
                  **settings.public_summary(), "demo": demo, "versions": _versions(),
                  "metrics_enabled": metrics_enabled,
                  "tracking": asdict(tracker_cfg), "control": asdict(control_cfg), "runtime": runtime_cfg})
    def event_callback(item):
        item = dict(item)
        event = item.pop("event")
        if event in {"completed", "stopped"} and "latency_ms" in item:
            item["backend_latency_ms"] = item["latency_ms"]
        telemetry.emit(event, **item)
    runtime = None
    if demo or settings.hardware_enabled:
        runtime = MotionRuntime(stream if demo else create_backend(network), limits=limits,
                                event_callback=event_callback, **runtime_cfg)
    options = TrackingOptions(detector.class_names, min_confidence=settings.min_confidence,
        allowed_classes=settings.allowed_classes)
    control = OperatorControl(runtime, lambda: stream.last_frame_age if stream.is_connected() else None,
        controller=Controller(control_cfg), tracking_options=options, telemetry=telemetry,
        on_emergency=lambda: telemetry.emit("operator_emergency"), max_video_age_s=control_cfg.max_frame_age_s)
    mailbox = FrameMailbox()
    pipeline = VisionPipeline(stream, detector, Tracker(tracker_cfg), control, mailbox, options, telemetry,
        dead_zone=control_cfg.dead_zone, max_frame_age_s=control_cfg.max_frame_age_s,
        warmup=not demo and settings.sections.get("application", {}).get("warmup", True))
    stop = threading.Event()
    worker = None
    dashboard = None
    handler = UILogHandler()
    logging.getLogger().addHandler(handler)
    try:
        if not duration:
            dashboard = DetectcamDashboard(control, mailbox, handler, preview_source=stream,
                tracking_options=options, media_store=MediaStore(settings.media_directory))
        control.start()
        stream.start()
        worker = threading.Thread(target=pipeline.run, args=(stop,), name="detectcam-vision", daemon=True)
        worker.start()
        LOGGER.info("Detectcam 2 listo · %s · siempre inicia desarmado", "DEMO LOCAL" if demo else "cámara configurada")
        if duration:
            # Únicamente la cámara sintética puede armarse desde este harness.
            control.request("mode", "automatic")
            started = monotonic()
            requested_arm = False
            while monotonic()-started < duration and not stop.is_set():
                control.heartbeat()
                packet = stream.read_packet(copy=False)
                control.preview_seen(packet.received_at if packet else None)
                state = runtime.snapshot
                if not requested_arm and state.connected and state.stop_confirmed and not state.stop_pending:
                    control.request("arm", "MOVER")
                    requested_arm = True
                sleep(.02)
            print(json.dumps({"demo": True, "duration_s": round(monotonic()-started, 2),
                              "control": asdict(control.status), "metrics": telemetry.snapshot},
                             ensure_ascii=False, indent=2))
        else:
            dashboard.run()
    except KeyboardInterrupt:
        LOGGER.info("Interrupción del operador")
    except Exception:
        LOGGER.exception("La aplicación no pudo continuar")
        return 1
    finally:
        stop.set()
        control.close()
        if demo:
            stream.stop_stream()
        else:
            stream.stop()
        if worker:
            worker.join(timeout=3.)
        if dashboard is not None and not dashboard.closed:
            dashboard.close()
        telemetry.close()
        logging.getLogger().removeHandler(handler)
        handler.close()
    return 1 if pipeline.failed else 0
