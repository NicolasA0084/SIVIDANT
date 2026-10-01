"""Componentes reutilizables de Detectcam con exportaciones diferidas."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "Detection": (".detector", "Detection"),
    "YOLODetector": (".detector", "YOLODetector"),
    "RTSPStream": (".stream", "RTSPStream"),
    "StreamConfigurationError": (".stream", "StreamConfigurationError"),
    "PTZController": (".ptz_controller", "PTZController"),
    "PTZError": (".ptz_controller", "PTZError"),
    "TargetSelection": (".target_selector", "TargetSelection"),
    "TargetSelector": (".target_selector", "TargetSelector"),
    "TargetAssociation": (".target_tracker", "TargetAssociation"),
    "TargetTracker": (".target_tracker", "TargetTracker"),
    "TrackedTarget": (".target_tracker", "TrackedTarget"),
    "TrackState": (".target_tracker", "TrackState"),
    "intersection_over_union": (".target_tracker", "intersection_over_union"),
    "normalized_center_distance": (".target_tracker", "normalized_center_distance"),
    "ISAPICommandResult": (".isapi_ptz", "ISAPICommandResult"),
    "ISAPIPTZCapabilities": (".isapi_ptz", "ISAPIPTZCapabilities"),
    "ISAPIPTZController": (".isapi_ptz", "ISAPIPTZController"),
    "ISAPIPTZError": (".isapi_ptz", "ISAPIPTZError"),
    "ISAPIPTZStatus": (".isapi_ptz", "ISAPIPTZStatus"),
    "PTZBackend": (".ptz_backend", "PTZBackend"),
    "PTZBackendCapabilityError": (".ptz_backend", "PTZBackendCapabilityError"),
    "PTZBackendError": (".ptz_backend", "PTZBackendError"),
    "PTZCapabilities": (".ptz_backend", "PTZCapabilities"),
    "create_ptz_controller": (".ptz_backend", "create_ptz_controller"),
    "DecisionAction": (".ptz_decision", "DecisionAction"),
    "OneEuroFilter": (".ptz_decision", "OneEuroFilter"),
    "PTZDecision": (".ptz_decision", "PTZDecision"),
    "PTZDecisionConfigurationError": (
        ".ptz_decision", "PTZDecisionConfigurationError"
    ),
    "PTZDecisionEngine": (".ptz_decision", "PTZDecisionEngine"),
    "PTZExecutionContext": (".ptz_execution_policy", "PTZExecutionContext"),
    "PTZExecutionPolicy": (".ptz_execution_policy", "PTZExecutionPolicy"),
    "PTZExecutionPolicyConfigurationError": (
        ".ptz_execution_policy", "PTZExecutionPolicyConfigurationError"
    ),
    "PTZPolicyResult": (".ptz_execution_policy", "PTZPolicyResult"),
    "PTZArmError": (".ptz_executor", "PTZArmError"),
    "PTZCommandExecutor": (".ptz_executor", "PTZCommandExecutor"),
    "PTZExecutionResult": (".ptz_executor", "PTZExecutionResult"),
    "PTZExecutorConfigurationError": (
        ".ptz_executor", "PTZExecutorConfigurationError"
    ),
    "PTZExecutorError": (".ptz_executor", "PTZExecutorError"),
    "PTZExecutorState": (".ptz_executor", "PTZExecutorState"),
    "PTZTuningConfiguration": (".ptz_tuning", "PTZTuningConfiguration"),
    "PTZTuningConfigurationError": (
        ".ptz_tuning", "PTZTuningConfigurationError"
    ),
    "PTZTuningProfile": (".ptz_tuning", "PTZTuningProfile"),
    "PTZTuningProfileName": (".ptz_tuning", "PTZTuningProfileName"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Carga un componente sólo cuando se solicita explícitamente."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
