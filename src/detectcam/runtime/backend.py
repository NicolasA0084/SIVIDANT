"""Physical transports with independent motion and urgent STOP connections.

These transports acknowledge protocol responses, never physical movement.
No movement is retried or switched to another backend after dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import math
import threading
from typing import Protocol
import xml.etree.ElementTree as ET

from .models import MotionCapabilities, NetworkConfig


class TransportError(RuntimeError):
    """A sanitized network/protocol error without credentials or camera URLs."""


class MotionBackend(Protocol):
    def connect(self) -> MotionCapabilities: ...
    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0,
                      *, duration_s: float = 0.35) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class _Range:
    minimum: float
    maximum: float

    def __post_init__(self):
        if not (math.isfinite(self.minimum) and math.isfinite(self.maximum)
                and self.minimum < 0 < self.maximum):
            raise TransportError("camera velocity range must be finite and span zero")

    def map(self, value: float) -> float:
        # Zero must remain zero even for asymmetric manufacturer ranges.
        return value * (self.maximum if value >= 0 else -self.minimum)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


class ISAPIBackend:
    """Digest keep-alive sessions; STOP never waits on the motion session."""

    def __init__(self, config: NetworkConfig, *, session_factory=None) -> None:
        import requests
        from requests.auth import HTTPDigestAuth

        self.config = config
        factory = session_factory or requests.Session
        self._motion = factory()
        self._urgent = factory()
        for session in (self._motion, self._urgent):
            session.auth = HTTPDigestAuth(config.username, config.password)
            session.headers.update({"Accept": "application/xml"})
            # Camera hosts are explicit; environment HTTP proxies must not route them.
            session.trust_env = False
        host = f"[{config.host}]" if ":" in config.host else config.host
        self._base = f"http://{host}:{config.port}/ISAPI/PTZCtrl/channels/{config.channel}"
        self._stop_lock = threading.Lock()
        self._ranges: tuple[_Range, _Range, _Range | None] | None = None

    def _request(self, session, method: str, endpoint: str, data=None):
        try:
            response = session.request(
                method, self._base + "/" + endpoint, data=data,
                headers={"Content-Type": "application/xml"},
                timeout=(self.config.connect_timeout_s, self.config.read_timeout_s),
                allow_redirects=False,
            )
        except Exception as error:
            raise TransportError(f"ISAPI {method} failed ({type(error).__name__})") from None
        if not 200 <= response.status_code < 300:
            raise TransportError(f"ISAPI {method} returned HTTP {response.status_code}")
        if response.content:
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                raise TransportError("ISAPI returned malformed XML") from None
            values = {_local(e.tag): (e.text or "").strip() for e in root.iter()}
            if values.get("statusCode") not in (None, "0", "1"):
                raise TransportError("ISAPI rejected command")
            if "statusString" in values and values["statusString"].casefold() != "ok":
                raise TransportError("ISAPI rejected command")
            return root
        return None

    @staticmethod
    def _range(parent, axis: str | None = None) -> _Range:
        if parent is None:
            raise TransportError("ISAPI lacks continuous movement capabilities")
        if axis is not None:
            parent = next((e for e in parent.iter() if _local(e.tag) == axis), None)
        if parent is None:
            raise TransportError("ISAPI velocity range missing")
        values = {_local(e.tag): e.text for e in parent.iter()}
        try:
            result = _Range(float(values["Min"]), float(values["Max"]))
        except (KeyError, TypeError, ValueError):
            raise TransportError("ISAPI velocity range invalid") from None
        if not result.minimum < 0 < result.maximum:
            raise TransportError("ISAPI velocity range does not span zero")
        return result

    def connect(self) -> MotionCapabilities:
        root = self._request(self._motion, "GET", "capabilities")
        if root is None:
            raise TransportError("ISAPI capabilities are empty")
        xy = next((e for e in root.iter() if _local(e.tag) == "ContinuousPanTiltSpace"), None)
        zoom = next((e for e in root.iter() if _local(e.tag) == "ContinuousZoomSpace"), None)
        zoom_axis = None if zoom is None else next(
            (e for e in zoom.iter() if _local(e.tag) in {"ZRange", "XRange"}), None,
        )
        self._ranges = (self._range(xy, "XRange"), self._range(xy, "YRange"),
                        None if zoom is None else self._range(zoom_axis))
        return MotionCapabilities(True, self._ranges[2] is not None, False, "isapi")

    @staticmethod
    def _payload(pan: float, tilt: float, zoom: float) -> bytes:
        root = ET.Element("PTZData", {"version": "2.0", "xmlns": "http://www.isapi.org/ver20/XMLSchema"})
        for name, value in (("pan", pan), ("tilt", tilt), ("zoom", zoom)):
            ET.SubElement(root, name).text = str(round(value))
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0,
                      *, duration_s: float = 0.35) -> None:
        # ISAPI continuous does not advertise a portable server-side TTL.
        if self._ranges is None:
            raise TransportError("ISAPI is not connected")
        x, y, z = self._ranges
        if zoom and z is None:
            raise TransportError("ISAPI camera does not support continuous zoom")
        self._request(self._motion, "PUT", "continuous", self._payload(
            x.map(pan), y.map(-tilt), 0 if z is None else z.map(zoom),
        ))

    def stop(self) -> None:
        with self._stop_lock:
            self._request(self._urgent, "PUT", "continuous", self._payload(0, 0, 0))

    def close(self) -> None:
        self._ranges = None
        self._motion.close()
        self._urgent.close()


class ONVIFBackend:
    """Two SOAP transports with discovered spaces and server TTL when advertised."""

    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self._service = self._urgent_service = self._token = None
        self._range_x = self._range_y = self._range_z = None
        self._xy_uri = self._zoom_uri = None
        self._timeout_range = None
        self._sessions = []
        self._stop_lock = threading.Lock()

    def connect(self) -> MotionCapabilities:
        try:
            import requests
            from onvif import ONVIFCamera
            from zeep.transports import Transport

            services = []
            for _ in range(2):
                session = requests.Session()
                session.trust_env = False
                self._sessions.append(session)
                transport = Transport(session=session, timeout=self.config.connect_timeout_s,
                                      operation_timeout=(self.config.connect_timeout_s, self.config.read_timeout_s))
                camera = ONVIFCamera(self.config.host, self.config.onvif_port or self.config.port,
                                     self.config.username, self.config.password, transport=transport)
                services.append((camera, camera.create_ptz_service()))
            profiles = services[0][0].create_media_service().GetProfiles()
            profile = next((p for p in profiles if getattr(p, "PTZConfiguration", None)), None)
            if profile is None:
                raise TransportError("ONVIF profile has no PTZ configuration")
            self._token = profile.token
            self._service, self._urgent_service = services[0][1], services[1][1]
            options = self._service.GetConfigurationOptions({"ConfigurationToken": profile.PTZConfiguration.token})
            spaces = options.Spaces
            xy = spaces.ContinuousPanTiltVelocitySpace[0]
            self._range_x = _Range(float(xy.XRange.Min), float(xy.XRange.Max))
            self._range_y = _Range(float(xy.YRange.Min), float(xy.YRange.Max))
            self._xy_uri = xy.URI
            zoom_spaces = getattr(spaces, "ContinuousZoomVelocitySpace", None)
            self._range_z = None
            if zoom_spaces:
                z = zoom_spaces[0]
                self._range_z = _Range(float(z.XRange.Min), float(z.XRange.Max))
                self._zoom_uri = z.URI
            timeout_range = getattr(options, "PTZTimeout", None)
            self._timeout_range = None if timeout_range is None else (
                timeout_range.Min.total_seconds(), timeout_range.Max.total_seconds(),
            )
            if self._timeout_range is not None:
                minimum, maximum = self._timeout_range
                if not (math.isfinite(minimum) and math.isfinite(maximum)
                        and 0 <= minimum <= maximum and maximum > 0):
                    raise TransportError("ONVIF server motion timeout range is invalid")
            return MotionCapabilities(True, self._range_z is not None,
                                      self._timeout_range is not None, "onvif",
                                      None if self._timeout_range is None else self._timeout_range[0],
                                      None if self._timeout_range is None else self._timeout_range[1])
        except TransportError:
            raise
        except Exception as error:
            raise TransportError(f"ONVIF connection failed ({type(error).__name__})") from None

    def move_velocity(self, pan: float, tilt: float, zoom: float = 0.0,
                      *, duration_s: float = 0.35) -> None:
        if self._service is None:
            raise TransportError("ONVIF is not connected")
        try:
            request = self._service.create_type("ContinuousMove")
            request.ProfileToken = self._token
            velocity = {"PanTilt": {"x": self._range_x.map(pan), "y": self._range_y.map(-tilt), "space": self._xy_uri}}
            if self._range_z is not None:
                velocity["Zoom"] = {"x": self._range_z.map(zoom), "space": self._zoom_uri}
            elif zoom:
                raise TransportError("ONVIF camera does not support continuous zoom")
            request.Velocity = velocity
            if self._timeout_range is not None:
                minimum, maximum = self._timeout_range
                # Some cameras advertise a one-second minimum. That is useful as
                # a remote backstop but does NOT enforce a shorter source TTL;
                # the independent local STOP path still owns that deadline.
                request.Timeout = timedelta(seconds=max(minimum, min(maximum, duration_s,
                                                                      self.config.server_motion_timeout_s)))
            self._service.ContinuousMove(request)
        except TransportError:
            raise
        except Exception as error:
            raise TransportError(f"ONVIF movement failed ({type(error).__name__})") from None

    def stop(self) -> None:
        with self._stop_lock:
            if self._urgent_service is None:
                raise TransportError("ONVIF STOP transport is not connected")
            try:
                self._urgent_service.Stop({"ProfileToken": self._token, "PanTilt": True, "Zoom": True})
            except Exception as error:
                raise TransportError(f"ONVIF STOP failed ({type(error).__name__})") from None

    def close(self) -> None:
        self._service = self._urgent_service = self._token = None
        for session in self._sessions:
            session.close()
        self._sessions.clear()


class _AutoBackend:
    def __init__(self, config: NetworkConfig) -> None:
        self.config = config
        self._selected: MotionBackend | None = None

    def connect(self) -> MotionCapabilities:
        # Fallback occurs only while disarmed during explicit connection.
        if self._selected is not None:
            return self._selected.connect()
        for kind in ("isapi", "onvif"):
            candidate = ISAPIBackend(self.config) if kind == "isapi" else ONVIFBackend(self.config)
            try:
                capabilities = candidate.connect()
            except Exception:
                candidate.close()
                if kind == "onvif":
                    raise TransportError("Neither ISAPI nor ONVIF could connect") from None
            else:
                self._selected = candidate
                return capabilities
        raise TransportError("No transport available")

    def move_velocity(self, pan, tilt, zoom=0.0, *, duration_s=0.35):
        if self._selected is None:
            raise TransportError("Transport is not connected")
        return self._selected.move_velocity(pan, tilt, zoom, duration_s=duration_s)

    def stop(self):
        if self._selected is None:
            raise TransportError("Transport is not connected")
        return self._selected.stop()

    def close(self):
        if self._selected is not None:
            self._selected.close()
        self._selected = None


def create_backend(config: NetworkConfig) -> MotionBackend:
    if config.backend == "isapi":
        return ISAPIBackend(config)
    if config.backend == "onvif":
        return ONVIFBackend(config)
    return _AutoBackend(config)
