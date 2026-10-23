"""Bounded latest-command control with a separate urgent STOP worker.

All network calls run outside the state lock. A STOP invalidates queued work,
is sent on an independent connection, and is repeated after an in-flight MOVE
returns. No client can guarantee motor stopping while the camera/network is
unreachable; the snapshot keeps STOP acknowledgement separate from intent.
"""

from __future__ import annotations

from dataclasses import replace
import math
import queue
import threading
import time
from typing import Callable

from .backend import MotionBackend
from .models import MotionCapabilities, MotionCommand, MotionLimits, MotionSnapshot


class MotionRuntime:
    def __init__(
        self, backend: MotionBackend, *, limits: MotionLimits | None = None,
        command_interval_s: float = 0.08, heartbeat_timeout_s: float = 0.4,
        max_command_ttl_s: float = 0.35, reconnect_delay_s: float = 2.0,
        event_callback: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for value in (command_interval_s, heartbeat_timeout_s, max_command_ttl_s, reconnect_delay_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("runtime timing budgets must be finite and positive")
        self._backend = backend
        self._limits = limits or MotionLimits()
        self._interval = command_interval_s
        self._heartbeat_timeout = heartbeat_timeout_s
        self._max_ttl = max_command_ttl_s
        self._reconnect_delay = reconnect_delay_s
        self._callback = event_callback
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._urgent = threading.Event()
        self._worker_done = threading.Event()
        self._transport_done = threading.Event()
        self._closing = threading.Event()
        self._events: queue.Queue[dict] = queue.Queue(maxsize=256)
        self._threads: list[threading.Thread] = []
        self._snapshot = MotionSnapshot()
        self._caps = MotionCapabilities(False, False)
        self._started = False
        self._epoch = 0
        self._next_id = 0
        self._latest: MotionCommand | None = None
        self._inflight = False
        self._stop_inflight = False
        self._stop_seq = 0
        self._stop_attempts = 0
        self._stop_retry_at = 0.0
        self._awaiting_postflight_stop = False
        self._connect_requested = True
        self._next_connect_at = 0.0
        self._last_dispatch_at: float | None = None
        self._presence_at = self._clock()
        self._evidence_deadline: float | None = None

    @property
    def snapshot(self) -> MotionSnapshot:
        # Published objects are frozen; readers never wait for state or network locks.
        return self._snapshot

    def _publish(self, **changes) -> None:
        self._snapshot = replace(self._snapshot, version=self._snapshot.version + 1, **changes)

    def _emit(self, event: str, **fields) -> None:
        if self._callback is None:
            return
        try:
            self._events.put_nowait({"event": event, "timestamp": self._clock(), **fields})
        except queue.Full:
            # Telemetry may be dropped; actuator dispatch must never wait for it.
            pass

    def _event_loop(self) -> None:
        while not self._transport_done.is_set() or not self._events.empty():
            try:
                event = self._events.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                callback = self._callback
                if callback is not None:
                    callback(event)
            except Exception:
                pass

    def start(self) -> "MotionRuntime":
        with self._lock:
            if self._started or self._closing.is_set():
                return self
            self._started = True
            self._threads = [
                threading.Thread(target=self._motion_loop, name="motion-transport", daemon=True),
                threading.Thread(target=self._stop_loop, name="motion-urgent-stop", daemon=True),
                threading.Thread(target=self._watchdog_loop, name="motion-deadlines", daemon=True),
                threading.Thread(target=self._event_loop, name="motion-events", daemon=True),
            ]
            threads = tuple(self._threads)
        for thread in threads:
            thread.start()
        self._wake.set()
        return self

    def arm(self) -> bool:
        with self._lock:
            s = self._snapshot
            if (self._closing.is_set() or not s.connected or s.state == "FAULT"
                    or s.stop_pending or self._inflight or not s.stop_confirmed):
                return False
            self._presence_at = self._clock()
            self._publish(armed=True, state="ARMED_IDLE", reason="PTZ armed")
        self._emit("armed")
        return True

    def heartbeat(self) -> None:
        """Renew controller presence without extending any motion evidence TTL."""
        with self._lock:
            self._presence_at = self._clock()

    def _request_stop_locked(self, reason: str, *, disarm=False, force=False) -> None:
        s = self._snapshot
        active = self._inflight or self._latest is not None or any((s.pan, s.tilt, s.zoom))
        self._epoch += 1
        self._latest = None
        self._evidence_deadline = None
        armed = s.armed and not disarm
        if s.stop_pending and not force:
            self._publish(armed=armed, reason=reason)
            return
        if not active and s.stop_confirmed and not force:
            self._publish(armed=armed, reason=reason,
                          state="ARMED_IDLE" if armed else ("FAULT" if s.state == "FAULT" else "DISARMED"))
            return
        self._stop_seq += 1
        self._stop_attempts = 0
        self._stop_retry_at = 0.0
        self._publish(armed=armed, state="FAULT" if s.state == "FAULT" else "STOPPING",
                      reason=reason, stop_pending=True, stop_confirmed=False)
        self._urgent.set()

    def stop(self, reason: str = "motion stopped") -> None:
        """Cancel pending MOVE and prioritize STOP, retaining the current arm state."""
        with self._lock:
            if self._closing.is_set():
                return
            self._request_stop_locked(reason)
        self._wake.set()

    def disarm(self, reason: str = "PTZ disarmed") -> None:
        with self._lock:
            if self._closing.is_set():
                return
            self._request_stop_locked(reason, disarm=True)
        self._wake.set()
        self._emit("disarmed", reason=reason)

    def emergency_stop(self, reason: str = "emergency stop") -> None:
        with self._lock:
            if self._closing.is_set():
                return
            self._request_stop_locked(reason, disarm=True, force=True)
        self._wake.set()
        self._emit("stop_requested", reason=reason)

    def submit_velocity(
        self, pan: float, tilt: float, zoom: float = 0.0, *, expires_at: float,
        frame_id: int | None = None, settings_revision: int | None = None,
    ) -> int | None:
        velocity = self._limits.clamp(pan, tilt, zoom)
        if not math.isfinite(expires_at):
            raise ValueError("command expiry must be finite")
        if not any(velocity):
            self.stop("zero velocity requested")
            return None
        now = self._clock()
        with self._lock:
            s = self._snapshot
            if (self._closing.is_set() or not s.armed or not s.connected
                    or s.state == "FAULT" or expires_at <= now):
                return None
            if zoom and not self._caps.zoom:
                self._request_stop_locked("continuous zoom is unavailable")
                return None
            self._next_id += 1
            command = MotionCommand(self._next_id, *velocity, now,
                                    min(expires_at, now + self._max_ttl), self._epoch,
                                    frame_id, settings_revision)
            self._latest = command
            self._presence_at = now
            self._evidence_deadline = command.expires_at
            self._publish(command_id=command.command_id, frame_id=frame_id)
        self._wake.set()
        self._emit("accepted", command_id=command.command_id, frame_id=frame_id,
                   accepted_at=now, deadline=command.expires_at,
                   settings_revision=settings_revision)
        return command.command_id

    def recover(self) -> None:
        """Explicit reconnection; successful recovery always remains disarmed."""
        with self._lock:
            if self._closing.is_set():
                return
            self._request_stop_locked("recovering transport", disarm=True)
            self._connect_requested = True
            self._next_connect_at = 0.0
            self._publish(connected=False, armed=False, state="CONNECTING", reason="reconnecting; PTZ disarmed")
        self._wake.set()

    def _fault_locked(self, reason: str) -> None:
        self._publish(state="FAULT", connected=False, armed=False, reason=reason)
        self._request_stop_locked(reason, disarm=True, force=True)

    def _connect(self) -> None:
        started = self._clock()
        try:
            self._backend.close()
            caps = self._backend.connect()
            # A connection/recovery is usable only after a successful STOP response.
            self._backend.stop()
        except Exception as error:
            with self._lock:
                self._next_connect_at = self._clock() + self._reconnect_delay
                self._publish(connected=False, armed=False, state="CONNECTING",
                              reason=f"connection failed ({type(error).__name__}); retrying")
            self._emit("connection_failed", elapsed_ms=(self._clock() - started) * 1000)
            return
        with self._lock:
            self._caps = caps
            self._connect_requested = False
            self._latest = None
            self._publish(connected=not self._closing.is_set(), armed=False,
                          state="DISARMED", reason="connected; PTZ disarmed",
                          zoom_supported=caps.zoom, backend=caps.backend,
                          stop_confirmed=True, pan=0.0, tilt=0.0, zoom=0.0)
        self._emit("connected", backend=caps.backend, server_motion_timeout=caps.server_motion_timeout,
                   server_timeout_min_s=caps.server_timeout_min_s,
                   server_timeout_max_s=caps.server_timeout_max_s,
                   source_deadline_enforced_by="client-stop")

    def _motion_loop(self) -> None:
        try:
            while not self._closing.is_set():
                self._wake.wait(0.01)
                self._wake.clear()
                command = None
                connect = False
                with self._lock:
                    now = self._clock()
                    if (self._connect_requested and now >= self._next_connect_at
                            and not self._snapshot.stop_pending and not self._stop_inflight):
                        connect = True
                    elif (self._snapshot.armed and self._snapshot.connected
                          and not self._snapshot.stop_pending and self._latest is not None
                          and (self._last_dispatch_at is None or now - self._last_dispatch_at >= self._interval)):
                        candidate = self._latest
                        if candidate.epoch != self._epoch or candidate.expires_at <= now:
                            self._request_stop_locked("command evidence expired")
                        else:
                            command = candidate
                            self._latest = None
                            self._inflight = True
                            previous = self._last_dispatch_at
                            self._last_dispatch_at = now
                            self._publish(inflight=True, state="MOVING_CONTINUOUS",
                                          pan=command.pan, tilt=command.tilt, zoom=command.zoom,
                                          stop_confirmed=False, last_command_at=now,
                                          last_command_interval_ms=None if previous is None else (now - previous) * 1000,
                                          reason="velocity dispatched; mechanical response unmeasured")
                if connect:
                    self._connect()
                elif command is not None:
                    self._dispatch(command)
        finally:
            self._worker_done.set()
            self._urgent.set()

    def _dispatch(self, command: MotionCommand) -> None:
        started = self._clock()
        error = None
        sent = False
        # A local fence catches STOP/recovery that arrived between selection and I/O.
        # Remote ordering remains unknowable when two requests cross on the wire.
        with self._lock:
            valid = (command.epoch == self._epoch and self._snapshot.armed
                     and not self._closing.is_set() and command.expires_at > self._clock())
        try:
            if valid:
                sent = True
                self._emit("dispatched", command_id=command.command_id, frame_id=command.frame_id,
                           started_at=started, deadline=command.expires_at)
                self._backend.move_velocity(command.pan, command.tilt, command.zoom,
                                            duration_s=max(0.001, command.expires_at - self._clock()))
        except Exception as caught:
            error = caught
        completed = self._clock()
        with self._lock:
            self._inflight = False
            self._publish(inflight=False, last_backend_latency_ms=(completed - started) * 1000)
            if error is not None:
                self._fault_locked(f"movement response failed ({type(error).__name__}); STOP unconfirmed")
            elif command.epoch != self._epoch or self._closing.is_set():
                # A MOVE may arrive after an urgent STOP; always fence it with a
                # second STOP after the movement request has returned.
                self._request_stop_locked(self._snapshot.reason, disarm=not self._snapshot.armed, force=True)
            elif sent:
                self._publish(acknowledged_command_id=command.command_id)
                if self._evidence_deadline is None or self._evidence_deadline <= completed:
                    self._request_stop_locked("command evidence expired")
            self._awaiting_postflight_stop = False
        self._urgent.set()
        self._emit("completed" if error is None and sent else "movement_failed" if error else "cancelled",
                   command_id=command.command_id, frame_id=command.frame_id,
                   completed_at=completed, latency_ms=(completed - started) * 1000,
                   acknowledged=error is None and sent)

    def _stop_loop(self) -> None:
        try:
            while True:
                self._urgent.wait(0.02)
                self._urgent.clear()
                with self._lock:
                    s = self._snapshot
                    if self._closing.is_set() and self._worker_done.is_set() and not s.stop_pending:
                        break
                    if (not s.stop_pending or self._stop_inflight
                            or self._awaiting_postflight_stop or self._clock() < self._stop_retry_at):
                        continue
                    seq = self._stop_seq
                    self._stop_inflight = True
                started = self._clock()
                error = None
                try:
                    self._backend.stop()
                except Exception as caught:
                    error = caught
                completed = self._clock()
                with self._lock:
                    self._stop_inflight = False
                    if seq != self._stop_seq:
                        self._urgent.set()
                        continue
                    self._stop_attempts += 1
                    if error is None:
                        self._awaiting_postflight_stop = self._inflight
                        self._publish(pan=0.0, tilt=0.0, zoom=0.0,
                                      stop_pending=self._inflight, stop_confirmed=not self._inflight,
                                      state="FAULT" if self._snapshot.state == "FAULT" else
                                      "STOPPING" if self._inflight else
                                      "ARMED_IDLE" if self._snapshot.armed else "DISARMED")
                    else:
                        self._epoch += 1
                        self._latest = None
                        self._evidence_deadline = None
                        self._stop_retry_at = completed + 0.1
                        self._publish(connected=False, armed=False, state="FAULT",
                                      reason=f"STOP response failed ({type(error).__name__}); physical stop unconfirmed",
                                      stop_pending=self._stop_attempts < 3, stop_confirmed=False)
                self._emit("stopped" if error is None else "stop_failed", completed_at=completed,
                           command_id=s.command_id, frame_id=s.frame_id, stop_request_id=seq,
                           latency_ms=(completed - started) * 1000,
                           acknowledged=error is None, postflight_pending=self._inflight)
        finally:
            # The motion worker has completed before sessions are disposed.
            try:
                self._backend.close()
            finally:
                with self._lock:
                    self._publish(connected=False, armed=False, state="CLOSED", reason="transport closed")
                self._transport_done.set()

    def check_deadlines(self) -> None:
        """Local-only deadline check, independent of all transport locks."""
        with self._lock:
            if not self._snapshot.armed or self._closing.is_set():
                return
            now = self._clock()
            if now - self._presence_at > self._heartbeat_timeout:
                self._request_stop_locked("controller heartbeat expired", disarm=True)
            elif self._evidence_deadline is not None and now >= self._evidence_deadline:
                self._request_stop_locked("command evidence expired")

    def _watchdog_loop(self) -> None:
        while not self._closing.wait(0.01):
            self.check_deadlines()

    def close(self, timeout: float = 2.0) -> None:
        """Bound caller wait; background STOP completes even after this budget."""
        with self._lock:
            if not self._closing.is_set():
                self._request_stop_locked("runtime closing", disarm=True, force=True)
                self._closing.set()
            started = self._started
        self._wake.set()
        self._urgent.set()
        if not started:
            self._backend.close()
            self._transport_done.set()
            with self._lock:
                self._publish(connected=False, armed=False, state="CLOSED", stop_pending=False,
                              reason="transport closed before start")
            return
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in self._threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - time.monotonic()))
