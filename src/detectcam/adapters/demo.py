"""Cámara sintética local para revisar toda la aplicación sin red ni motores.

La dinámica es ilustrativa (píxeles/segundo), no un modelo de la Hikvision.
Incluye pérdidas de detección para observar TRACKING -> RECOVERING -> TRACKING.
"""
from __future__ import annotations
from math import sin
import threading
from time import monotonic

import cv2
import numpy as np

from .detector import Detection
from .stream import ReceivedFrame
from detectcam.runtime.models import MotionCapabilities


class DemoCamera:
    frame_rate = 25.

    def __init__(self):
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._packet = None
        self._velocity = (0., 0., 0.)
        self._until = 0.
        self._id = 0

    def connect(self):
        return MotionCapabilities(True, True, True, "demo-local")

    def move_velocity(self, pan, tilt, zoom=0., *, duration_s=.35):
        with self._lock:
            self._velocity, self._until = (pan, tilt, zoom), monotonic()+duration_s

    def stop(self, timeout=0.):
        # Backend STOP and stream stop have separate entry points in the app.
        with self._lock:
            self._velocity, self._until = (0., 0., 0.), 0.

    def stop_stream(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.)

    def close(self):
        self.stop()

    def start(self):
        self._thread = threading.Thread(target=self._run, name="demo-camera", daemon=True)
        self._thread.start()
        return self

    def read_packet(self, *, copy=True, after_id=None):
        with self._lock:
            packet = self._packet
            if packet is None or packet.frame_id == after_id:
                return None
            return ReceivedFrame(packet.frame_id, packet.frame.copy() if copy else packet.frame, packet.received_at)

    def is_connected(self):
        return self._packet is not None and not self._stop.is_set()

    @property
    def last_frame_age(self):
        with self._lock:
            return None if self._packet is None else monotonic()-self._packet.received_at

    def _run(self):
        started = previous = monotonic()
        camera_x = camera_y = 0.
        while not self._stop.is_set():
            now = monotonic()
            dt, t = now-previous, now-started
            previous = now
            with self._lock:
                pan, tilt, zoom = self._velocity if now < self._until else (0., 0., 0.)
            camera_x += pan*500*dt
            camera_y += tilt*350*dt
            frame = np.full((540, 960, 3), (42, 29, 20), np.uint8)
            for x in range(-1920, 2880, 80):
                screen_x = round(x-camera_x)
                cv2.line(frame, (screen_x, 0), (screen_x, 540), (65, 51, 40), 1)
            for y in range(-1080, 1620, 80):
                screen_y = round(y-camera_y)
                cv2.line(frame, (0, screen_y), (960, screen_y), (65, 51, 40), 1)
            x = round(650+180*sin(t*.32)-camera_x)
            y = round(190+90*sin(t*.23)-camera_y)
            hidden = 4.0 < t % 10 < 4.28 or 7.0 < t % 10 < 8.1
            if not hidden:
                cv2.rectangle(frame, (x-14, y-7), (x+14, y+7), (0, 240, 0), -1)
            cv2.putText(frame, "DEMO LOCAL - SIN CAMARA REAL", (20, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (230, 230, 230), 1, cv2.LINE_AA)
            with self._lock:
                self._id += 1
                self._packet = ReceivedFrame(self._id, frame, now)
            self._stop.wait(1/self.frame_rate)


class DemoDetector:
    class_names = {0: "drone"}
    confidence = .25
    last_inference_ms = 0.

    def set_confidence(self, value):
        self.confidence = value

    def detect(self, frame):
        started = monotonic()
        mask = cv2.inRange(frame, (0, 200, 0), (30, 255, 30))
        points = cv2.findNonZero(mask)
        self.last_inference_ms = (monotonic()-started)*1000
        if points is None:
            return []
        x, y, w, h = cv2.boundingRect(points)
        return [Detection(0, "drone", .95, x, y, x+w, y+h)]
