from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ptz.operator_control import OperatorControl, OperatorStatus
from ptz.ui import DetectcamDashboard


class Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class KeyboardTests(unittest.TestCase):
    """Comprueba el control de teclas también en equipos sin servidor gráfico."""

    def setUp(self):
        self.control = OperatorControl(None, lambda: None)
        self.control._status = OperatorStatus(hardware=True, armed=True, zoom_supported=True)
        self.app = object.__new__(DetectcamDashboard)
        self.app.control = self.control
        self.app.root = Mock()
        self.app.root.after.return_value = "pending"
        self.app._pressed_keys = set()
        self.app._key_releases = {}
        self.app._input_paused = False
        self.app._holding = (0, 0)
        self.app._held_button = None
        self.app.mode = Value("manual")
        self.app.speed = Value(0.2)
        self.app.zoom = Value(0.0)
        self.app.capture_image = Mock()
        self.app.toggle_recording = Mock()
        self.app.close = Mock()

    def event(self, key, **options):
        return SimpleNamespace(keysym=key, state=options.get("state", 0), widget=options.get("widget", object()))

    def test_wasd_and_arrows_combine_and_releasing_one_keeps_the_other(self):
        with patch.object(self.control, "set_manual") as move, patch.object(self.control, "request") as request:
            self.app._key_press(self.event("w"))
            move.assert_called_with(0.0, -0.2)
            self.app._key_press(self.event("Right"))
            move.assert_called_with(0.2, -0.2)
            self.app._key_press(self.event("a"))
            move.assert_called_with(0.0, -0.2)
            request.assert_not_called()
            self.app._finish_key_release("w")
            move.assert_called_with(0.0, 0.0)
            self.app._finish_key_release("a")
            move.assert_called_with(0.2, 0.0)
            self.app._finish_key_release("right")
            move.assert_called_with(0.0, 0.0)

    def test_numpad_zoom_updates_bar_and_stops_on_release(self):
        with patch.object(self.control, "set_manual") as move:
            self.app._key_press(self.event("KP_Add"))
            self.assertEqual(self.app.zoom.get(), 0.3)
            move.assert_called_with(0.0, 0.0, zoom=0.3)
            self.app._finish_key_release("kp_add")
            self.assertEqual(self.app.zoom.get(), 0)
            move.assert_called_with(0.0, 0.0)
            self.app._key_press(self.event("KP_Subtract"))
            move.assert_called_with(0.0, 0.0, zoom=-0.3)

    def test_key_repeat_does_not_toggle_recording_or_capture_repeatedly(self):
        for key, action in (("V", self.app.toggle_recording), ("c", self.app.capture_image)):
            self.app._key_press(self.event(key))
            self.app._key_release(self.event(key))
            self.app._key_press(self.event(key))
            self.app._key_press(self.event(key))
            action.assert_called_once()
            self.app._finish_key_release(key.lower())
            self.app._key_press(self.event(key))
            self.assertEqual(action.call_count, 2)

    def test_f2_f3_replace_arm_shortcuts_and_focus_loss_clears_motion(self):
        with patch.object(self.control, "request") as request, patch.object(self.control, "set_manual") as move:
            self.app._key_press(self.event("d"))
            move.assert_called_with(0.2, 0.0)
            request.assert_not_called()
            self.app._key_press(self.event("F3"))
            request.assert_called_with("disarm")
            self.assertEqual(self.app._pressed_keys, set())
            self.app._key_press(self.event("w"))
            self.app._key_press(self.event("KP_Add"))
            self.app._focus_out()
            self.assertEqual(self.app.zoom.get(), 0)
            self.assertEqual(self.app._pressed_keys, set())
            move.assert_called_with(0.0, 0.0)

    def test_motion_requires_manual_mode_and_armed_hardware(self):
        for status in (OperatorStatus(), OperatorStatus(armed=True, mode="automatic")):
            self.control._status = status
            with patch.object(self.control, "set_manual") as move:
                self.app._key_press(self.event("Up"))
                self.app._key_press(self.event("KP_Add"))
                move.assert_not_called()
        # Capture and record do not require arming the PTZ.
        self.app._key_press(self.event("c"))
        self.app.capture_image.assert_called_once()

    def test_shortcuts_do_not_interfere_with_editing_or_control_combinations(self):
        self.app._key_press(self.event("v", state=0x4))
        with patch.object(self.app, "_editing", return_value=True):
            self.app._key_press(self.event("c"))
            self.app._key_press(self.event("w"))
        self.app.capture_image.assert_not_called()
        self.app.toggle_recording.assert_not_called()
        self.assertEqual(self.app._pressed_keys, set())


if __name__ == "__main__":
    unittest.main()
