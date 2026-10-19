from __future__ import annotations

import logging
import os
import tkinter as tk
import unittest
from time import monotonic
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from ptz.operator_control import OperatorControl, OperatorStatus
from ptz.tracking_settings import TrackingOptions
from ptz.ui import DashboardFrame, DetectcamDashboard, FrameMailbox, UILogHandler


class MailboxTests(unittest.TestCase):
    def test_latest_frame_only(self):
        mailbox = FrameMailbox()
        for number in range(10):
            mailbox.publish(DashboardFrame(None, (str(number),), (), False))
        self.assertEqual(mailbox.take().telemetry, ("9",))
        self.assertIsNone(mailbox.take())

    def test_bounded_log_redacts_credentials(self):
        handler = UILogHandler()
        with patch.dict(os.environ, {"DETECTCAM_CAMERA_USER": "sensitive-user",
                                    "DETECTCAM_CAMERA_PASSWORD": "sensitive-password"}):
            for _ in range(2100):
                handler.handle(logging.LogRecord("test", logging.INFO, "", 0,
                               "rtsp://sensitive-user:sensitive-password@camera/live", (), None))
        lines = handler.drain()
        self.assertEqual(len(lines), 2000)
        self.assertNotIn("sensitive", lines[-1][0])
        self.assertIn("rtsp://***@camera/live", lines[-1][0])


class DashboardTests(unittest.TestCase):
    def setUp(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Sin pantalla gráfica: {error}")
        self.control = OperatorControl(None, lambda: None)
        self.mailbox = FrameMailbox()
        self.handler = UILogHandler()
        self.tracking_options = TrackingOptions({0: "Drone", 1: "Bird", 2: "Person"})
        self.app = DetectcamDashboard(
            self.control, self.mailbox, self.handler, root=root,
            tracking_options=self.tracking_options,
        )
        root.update()

    def tearDown(self):
        if hasattr(self, "app") and not self.app.closed:
            self.app.close()

    def test_video_log_toggle_and_live_data(self):
        self.mailbox.publish(DashboardFrame(
            np.full((360, 640, 3), (45, 33, 20), dtype=np.uint8),
            ("Resolución: 640 × 360", "Detecciones: 2"),
            ("PTZ: DESARMADO",), True, fps=9.5, detections=2,
        ))
        self.app._last_data_update = 0
        self.app.root.after(100, self.app.root.quit)
        self.app.run()
        self.assertIsNotNone(self.app._photo)
        self.assertIn("Detecciones: 2", self.app.telemetry.get("1.0", "end"))
        self.app.toggle_log()
        self.app.root.update_idletasks()
        self.assertFalse(self.app.log_panel.winfo_ismapped())
        self.app.toggle_log()
        self.app.root.update_idletasks()
        self.assertTrue(self.app.log_panel.winfo_ismapped())
        self.assertEqual(self.app.arm_button["state"], "disabled")

    def test_arm_submits_directly_without_confirmation_dialog(self):
        self.control._status = OperatorStatus(hardware=True)
        with patch("tkinter.messagebox.askyesno") as dialog, patch.object(self.control, "request") as request:
            self.app.arm()
            request.assert_called_once_with("arm", "MOVER")
            dialog.assert_not_called()

    def test_camera_missing_keeps_dashboard_open_and_arm_disabled(self):
        self.control._status = OperatorStatus(
            state="CONNECTING", reason="Cámara PTZ sin conexión · reintentando",
        )
        self.app._tick()
        self.assertFalse(self.app.closed)
        self.assertIn("Sin señal", self.app.connection["text"])
        self.assertIn("reintentando", self.app.hardware_label["text"])
        self.assertEqual(self.app.arm_button["state"], "disabled")
        with patch.object(self.control, "request") as request:
            self.app.arm()
            request.assert_not_called()

    def test_press_release_focus_and_close_clear_manual_input(self):
        self.control._status = OperatorStatus(hardware=True, armed=True)
        event = SimpleNamespace(widget=self.app.direction_buttons[0])
        self.app._press(event, -1, -1)
        self.assertEqual(self.app._holding, (-1, -1))
        self.app.root.event_generate("<ButtonRelease-1>")
        self.app.root.update()
        self.assertEqual(self.app._holding, (0, 0))
        self.app._press(event, 1, 0)
        self.app._focus_out()
        self.assertEqual(self.app._holding, (0, 0))
        with patch.object(self.control, "request") as request:
            self.app.close()
            request.assert_called_once_with("emergency")

    def test_emergency_remains_visible_in_compact_window(self):
        self.app.root.geometry("900x640")
        self.app.root.update()
        button = self.app.emergency_button
        self.assertTrue(button.winfo_ismapped())
        bottom = button.winfo_rooty() + button.winfo_height()
        self.assertLessEqual(bottom, self.app.root.winfo_rooty() + self.app.root.winfo_height())
        self.assertTrue(self.app.zoom_scale.winfo_ismapped())
        self.assertLess(self.app.zoom_scale.winfo_rooty(), button.winfo_rooty())
        self.app.controls_canvas.yview_moveto(1)
        self.app.root.update()
        self.assertTrue(self.app.speed_scale.winfo_ismapped())

    def test_actual_keyboard_bindings_move_and_release_with_numpad_zoom(self):
        self.control._status = OperatorStatus(hardware=True, armed=True, zoom_supported=True)
        self.app._tick()
        self.app.root.focus_force()
        self.app.root.update()
        with patch.object(self.control, "set_manual") as move, patch.object(self.control, "request") as request:
            self.app.root.event_generate("<KeyPress-w>")
            self.app.root.event_generate("<KeyPress-Right>")
            self.app.root.update()
            move.assert_called_with(0.15, -0.15)
            request.assert_not_called()
            self.app.root.event_generate("<KeyPress>", keysym="KP_Add")
            self.app.root.update()
            self.assertEqual(self.app.zoom.get(), 0.3)
            move.assert_called_with(0.15, -0.15, zoom=0.3)
            for key in ("w", "Right", "KP_Add"):
                self.app.root.event_generate("<KeyRelease>", keysym=key)
            self.app.root.after(50, self.app.root.quit)
            self.app.run()
            self.assertEqual(self.app.zoom.get(), 0)
            move.assert_called_with(0.0, 0.0)

    def test_capture_and_video_shortcuts_do_not_repeat_when_held(self):
        self.app.root.focus_force()
        self.app.root.update()
        with patch.object(self.app, "capture_image") as capture, \
             patch.object(self.app, "toggle_recording") as record:
            for key in ("c", "v"):
                self.app.root.event_generate(f"<KeyPress-{key}>")
                self.app.root.event_generate(f"<KeyPress-{key}>")
                self.app.root.event_generate(f"<KeyRelease-{key}>")
            self.app.root.update()
            capture.assert_called_once()
            record.assert_called_once()

    def test_capture_uses_full_camera_frame_and_rejects_stale_images(self):
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        self.app._preview_frame = frame
        self.app._preview_received_at = monotonic()
        with patch.object(self.app.media, "capture") as capture:
            self.app.capture_image()
            self.assertIs(capture.call_args.args[0], frame)
            self.app._preview_received_at -= 2
            self.app.capture_image()
            capture.assert_called_once()

    def test_zoom_slider_returns_to_zero_on_mouse_release(self):
        self.control._status = OperatorStatus(hardware=True, armed=True, zoom_supported=True)
        self.app._tick()
        with patch.object(self.control, "set_manual") as move:
            self.app.zoom_scale.set(-0.22)
            self.app.root.update()
            move.assert_called_with(0.0, 0.0, zoom=-0.22)
            self.app.root.event_generate("<ButtonRelease-1>")
            self.app.root.update()
            self.assertEqual(self.app.zoom.get(), 0)
            move.assert_called_with(0.0, 0.0)

    def test_manual_speed_reaches_75_percent(self):
        self.control._status = OperatorStatus(hardware=True, armed=True)
        self.app.speed_scale.set(0.75)
        self.app.root.update()
        self.assertEqual(self.app.speed_label["text"], "75 %")
        with patch.object(self.control, "set_manual") as set_manual:
            self.app._press(SimpleNamespace(widget=self.app.direction_buttons[0]), 1, -1)
            set_manual.assert_called_once_with(0.75, -0.75)

    def test_confidence_and_multiple_classes_require_apply(self):
        before = self.tracking_options.snapshot
        self.app.notebook.select(self.app.tracking_tab)
        self.app.tracking_confidence_scale.set(0.64)
        self.app.tracking_class_list.selection_clear(0, "end")
        self.app.tracking_class_list.selection_set(0)
        self.app.tracking_class_list.selection_set(2)
        self.app.root.update()
        self.assertEqual(self.tracking_options.snapshot, before)
        self.assertEqual(self.app.tracking_confidence_label["text"], "64 %")
        self.app.tracking_apply_button.invoke()
        after = self.tracking_options.snapshot
        self.assertEqual(after.min_confidence, 0.64)
        self.assertEqual(after.allowed_classes, frozenset({"drone", "person"}))
        self.assertIn("Sólo esta sesión", self.app.tracking_status["text"])

    def test_select_all_and_none_have_explicit_meaning(self):
        self.app._select_tracking_classes(False)
        self.app.tracking_apply_button.invoke()
        self.assertEqual(self.tracking_options.snapshot.allowed_classes, frozenset())
        self.assertIn("seguimiento automático detenido", self.app.tracking_status["text"])
        self.app._select_tracking_classes(True)
        self.app.tracking_apply_button.invoke()
        self.assertEqual(self.tracking_options.snapshot.allowed_classes,
                         frozenset({"drone", "bird", "person"}))

    def test_mouse_toggles_multiple_classes_without_modifier(self):
        self.app.notebook.select(self.app.tracking_tab)
        self.app._select_tracking_classes(False)
        self.app.root.update()
        choices = self.app.tracking_class_list
        for index in (0, 1):
            x, y, _, height = choices.bbox(index)
            choices.event_generate("<ButtonPress-1>", x=x + 3, y=y + height // 2)
            choices.event_generate("<ButtonRelease-1>", x=x + 3, y=y + height // 2)
            self.app.root.update()
        self.assertEqual(choices.curselection(), (0, 1))
        x, y, _, height = choices.bbox(0)
        choices.event_generate("<ButtonPress-1>", x=x + 3, y=y + height // 2)
        choices.event_generate("<ButtonRelease-1>", x=x + 3, y=y + height // 2)
        self.app.root.update()
        self.assertEqual(choices.curselection(), (1,))
        self.app.tracking_apply_button.invoke()
        self.assertEqual(self.tracking_options.snapshot.allowed_classes, frozenset({"bird"}))

    def test_tracking_controls_fit_compact_window(self):
        self.app.root.geometry("900x640")
        self.app.notebook.select(self.app.tracking_tab)
        self.app.root.update()
        panel = self.app.tracking_tab
        for widget in (self.app.tracking_confidence_scale, self.app.tracking_class_list,
                       self.app.tracking_apply_button):
            with self.subTest(widget=widget):
                self.assertTrue(widget.winfo_ismapped())
                self.assertGreaterEqual(widget.winfo_rootx(), panel.winfo_rootx())
                self.assertGreaterEqual(widget.winfo_rooty(), panel.winfo_rooty())
                self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                     panel.winfo_rootx() + panel.winfo_width())
                self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                     panel.winfo_rooty() + panel.winfo_height())

    def test_space_stops_when_tracking_classes_have_focus(self):
        self.app.notebook.select(self.app.tracking_tab)
        self.app.tracking_class_list.focus_force()
        self.app.root.update()
        selected = self.app.tracking_class_list.curselection()
        with patch.object(self.control, "request") as request:
            self.app.tracking_class_list.event_generate("<KeyPress-space>")
            self.app.tracking_class_list.event_generate("<KeyRelease-space>")
            self.app.root.update()
            request.assert_called_once_with("emergency")
        self.assertEqual(self.app.tracking_class_list.curselection(), selected)

    def test_space_stops_even_when_arm_button_has_focus(self):
        self.control._status = OperatorStatus(hardware=True)
        self.app.arm_button.configure(state="normal")
        self.app.arm_button.focus_force()
        self.app.root.update()
        with patch.object(self.control, "request") as request, patch("tkinter.messagebox.askyesno") as dialog:
            self.app.arm_button.event_generate("<KeyPress-space>")
            self.app.arm_button.event_generate("<KeyRelease-space>")
            self.app.root.update()
            request.assert_called_once_with("emergency")
            dialog.assert_not_called()


if __name__ == "__main__":
    unittest.main()
