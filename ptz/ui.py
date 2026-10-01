"""Panel de operación Tk: video limpio, control PTZ y registro plegable."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
from time import monotonic
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk

from ptz.config import SecretRedactionFilter
from ptz.operator_control import OperatorControl
from ptz.ptz_executor import ARM_CONFIRMATION
from ptz.preview import PreviewAnnotations, render_preview
from ptz.stream import RTSPStream
from ptz.tracking_settings import TrackingOptions

BG = "#0b1120"
PANEL = "#131e30"
SURFACE = "#1e2d43"
TEXT = "#e7edf6"
MUTED = "#93a6bf"
ACCENT = "#38bdf8"
GREEN = "#34d399"
RED = "#fb7185"


@dataclass(frozen=True)
class DashboardFrame:
    frame: NDArray[np.uint8] | None
    telemetry: tuple[str, ...]
    ptz: tuple[str, ...]
    connected: bool
    fps: float = 0.0
    detections: int = 0
    captured_at: float | None = None
    annotations: PreviewAnnotations | None = None


class FrameMailbox:
    """Conserva sólo el último resultado; nunca forma una cola de video."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: DashboardFrame | None = None

    def publish(self, snapshot: DashboardFrame) -> None:
        with self._lock:
            self._pending = snapshot

    def take(self) -> DashboardFrame | None:
        with self._lock:
            snapshot, self._pending = self._pending, None
            return snapshot


class UILogHandler(logging.Handler):
    """Los hilos publican texto; sólo el hilo de Tk modifica widgets."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.addFilter(SecretRedactionFilter())
        self.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
        self._lines: deque[tuple[str, str]] = deque(maxlen=2000)

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append((self.format(record), record.levelname))

    def drain(self) -> list[tuple[str, str]]:
        with self.lock:
            lines = list(self._lines)
            self._lines.clear()
        return lines


class DetectcamDashboard:
    """Toda operación sobre Tk permanece en el hilo principal."""

    def __init__(
        self, control: OperatorControl, mailbox: FrameMailbox,
        log_handler: UILogHandler, *, root: tk.Tk | None = None,
        preview_source: RTSPStream | None = None,
        tracking_options: TrackingOptions | None = None,
    ) -> None:
        self.root = root or tk.Tk()
        self.control = control
        self.mailbox = mailbox
        self.log_handler = log_handler
        self.preview_source = preview_source
        self.tracking_options = tracking_options if tracking_options is not None else TrackingOptions()
        self._preview_frame_id = -1
        self._preview_received_at: float | None = None
        self._preview_shape: tuple[int, ...] | None = None
        self.closed = False
        self._holding = (0, 0)
        self._held_button: tk.Button | None = None
        self._image: Image.Image | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._last_tick = monotonic()
        self._last_data_update = 0.0
        self._snapshot: DashboardFrame | None = None
        self._log_visible = True
        self._dialog_open = False
        self._after_id: str | None = None
        self.root.title("Detectcam · Centro de control PTZ")
        self.root.configure(bg=BG)
        width = min(1440, self.root.winfo_screenwidth() - 60)
        height = min(920, self.root.winfo_screenheight() - 80)
        self.root.geometry(f"{max(900, width)}x{max(640, height)}")
        self.root.minsize(900, 640)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        self.root.bind("<ButtonRelease-1>", self._release, add="+")
        self.root.bind("<FocusOut>", self._focus_out, add="+")
        self.root.bind("<Unmap>", self._unmapped, add="+")
        self.root.bind_class("DetectcamEmergency", "<KeyPress-space>", self._emergency_key)
        self.root.bind_class("DetectcamEmergency", "<KeyRelease-space>", lambda _: "break")
        self._install_emergency_binding(self.root)
        self.root.bind("<KeyPress-a>", lambda event: self._shortcut(event, self.arm))
        self.root.bind("<KeyPress-d>", lambda event: self._shortcut(event, self.disarm))
        self.root.bind("<KeyPress-q>", lambda event: self._shortcut(event, self.close))
        self._tick()

    def _install_emergency_binding(self, widget) -> None:
        widget.bindtags(("DetectcamEmergency",) + widget.bindtags())
        for child in widget.winfo_children():
            self._install_emergency_binding(child)

    def _style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook", background=PANEL, borderwidth=0)
        style.configure("TNotebook.Tab", background=SURFACE, foreground=MUTED, padding=(18, 8))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", ACCENT)])
        style.configure("Vertical.TScrollbar", background=SURFACE, troughcolor=PANEL, borderwidth=0, arrowcolor=MUTED)

    @staticmethod
    def _label(parent, text="", *, color=TEXT, size=10, bold=False, **kwargs):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=("DejaVu Sans", size, "bold" if bold else "normal"), **kwargs)

    @staticmethod
    def _button(parent, text, command=None, *, bg=SURFACE, fg=TEXT, **kwargs):
        return tk.Button(parent, text=text, command=command, bg=bg, fg=fg,
                         activebackground="#304864", activeforeground=TEXT,
                         disabledforeground="#64748b", relief="flat", bd=0,
                         cursor="hand2", font=("DejaVu Sans", 10, "bold"),
                         padx=12, pady=10, highlightthickness=0, **kwargs)

    def _build(self) -> None:
        header = tk.Frame(self.root, bg=BG, padx=22, pady=16)
        header.pack(fill="x")
        self._label(header, "◉  DETECTCAM", size=17, bold=True).pack(side="left")
        self._label(header, "  /  Centro de control PTZ", color=MUTED).pack(side="left", padx=12)
        self.connection = self._label(header, "●  Conectando video", color=MUTED)
        self.connection.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 16))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, bg="#050a13", highlightthickness=1, highlightbackground=SURFACE)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self._video_item = self.canvas.create_image(0, 0, anchor="center")
        self._placeholder = self.canvas.create_text(0, 0, text="Esperando video de la cámara…",
                                                   fill=MUTED, font=("DejaVu Sans", 14))
        self.canvas.bind("<Configure>", lambda _: self._render_video())
        video_bar = tk.Frame(left, bg=PANEL, padx=12, pady=8)
        video_bar.grid(row=1, column=0, sticky="ew")
        self.video_info = self._label(video_bar, "VIDEO EN VIVO", color=MUTED, size=9)
        self.video_info.pack(side="left")
        self.log_toggle = tk.Button(
            video_bar, text="▾  Registro", command=self.toggle_log, bg=PANEL, fg=ACCENT,
            activebackground=SURFACE, activeforeground=TEXT, relief="flat", bd=0,
            font=("DejaVu Sans", 10, "bold"), cursor="hand2", highlightthickness=0)
        self.log_toggle.pack(side="right")

        self.log_panel = tk.Frame(left, bg=PANEL, height=260)
        self.log_panel.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.log_panel.grid_propagate(False)
        self.log_panel.columnconfigure(0, weight=1)
        self.log_panel.rowconfigure(0, weight=1)
        self.notebook = ttk.Notebook(self.log_panel)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        data = tk.Frame(self.notebook, bg=PANEL)
        events = tk.Frame(self.notebook, bg=PANEL)
        self.notebook.add(data, text="Estado actual")
        self.notebook.add(events, text="Eventos")
        data.columnconfigure((0, 1), weight=1)
        data.rowconfigure(1, weight=1)
        self._label(data, "VIDEO Y SEGUIMIENTO", color=ACCENT, size=9, bold=True).grid(row=0, column=0, sticky="w", padx=12, pady=6)
        self._label(data, "TELEMETRÍA PTZ", color=ACCENT, size=9, bold=True).grid(row=0, column=1, sticky="w", padx=12, pady=6)
        self.telemetry = self._text_area(data, row=1, column=0)
        self.ptz_telemetry = self._text_area(data, row=1, column=1)
        events.columnconfigure(0, weight=1)
        events.rowconfigure(1, weight=1)
        event_bar = tk.Frame(events, bg=PANEL)
        event_bar.grid(row=0, column=0, sticky="ew")
        self.pause_log = tk.BooleanVar(value=False)
        tk.Checkbutton(event_bar, text="Pausar registro", variable=self.pause_log,
                       bg=PANEL, fg=MUTED, selectcolor=SURFACE, activebackground=PANEL,
                       activeforeground=TEXT).pack(side="left")
        tk.Button(event_bar, text="Limpiar", command=self.clear_log, bg=SURFACE, fg=TEXT,
                  relief="flat", padx=10).pack(side="right", padx=8, pady=4)
        self.events = self._text_area(events, row=1, column=0)
        self.events.tag_configure("WARNING", foreground="#fbbf24")
        self.events.tag_configure("ERROR", foreground=RED)
        self.events.tag_configure("CRITICAL", foreground=RED)
        self._build_tracking_tab()

        sidebar_shell = tk.Frame(body, bg=PANEL, width=282)
        sidebar_shell.grid(row=0, column=1, sticky="ns")
        sidebar_shell.pack_propagate(False)
        # La parada siempre queda visible, incluso en ventanas pequeñas.
        footer = tk.Frame(sidebar_shell, bg=PANEL, padx=14, pady=12)
        footer.pack(side="bottom", fill="x")
        self._label(footer, "Espacio: parar y desarmar", color=MUTED, size=8).pack(pady=(0, 8))
        self.emergency_button = self._button(
            footer, "■  PARADA DE EMERGENCIA", self.emergency, bg="#9f1239",
        )
        self.emergency_button.pack(fill="x")
        self.controls_canvas = tk.Canvas(sidebar_shell, bg=PANEL, highlightthickness=0, width=260)
        controls_scroll = ttk.Scrollbar(sidebar_shell, command=self.controls_canvas.yview)
        controls_scroll.pack(side="right", fill="y")
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        self.controls_canvas.configure(yscrollcommand=controls_scroll.set)
        sidebar = tk.Frame(self.controls_canvas, bg=PANEL, padx=14, pady=18)
        content = self.controls_canvas.create_window(0, 0, anchor="nw", window=sidebar)
        sidebar.bind("<Configure>", lambda _: self.controls_canvas.configure(
            scrollregion=self.controls_canvas.bbox("all")))
        self.controls_canvas.bind("<Configure>", lambda event: self.controls_canvas.itemconfigure(
            content, width=event.width))
        self._label(sidebar, "CONTROL PTZ", color=MUTED, size=10, bold=True).pack(anchor="w")
        self.armed_label = self._label(sidebar, "●  DESARMADO", color=MUTED, size=18, bold=True)
        self.armed_label.pack(anchor="w", pady=(12, 2))
        self.hardware_label = self._label(sidebar, "Comprobando hardware", color=MUTED, size=9)
        self.hardware_label.pack(anchor="w", pady=(0, 16))
        self.arm_button = self._button(sidebar, "Armar PTZ", self.arm, bg="#075985")
        self.arm_button.pack(fill="x", pady=(0, 7))
        self.disarm_button = self._button(sidebar, "Desarmar", self.disarm)
        self.disarm_button.pack(fill="x")
        tk.Frame(sidebar, bg=SURFACE, height=1).pack(fill="x", pady=18)
        self._label(sidebar, "MODO DE CONTROL", color=MUTED, size=9, bold=True).pack(anchor="w")
        self.mode = tk.StringVar(value="manual")
        for label, value in (("Manual", "manual"), ("Seguimiento automático", "automatic")):
            tk.Radiobutton(sidebar, text=label, value=value, variable=self.mode,
                           command=self.change_mode, bg=PANEL, fg=TEXT, selectcolor=SURFACE,
                           activebackground=PANEL, activeforeground=ACCENT,
                           font=("DejaVu Sans", 10), highlightthickness=0).pack(anchor="w", pady=(8, 0))
        self._label(sidebar, "Cambiar de modo desarma el PTZ.", color=MUTED, size=8).pack(anchor="w", pady=(6, 16))

        pad = tk.Frame(sidebar, bg=PANEL)
        pad.pack()
        self.direction_buttons: list[tk.Button] = []
        for label, row, column, dx, dy in (
            ("↖", 0, 0, -1, -1), ("↑", 0, 1, 0, -1), ("↗", 0, 2, 1, -1),
            ("←", 1, 0, -1, 0), ("→", 1, 2, 1, 0),
            ("↙", 2, 0, -1, 1), ("↓", 2, 1, 0, 1), ("↘", 2, 2, 1, 1),
        ):
            button = self._button(pad, label, width=2)
            button.grid(row=row, column=column, padx=3, pady=3)
            button.bind("<ButtonPress-1>", lambda event, x=dx, y=dy: self._press(event, x, y))
            button.bind("<Leave>", self._release)
            self.direction_buttons.append(button)
        self._button(pad, "■", self._release, width=2, fg=RED).grid(row=1, column=1, padx=3, pady=3)
        self._label(sidebar, "Mantén pulsado para mover.", color=MUTED, size=9).pack(pady=(10, 6))
        speed_row = tk.Frame(sidebar, bg=PANEL)
        speed_row.pack(fill="x")
        self._label(speed_row, "Velocidad", color=MUTED, size=9).pack(side="left")
        self.speed_label = self._label(speed_row, "15 %", color=ACCENT, size=9)
        self.speed_label.pack(side="right")
        self.speed = tk.DoubleVar(value=0.15)
        self.speed_scale = tk.Scale(sidebar, from_=0.05, to=0.75, resolution=0.01,
                                   variable=self.speed, orient="horizontal", showvalue=False,
                                   command=lambda value: self.speed_label.config(text=f"{float(value)*100:.0f} %"),
                                   bg=PANEL, fg=TEXT, troughcolor=SURFACE, activebackground=ACCENT,
                                   highlightthickness=0, bd=0)
        self.speed_scale.pack(fill="x")
        self._label(sidebar, "Pan / tilt · máximo 75 %", color=MUTED, size=8).pack(anchor="w")
        self.reason = self._label(sidebar, "", color=MUTED, size=9, wraplength=242, justify="left")
        self.reason.pack(anchor="w", pady=12)

    def _build_tracking_tab(self) -> None:
        self.tracking_tab = tk.Frame(self.notebook, bg=PANEL, padx=10, pady=6)
        self.notebook.add(self.tracking_tab, text="Seguimiento")
        self.tracking_tab.columnconfigure((0, 1), weight=1, uniform="tracking")
        self.tracking_tab.rowconfigure(0, weight=1)
        settings = self.tracking_options.snapshot

        confidence_panel = tk.Frame(self.tracking_tab, bg=PANEL)
        confidence_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._label(confidence_panel, "Confianza mínima para mover", color=ACCENT,
                    size=9, bold=True).pack(anchor="w")
        self.tracking_confidence = tk.DoubleVar(value=settings.min_confidence)
        self.tracking_confidence_label = self._label(
            confidence_panel, f"{settings.min_confidence * 100:.0f} %", size=12, bold=True,
        )
        self.tracking_confidence_label.pack(anchor="w", pady=(8, 0))
        self.tracking_confidence_scale = tk.Scale(
            confidence_panel, from_=0.01, to=1.0, resolution=0.01,
            variable=self.tracking_confidence, orient="horizontal", showvalue=False,
            command=self._tracking_confidence_changed, bg=PANEL, fg=TEXT,
            troughcolor=SURFACE, activebackground=ACCENT, highlightthickness=0, bd=0,
        )
        self.tracking_confidence_scale.pack(fill="x")
        self._label(confidence_panel, "Sin objetos seleccionados, el\nseguimiento automático se detiene.",
                    color=MUTED, size=8, justify="left").pack(anchor="w", pady=(6, 0))

        objects_panel = tk.Frame(self.tracking_tab, bg=PANEL)
        objects_panel.grid(row=0, column=1, sticky="nsew")
        objects_panel.columnconfigure(0, weight=1)
        objects_panel.rowconfigure(1, weight=1)
        self._label(objects_panel, "Objetos a seguir", color=ACCENT,
                    size=9, bold=True).grid(row=0, column=0, columnspan=2, sticky="w")
        self.tracking_classes = tuple(dict.fromkeys(self.tracking_options.class_names.values()))
        self.tracking_class_list = tk.Listbox(
            objects_panel, selectmode=tk.MULTIPLE, exportselection=False, height=3, width=16,
            bg=SURFACE, fg=TEXT, selectbackground="#075985", selectforeground=TEXT,
            activestyle="dotbox", font=("DejaVu Sans", 9), relief="flat", bd=0,
            highlightthickness=0,
        )
        self.tracking_class_list.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        scroll = ttk.Scrollbar(objects_panel, command=self.tracking_class_list.yview)
        scroll.grid(row=1, column=1, sticky="ns", pady=(5, 0))
        self.tracking_class_list.configure(yscrollcommand=scroll.set)
        for index, label in enumerate(self.tracking_classes):
            self.tracking_class_list.insert("end", label)
            if label.strip().casefold() in settings.allowed_classes:
                self.tracking_class_list.selection_set(index)
        if not self.tracking_classes:
            self.tracking_class_list.insert("end", "No hay clases disponibles")
            self.tracking_class_list.configure(state="disabled")
        self.tracking_class_list.bind("<<ListboxSelect>>", self._tracking_pending)
        selection_bar = tk.Frame(objects_panel, bg=PANEL)
        selection_bar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(3, 0))
        for label, select in (("Todos", True), ("Ninguno", False)):
            tk.Button(selection_bar, text=label, command=lambda value=select: self._select_tracking_classes(value),
                      bg=SURFACE, fg=TEXT, relief="flat", padx=8, pady=2,
                      state="normal" if self.tracking_classes else "disabled").pack(side="left", padx=(0, 5))
        self._label(objects_panel, "Clic: marcar / desmarcar uno o varios", color=MUTED,
                    size=8).grid(row=3, column=0, columnspan=2, sticky="w", pady=(3, 0))

        actions = tk.Frame(self.tracking_tab, bg=PANEL)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        actions.columnconfigure(0, weight=1)
        self.tracking_status = self._label(actions, "Ajustes sólo para esta sesión.",
                                           color=MUTED, size=8, justify="left", anchor="w")
        self.tracking_status.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.tracking_status.bind("<Configure>", lambda event: self.tracking_status.configure(
            wraplength=max(1, event.width)))
        self.tracking_apply_button = self._button(actions, "Aplicar", self.apply_tracking, bg="#075985")
        self.tracking_apply_button.grid(row=0, column=1, sticky="e")

    def _tracking_confidence_changed(self, value: str) -> None:
        self.tracking_confidence_label.configure(text=f"{float(value) * 100:.0f} %")
        self._tracking_pending()

    def _tracking_pending(self, _event=None) -> None:
        self.tracking_status.configure(text="Cambios pendientes · Pulsa Aplicar.\nSólo esta sesión.", fg=MUTED)

    def _select_tracking_classes(self, select: bool) -> None:
        if select:
            self.tracking_class_list.selection_set(0, "end")
        else:
            self.tracking_class_list.selection_clear(0, "end")
        self._tracking_pending()

    def apply_tracking(self) -> None:
        selected = [self.tracking_classes[index] for index in self.tracking_class_list.curselection()]
        try:
            settings = self.tracking_options.update(
                min_confidence=self.tracking_confidence.get(), allowed_classes=selected,
            )
        except (ValueError, tk.TclError) as error:
            self.tracking_status.configure(text=str(error), fg=RED)
            return
        if selected:
            text = f"Aplicado · {settings.min_confidence * 100:.0f} % · {len(selected)} clases.\nSólo esta sesión."
        else:
            text = "Sin objetos: seguimiento automático detenido.\nSólo esta sesión."
        self.tracking_status.configure(text=text, fg=GREEN)

    @staticmethod
    def _text_area(parent, *, row: int, column: int) -> tk.Text:
        container = tk.Frame(parent, bg=PANEL)
        container.grid(row=row, column=column, sticky="nsew", padx=6, pady=(0, 6))
        text = tk.Text(container, height=8, width=20, bg=PANEL, fg=TEXT,
                       insertbackground=TEXT, relief="flat", wrap="word",
                       font=("DejaVu Sans Mono", 9), padx=6, pady=4, state="disabled",
                       highlightthickness=0)
        scroll = ttk.Scrollbar(container, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        return text

    def toggle_log(self) -> None:
        self._log_visible = not self._log_visible
        if self._log_visible:
            self.log_panel.grid()
        else:
            self.log_panel.grid_remove()
        self.log_toggle.configure(text=("▾" if self._log_visible else "▸") + "  Registro")

    def clear_log(self) -> None:
        self.log_handler.drain()
        self._set_text(self.events, "")

    def arm(self) -> None:
        if self._dialog_open or not self.control.status.hardware or self.control.status.armed:
            return
        self._release()
        self._dialog_open = True
        try:
            confirmed = messagebox.askyesno(
                "Armar PTZ",
                "¿Seguro que quieres armar el PTZ?\nLa cámara podrá moverse en el modo seleccionado.",
                parent=self.root, default=messagebox.NO,
            )
            if confirmed:
                self.control.request("arm", ARM_CONFIRMATION)
        finally:
            self._dialog_open = False

    def disarm(self) -> None:
        self._release()
        self.control.request("disarm")

    def emergency(self) -> None:
        self._release()
        self.control.request("emergency")

    def change_mode(self) -> None:
        self._release()
        self.control.request("mode", self.mode.get())

    def _press(self, event, dx: int, dy: int) -> str:
        status = self.control.status
        if status.armed and status.mode == "manual" and self.mode.get() == "manual":
            self._holding = (dx, dy)
            self._held_button = event.widget
            event.widget.configure(bg="#075985")
            self._send_manual()
        return "break"

    def _send_manual(self) -> None:
        dx, dy = self._holding
        self.control.set_manual(dx * self.speed.get(), dy * self.speed.get())

    def _release(self, _event=None) -> None:
        self._holding = (0, 0)
        if self._held_button is not None:
            self._held_button.configure(bg=SURFACE)
            self._held_button = None
        self.control.set_manual(0.0, 0.0)

    def _focus_out(self, _event=None) -> None:
        # Se libera incluso al pasar a un diálogo de la aplicación.
        self._release()

    def _unmapped(self, event) -> None:
        if event.widget is self.root:
            self._release()

    def _shortcut(self, event, action) -> str | None:
        if self._dialog_open or isinstance(event.widget, (tk.Entry, tk.Text)):
            return None
        action()
        return "break"

    def _emergency_key(self, event) -> str | None:
        if self._dialog_open:
            return None
        self.emergency()
        return "break"

    def _tick(self) -> None:
        if self.closed:
            return
        now = monotonic()
        if now - self._last_tick > 0.25:
            self._release()
        self._last_tick = now
        self.control.heartbeat()
        status = self.control.status
        self.mode.set(status.mode)
        if not status.armed or status.mode != "manual":
            if self._holding != (0, 0):
                self._release()
        elif self._holding != (0, 0):
            self._send_manual()
        self.armed_label.configure(
            text="●  FALLO" if status.state == "FAULT" else ("●  ARMADO" if status.armed else "●  DESARMADO"),
            fg=RED if status.state == "FAULT" else (GREEN if status.armed else MUTED),
        )
        self.hardware_label.configure(text="Hardware PTZ habilitado" if status.hardware else "Simulación · hardware deshabilitado")
        self.arm_button.configure(state="normal" if status.hardware and not status.armed and status.state != "FAULT" else "disabled")
        self.disarm_button.configure(state="normal" if status.hardware else "disabled")
        manual = status.armed and status.mode == "manual" and self.mode.get() == "manual"
        for button in self.direction_buttons:
            button.configure(state="normal" if manual else "disabled")
        record = logging.LogRecord("ptz.ui", logging.INFO, "", 0, status.reason, (), None)
        SecretRedactionFilter().filter(record)
        self.reason.configure(text=record.getMessage())

        snapshot = self.mailbox.take()
        if snapshot is not None:
            self._snapshot = snapshot
        self._update_preview(snapshot, now)
        if self._snapshot is not None and now - self._last_data_update >= 0.25:
            self._last_data_update = now
            self._set_text(self.telemetry, "\n".join(self._snapshot.telemetry))
            self._set_text(self.ptz_telemetry, "\n".join(self._snapshot.ptz))
        if not self.pause_log.get():
            lines = self.log_handler.drain()
            if lines:
                at_bottom = self.events.yview()[1] >= 0.99
                self.events.configure(state="normal")
                for text, level in lines:
                    self.events.insert("end", text + "\n", level)
                excess = int(self.events.index("end-1c").split(".")[0]) - 2000
                if excess > 0:
                    self.events.delete("1.0", f"{excess + 1}.0")
                self.events.configure(state="disabled")
                if at_bottom:
                    self.events.see("end")
        self._after_id = self.root.after(40, self._tick)

    def _update_preview(self, result: DashboardFrame | None, now: float) -> None:
        """Lee la captura directamente: YOLO nunca bloquea la vista en vivo."""
        frame = None
        if self.preview_source is not None:
            packet = self.preview_source.read_packet(copy=False, after_id=self._preview_frame_id)
            connected = self.preview_source.is_connected()
            if packet is not None:
                frame = packet.frame
                self._preview_frame_id = packet.frame_id
                self._preview_received_at = packet.received_at
        else:
            connected = self._snapshot is not None and self._snapshot.connected
            if result is not None:
                frame = result.frame
                self._preview_received_at = result.captured_at
        if frame is not None and connected:
            annotations = None if self._snapshot is None else self._snapshot.annotations
            display = render_preview(frame, annotations, timestamp=now)
            self._image = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
            self._preview_shape = frame.shape
            self._render_video()
        if not connected:
            self._preview_received_at = None
            if self._image is not None:
                self._image = None
                self._render_video()
        self.control.preview_seen(self._preview_received_at if connected else None)
        age = None if self._preview_received_at is None else max(0.0, now - self._preview_received_at)
        stale = age is None or age > 0.5
        self.connection.configure(
            text=("●  Sin señal · reconectando" if not connected else
                  ("●  Video atrasado" if stale else "●  Video conectado")),
            fg="#fbbf24" if not connected or stale else GREEN,
        )
        resolution = "" if self._preview_shape is None else f"{self._preview_shape[1]} × {self._preview_shape[0]}  ·  "
        fps = 0.0 if self._snapshot is None else self._snapshot.fps
        count = 0 if self._snapshot is None else self._snapshot.detections
        age_text = "—" if age is None else f"{age * 1000:.0f} ms"
        self.video_info.configure(
            text=f"{resolution}IA: {fps:.1f} FPS · {count} detecciones · Edad local: {age_text}",
        )

    @staticmethod
    def _set_text(widget: tk.Text, value: str) -> None:
        position = widget.yview()[0]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")
        widget.yview_moveto(position)

    def _render_video(self) -> None:
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.canvas.coords(self._video_item, width / 2, height / 2)
        self.canvas.coords(self._placeholder, width / 2, height / 2)
        self.canvas.itemconfigure(self._placeholder, state="normal" if self._image is None else "hidden")
        if self._image is None:
            self.canvas.itemconfigure(self._video_item, image="")
            return
        ratio = min(width / self._image.width, height / self._image.height)
        size = (max(1, int(self._image.width * ratio)), max(1, int(self._image.height * ratio)))
        resized = self._image.resize(size, Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(resized, master=self.root)
        self.canvas.itemconfigure(self._video_item, image=self._photo)

    def close(self) -> None:
        if self.closed:
            return
        self._release()
        self.control.request("emergency")
        self.closed = True
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
