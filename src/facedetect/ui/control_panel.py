from __future__ import annotations

import logging
import threading
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

from ..capture.webcam import WebcamSource
from ..config import AppConfig
from ..pipeline.detector import InsightFaceAnalyzer
from ..pipeline.liveness import BlinkDetector, PassiveAntiSpoof
from ..services.events import EventLogger
from ..services.recognition import RecognitionService
from ..storage.repo import PersonRepo
from ..util.metrics import SystemMetrics
from .enrollment_wizard import EnrollmentWizard
from .events_viewer import EventsViewer
from .live_view import run_live_view
from .people_manager import PeopleManager

log = logging.getLogger(__name__)

# Windows 11 Fluent-ish palette (stays readable without a custom theme dep).
BG = "#F3F3F3"
CARD_BG = "#FFFFFF"
BORDER = "#E5E5E5"
FG = "#1F1F1F"
FG_MUTED = "#616161"
ACCENT = "#0F6CBD"


class ControlPanel(tk.Tk):
    def __init__(
        self,
        *,
        cfg: AppConfig,
        repo: PersonRepo,
        analyzer: InsightFaceAnalyzer,
        antispoof: PassiveAntiSpoof | None,
        blink: BlinkDetector,
        event_logger: EventLogger,
        recognition: RecognitionService,
        metrics: SystemMetrics,
    ):
        super().__init__()
        self.title("FaceDetect")
        self.geometry("460x640")
        self.minsize(460, 640)
        self.configure(bg=BG)

        self.cfg = cfg
        self.repo = repo
        self.analyzer = analyzer
        self.antispoof = antispoof
        self.blink = blink
        self.event_logger = event_logger
        self.recognition = recognition
        self.metrics = metrics
        self._webcam: WebcamSource | None = None

        # Live-view thread state.
        self._live_thread: threading.Thread | None = None
        self._live_stop: threading.Event | None = None

        self._apply_styles()
        self._build_ui()
        self._schedule_metrics_update()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # --- styling ---

    def _apply_styles(self) -> None:
        style = ttk.Style(self)
        # 'vista' is native on Windows; 'clam' gives us more color control for
        # progress bars. We use 'clam' so the metrics bars can be tinted.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD_BG, relief="flat", borderwidth=1)

        style.configure("Brand.TLabel", background=BG, foreground=FG,
                        font=("Segoe UI Semibold", 22))
        style.configure("Subtitle.TLabel", background=BG, foreground=FG_MUTED,
                        font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=CARD_BG, foreground=FG_MUTED,
                        font=("Segoe UI Semibold", 9))
        style.configure("Metric.TLabel", background=CARD_BG, foreground=FG,
                        font=("Segoe UI", 10))
        style.configure("MetricValue.TLabel", background=CARD_BG, foreground=FG,
                        font=("Consolas", 10))

        # Action buttons — padded, left-aligned for icon + text.
        style.configure(
            "Action.TButton",
            font=("Segoe UI", 11),
            padding=(14, 10),
            anchor="w",
            background=CARD_BG,
            foreground=FG,
            borderwidth=1,
            relief="flat",
        )
        style.map(
            "Action.TButton",
            background=[("active", "#EDEDED"), ("pressed", "#E0E0E0")],
            bordercolor=[("!active", BORDER), ("active", ACCENT)],
        )

        # Primary (accent) button for the most common action.
        style.configure(
            "Primary.TButton",
            font=("Segoe UI Semibold", 11),
            padding=(14, 10),
            anchor="w",
            background=ACCENT,
            foreground="#FFFFFF",
            borderwidth=0,
            relief="flat",
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#0B5BA0"), ("pressed", "#094B84")],
            foreground=[("active", "#FFFFFF")],
        )

        style.configure(
            "Quit.TButton",
            font=("Segoe UI", 10),
            padding=(10, 6),
            background=BG,
            foreground=FG_MUTED,
            borderwidth=1,
            relief="flat",
        )
        style.map(
            "Quit.TButton",
            background=[("active", "#EDEDED")],
            foreground=[("active", FG)],
        )

        # Colored progress bars for each metric row.
        for name, color in (
            ("CPU.Horizontal.TProgressbar", "#0F6CBD"),
            ("GPU.Horizontal.TProgressbar", "#7A4FD0"),
            ("Mem.Horizontal.TProgressbar", "#107C10"),
        ):
            style.configure(
                name,
                troughcolor="#EFEFEF",
                background=color,
                bordercolor="#EFEFEF",
                lightcolor=color,
                darkcolor=color,
                thickness=6,
            )

    # --- layout ---

    def _build_ui(self) -> None:
        outer = ttk.Frame(self, padding=(20, 18, 20, 16))
        outer.pack(fill="both", expand=True)

        # Brand header
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="FaceDetect", style="Brand.TLabel").pack(anchor="w")
        subtitle = "AMD Radeon  •  DirectML  •  InsightFace buffalo_l"
        ttk.Label(header, text=subtitle, style="Subtitle.TLabel").pack(anchor="w")

        # Action buttons. We hold refs to the ones whose state changes while
        # live recognition is active.
        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(4, 12))
        self._start_btn_text_idle = "\u25B6    Start live recognition"
        self._start_btn_text_active = "\u25A0    Stop live recognition"
        self._start_btn = self._make_button(
            actions, self._start_btn_text_idle, self._toggle_recognition,
            style="Primary.TButton")
        self._enroll_btn = self._make_button(
            actions, "\uFF0B    Enroll new person", self._enroll_new,
            style="Action.TButton")
        self._manage_btn = self._make_button(
            actions, "\u2630    Manage people", self._manage_people,
            style="Action.TButton")
        self._events_btn = self._make_button(
            actions, "\u29D6    Recent events", self._show_events,
            style="Action.TButton")

        # Metrics card
        card = tk.Frame(outer, bg=CARD_BG, highlightbackground=BORDER,
                        highlightthickness=1, bd=0)
        card.pack(fill="x", pady=(4, 12))
        inner = ttk.Frame(card, style="Card.TFrame", padding=(14, 12))
        inner.pack(fill="x")
        ttk.Label(inner, text="SYSTEM USAGE", style="Section.TLabel").grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        self._cpu_label = ttk.Label(inner, text="CPU", style="Metric.TLabel")
        self._cpu_value = ttk.Label(inner, text="—", style="MetricValue.TLabel", anchor="e")
        self._cpu_bar = ttk.Progressbar(inner, style="CPU.Horizontal.TProgressbar",
                                        orient="horizontal", maximum=100)

        self._gpu_label = ttk.Label(inner, text="GPU", style="Metric.TLabel")
        self._gpu_value = ttk.Label(inner, text="—", style="MetricValue.TLabel", anchor="e")
        self._gpu_bar = ttk.Progressbar(inner, style="GPU.Horizontal.TProgressbar",
                                        orient="horizontal", maximum=100)

        self._mem_label = ttk.Label(inner, text="Memory", style="Metric.TLabel")
        self._mem_value = ttk.Label(inner, text="—", style="MetricValue.TLabel", anchor="e")
        self._mem_bar = ttk.Progressbar(inner, style="Mem.Horizontal.TProgressbar",
                                        orient="horizontal", maximum=100)

        inner.columnconfigure(1, weight=1)
        self._cpu_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=3)
        self._cpu_bar.grid(row=1, column=1, sticky="ew", pady=3)
        self._cpu_value.grid(row=1, column=2, sticky="e", padx=(10, 0), pady=3)
        self._gpu_label.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=3)
        self._gpu_bar.grid(row=2, column=1, sticky="ew", pady=3)
        self._gpu_value.grid(row=2, column=2, sticky="e", padx=(10, 0), pady=3)
        self._mem_label.grid(row=3, column=0, sticky="w", padx=(0, 10), pady=3)
        self._mem_bar.grid(row=3, column=1, sticky="ew", pady=3)
        self._mem_value.grid(row=3, column=2, sticky="e", padx=(10, 0), pady=3)

        # Push Quit to the bottom
        ttk.Frame(outer).pack(fill="both", expand=True)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", side="bottom")
        ttk.Button(footer, text="Quit", command=self._quit,
                   style="Quit.TButton").pack(side="right")

    def _make_button(self, parent: tk.Misc, text: str, command, *, style: str) -> ttk.Button:
        btn = ttk.Button(parent, text=text, command=command, style=style)
        btn.pack(fill="x", pady=3)
        return btn

    # --- metrics polling ---

    def _schedule_metrics_update(self) -> None:
        self._update_metrics()
        self.after(1000, self._schedule_metrics_update)

    def _update_metrics(self) -> None:
        try:
            s = self.metrics.snapshot()
        except Exception as e:  # noqa: BLE001
            log.debug("metrics snapshot failed: %s", e)
            return

        self._cpu_bar["value"] = s.cpu_proc_pct
        self._cpu_value.configure(text=f"{s.cpu_proc_pct:4.1f}%")

        if s.gpu_available and s.gpu_pct is not None:
            self._gpu_bar["value"] = s.gpu_pct
            if s.gpu_mem_mb is not None and s.gpu_mem_mb > 0:
                self._gpu_value.configure(text=f"{s.gpu_pct:4.1f}% • {s.gpu_mem_mb:,.0f} MB")
            else:
                self._gpu_value.configure(text=f"{s.gpu_pct:4.1f}%")
        elif not s.gpu_available:
            self._gpu_bar["value"] = 0
            self._gpu_value.configure(text="n/a")
        else:
            # Available but first sample hasn't landed yet.
            self._gpu_value.configure(text="…")

        # Show process RSS as the headline, with system % on the bar.
        self._mem_bar["value"] = s.mem_system_pct
        self._mem_value.configure(
            text=f"{s.mem_proc_mb:,.0f} MB  •  sys {s.mem_system_pct:4.1f}%"
        )

    # --- webcam lifecycle (shared across all actions) ---

    def _get_webcam(self) -> WebcamSource | None:
        if self._webcam is not None and self._webcam.is_open():
            return self._webcam
        try:
            w = WebcamSource(
                device_index=self.cfg.camera.device_index,
                frame_width=self.cfg.camera.frame_width,
                frame_height=self.cfg.camera.frame_height,
            )
            w.open()
            self._webcam = w
            return w
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Webcam error", f"Cannot open webcam: {e}", parent=self)
            return None

    def _release_webcam(self) -> None:
        if self._webcam is not None:
            self._webcam.close()
            self._webcam = None

    # --- actions ---

    # --- live recognition (runs on a background thread) ---

    def is_live_active(self) -> bool:
        return self._live_thread is not None and self._live_thread.is_alive()

    def _toggle_recognition(self) -> None:
        if self.is_live_active():
            self._stop_recognition()
        else:
            self._start_recognition()

    def _start_recognition(self) -> None:
        w = self._get_webcam()
        if w is None:
            return
        # Reload index in case we enrolled someone during this session.
        self.recognition.reload_index()

        self._live_stop = threading.Event()
        stop_event = self._live_stop
        webcam = w

        def worker() -> None:
            try:
                run_live_view(webcam, self.recognition, self.cfg, stop_event=stop_event)
            except Exception:  # noqa: BLE001
                log.exception("live recognition crashed")

        self._live_thread = threading.Thread(target=worker, name="live-view", daemon=True)
        self._live_thread.start()
        self._set_live_ui_active(True)
        self.after(200, self._poll_live_thread)

    def _stop_recognition(self) -> None:
        if self._live_stop is not None:
            self._live_stop.set()
        # The poll loop will notice the thread has exited and restore UI.

    def _poll_live_thread(self) -> None:
        if self.is_live_active():
            self.after(200, self._poll_live_thread)
            return
        # Thread finished — reclaim the camera and re-enable UI.
        self._live_thread = None
        self._live_stop = None
        self._release_webcam()
        self._set_live_ui_active(False)

    def _set_live_ui_active(self, active: bool) -> None:
        if active:
            self._start_btn.configure(text=self._start_btn_text_active)
            self._enroll_btn.state(["disabled"])
        else:
            self._start_btn.configure(text=self._start_btn_text_idle)
            self._enroll_btn.state(["!disabled"])

    def _enroll_new(self) -> None:
        if self.is_live_active():
            messagebox.showinfo(
                "Camera busy",
                "Stop live recognition before enrolling — they share the webcam.",
                parent=self,
            )
            return
        name = simpledialog.askstring("Enroll new person", "Name:", parent=self)
        if not name:
            return
        name = name.strip()
        if not name:
            return
        existing = self.repo.get_person_by_name(name)
        if existing is not None:
            if not messagebox.askyesno(
                "Already enrolled",
                f"'{name}' already exists. Add more samples instead?",
                parent=self,
            ):
                return
            self._run_wizard(name, existing.id)
            return
        self._run_wizard(name, None)

    def _add_samples_for(self, person_id: str, name: str) -> None:
        self._run_wizard(name, person_id)

    def _run_wizard(self, name: str, existing_person_id: str | None) -> None:
        if self.is_live_active():
            messagebox.showinfo(
                "Camera busy",
                "Stop live recognition before adding samples — they share the webcam.",
                parent=self,
            )
            return
        w = self._get_webcam()
        if w is None:
            return
        wizard = EnrollmentWizard(
            self,
            name=name,
            existing_person_id=existing_person_id,
            webcam=w,
            analyzer=self.analyzer,
            antispoof=self.antispoof,
            blink=self.blink,
            repo=self.repo,
            cfg=self.cfg,
        )
        wizard.grab_set()
        wizard.wait_window()
        self._release_webcam()
        if wizard.result_person_id is not None:
            self.recognition.reload_index()

    def _manage_people(self) -> None:
        mgr = PeopleManager(
            self, self.repo, self._add_samples_for,
            is_live_active=self.is_live_active,
        )
        mgr.grab_set()
        mgr.wait_window()
        self.recognition.reload_index()

    def _show_events(self) -> None:
        viewer = EventsViewer(self, self.repo)
        viewer.grab_set()

    def _quit(self) -> None:
        # Stop live recognition first if it's running — it owns the webcam.
        if self.is_live_active() and self._live_stop is not None:
            self._live_stop.set()
            if self._live_thread is not None:
                self._live_thread.join(timeout=2.0)
        self.metrics.stop()
        self._release_webcam()
        self.destroy()
