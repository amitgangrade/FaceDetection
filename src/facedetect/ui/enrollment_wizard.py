from __future__ import annotations

import logging
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from ..capture.webcam import WebcamSource
from ..config import AppConfig
from ..pipeline.detector import InsightFaceAnalyzer
from ..pipeline.liveness import BlinkDetector, PassiveAntiSpoof
from ..services.enrollment import EnrollmentSession, Phase
from ..storage.repo import PersonRepo

log = logging.getLogger(__name__)


class EnrollmentWizard(tk.Toplevel):
    PREVIEW_W = 480
    PREVIEW_H = 360

    def __init__(
        self,
        parent: tk.Misc,
        *,
        name: str,
        existing_person_id: str | None,
        webcam: WebcamSource,
        analyzer: InsightFaceAnalyzer,
        antispoof: PassiveAntiSpoof | None,
        blink: BlinkDetector,
        repo: PersonRepo,
        cfg: AppConfig,
    ):
        super().__init__(parent)
        self.title(f"Enroll: {name}")
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.resizable(False, False)

        self.webcam = webcam
        self.cfg = cfg
        self.session = EnrollmentSession(
            name=name,
            existing_person_id=existing_person_id,
            analyzer=analyzer,
            antispoof=antispoof,
            blink=blink,
            repo=repo,
            cfg=cfg,
        )
        self._finished = False
        self._result_person_id: str | None = None

        self._build_ui()
        self.after(30, self._tick)

    def _build_ui(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.grid(sticky="nsew")

        self.canvas = tk.Canvas(container, width=self.PREVIEW_W, height=self.PREVIEW_H, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=2)

        self.phase_var = tk.StringVar(value="Preparing...")
        ttk.Label(container, textvariable=self.phase_var, font=("Segoe UI", 14, "bold")).grid(row=1, column=0, columnspan=2, pady=(10, 4), sticky="w")

        self.prompt_var = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.prompt_var, font=("Segoe UI", 11)).grid(row=2, column=0, columnspan=2, sticky="w")

        self.feedback_var = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.feedback_var, foreground="#666").grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 8))

        self.progress = ttk.Progressbar(container, orient="horizontal", length=self.PREVIEW_W, mode="determinate")
        self.progress.grid(row=4, column=0, columnspan=2, pady=(0, 8), sticky="we")

        self.stats_var = tk.StringVar(value="")
        ttk.Label(container, textvariable=self.stats_var, font=("Consolas", 9)).grid(row=5, column=0, columnspan=2, sticky="w")

        button_row = ttk.Frame(container)
        button_row.grid(row=6, column=0, columnspan=2, pady=(10, 0), sticky="we")
        ttk.Button(button_row, text="Cancel", command=self._cancel).pack(side="right")

    # ------------- main loop --------------

    def _tick(self) -> None:
        if self._finished:
            return
        frame = self.webcam.read()
        if frame is None:
            self.feedback_var.set("No webcam frame — is the camera in use elsewhere?")
            self.after(50, self._tick)
            return

        try:
            step = self.session.push_frame(frame)
        except Exception:  # noqa: BLE001
            log.exception("enrollment step failed")
            self.feedback_var.set("Internal error — see logs")
            self.after(50, self._tick)
            return

        # Update preview (show full frame, not face crop, so user sees themselves).
        self._update_preview(frame)

        # Update text fields.
        phase_label = {
            Phase.PREFLIGHT: "Preflight: line up",
            Phase.CAPTURE: f"Pose {step.pose_index + 1} of {len(self.cfg.enrollment.poses)}",
            Phase.FINALIZING: "Finalizing...",
            Phase.DONE: "Done!",
            Phase.FAILED: "Failed",
        }[step.phase]
        self.phase_var.set(phase_label)
        self.prompt_var.set(step.prompt)
        self.feedback_var.set(step.feedback)
        self.progress["value"] = int(step.progress * 100)
        self.stats_var.set(
            f"samples: {step.total_samples}   pose samples: {step.samples_for_pose}   blinks: {step.blink_count}"
        )

        if step.phase is Phase.DONE:
            self._finished = True
            self._result_person_id = step.final_person_id
            messagebox.showinfo(
                "Enrollment complete",
                f"{self.session.name} enrolled with {step.total_samples} samples.",
                parent=self,
            )
            self.destroy()
            return
        if step.phase is Phase.FAILED:
            self._finished = True
            messagebox.showerror(
                "Enrollment failed",
                step.error or "Unknown error",
                parent=self,
            )
            self.destroy()
            return

        self.after(33, self._tick)

    def _update_preview(self, frame_bgr) -> None:
        # Fit BGR frame into preview box while preserving aspect ratio.
        h, w = frame_bgr.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self._photo_ref = photo  # prevent GC
        self.canvas.delete("all")
        self.canvas.create_image(
            self.PREVIEW_W // 2,
            self.PREVIEW_H // 2,
            image=photo,
            anchor="center",
        )

    # ------------- cancel --------------

    def _cancel(self) -> None:
        if not self._finished:
            self.session.cancel()
        self._finished = True
        self.destroy()

    # ------------- API --------------

    @property
    def result_person_id(self) -> str | None:
        return self._result_person_id

    def wait(self) -> str | None:
        self.wait_window()
        return self._result_person_id
