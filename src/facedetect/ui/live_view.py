from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from ..capture.webcam import WebcamSource
from ..config import AppConfig
from ..pipeline.types import RecognitionResult
from ..services.recognition import RecognitionService
from ..util.image_ops import draw_face_label, draw_status

log = logging.getLogger(__name__)


def _color_for_state(state: str) -> tuple[int, int, int]:
    # BGR
    return {
        "known": (80, 200, 80),
        "uncertain": (0, 200, 220),
        "unknown": (60, 60, 220),
        "spoof": (200, 60, 200),
        "warming": (180, 180, 180),
    }.get(state, (200, 200, 200))


def _format_label(r: RecognitionResult) -> str:
    if r.state == "known":
        return f"{r.person_name} ({r.similarity:.2f})"
    if r.state == "uncertain":
        name = r.person_name or "?"
        return f"maybe {name} ({r.similarity:.2f})"
    if r.state == "spoof":
        return "spoof attempt"
    if r.state == "warming":
        return "..."
    return f"unknown ({r.similarity:.2f})"


def run_live_view(
    webcam: WebcamSource,
    recognition: RecognitionService,
    cfg: AppConfig,
    stop_event: threading.Event | None = None,
) -> None:
    """Blocking loop: opens the OpenCV window, runs until the user presses 'q',
    closes the window, or the caller sets ``stop_event``."""
    window_name = cfg.ui.window_title
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Warm up the models once before the loop so the first frame isn't slow.
    frame = webcam.read()
    if frame is not None:
        recognition.analyzer.warmup(frame)

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        frame = webcam.read()
        if frame is None:
            log.warning("No frame from webcam — exiting live view")
            break

        try:
            results = recognition.step(frame)
        except Exception:  # noqa: BLE001
            log.exception("recognition step failed")
            results = []

        for r in results:
            draw_face_label(frame, r.bbox, _format_label(r), _color_for_state(r.state))

        draw_status(frame, f"FaceDetect — {len(results)} face(s)  [q to quit]", (255, 255, 255))

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        # If the user clicked the X on the window:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)
    # Process any leftover window events so the OS actually tears down the window.
    for _ in range(3):
        cv2.waitKey(1)
