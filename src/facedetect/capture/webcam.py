from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class WebcamSource:
    """Simple cv2.VideoCapture wrapper with DirectShow on Windows.

    Use as a context manager or call open/close explicitly.
    """

    def __init__(self, device_index: int = 0, frame_width: int | None = None, frame_height: int | None = None):
        self.device_index = device_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fall back to default backend if DirectShow failed (rare, but possible on some USB cams).
            self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam index {self.device_index}")
        if self.frame_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        if self.frame_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        log.info(
            "Webcam opened: index=%d requested=%sx%s actual=%.0fx%.0f",
            self.device_index,
            self.frame_width, self.frame_height,
            self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamSource":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
