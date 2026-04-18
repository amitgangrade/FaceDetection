from __future__ import annotations

import cv2
import numpy as np


def draw_face_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    label: str,
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
    pad = 4
    ly2 = y1
    ly1 = y1 - th - baseline - pad
    if ly1 < 0:
        ly1 = y2
        ly2 = y2 + th + baseline + pad
    cv2.rectangle(frame, (x1, ly1), (x1 + tw + 2 * pad, ly2), color_bgr, -1)
    text_y = ly2 - pad if ly1 < y1 else ly1 + th + pad
    cv2.putText(frame, label, (x1 + pad, text_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(frame: np.ndarray, text: str, color_bgr: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_bgr, 1, cv2.LINE_AA)


def save_jpeg(frame: np.ndarray, path, quality: int = 85) -> None:
    from pathlib import Path as _P
    p = _P(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])


def make_thumbnail(crop_bgr: np.ndarray, size: int = 200) -> np.ndarray:
    h, w = crop_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    square = crop_bgr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(square, (size, size))
