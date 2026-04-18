from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityConfig
from .types import DetectedFace, QualityReport


def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_brightness(gray: np.ndarray) -> float:
    return float(gray.mean())


def assess(
    face: DetectedFace,
    frame: np.ndarray,
    cfg: QualityConfig,
    *,
    strict: bool,
) -> QualityReport:
    """Evaluate whether a detected face is good enough to use.

    strict=True applies enrollment thresholds; strict=False uses relaxed recognition thresholds.
    """
    min_size = cfg.min_face_size_px if strict else cfg.min_face_size_px_recognition
    min_lap = cfg.min_laplacian_var if strict else cfg.min_laplacian_var_recognition

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox

    if strict:
        if x1 < cfg.edge_margin_px or y1 < cfg.edge_margin_px:
            return QualityReport(ok=False, reason="face too close to edge", face_size=face.min_side)
        if x2 > w - cfg.edge_margin_px or y2 > h - cfg.edge_margin_px:
            return QualityReport(ok=False, reason="face too close to edge", face_size=face.min_side)

    if face.min_side < min_size:
        return QualityReport(ok=False, reason=f"face too small ({face.min_side}px)", face_size=face.min_side)

    crop = face.crop(frame)
    if crop.size == 0:
        return QualityReport(ok=False, reason="empty crop", face_size=face.min_side)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = laplacian_variance(gray)
    bright = mean_brightness(gray)

    if lap < min_lap:
        return QualityReport(ok=False, reason=f"too blurry (lap {lap:.1f})", laplacian_var=lap, brightness=bright, face_size=face.min_side)
    if bright < cfg.min_brightness:
        return QualityReport(ok=False, reason=f"too dark ({bright:.0f})", laplacian_var=lap, brightness=bright, face_size=face.min_side)
    if bright > cfg.max_brightness:
        return QualityReport(ok=False, reason=f"too bright ({bright:.0f})", laplacian_var=lap, brightness=bright, face_size=face.min_side)

    return QualityReport(ok=True, laplacian_var=lap, brightness=bright, face_size=face.min_side)


def quality_score(report: QualityReport) -> float:
    """A bounded [0,1] score for ranking accepted samples — not used for gating.

    Rewards higher sharpness (laplacian variance) and mid-range brightness.
    """
    if not report.ok:
        return 0.0
    # Saturating laplacian at 500 (very sharp).
    sharp = min(report.laplacian_var / 500.0, 1.0)
    # Brightness closeness to 128 on 0-255 scale.
    bright_dist = abs(report.brightness - 128.0) / 128.0
    bright_score = max(0.0, 1.0 - bright_dist)
    return 0.7 * sharp + 0.3 * bright_score
