from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectedFace:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    landmarks: np.ndarray | None = None  # shape (5, 2), five-point keypoints
    landmark_2d_106: np.ndarray | None = None  # shape (106, 2), dense landmarks
    # InsightFace returns aligned embedding on the Face object directly
    embedding: np.ndarray | None = None  # 512-D L2-normalized
    # Aligned 112x112 crop for downstream models (anti-spoof, etc.)
    aligned: np.ndarray | None = None

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def min_side(self) -> int:
        return min(self.width, self.height)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        h, w = frame.shape[:2]
        x1c = max(0, x1)
        y1c = max(0, y1)
        x2c = min(w, x2)
        y2c = min(h, y2)
        return frame[y1c:y2c, x1c:x2c]


@dataclass
class QualityReport:
    ok: bool
    reason: str | None = None
    laplacian_var: float = 0.0
    brightness: float = 0.0
    face_size: int = 0


@dataclass
class RecognitionResult:
    bbox: tuple[int, int, int, int]
    person_id: str | None  # None = unknown
    person_name: str | None
    similarity: float
    state: str  # "known" | "uncertain" | "unknown" | "spoof" | "warming"
    track_id: int
