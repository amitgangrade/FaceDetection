from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..config import LivenessConfig
from .types import DetectedFace

log = logging.getLogger(__name__)

# InsightFace 2d106det landmark indices covering the two eyes.
# Source: the MenpoBenchmark-style 106-point face annotation used by InsightFace.
# Each eye is covered by a 10-point contour.
RIGHT_EYE_IDX = list(range(33, 43))   # subject's right (viewer's left)
LEFT_EYE_IDX = list(range(87, 97))    # subject's left  (viewer's right)


def _eye_openness(pts: np.ndarray) -> float:
    """Height / width ratio for a set of eye contour points. Higher = more open."""
    if pts.shape[0] < 4:
        return 0.0
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    w = x_max - x_min
    h = y_max - y_min
    if w < 1e-3:
        return 0.0
    return float(h / w)


class BlinkDetector:
    """Detects blinks using the 10-point eye contours from InsightFace's 106-point landmarks.

    Tracks an "openness" ratio (height/width) per eye, averaged across both eyes.
    A blink is recorded as a transition from closed (< closed_threshold) back to
    open (> open_threshold).
    """

    def __init__(self, closed_threshold: float, open_threshold: float):
        self.closed_threshold = closed_threshold
        self.open_threshold = open_threshold
        self._state_closed = False
        self._blink_count = 0
        self.last_openness: float | None = None

    def reset(self) -> None:
        self._state_closed = False
        self._blink_count = 0
        self.last_openness = None

    @property
    def blink_count(self) -> int:
        return self._blink_count

    def update_from_face(self, face: DetectedFace) -> float | None:
        """Feed one detected face; returns the combined eye openness ratio (or None if no landmarks)."""
        if face.landmark_2d_106 is None or face.landmark_2d_106.shape[0] < 106:
            self.last_openness = None
            return None
        lm = face.landmark_2d_106
        right = lm[RIGHT_EYE_IDX]
        left = lm[LEFT_EYE_IDX]
        openness = (_eye_openness(left) + _eye_openness(right)) / 2.0
        self.last_openness = openness

        if not self._state_closed and openness < self.closed_threshold:
            self._state_closed = True
        elif self._state_closed and openness > self.open_threshold:
            self._state_closed = False
            self._blink_count += 1
        return openness


class PassiveAntiSpoof:
    """MiniFASNet ONNX inference. Degrades gracefully if the model is absent.

    Expected input: HxW BGR face crop (aligned). Output: 2-way softmax (real, spoof)
    or a single sigmoid logit.
    """

    def __init__(self, model_path: Path, spoof_threshold: float):
        self.model_path = model_path
        self.spoof_threshold = spoof_threshold
        self._session = None
        self._input_name: str | None = None
        self._input_shape: tuple | None = None
        self.enabled = False
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            log.warning("Anti-spoof model not found at %s — passive liveness disabled.", self.model_path)
            return
        try:
            import onnxruntime as ort  # lazy import
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._input_shape = tuple(self._session.get_inputs()[0].shape)
            log.info("Anti-spoof model loaded (%s), providers: %s", self.model_path.name, self._session.get_providers())
            self.enabled = True
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load anti-spoof model: %s", e)
            self._session = None
            self.enabled = False

    def _target_hw(self) -> tuple[int, int]:
        if self._input_shape and len(self._input_shape) == 4:
            h = self._input_shape[2] if isinstance(self._input_shape[2], int) and self._input_shape[2] > 0 else 80
            w = self._input_shape[3] if isinstance(self._input_shape[3], int) and self._input_shape[3] > 0 else 80
            return int(h), int(w)
        return 80, 80

    def spoof_probability(self, face_crop_bgr: np.ndarray) -> float | None:
        if not self.enabled or self._session is None or face_crop_bgr.size == 0:
            return None
        try:
            th, tw = self._target_hw()
            resized = cv2.resize(face_crop_bgr, (tw, th))
            x = resized.astype(np.float32) / 255.0
            x = np.transpose(x, (2, 0, 1))[None, ...]
            out = self._session.run(None, {self._input_name: x})[0]
            logits = np.asarray(out).reshape(-1)
            if logits.size == 2:
                probs = np.exp(logits - logits.max())
                probs /= probs.sum()
                return float(probs[1])
            if logits.size == 1:
                return float(1.0 / (1.0 + np.exp(-float(logits[0]))))
            log.debug("Unexpected anti-spoof output shape: %s", logits.shape)
            return None
        except Exception as e:  # noqa: BLE001
            log.warning("Anti-spoof inference failed: %s", e)
            return None

    def is_spoof(self, face_crop_bgr: np.ndarray) -> bool:
        p = self.spoof_probability(face_crop_bgr)
        if p is None:
            return False  # fail-open — blink challenge + quality gates are the fallback
        return p >= self.spoof_threshold


def build_liveness(cfg: LivenessConfig, antispoof_model_path: Path) -> tuple[PassiveAntiSpoof | None, BlinkDetector]:
    passive: PassiveAntiSpoof | None = None
    if cfg.passive_enabled:
        passive = PassiveAntiSpoof(antispoof_model_path, cfg.passive_spoof_threshold)
    blink = BlinkDetector(cfg.blink_ear_closed, cfg.blink_ear_open)
    return passive, blink
