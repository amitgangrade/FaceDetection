from __future__ import annotations

import logging

import numpy as np

from .types import DetectedFace

log = logging.getLogger(__name__)


class InsightFaceAnalyzer:
    """Wraps InsightFace FaceAnalysis to provide detection + alignment + embedding
    in a single pass. Uses DirectML if available, CPU fallback otherwise.
    """

    def __init__(self, det_size: int = 640, det_threshold: float = 0.5):
        from insightface.app import FaceAnalysis  # lazy import

        providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        self._app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=det_threshold)

        active = self._active_providers()
        log.info("InsightFace FaceAnalysis ready. Active ONNX providers: %s", active)

    def _active_providers(self) -> list[str]:
        providers: set[str] = set()
        for model in getattr(self._app, "models", {}).values():
            session = getattr(model, "session", None)
            if session is not None:
                try:
                    providers.update(session.get_providers())
                except Exception:  # noqa: BLE001
                    pass
        return sorted(providers)

    def analyze(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        faces = self._app.get(frame_bgr)
        out: list[DetectedFace] = []
        for f in faces:
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            lm106 = getattr(f, "landmark_2d_106", None)
            detected = DetectedFace(
                bbox=(x1, y1, x2, y2),
                score=float(f.det_score),
                landmarks=np.asarray(f.kps) if getattr(f, "kps", None) is not None else None,
                landmark_2d_106=np.asarray(lm106) if lm106 is not None else None,
                embedding=self._normalize(np.asarray(f.normed_embedding))
                    if getattr(f, "normed_embedding", None) is not None
                    else None,
            )
            # Aligned crop for anti-spoof downstream. Try the standard InsightFace helper.
            try:
                from insightface.utils import face_align  # type: ignore
                if detected.landmarks is not None:
                    detected.aligned = face_align.norm_crop(frame_bgr, landmark=detected.landmarks, image_size=112)
            except Exception:  # noqa: BLE001
                detected.aligned = None
            out.append(detected)
        return out

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    def warmup(self, frame_bgr: np.ndarray) -> None:
        """One dummy forward pass so first real inference isn't the slow one."""
        try:
            self._app.get(frame_bgr)
        except Exception as e:  # noqa: BLE001
            log.warning("Warmup failed: %s", e)
