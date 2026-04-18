import numpy as np

from facedetect.config import QualityConfig
from facedetect.pipeline.quality import assess, laplacian_variance, mean_brightness, quality_score
from facedetect.pipeline.types import DetectedFace


def _cfg(**overrides):
    defaults = dict(
        min_face_size_px=112,
        min_laplacian_var=60.0,
        min_brightness=40.0,
        max_brightness=220.0,
        edge_margin_px=8,
        min_face_size_px_recognition=80,
        min_laplacian_var_recognition=30.0,
    )
    defaults.update(overrides)
    return QualityConfig(**defaults)


def _make_frame(w=640, h=480, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _make_noisy_frame(w=640, h=480, mean=128, spread=40):
    """Frame with mean ≈ `mean` and a small uniform noise band of +/- spread/2.

    Small spread keeps `mean_brightness` predictable while providing enough
    Laplacian variance to satisfy the blur gate.
    """
    rng = np.random.default_rng(42)
    low = max(0, mean - spread // 2)
    high = min(255, mean + spread // 2)
    return rng.integers(low, high + 1, size=(h, w, 3), dtype=np.uint8)


def test_laplacian_variance_flat_is_zero():
    gray = np.full((100, 100), 128, dtype=np.uint8)
    assert laplacian_variance(gray) == 0.0


def test_mean_brightness():
    gray = np.full((10, 10), 150, dtype=np.uint8)
    assert mean_brightness(gray) == 150.0


def test_assess_rejects_too_small():
    cfg = _cfg()
    frame = _make_noisy_frame()
    face = DetectedFace(bbox=(100, 100, 160, 160), score=0.95)  # 60x60 < 112
    report = assess(face, frame, cfg, strict=True)
    assert not report.ok
    assert "too small" in report.reason.lower()


def test_assess_rejects_blurry_face():
    cfg = _cfg()
    frame = _make_frame()  # flat image -> laplacian var == 0
    face = DetectedFace(bbox=(100, 100, 250, 250), score=0.95)
    report = assess(face, frame, cfg, strict=True)
    assert not report.ok
    assert "blurry" in report.reason.lower()


def test_assess_rejects_too_dark():
    cfg = _cfg()
    frame = _make_noisy_frame(mean=10)
    face = DetectedFace(bbox=(100, 100, 250, 250), score=0.95)
    report = assess(face, frame, cfg, strict=True)
    assert not report.ok
    assert "dark" in report.reason.lower()


def test_assess_rejects_edge_face_strict():
    cfg = _cfg()
    frame = _make_noisy_frame()
    face = DetectedFace(bbox=(0, 0, 200, 200), score=0.95)
    report = assess(face, frame, cfg, strict=True)
    assert not report.ok
    assert "edge" in report.reason.lower()


def test_assess_allows_edge_face_relaxed():
    cfg = _cfg()
    frame = _make_noisy_frame()
    face = DetectedFace(bbox=(0, 0, 200, 200), score=0.95)
    # Relaxed mode doesn't enforce edge margin.
    report = assess(face, frame, cfg, strict=False)
    assert report.ok or "edge" not in (report.reason or "")


def test_quality_score_monotonic():
    cfg = _cfg()
    frame = _make_noisy_frame()
    face = DetectedFace(bbox=(100, 100, 250, 250), score=0.95)
    report = assess(face, frame, cfg, strict=True)
    if report.ok:
        s = quality_score(report)
        assert 0.0 <= s <= 1.0
