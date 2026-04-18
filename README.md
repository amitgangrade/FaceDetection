# FaceDetect

A local Windows desktop app to register faces and identify people on camera.
Runs on AMD Radeon (or any DirectX-12) GPU via DirectML, falling back to CPU.

## Features

- Guided multi-pose enrollment with per-frame quality gates (blur, brightness, face size, edge margin).
- Passive anti-spoof (MiniFASNet ONNX, optional) plus active blink-challenge liveness
  using InsightFace's 106-point face landmarks (no extra model required).
- Continuous recognition with IOU tracker and temporal smoothing (4-of-7 consecutive frames agree).
- SQLite persistence of people, embeddings, enrollment sessions, and recognition events.
- Unknown-person snapshot archive under `data/snapshots/YYYY/MM/DD/`.
- Per-person visit log with last-seen, similarity, and optional snapshots.
- Tkinter control panel: enroll, manage people, view recent events, start live recognition.

## Install

Requires Python 3.11 or 3.12 (64-bit).

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Optional — fetch the passive anti-spoof model (the app runs without it, relying on the blink challenge):

```bash
python scripts\download_models.py
```

## Run

```bash
python -m facedetect
```

On first launch, InsightFace auto-downloads the `buffalo_l` model pack
(~280 MB) into `%USERPROFILE%\.insightface\models`. Subsequent starts are fast.

## Configure

Copy `config.defaults.toml` to `config.toml` to override any setting (thresholds,
camera index, frame size, etc.). See the defaults file for every tunable.

Key thresholds:

| Key | Default | Purpose |
|---|---|---|
| `recognition.match_threshold` | 0.40 | ArcFace cosine similarity cutoff for a confirmed match |
| `recognition.uncertain_threshold` | 0.30 | Below this, show "unknown"; above but below `match_threshold`, show nothing (uncertain) |
| `liveness.passive_spoof_threshold` | 0.50 | P(spoof) above this rejects the frame |
| `enrollment.min_accepted_samples` | 8 | Enrollment fails with fewer good samples |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the tracker, storage repo, and quality gates (the non-model parts).

## Troubleshooting

- **"No face detected" during enrollment**: improve lighting; move closer so the face is at least 112 px wide.
- **Enrollment fails with "no blink detected"**: you must blink at least once during capture. Toggle `liveness.blink_required_for_enrollment` in `config.toml` if you need to disable it.
- **Passive anti-spoof disabled**: `models/antispoof/antispoof.onnx` is missing. Either run `scripts/download_models.py` or drop an ONNX file there manually.
- **DirectML not used**: check `logs/app.log` — look for `Active ONNX providers`. If only CPU is listed, update your GPU driver; DirectML requires DirectX 12.
