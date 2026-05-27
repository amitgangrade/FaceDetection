# FaceDetect — Build Spec

A local Windows desktop app that enrolls faces from a webcam and recognizes them on a live video feed. Single-user, fully offline after first model download. No network calls at runtime.

## Stack & runtime
- **Python** 3.11–3.12, 64-bit, Windows 11.
- **Inference**: `onnxruntime-directml` (AMD/Intel/NVIDIA via DirectX 12); CPU fallback. Never CUDA/ROCm.
- **Face model**: InsightFace `buffalo_l` pack (RetinaFace det + ArcFace 512-D embedding + 106-pt landmarks). Auto-downloaded by InsightFace into `%USERPROFILE%\.insightface\models` on first run.
- **Anti-spoof model** (optional): MiniFASNet ONNX at `models/antispoof/antispoof.onnx`; absent ⇒ degrade gracefully.
- **Other deps**: `opencv-python`, `numpy`, `pillow`, `requests`, `psutil`. UI uses stdlib **Tkinter** (no extra theme dep). Tests: `pytest`.
- **Persistence**: SQLite (WAL) at `data/faces.db`. Snapshots, thumbnails on disk under `data/`. Logs under `logs/app.log`.
- **Config**: `config.defaults.toml` (committed) + optional `config.toml` (user override, deep-merged).

## Package layout (`src/facedetect/`)
- `__main__.py` → `app.build_and_run`: loads config, opens DB, builds analyzer + liveness + recognition + metrics, launches `ControlPanel.mainloop()`.
- `config.py` — typed dataclasses (`AppConfig` with `paths/camera/detector/quality/liveness/enrollment/recognition/events/ui`); `load_config()` deep-merges defaults with user TOML.
- `capture/webcam.py` — `WebcamSource` wraps `cv2.VideoCapture` (configurable index + resolution).
- `pipeline/detector.py` — `InsightFaceAnalyzer.analyze(frame_bgr)` → `list[DetectedFace]` with bbox, det_score, 5-pt kps, 106-pt landmarks, 512-D L2-normalized embedding, 112×112 aligned crop (via `insightface.utils.face_align.norm_crop`). Providers: `["DmlExecutionProvider","CPUExecutionProvider"]`.
- `pipeline/quality.py` — `assess(face, frame, cfg, strict)`: gates min face size, Laplacian variance (sharpness), mean brightness, and (strict only) edge-margin distance. `quality_score()` = 0.7·sharpness + 0.3·brightness-closeness-to-128, used only for ranking accepted samples.
- `pipeline/liveness.py` —
  - `BlinkDetector`: tracks averaged eye openness ratio (h/w) from landmark indices `33..43` (right eye) and `87..97` (left eye); counts a blink on closed→open transition with hysteresis (`blink_ear_closed`/`open`).
  - `PassiveAntiSpoof`: lazy-loads MiniFASNet ONNX (DirectML), resizes aligned crop to model HW, NCHW float [0,1]; returns sigmoid or 2-way softmax spoof prob; threshold-based. Fail-open if model absent.
- `util/tracker.py` — `IOUTracker` (greedy IOU matching, `max_missed=5`); each `Track` holds a deque of per-frame decision strings for temporal smoothing.
- `services/recognition.py` — `RecognitionService.step(frame)`:
  1. detect+embed via analyzer, 2. update tracker, 3. for each face: relaxed quality gate; passive spoof every N frames (cached per track); cosine sim against `_EmbeddingIndex` matrix; record decision (`person_id` | `"uncertain"` | `"unknown"` | `"spoof"`); ask track for dominant label with `smoothing_min_agree` of `smoothing_window`; emit `RecognitionResult(state ∈ {known,uncertain,unknown,spoof,warming})` and log event.
- `services/enrollment.py` — `EnrollmentSession` state machine: `PREFLIGHT → CAPTURE → FINALIZING → DONE|FAILED`. UI calls `push_frame(bgr)`; returns `StepResult` with prompt/progress/feedback/blink_count. Captures samples across 6 poses (`front,left,right,up,down,expression`); per-pose hold + min samples. Finalize: require ≥1 blink (configurable), ≥`min_accepted_samples` good frames; reject outliers (cosine distance > min(mean+2σ, `outlier_cosine_cap`)); cap at `max_samples` by quality; pick best frontal as thumbnail; persist person + embeddings + enrollment_session.
- `services/events.py` — `EventLogger.log_known/log_unknown` with debounce keyed by `(person_id|"unknown", track_id)` over `event_debounce_seconds` (default 300). Unknown frames saved to `data/snapshots/YYYY/MM/DD/HHMMSS_<rand>.jpg` at configured JPEG quality.
- `storage/repo.py` + `schema.sql` — tables `person(id TEXT PK uuid, name UNIQUE, created_at, thumbnail_path, notes)`, `embedding(id INT PK, person_id FK CASCADE, vector BLOB 512×float32, pose_tag, quality, created_at)`, `enrollment_session(id, person_id, n_samples, avg_quality, created_at)`, `recognition_event(id, person_id NULLABLE SET NULL, similarity, frame_ts, snapshot_path)`. Indexes on embedding(person_id), event(person_id,frame_ts), event(frame_ts). Embeddings packed as raw float32 little-endian bytes.
- `util/metrics.py` — `SystemMetrics` polls CPU/GPU/memory (psutil + DXGI counters where available) for the live status card; 1 Hz.
- `ui/` — Tkinter, Windows 11 Fluent-ish palette (`#F3F3F3`/`#FFFFFF`/`#0F6CBD`), "clam" ttk theme so progress bars tint.
  - `control_panel.py` — 460×640 main window: brand header, four action buttons (Start/Stop live, Enroll, Manage people, Recent events), system-usage card (CPU/GPU/Mem bars), Quit. Webcam is shared by enrollment + live view; mutually exclusive.
  - `enrollment_wizard.py` — modal wizard around `EnrollmentSession` showing live preview, current pose prompt, progress, feedback text, blink counter.
  - `live_view.py` — runs on a background thread; draws bbox + label colored by state (green known, orange uncertain, gray unknown, red spoof) plus optional landmarks; cooperative stop via `threading.Event`.
  - `people_manager.py` / `events_viewer.py` — list/delete people, add samples, browse recent events with thumbnails.

## Key defaults (`config.defaults.toml`)
- camera: index 0, 1280×720.
- detector: `det_size=640`, `det_score_threshold=0.8`.
- quality (enroll/recog): `min_face_size_px=112/80`, `min_laplacian_var=60/30`, brightness `[40,220]`, `edge_margin_px=8`.
- liveness: `passive_enabled=true`, `passive_spoof_threshold=0.5`, `passive_every_n_frames=5`, `blink_ear_closed=0.15`, `blink_ear_open=0.22`, `blink_required_for_enrollment=true`.
- enrollment: `preflight_hold=3.0s`, `per_pose_hold=3.0s`, `max_samples=30`, `min_accepted_samples=8`, `outlier_cosine_cap=0.5`, poses `[front,left,right,up,down,expression]`.
- recognition: `match_threshold=0.40`, `uncertain_threshold=0.30`, `smoothing_window=7`, `smoothing_min_agree=4`, `tracker_iou_threshold=0.3`, `event_debounce_seconds=300`.
- events: save unknown snapshots, JPEG quality 85.

## Install / run / test
```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
python scripts\download_models.py   # optional anti-spoof
python -m facedetect                # entrypoint: facedetect = facedetect.__main__:main
pytest                              # tracker, repo, quality gates
```

## Non-goals & invariants
- No cloud, no telemetry, no auth, no multi-user.
- Embeddings are exactly 512-D float32, L2-normalized; matching = pure cosine (dot of normalized vectors).
- DirectML first, CPU fallback; never CUDA.
- Recognition must not display a name until temporal smoothing agrees (≥`min_agree` of last `window` frames); below `match_threshold` but above `uncertain_threshold` shows nothing rather than a wrong name.
- Active spoof during enrollment hard-fails the session; passive-spoof model missing is OK (fail-open, blink challenge + quality gates remain).
