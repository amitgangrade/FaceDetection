from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    data_dir: Path
    models_dir: Path
    logs_dir: Path


@dataclass
class CameraConfig:
    device_index: int
    frame_width: int
    frame_height: int


@dataclass
class DetectorConfig:
    det_size: int
    det_score_threshold: float


@dataclass
class QualityConfig:
    min_face_size_px: int
    min_laplacian_var: float
    min_brightness: float
    max_brightness: float
    edge_margin_px: int
    min_face_size_px_recognition: int
    min_laplacian_var_recognition: float


@dataclass
class LivenessConfig:
    passive_enabled: bool
    passive_spoof_threshold: float
    passive_every_n_frames: int
    blink_ear_closed: float
    blink_ear_open: float
    blink_required_for_enrollment: bool


@dataclass
class EnrollmentConfig:
    preflight_hold_seconds: float
    per_pose_hold_seconds: float
    max_samples: int
    min_accepted_samples: int
    outlier_cosine_cap: float
    poses: list[str]


@dataclass
class RecognitionConfig:
    match_threshold: float
    uncertain_threshold: float
    smoothing_window: int
    smoothing_min_agree: int
    tracker_iou_threshold: float
    event_debounce_seconds: int


@dataclass
class EventsConfig:
    save_unknown_snapshots: bool
    snapshot_jpeg_quality: int


@dataclass
class UIConfig:
    window_title: str
    draw_landmarks: bool


@dataclass
class AppConfig:
    paths: PathsConfig
    camera: CameraConfig
    detector: DetectorConfig
    quality: QualityConfig
    liveness: LivenessConfig
    enrollment: EnrollmentConfig
    recognition: RecognitionConfig
    events: EventsConfig
    ui: UIConfig
    project_root: Path = field(default_factory=lambda: Path.cwd())

    @property
    def antispoof_model_path(self) -> Path:
        return self.paths.models_dir / "antispoof" / "antispoof.onnx"


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project_root: Path | None = None) -> AppConfig:
    root = project_root or Path.cwd()
    defaults_path = root / "config.defaults.toml"
    user_path = root / "config.toml"

    with open(defaults_path, "rb") as f:
        data = tomllib.load(f)
    if user_path.exists():
        with open(user_path, "rb") as f:
            user_data = tomllib.load(f)
        data = _merge(data, user_data)

    paths = PathsConfig(
        data_dir=root / data["paths"]["data_dir"],
        models_dir=root / data["paths"]["models_dir"],
        logs_dir=root / data["paths"]["logs_dir"],
    )
    return AppConfig(
        paths=paths,
        camera=CameraConfig(**data["camera"]),
        detector=DetectorConfig(**data["detector"]),
        quality=QualityConfig(**data["quality"]),
        liveness=LivenessConfig(**data["liveness"]),
        enrollment=EnrollmentConfig(**data["enrollment"]),
        recognition=RecognitionConfig(**data["recognition"]),
        events=EventsConfig(**data["events"]),
        ui=UIConfig(**data["ui"]),
        project_root=root,
    )
