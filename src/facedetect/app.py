from __future__ import annotations

import logging
from pathlib import Path

from .config import AppConfig, load_config
from .pipeline.detector import InsightFaceAnalyzer
from .pipeline.liveness import build_liveness
from .services.events import EventLogger
from .services.recognition import RecognitionService
from .storage.repo import PersonRepo
from .ui.control_panel import ControlPanel
from .util.logging import setup_logging
from .util.metrics import SystemMetrics

log = logging.getLogger(__name__)


def build_and_run(project_root: Path | None = None) -> None:
    cfg: AppConfig = load_config(project_root)
    for p in (cfg.paths.data_dir, cfg.paths.models_dir, cfg.paths.logs_dir):
        p.mkdir(parents=True, exist_ok=True)

    setup_logging(cfg.paths.logs_dir)
    log.info("FaceDetect starting. project=%s", cfg.project_root)

    repo = PersonRepo(cfg.paths.data_dir / "faces.db")

    analyzer = InsightFaceAnalyzer(
        det_size=cfg.detector.det_size,
        det_threshold=cfg.detector.det_score_threshold,
    )

    antispoof, blink = build_liveness(cfg.liveness, cfg.antispoof_model_path)
    if antispoof is None or not antispoof.enabled:
        log.warning(
            "Passive anti-spoof is not active. Enrollment relies on the blink challenge "
            "and quality gates only. See scripts/download_models.py."
        )

    event_logger = EventLogger(repo, cfg)
    recognition = RecognitionService(
        analyzer=analyzer,
        antispoof=antispoof,
        repo=repo,
        logger=event_logger,
        cfg=cfg,
    )

    metrics = SystemMetrics()
    metrics.start()

    panel = ControlPanel(
        cfg=cfg,
        repo=repo,
        analyzer=analyzer,
        antispoof=antispoof,
        blink=blink,
        event_logger=event_logger,
        recognition=recognition,
        metrics=metrics,
    )
    try:
        panel.mainloop()
    finally:
        metrics.stop()
        repo.close()
        log.info("FaceDetect exiting.")
