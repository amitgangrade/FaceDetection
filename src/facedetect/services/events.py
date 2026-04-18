from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..config import AppConfig
from ..storage.repo import PersonRepo
from ..util.image_ops import save_jpeg

log = logging.getLogger(__name__)


class EventLogger:
    """Records recognition events to the DB, saves unknown-person snapshots.

    Applies debouncing: the same person (or unknown from the same track) isn't re-logged
    more than once per `event_debounce_seconds`, keyed by (person_id|'unknown', track_id).
    """

    def __init__(self, repo: PersonRepo, cfg: AppConfig):
        self.repo = repo
        self.cfg = cfg
        self._last_seen: dict[tuple[str, int], float] = {}

    def _debounce_key(self, person_id: str | None, track_id: int) -> tuple[str, int]:
        return (person_id or "unknown", track_id)

    def _debounced(self, key: tuple[str, int]) -> bool:
        last = self._last_seen.get(key)
        if last is None:
            return False
        return (time.time() - last) < self.cfg.recognition.event_debounce_seconds

    def log_known(self, person_id: str, similarity: float, track_id: int) -> None:
        key = self._debounce_key(person_id, track_id)
        if self._debounced(key):
            return
        self.repo.log_event(person_id=person_id, similarity=similarity)
        self._last_seen[key] = time.time()
        log.info("event: known person=%s sim=%.3f track=%d", person_id, similarity, track_id)

    def log_unknown(self, similarity: float, frame: np.ndarray, track_id: int) -> None:
        key = self._debounce_key(None, track_id)
        if self._debounced(key):
            return
        snapshot_path: str | None = None
        if self.cfg.events.save_unknown_snapshots:
            now = datetime.now()
            rel = Path("snapshots") / f"{now:%Y}" / f"{now:%m}" / f"{now:%d}" / f"{now:%H%M%S}_{uuid4().hex[:8]}.jpg"
            full = self.cfg.paths.data_dir / rel
            save_jpeg(frame, full, quality=self.cfg.events.snapshot_jpeg_quality)
            snapshot_path = str(full)
        self.repo.log_event(person_id=None, similarity=similarity, snapshot_path=snapshot_path)
        self._last_seen[key] = time.time()
        log.info("event: unknown sim=%.3f track=%d snapshot=%s", similarity, track_id, snapshot_path or "-")

    def drop_track(self, track_id: int) -> None:
        """Remove debounce entries for a track that has ended."""
        for k in list(self._last_seen):
            if k[1] == track_id:
                del self._last_seen[k]
