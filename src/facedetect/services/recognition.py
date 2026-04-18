from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from ..config import AppConfig
from ..pipeline import quality as quality_mod
from ..pipeline.detector import InsightFaceAnalyzer
from ..pipeline.liveness import PassiveAntiSpoof
from ..pipeline.types import DetectedFace, RecognitionResult
from ..storage.repo import PersonRepo
from ..util.tracker import IOUTracker
from .events import EventLogger

log = logging.getLogger(__name__)


@dataclass
class _EmbeddingIndex:
    matrix: np.ndarray  # (N, 512), L2-normalized
    person_ids: list[str]

    @classmethod
    def empty(cls) -> "_EmbeddingIndex":
        return cls(matrix=np.zeros((0, 512), dtype=np.float32), person_ids=[])

    @property
    def size(self) -> int:
        return self.matrix.shape[0]


class RecognitionService:
    """Runs the detect-embed-match loop on successive frames.

    `step(frame)` returns a list of `RecognitionResult` (one per detected face).
    Event logging is a side effect performed internally.
    """

    def __init__(
        self,
        analyzer: InsightFaceAnalyzer,
        antispoof: PassiveAntiSpoof | None,
        repo: PersonRepo,
        logger: EventLogger,
        cfg: AppConfig,
    ):
        self.analyzer = analyzer
        self.antispoof = antispoof
        self.repo = repo
        self.events = logger
        self.cfg = cfg

        self.tracker = IOUTracker(
            iou_threshold=cfg.recognition.tracker_iou_threshold,
            max_missed=5,
            history_len=cfg.recognition.smoothing_window,
        )
        self._index = _EmbeddingIndex.empty()
        self._name_cache: dict[str, str] = {}  # person_id -> name
        self._frame_counter = 0
        self._spoof_cache: dict[int, bool] = {}  # track_id -> last spoof decision
        self._last_refresh = 0.0

        self.reload_index()

    # ---- index management ----

    def reload_index(self) -> None:
        mat, ids = self.repo.load_all_embeddings()
        self._index = _EmbeddingIndex(matrix=mat.astype(np.float32), person_ids=ids)
        self._name_cache = {p.id: p.name for p in self.repo.list_people()}
        self._last_refresh = time.time()
        log.info("Recognition index loaded: %d embeddings across %d people", mat.shape[0], len(self._name_cache))

    # ---- per-frame step ----

    def step(self, frame: np.ndarray) -> list[RecognitionResult]:
        self._frame_counter += 1

        faces = self.analyzer.analyze(frame)
        bboxes = [f.bbox for f in faces]
        track_ids = self.tracker.update(bboxes)

        # Drop debounce entries for tracks that are gone.
        active_ids = {t.id for t in self.tracker.tracks}
        for tid in list(self._spoof_cache):
            if tid not in active_ids:
                del self._spoof_cache[tid]
                self.events.drop_track(tid)

        results: list[RecognitionResult] = []
        for face, track_id in zip(faces, track_ids):
            result = self._process_face(frame, face, track_id)
            results.append(result)
        return results

    def _process_face(self, frame: np.ndarray, face: DetectedFace, track_id: int) -> RecognitionResult:
        # Relaxed quality gate.
        q = quality_mod.assess(face, frame, self.cfg.quality, strict=False)

        # Passive liveness: run on every Nth frame per pipeline (not per track — simpler).
        spoof = self._spoof_cache.get(track_id, False)
        if (
            self.antispoof is not None
            and self.antispoof.enabled
            and face.aligned is not None
            and (self._frame_counter % max(1, self.cfg.liveness.passive_every_n_frames) == 0)
        ):
            spoof = self.antispoof.is_spoof(face.aligned)
            self._spoof_cache[track_id] = spoof

        if spoof:
            self._record_track(track_id, "spoof")
            return RecognitionResult(bbox=face.bbox, person_id=None, person_name="spoof?", similarity=0.0, state="spoof", track_id=track_id)

        if not q.ok or face.embedding is None:
            # Face present but unusable; count as "unknown" for smoothing.
            self._record_track(track_id, "unknown")
            return RecognitionResult(bbox=face.bbox, person_id=None, person_name=None, similarity=0.0, state="warming", track_id=track_id)

        # Cosine similarity against the whole index.
        if self._index.size == 0:
            self._record_track(track_id, "unknown")
            decision = self._track_decision(track_id)
            state = "unknown" if decision == "unknown" else "warming"
            return RecognitionResult(bbox=face.bbox, person_id=None, person_name=None, similarity=0.0, state=state, track_id=track_id)

        sims = self._index.matrix @ face.embedding.astype(np.float32)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_pid = self._index.person_ids[best_idx]
        best_name = self._name_cache.get(best_pid)

        if best_sim >= self.cfg.recognition.match_threshold:
            self._record_track(track_id, best_pid)
        elif best_sim >= self.cfg.recognition.uncertain_threshold:
            self._record_track(track_id, "uncertain")
        else:
            self._record_track(track_id, "unknown")

        decision = self._track_decision(track_id)

        if decision is None:
            return RecognitionResult(bbox=face.bbox, person_id=best_pid, person_name=best_name, similarity=best_sim, state="warming", track_id=track_id)

        if decision == "uncertain":
            return RecognitionResult(bbox=face.bbox, person_id=None, person_name=best_name, similarity=best_sim, state="uncertain", track_id=track_id)

        if decision == "unknown":
            self.events.log_unknown(similarity=best_sim, frame=frame, track_id=track_id)
            return RecognitionResult(bbox=face.bbox, person_id=None, person_name=None, similarity=best_sim, state="unknown", track_id=track_id)

        # Known person (decision is a person_id).
        pid = decision
        name = self._name_cache.get(pid, "?")
        self.events.log_known(person_id=pid, similarity=best_sim, track_id=track_id)
        return RecognitionResult(bbox=face.bbox, person_id=pid, person_name=name, similarity=best_sim, state="known", track_id=track_id)

    # ---- tracker helpers ----

    def _record_track(self, track_id: int, decision: str) -> None:
        t = self.tracker.get(track_id)
        if t is not None:
            t.record(decision)

    def _track_decision(self, track_id: int) -> str | None:
        t = self.tracker.get(track_id)
        if t is None:
            return None
        return t.dominant(self.cfg.recognition.smoothing_min_agree)
