from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from ..config import AppConfig
from ..pipeline import quality as quality_mod
from ..pipeline.detector import InsightFaceAnalyzer
from ..pipeline.liveness import BlinkDetector, PassiveAntiSpoof
from ..pipeline.types import DetectedFace
from ..storage.repo import PersonRepo
from ..util.image_ops import make_thumbnail, save_jpeg

log = logging.getLogger(__name__)


POSE_INSTRUCTIONS: dict[str, str] = {
    "front": "Look straight at the camera",
    "left": "Turn your head slightly LEFT",
    "right": "Turn your head slightly RIGHT",
    "up": "Tilt your head slightly UP",
    "down": "Tilt your head slightly DOWN",
    "expression": "Smile naturally",
}


class Phase(str, Enum):
    PREFLIGHT = "preflight"
    CAPTURE = "capture"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Sample:
    embedding: np.ndarray
    pose_tag: str
    quality_score: float
    crop_bgr: np.ndarray  # for thumbnail selection


@dataclass
class StepResult:
    phase: Phase
    pose_index: int  # index into cfg.enrollment.poses, or -1 in preflight
    pose_tag: str | None
    prompt: str
    progress: float  # 0..1 within current phase
    samples_for_pose: int
    total_samples: int
    feedback: str  # live guidance ("too dark", "too close to edge", etc.)
    blink_count: int
    error: str | None = None
    final_person_id: str | None = None


@dataclass
class EnrollmentSession:
    """Stateful enrollment driver. UI calls push_frame(frame_bgr) repeatedly.

    Failure modes reported via StepResult.error (non-None when Phase.FAILED).
    """

    name: str
    existing_person_id: str | None  # if not None, add samples to this person
    analyzer: InsightFaceAnalyzer
    antispoof: PassiveAntiSpoof | None
    blink: BlinkDetector
    repo: PersonRepo
    cfg: AppConfig

    phase: Phase = Phase.PREFLIGHT
    _pose_index: int = 0
    _samples: list[Sample] = field(default_factory=list)
    _samples_for_current_pose: int = 0
    _phase_start: float = field(default_factory=time.time)
    _preflight_ok_start: float | None = None
    _error: str | None = None
    _final_person_id: str | None = None
    _last_feedback: str = ""

    def __post_init__(self):
        self.blink.reset()

    @property
    def current_pose_tag(self) -> str | None:
        if self.phase is not Phase.CAPTURE:
            return None
        poses = self.cfg.enrollment.poses
        if 0 <= self._pose_index < len(poses):
            return poses[self._pose_index]
        return None

    # ------------- main entry --------------

    def push_frame(self, frame_bgr: np.ndarray) -> StepResult:
        if self.phase is Phase.PREFLIGHT:
            return self._handle_preflight(frame_bgr)
        if self.phase is Phase.CAPTURE:
            return self._handle_capture(frame_bgr)
        return self._make_result()

    def cancel(self, reason: str = "cancelled by user") -> StepResult:
        self.phase = Phase.FAILED
        self._error = reason
        self._last_feedback = reason
        return self._make_result()

    # ------------- preflight --------------

    def _handle_preflight(self, frame: np.ndarray) -> StepResult:
        faces = self.analyzer.analyze(frame)
        if len(faces) == 0:
            self._preflight_ok_start = None
            self._last_feedback = "No face detected — move into view"
            return self._make_result()
        if len(faces) > 1:
            self._preflight_ok_start = None
            self._last_feedback = "Only the enrollee should be visible"
            return self._make_result()

        face = faces[0]
        self.blink.update_from_face(face)
        q = quality_mod.assess(face, frame, self.cfg.quality, strict=True)
        if not q.ok:
            self._preflight_ok_start = None
            self._last_feedback = f"Adjust: {q.reason}"
            return self._make_result()

        # Passive liveness check on aligned crop.
        if self.antispoof is not None and self.antispoof.enabled and face.aligned is not None:
            if self.antispoof.is_spoof(face.aligned):
                self._preflight_ok_start = None
                self._last_feedback = "Liveness check failed — please show a real face"
                return self._make_result()

        now = time.time()
        if self._preflight_ok_start is None:
            self._preflight_ok_start = now
            self._last_feedback = "Hold still..."
            return self._make_result()

        held = now - self._preflight_ok_start
        if held >= self.cfg.enrollment.preflight_hold_seconds:
            self._advance_to_capture()
            return self._make_result()
        self._last_feedback = f"Good — hold still ({held:.1f}s)"
        return self._make_result()

    def _advance_to_capture(self) -> None:
        self.phase = Phase.CAPTURE
        self._pose_index = 0
        self._samples_for_current_pose = 0
        self._phase_start = time.time()
        self._last_feedback = POSE_INSTRUCTIONS.get(self.current_pose_tag or "", "")
        log.info("Enrollment: preflight passed, starting capture (%d poses)", len(self.cfg.enrollment.poses))

    # ------------- capture --------------

    def _handle_capture(self, frame: np.ndarray) -> StepResult:
        pose_tag = self.current_pose_tag
        if pose_tag is None:
            self._finalize()
            return self._make_result()

        faces = self.analyzer.analyze(frame)
        if len(faces) == 0:
            self._last_feedback = "Face lost — move back into view"
            return self._make_result()
        if len(faces) > 1:
            self._last_feedback = "Only the enrollee should be visible"
            return self._make_result()

        face = faces[0]
        self.blink.update_from_face(face)
        q = quality_mod.assess(face, frame, self.cfg.quality, strict=True)
        if not q.ok:
            self._last_feedback = f"Adjust: {q.reason}"
            return self._maybe_advance_pose(pose_tag)

        if self.antispoof is not None and self.antispoof.enabled and face.aligned is not None:
            if self.antispoof.is_spoof(face.aligned):
                self._last_feedback = "Liveness check failed"
                # Hard fail the session: an active spoof attempt should not be tolerated.
                self.phase = Phase.FAILED
                self._error = "liveness check failed during enrollment"
                return self._make_result()

        if face.embedding is None:
            self._last_feedback = "Embedding unavailable — try again"
            return self._maybe_advance_pose(pose_tag)

        # Accept this frame as a sample.
        score = quality_mod.quality_score(q)
        self._samples.append(
            Sample(
                embedding=face.embedding.astype(np.float32),
                pose_tag=pose_tag,
                quality_score=score,
                crop_bgr=face.crop(frame).copy(),
            )
        )
        self._samples_for_current_pose += 1
        self._last_feedback = POSE_INSTRUCTIONS.get(pose_tag, "")

        return self._maybe_advance_pose(pose_tag)

    def _maybe_advance_pose(self, pose_tag: str) -> StepResult:
        held = time.time() - self._phase_start
        if held >= self.cfg.enrollment.per_pose_hold_seconds and self._samples_for_current_pose >= 1:
            self._pose_index += 1
            self._samples_for_current_pose = 0
            self._phase_start = time.time()
            if self._pose_index >= len(self.cfg.enrollment.poses):
                self._finalize()
            else:
                self._last_feedback = POSE_INSTRUCTIONS.get(self.current_pose_tag or "", "")
        return self._make_result()

    # ------------- finalize --------------

    def _finalize(self) -> None:
        self.phase = Phase.FINALIZING

        if self.cfg.liveness.blink_required_for_enrollment and self.blink.blink_count < 1:
            self.phase = Phase.FAILED
            self._error = "no blink detected — liveness failed. Please try again."
            return

        if len(self._samples) < self.cfg.enrollment.min_accepted_samples:
            self.phase = Phase.FAILED
            self._error = f"only {len(self._samples)} good samples captured (need {self.cfg.enrollment.min_accepted_samples})"
            return

        kept = self._reject_outliers(self._samples)
        if len(kept) < self.cfg.enrollment.min_accepted_samples:
            self.phase = Phase.FAILED
            self._error = f"after outlier rejection only {len(kept)} samples remain"
            return

        # Cap at max_samples (keep highest quality).
        if len(kept) > self.cfg.enrollment.max_samples:
            kept.sort(key=lambda s: s.quality_score, reverse=True)
            kept = kept[: self.cfg.enrollment.max_samples]

        try:
            person_id = self._persist(kept)
        except Exception as e:  # noqa: BLE001
            log.exception("enrollment persist failed")
            self.phase = Phase.FAILED
            self._error = f"database error: {e}"
            return

        self._final_person_id = person_id
        self.phase = Phase.DONE
        self._last_feedback = f"Enrolled {self.name} with {len(kept)} samples"
        log.info("Enrollment complete: person_id=%s samples=%d", person_id, len(kept))

    def _reject_outliers(self, samples: list[Sample]) -> list[Sample]:
        if len(samples) <= 3:
            return list(samples)
        mat = np.stack([s.embedding for s in samples])
        # Normalize just in case.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        mat = mat / norms

        centroid = mat.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-8)
        sims = mat @ centroid  # higher = closer
        dists = 1.0 - sims
        mean = dists.mean()
        std = dists.std()
        threshold = min(mean + 2.0 * std, self.cfg.enrollment.outlier_cosine_cap)
        kept = [s for s, d in zip(samples, dists) if d <= threshold]
        dropped = len(samples) - len(kept)
        if dropped:
            log.info("Outlier rejection: dropped %d of %d samples (threshold=%.3f)", dropped, len(samples), threshold)
        return kept

    def _persist(self, kept: list[Sample]) -> str:
        if self.existing_person_id is not None:
            person_id = self.existing_person_id
            person = self.repo.get_person(person_id)
            if person is None:
                raise RuntimeError(f"existing person {person_id} no longer exists")
        else:
            person = self.repo.create_person(name=self.name)
            person_id = person.id

        # Save thumbnail from the highest-quality front sample if possible.
        front_samples = [s for s in kept if s.pose_tag == "front"]
        best = max(front_samples or kept, key=lambda s: s.quality_score)
        thumb = make_thumbnail(best.crop_bgr, size=200)
        thumb_path = self.cfg.paths.data_dir / "thumbnails" / f"{person_id}.jpg"
        save_jpeg(thumb, thumb_path, quality=90)
        self.repo.update_thumbnail(person_id, str(thumb_path))

        self.repo.add_embeddings(
            person_id=person_id,
            vectors=[s.embedding for s in kept],
            pose_tags=[s.pose_tag for s in kept],
            qualities=[s.quality_score for s in kept],
        )
        avg_q = float(np.mean([s.quality_score for s in kept])) if kept else None
        self.repo.create_enrollment_session(person_id=person_id, n_samples=len(kept), avg_quality=avg_q)
        return person_id

    # ------------- result packaging --------------

    def _make_result(self) -> StepResult:
        poses = self.cfg.enrollment.poses
        if self.phase is Phase.PREFLIGHT:
            held = 0.0
            if self._preflight_ok_start is not None:
                held = min(time.time() - self._preflight_ok_start, self.cfg.enrollment.preflight_hold_seconds)
            progress = held / self.cfg.enrollment.preflight_hold_seconds
            return StepResult(
                phase=self.phase,
                pose_index=-1,
                pose_tag=None,
                prompt="Line up your face in the center",
                progress=progress,
                samples_for_pose=0,
                total_samples=len(self._samples),
                feedback=self._last_feedback,
                blink_count=self.blink.blink_count,
            )
        if self.phase is Phase.CAPTURE:
            pose_tag = self.current_pose_tag or ""
            elapsed = time.time() - self._phase_start
            progress = min(elapsed / self.cfg.enrollment.per_pose_hold_seconds, 1.0)
            overall = (self._pose_index + progress) / max(len(poses), 1)
            return StepResult(
                phase=self.phase,
                pose_index=self._pose_index,
                pose_tag=pose_tag,
                prompt=POSE_INSTRUCTIONS.get(pose_tag, ""),
                progress=overall,
                samples_for_pose=self._samples_for_current_pose,
                total_samples=len(self._samples),
                feedback=self._last_feedback,
                blink_count=self.blink.blink_count,
            )
        return StepResult(
            phase=self.phase,
            pose_index=self._pose_index,
            pose_tag=None,
            prompt="",
            progress=1.0,
            samples_for_pose=0,
            total_samples=len(self._samples),
            feedback=self._last_feedback,
            blink_count=self.blink.blink_count,
            error=self._error,
            final_person_id=self._final_person_id,
        )
