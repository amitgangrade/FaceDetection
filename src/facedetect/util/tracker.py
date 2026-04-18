from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


BBox = tuple[int, int, int, int]


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    id: int
    bbox: BBox
    history: deque = field(default_factory=lambda: deque(maxlen=7))
    missed: int = 0

    def record(self, decision: str | None) -> None:
        self.history.append(decision)

    def dominant(self, min_agree: int) -> str | None:
        """Return the label that appears >= min_agree times, else None."""
        if not self.history:
            return None
        counts: dict[str | None, int] = {}
        for d in self.history:
            counts[d] = counts.get(d, 0) + 1
        best = max(counts.items(), key=lambda kv: kv[1])
        if best[1] >= min_agree:
            return best[0]
        return None


class IOUTracker:
    """Very simple IOU-based tracker with history-per-track for temporal smoothing."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 5, history_len: int = 7):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.history_len = history_len
        self._tracks: list[Track] = []
        self._next_id = 1

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def update(self, detections: Iterable[BBox]) -> list[int]:
        """Assign track IDs to each detection. Returns a list of track IDs in the same order."""
        dets = list(detections)
        assigned_det = [False] * len(dets)
        track_ids: list[int | None] = [None] * len(dets)

        # Greedy match: for each track, find the best-IOU unassigned detection.
        for t in self._tracks:
            best_idx = -1
            best_iou = self.iou_threshold
            for i, d in enumerate(dets):
                if assigned_det[i]:
                    continue
                score = iou(t.bbox, d)
                if score > best_iou:
                    best_iou = score
                    best_idx = i
            if best_idx >= 0:
                t.bbox = dets[best_idx]
                t.missed = 0
                track_ids[best_idx] = t.id
                assigned_det[best_idx] = True
            else:
                t.missed += 1

        # New tracks for unassigned detections.
        for i, d in enumerate(dets):
            if not assigned_det[i]:
                new = Track(id=self._next_id, bbox=d, history=deque(maxlen=self.history_len))
                self._next_id += 1
                self._tracks.append(new)
                track_ids[i] = new.id

        # Drop stale tracks.
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        return [tid for tid in track_ids if tid is not None]

    def get(self, track_id: int) -> Track | None:
        for t in self._tracks:
            if t.id == track_id:
                return t
        return None
