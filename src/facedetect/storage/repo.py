from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class Person:
    id: str
    name: str
    created_at: str
    thumbnail_path: str | None
    notes: str | None


@dataclass
class Embedding:
    id: int
    person_id: str
    vector: np.ndarray
    pose_tag: str | None
    quality: float | None
    created_at: str


@dataclass
class RecognitionEvent:
    id: int
    person_id: str | None
    similarity: float | None
    frame_ts: str
    snapshot_path: str | None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _vector_to_blob(vec: np.ndarray) -> bytes:
    v = np.ascontiguousarray(vec.astype(np.float32))
    if v.shape != (512,):
        raise ValueError(f"Embedding must be 512-D, got {v.shape}")
    return v.tobytes()


def _blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class PersonRepo:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            self._conn.executescript(f.read())

    def close(self) -> None:
        self._conn.close()

    # --- persons ---

    def create_person(self, name: str, thumbnail_path: str | None = None, notes: str | None = None) -> Person:
        pid = str(uuid.uuid4())
        created = _now_iso()
        self._conn.execute(
            "INSERT INTO person (id, name, created_at, thumbnail_path, notes) VALUES (?, ?, ?, ?, ?)",
            (pid, name, created, thumbnail_path, notes),
        )
        return Person(id=pid, name=name, created_at=created, thumbnail_path=thumbnail_path, notes=notes)

    def get_person_by_name(self, name: str) -> Person | None:
        row = self._conn.execute("SELECT id, name, created_at, thumbnail_path, notes FROM person WHERE name = ?", (name,)).fetchone()
        return Person(*row) if row else None

    def get_person(self, person_id: str) -> Person | None:
        row = self._conn.execute("SELECT id, name, created_at, thumbnail_path, notes FROM person WHERE id = ?", (person_id,)).fetchone()
        return Person(*row) if row else None

    def list_people(self) -> list[Person]:
        rows = self._conn.execute("SELECT id, name, created_at, thumbnail_path, notes FROM person ORDER BY name").fetchall()
        return [Person(*r) for r in rows]

    def update_thumbnail(self, person_id: str, thumbnail_path: str) -> None:
        self._conn.execute("UPDATE person SET thumbnail_path = ? WHERE id = ?", (thumbnail_path, person_id))

    def delete_person(self, person_id: str) -> None:
        self._conn.execute("DELETE FROM person WHERE id = ?", (person_id,))

    # --- embeddings ---

    def add_embeddings(
        self,
        person_id: str,
        vectors: Iterable[np.ndarray],
        pose_tags: Iterable[str | None] | None = None,
        qualities: Iterable[float | None] | None = None,
    ) -> int:
        vecs = list(vectors)
        tags = list(pose_tags) if pose_tags is not None else [None] * len(vecs)
        quals = list(qualities) if qualities is not None else [None] * len(vecs)
        now = _now_iso()
        rows = [
            (person_id, _vector_to_blob(v), t, q, now)
            for v, t, q in zip(vecs, tags, quals)
        ]
        self._conn.executemany(
            "INSERT INTO embedding (person_id, vector, pose_tag, quality, created_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def count_embeddings(self, person_id: str) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM embedding WHERE person_id = ?", (person_id,)).fetchone()
        return int(row[0])

    def load_all_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Return (matrix of shape (N, 512), list of person_ids in the same order)."""
        rows = self._conn.execute("SELECT person_id, vector FROM embedding").fetchall()
        if not rows:
            return np.zeros((0, 512), dtype=np.float32), []
        ids = [r[0] for r in rows]
        mat = np.stack([_blob_to_vector(r[1]) for r in rows]).astype(np.float32)
        return mat, ids

    # --- enrollment sessions ---

    def create_enrollment_session(self, person_id: str, n_samples: int, avg_quality: float | None) -> str:
        sid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO enrollment_session (id, person_id, n_samples, avg_quality, created_at) VALUES (?, ?, ?, ?, ?)",
            (sid, person_id, n_samples, avg_quality, _now_iso()),
        )
        return sid

    # --- recognition events ---

    def log_event(
        self,
        person_id: str | None,
        similarity: float | None,
        snapshot_path: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO recognition_event (person_id, similarity, frame_ts, snapshot_path) VALUES (?, ?, ?, ?)",
            (person_id, similarity, _now_iso(), snapshot_path),
        )
        return int(cur.lastrowid)

    def last_event_for_person(self, person_id: str) -> RecognitionEvent | None:
        row = self._conn.execute(
            "SELECT id, person_id, similarity, frame_ts, snapshot_path FROM recognition_event "
            "WHERE person_id = ? ORDER BY frame_ts DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        return RecognitionEvent(*row) if row else None

    def recent_events(self, limit: int = 50) -> list[RecognitionEvent]:
        rows = self._conn.execute(
            "SELECT id, person_id, similarity, frame_ts, snapshot_path FROM recognition_event "
            "ORDER BY frame_ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [RecognitionEvent(*r) for r in rows]

    def events_for_person(self, person_id: str, limit: int = 50) -> list[RecognitionEvent]:
        rows = self._conn.execute(
            "SELECT id, person_id, similarity, frame_ts, snapshot_path FROM recognition_event "
            "WHERE person_id = ? ORDER BY frame_ts DESC LIMIT ?",
            (person_id, limit),
        ).fetchall()
        return [RecognitionEvent(*r) for r in rows]
