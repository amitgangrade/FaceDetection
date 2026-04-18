PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS person (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    created_at     TEXT NOT NULL,
    thumbnail_path TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS embedding (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    vector     BLOB NOT NULL,
    pose_tag   TEXT,
    quality    REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_embedding_person ON embedding(person_id);

CREATE TABLE IF NOT EXISTS enrollment_session (
    id          TEXT PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    n_samples   INTEGER NOT NULL,
    avg_quality REAL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recognition_event (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id     TEXT REFERENCES person(id) ON DELETE SET NULL,
    similarity    REAL,
    frame_ts      TEXT NOT NULL,
    snapshot_path TEXT
);
CREATE INDEX IF NOT EXISTS ix_event_person_ts ON recognition_event(person_id, frame_ts);
CREATE INDEX IF NOT EXISTS ix_event_ts ON recognition_event(frame_ts);
