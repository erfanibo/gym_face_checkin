"""
SQLite access layer.

A single sqlite3 connection is shared between:
  - the FastAPI request-handling threads (uvicorn's threadpool for sync routes,
    plus the asyncio loop for async ones), and
  - the background camera-processing thread (app.face_engine.FaceEngine).

sqlite3 connections are NOT thread-safe by default, so every access goes
through `db_cursor()`, which serializes all reads/writes behind a single
`threading.Lock`. For <200 users and one camera this is more than fast enough
and avoids "database is locked" errors entirely.
"""
import sqlite3
import threading
from contextlib import contextmanager

from . import config

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS registered_users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    membership_code     TEXT UNIQUE,
    full_name           TEXT NOT NULL,
    phone               TEXT,
    encoding            BLOB NOT NULL,   -- 128 x float64, see face_engine.encoding_to_blob
    photo_path          TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_queue (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    encoding            BLOB NOT NULL,
    photo_path          TEXT NOT NULL,
    first_seen_at       TEXT NOT NULL,
    last_seen_at        TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',  -- pending | registered | rejected
    registered_user_id  INTEGER REFERENCES registered_users(id)
);

CREATE TABLE IF NOT EXISTS attendance_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES registered_users(id),
    checkin_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pending_status   ON pending_queue(status);
CREATE INDEX IF NOT EXISTS idx_attendance_user  ON attendance_log(user_id);
"""


def get_raw_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA foreign_keys = ON")
    return _connection


@contextmanager
def db_cursor(commit: bool = False):
    """Use as:  with db_cursor(commit=True) as cur: cur.execute(...)"""
    conn = get_raw_connection()
    with _lock:
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def init_db():
    conn = get_raw_connection()
    with _lock:
        conn.executescript(SCHEMA)
        conn.commit()
