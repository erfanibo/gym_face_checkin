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
import random
import sqlite3
import threading
from contextlib import contextmanager

from . import config

_lock = threading.Lock()
_connection: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS registered_users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    membership_code     TEXT UNIQUE,   -- auto-generated random number, see generate_unique_membership_code()
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
    event_type  TEXT NOT NULL DEFAULT 'in',   -- 'in' (ورود) یا 'out' (خروج)
    checkin_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Multiple face vectors per member (angle/lighting variants), so recognition
-- doesn't depend on the single snapshot taken at signup. Populated either at
-- registration time (source='registration') or later when an operator
-- manually links a pending-queue face to this member (source='assigned', see
-- routers/queue.py assign_pending_to_member + face_engine.add_face_sample).
CREATE TABLE IF NOT EXISTS member_face_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES registered_users(id),
    encoding    BLOB NOT NULL,
    source      TEXT NOT NULL DEFAULT 'registration',  -- 'registration' | 'assigned'
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Raw, near-real-time "X شناسایی شد" log -- every time a known member's face
-- is matched by the camera, independent of attendance_log's 5-minute
-- cooldown (only lightly debounced, see config.RECOGNITION_LOG_DEBOUNCE_SECONDS).
-- full_name is captured at write time (not JOINed later) so this log stays
-- readable even after a member is renamed or deleted.
CREATE TABLE IF NOT EXISTS recognition_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES registered_users(id),
    full_name   TEXT NOT NULL,
    distance    REAL NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pending_status       ON pending_queue(status);
CREATE INDEX IF NOT EXISTS idx_attendance_user      ON attendance_log(user_id);
CREATE INDEX IF NOT EXISTS idx_face_samples_user    ON member_face_samples(user_id);
CREATE INDEX IF NOT EXISTS idx_recognition_log_id   ON recognition_log(id DESC);
"""


def get_raw_connection() -> sqlite3.Connection:
    global _connection
    if _connection is None:
        _connection = sqlite3.connect(str(config.DB_PATH), check_same_thread=False)
        _connection.row_factory = sqlite3.Row
        _connection.execute("PRAGMA foreign_keys = ON")
    return _connection


def close_connection():
    """
    Closes and drops the shared connection so the underlying .db file can be
    safely replaced on disk (used by routers/backup.py during a restore).
    The next get_raw_connection() call transparently reopens a fresh
    connection against whatever file is at config.DB_PATH at that point.
    """
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None


def backup_database_to(dest_path):
    """
    Snapshots the live database into a new file at dest_path using sqlite3's
    built-in online backup API (Connection.backup), NOT a plain file copy --
    a plain `cp` of a SQLite file that's being written to concurrently (the
    camera thread writes recognition_log/attendance_log constantly) can
    produce a torn, corrupt copy. `backup()` takes the same lock db_cursor()
    uses, so it can't run concurrently with an in-progress write either.
    """
    conn = get_raw_connection()
    with _lock:
        dest_conn = sqlite3.connect(str(dest_path))
        try:
            conn.backup(dest_conn)
        finally:
            dest_conn.close()


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


def _migrate(conn: sqlite3.Connection):
    """Add columns that were introduced after the initial CREATE TABLE, for
    any database file created by an earlier version of this project."""
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(attendance_log)")}
    if "event_type" not in cols:
        conn.execute("ALTER TABLE attendance_log ADD COLUMN event_type TEXT NOT NULL DEFAULT 'in'")


def _backfill_face_samples(conn: sqlite3.Connection):
    """
    member_face_samples was introduced after members had a single encoding
    column. Any member that doesn't have at least one row there yet (every
    member registered before this feature existed) gets their original
    registered_users.encoding copied in as their first sample. Idempotent —
    only touches members with zero samples, safe to run on every startup,
    never deletes or overwrites anything.
    """
    rows = conn.execute(
        """SELECT u.id, u.encoding FROM registered_users u
           LEFT JOIN member_face_samples s ON s.user_id = u.id
           WHERE s.id IS NULL"""
    ).fetchall()
    for row in rows:
        conn.execute(
            "INSERT INTO member_face_samples (user_id, encoding, source) VALUES (?, ?, 'registration')",
            (row["id"], row["encoding"]),
        )


def _pick_unique_code(conn: sqlite3.Connection) -> str:
    low = 10 ** (config.MEMBERSHIP_CODE_DIGITS - 1)
    high = (10 ** config.MEMBERSHIP_CODE_DIGITS) - 1
    for _ in range(50):
        code = str(random.randint(low, high))
        exists = conn.execute(
            "SELECT 1 FROM registered_users WHERE membership_code=?", (code,)
        ).fetchone()
        if exists is None:
            return code
    raise RuntimeError("could not generate a unique membership code after 50 attempts")


def generate_unique_membership_code() -> str:
    """
    Public entry point used at registration time. Acquires the DB lock itself
    — call this OUTSIDE of any existing `with db_cursor(...)` block, since the
    lock is not reentrant and this would otherwise deadlock.
    """
    conn = get_raw_connection()
    with _lock:
        return _pick_unique_code(conn)


def _backfill_membership_codes(conn: sqlite3.Connection):
    """
    Membership codes used to be optional/manually typed; any member created
    before this changed would have NULL here. Assign each of them a code once
    at startup so every member always has one going forward.
    """
    rows = conn.execute("SELECT id FROM registered_users WHERE membership_code IS NULL").fetchall()
    for row in rows:
        code = _pick_unique_code(conn)
        conn.execute("UPDATE registered_users SET membership_code=? WHERE id=?", (code, row["id"]))


def init_db():
    conn = get_raw_connection()
    with _lock:
        conn.executescript(SCHEMA)
        _migrate(conn)
        _backfill_membership_codes(conn)
        _backfill_face_samples(conn)
        conn.commit()
