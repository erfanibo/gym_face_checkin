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
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    membership_code        TEXT UNIQUE,   -- auto-generated random number, see generate_unique_membership_code()
    full_name               TEXT NOT NULL,
    phone                   TEXT,
    encoding                BLOB NOT NULL,   -- 128 x float64, see face_engine.encoding_to_blob
    photo_path              TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    -- NULL means "no expiry date has ever been set for this member" (e.g.
    -- anyone registered before this feature existed, or never renewed) --
    -- the admin panel treats that as neutral, NOT expired. Set by the
    -- membership renew/end endpoints in routers/users.py, as a full UTC
    -- ISO-8601 timestamp (to match the convention used everywhere else in
    -- this project, see face_engine.py).
    membership_expires_at   TEXT
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

-- Raw, near-real-time log -- every time a known member's face is matched OR
-- an unrecognized face is seen (new or still-waiting-in-queue), independent
-- of attendance_log's 5-minute cooldown (only lightly debounced, see
-- config.RECOGNITION_LOG_DEBOUNCE_SECONDS). user_id is NULL for an unknown
-- sighting (there's no member to point to). full_name is captured at write
-- time (not JOINed later) so this log stays readable even after a member is
-- renamed/deleted; for unknown sightings it holds a display label like
-- "چهرهٔ ناشناس #<pending_id>". distance is the nearest known-member match
-- found, even when that wasn't close enough to count as a real match --
-- useful for an unknown entry ("closest known member was 0.58 away") --
-- and can be NULL if there were no enrolled members at all to compare against.
CREATE TABLE IF NOT EXISTS recognition_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES registered_users(id),
    full_name   TEXT NOT NULL,
    distance    REAL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- The member's current workout plan, as an ordered list of exercises
-- ("todo list" style: name + sets + reps, optional note). Only ONE plan
-- exists per member at a time, by design -- saving a new plan from the
-- admin panel deletes every existing row for that user_id and inserts the
-- new list in one go (see routers/workouts.py, added in a later step), it
-- never diffs/edits individual rows from the admin side. is_done lets the
-- member check an item off as they go; the manager's next save always
-- resets the whole list anyway, so there's no "undo a check" endpoint needed.
CREATE TABLE IF NOT EXISTS workout_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES registered_users(id),
    position       INTEGER NOT NULL,       -- display order, 0-based, set by the admin panel
    exercise_name  TEXT NOT NULL,
    sets           INTEGER NOT NULL,
    reps           TEXT NOT NULL,           -- TEXT not INTEGER: trainers write ranges too, e.g. "8-12" or "تا ناتوانی"
    notes          TEXT,                    -- optional, e.g. weight or form cue
    is_done        INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Small generic key/value store for things that aren't "a member" or "an
-- attendance event" -- right now just the manager's password hash (see
-- auth.py). Kept as its own table instead of a bunch of one-off tables so
-- future single-value settings don't need a new CREATE TABLE + migration each time.
CREATE TABLE IF NOT EXISTS app_settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_status       ON pending_queue(status);
CREATE INDEX IF NOT EXISTS idx_attendance_user      ON attendance_log(user_id);
CREATE INDEX IF NOT EXISTS idx_face_samples_user    ON member_face_samples(user_id);
CREATE INDEX IF NOT EXISTS idx_recognition_log_id   ON recognition_log(id DESC);
CREATE INDEX IF NOT EXISTS idx_users_phone          ON registered_users(phone);
CREATE INDEX IF NOT EXISTS idx_workout_items_user   ON workout_items(user_id, position);
"""


def get_setting(key: str) -> str | None:
    with db_cursor() as cur:
        cur.execute("SELECT value FROM app_settings WHERE key=?", (key,))
        row = cur.fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO app_settings (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


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

    user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(registered_users)")}
    if "membership_expires_at" not in user_cols:
        conn.execute("ALTER TABLE registered_users ADD COLUMN membership_expires_at TEXT")


def _migrate_recognition_log_nullable(conn: sqlite3.Connection):
    """
    recognition_log originally required user_id/distance (only known-member
    matches were logged). Now unknown-face sightings are logged too, with
    user_id/distance NULL -- but SQLite can't just relax a NOT NULL
    constraint with ALTER TABLE, so any database created before this change
    needs its recognition_log table rebuilt (rename -> recreate with the new
    nullable definition -> copy every existing row across -> drop the old
    one). Only runs when the OLD stricter definition is actually detected,
    so this is a no-op forever after the first upgrade.
    """
    cols = {row["name"]: row for row in conn.execute("PRAGMA table_info(recognition_log)")}
    user_id_col = cols.get("user_id")
    if user_id_col is None or user_id_col["notnull"] == 0:
        return  # table doesn't exist yet, or was already created with the new nullable schema
    conn.execute("ALTER TABLE recognition_log RENAME TO recognition_log_old")
    conn.execute(
        """CREATE TABLE recognition_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER REFERENCES registered_users(id),
            full_name   TEXT NOT NULL,
            distance    REAL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        )"""
    )
    conn.execute(
        """INSERT INTO recognition_log (id, user_id, full_name, distance, created_at)
           SELECT id, user_id, full_name, distance, created_at FROM recognition_log_old"""
    )
    conn.execute("DROP TABLE recognition_log_old")


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
        _migrate_recognition_log_nullable(conn)
        _backfill_membership_codes(conn)
        _backfill_face_samples(conn)
        conn.commit()
