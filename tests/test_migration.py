"""A database from before the recognizer layer (no `model` column) must upgrade in place."""
import sqlite3

from app import config, database

OLD_SCHEMA = """
CREATE TABLE registered_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT, membership_code TEXT UNIQUE, full_name TEXT NOT NULL,
    phone TEXT, encoding BLOB, photo_path TEXT, created_at TEXT NOT NULL DEFAULT (datetime('now')),
    membership_expires_at TEXT, member_password_hash TEXT);
CREATE TABLE pending_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT, encoding BLOB NOT NULL, photo_path TEXT NOT NULL,
    first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
    registered_user_id INTEGER REFERENCES registered_users(id));
CREATE TABLE member_face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES registered_users(id),
    encoding BLOB NOT NULL, source TEXT NOT NULL DEFAULT 'registration',
    created_at TEXT NOT NULL DEFAULT (datetime('now')));
"""


def _columns(table):
    conn = database.get_raw_connection()
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_old_database_gets_model_column_and_keeps_every_row(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(OLD_SCHEMA)
    old.execute("INSERT INTO registered_users (id, membership_code, full_name, encoding) VALUES (1, '111111', 'Ali', x'00')")
    old.execute("INSERT INTO member_face_samples (user_id, encoding) VALUES (1, x'0102')")
    old.execute("INSERT INTO pending_queue (encoding, photo_path, first_seen_at, last_seen_at) VALUES (x'03', 'a.jpg', 't', 't')")
    old.commit()
    old.close()

    monkeypatch.setattr(config, "DB_PATH", path)
    database.close_connection()
    database.init_db()
    try:
        assert "model" in _columns("member_face_samples") and "model" in _columns("pending_queue")
        conn = database.get_raw_connection()
        samples = conn.execute("SELECT user_id, encoding, model FROM member_face_samples").fetchall()
        assert [(r["user_id"], bytes(r["encoding"]), r["model"]) for r in samples] == [(1, b"\x01\x02", "dlib")]
        pending = conn.execute("SELECT model, photo_path FROM pending_queue").fetchall()
        assert [(r["model"], r["photo_path"]) for r in pending] == [("dlib", "a.jpg")]

        database.init_db()  # idempotent
        assert conn.execute("SELECT COUNT(*) AS n FROM member_face_samples").fetchone()["n"] == 1
    finally:
        database.close_connection()


def test_new_database_has_the_column_from_the_start(db):
    assert "model" in _columns("member_face_samples") and "model" in _columns("pending_queue")
