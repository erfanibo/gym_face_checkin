import os
import tempfile

# config creates its folders at import time, so point it at a scratch dir BEFORE anything imports app.*
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="gym_tests_"))
os.environ.setdefault("SESSION_SECRET_KEY", "test-secret")

import pytest

from app import config, database
from app.recognizer import set_recognizer
from tests.fakes import FakeRecognizer


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fresh, fully migrated SQLite database + private photo folders for one test."""
    pending, users = tmp_path / "pending_faces", tmp_path / "user_photos"
    pending.mkdir()
    users.mkdir()
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "gym_face.db")
    monkeypatch.setattr(config, "PENDING_PHOTOS_DIR", pending)
    monkeypatch.setattr(config, "USER_PHOTOS_DIR", users)
    database.close_connection()
    database.init_db()
    yield database
    database.close_connection()


@pytest.fixture
def fake():
    """The process-wide recognizer, replaced by a fake for the duration of a test."""
    recognizer = FakeRecognizer()
    set_recognizer(recognizer)
    yield recognizer
    set_recognizer(None)
