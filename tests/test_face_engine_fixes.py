"""
Regression tests for two FaceEngine fixes:

  1. Face encodings are computed from the FULL-RESOLUTION frame (detection alone
     runs on the downscaled copy).
  2. The in-memory caches are safe against concurrent reloads from request
     threads (known-faces snapshot is swapped atomically; pending-queue cache
     is updated under a lock together with its DB write).

Run from the project root:   pip install pytest && pytest tests -q

These tests never open a camera and don't need dlib to be installed: if the
real `face_recognition` package is missing, a tiny stand-in module is used, and
`face_locations` / `face_encodings` are faked per test either way. Everything
else (SQLite, threads, coordinate scaling, cropping) is the real code.
"""
import asyncio
import os
import sys
import tempfile
import threading
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

# must be set BEFORE app.config is imported: it creates its folders at import time
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="gym_test_")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import face_recognition  # noqa: F401
except ImportError:  # dlib not installed here -> minimal stand-in
    _stub = types.ModuleType("face_recognition")
    _stub.face_locations = lambda img, *a, **k: []
    _stub.face_encodings = lambda img, locs=None, *a, **k: []
    _stub.face_distance = lambda encs, e: (
        np.linalg.norm(np.asarray(encs) - e, axis=1) if len(encs) else np.empty(0)
    )
    sys.modules["face_recognition"] = _stub

from app import config, database, face_engine  # noqa: E402
from app.face_engine import FaceEngine  # noqa: E402


@pytest.fixture()
def engine(monkeypatch):
    database.close_connection()
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    database.init_db()

    loop = asyncio.new_event_loop()
    eng = FaceEngine(loop)
    monkeypatch.setattr(eng, "_broadcast_soon", lambda message: None)
    yield eng
    loop.close()
    database.close_connection()


@pytest.fixture()
def fast_thread_switching():
    """Make the interpreter switch threads ~1000x more often than normal, so
    a race that would normally hit once in a million operations shows up."""
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(old)


# --------------------------------------------------------------------------- #
# Fix 1: encode from the full-resolution frame
# --------------------------------------------------------------------------- #
def test_detection_runs_small_but_encoding_runs_on_full_frame(engine, monkeypatch):
    frame = np.zeros((480, 640, 3), np.uint8)
    frame[..., 0] = 10   # OpenCV frames are BGR: B=10 ...
    frame[..., 2] = 200  # ... R=200
    seen = {}

    def fake_locations(img, *a, **k):
        seen["detect_shape"] = img.shape
        return [(20, 60, 60, 20)]  # (top, right, bottom, left) on the 160x120 image

    def fake_encodings(img, locs=None, *a, **k):
        seen["encode_shape"] = img.shape
        seen["encode_locs"] = list(locs)
        seen["encode_px"] = tuple(int(v) for v in img[0, 0])
        return [np.zeros(128)]

    monkeypatch.setattr(face_engine.face_recognition, "face_locations", fake_locations)
    monkeypatch.setattr(face_engine.face_recognition, "face_encodings", fake_encodings)

    locations, encodings = engine._detect(frame)

    assert seen["detect_shape"] == (120, 160, 3)          # detection: downscaled (0.25x)
    assert seen["encode_shape"] == (480, 640, 3)          # encoding: native resolution
    assert seen["encode_locs"] == [(80, 240, 240, 80)]    # boxes scaled back up exactly 4x
    assert seen["encode_px"] == (200, 0, 10)              # and converted to RGB for dlib
    assert locations == [(20, 60, 60, 20)]                # callers still get DOWNSCALED boxes
    assert len(encodings) == 1


def test_upscale_handles_non_integer_scale_and_clamps(monkeypatch):
    monkeypatch.setattr(config, "FRAME_RESIZE_SCALE", 0.3)
    # 1/0.3 = 3.333..., the old int(round(1/scale)) would have used 3
    assert FaceEngine._upscale_location((10, 100, 50, 30), (480, 640, 3)) == (33, 333, 167, 100)

    monkeypatch.setattr(config, "FRAME_RESIZE_SCALE", 0.25)
    # a box that would land outside the frame after scaling is clamped to it
    assert FaceEngine._upscale_location((0, 200, 200, 0), (480, 640, 3)) == (0, 640, 480, 0)


def test_saved_crop_uses_same_box_as_encoder(engine):
    frame = np.full((480, 640, 3), 128, np.uint8)
    path = engine._save_face_crop(frame, (20, 60, 60, 20))  # -> full box (80,240,240,80) + 20px pad
    import cv2
    crop = cv2.imread(str(path))
    assert crop.shape == (200, 200, 3)


# --------------------------------------------------------------------------- #
# Fix 2a: known-faces cache is swapped atomically
# --------------------------------------------------------------------------- #
def _vec(user_id: int, k: int = 0) -> np.ndarray:
    # different users are ~11+ apart (far beyond tolerance); samples of the
    # same user are ~0.01 apart
    return np.full(128, user_id + 0.001 * k)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return self._rows


def test_known_cache_is_one_consistent_snapshot_under_concurrent_reload(
    engine, monkeypatch, fast_thread_switching
):
    # two very different DB states the reload thread flips between
    state_a = [{"user_id": 10, "encoding": _vec(10, 0).tobytes()},
               {"user_id": 10, "encoding": _vec(10, 1).tobytes()},
               {"user_id": 11, "encoding": _vec(11, 0).tobytes()}]
    state_b = [{"user_id": 20, "encoding": _vec(20, k).tobytes()} for k in range(3)] + \
              [{"user_id": 21, "encoding": _vec(21).tobytes()},
               {"user_id": 22, "encoding": _vec(22).tobytes()}]
    current = {"rows": state_a}

    @contextmanager
    def fake_db_cursor(commit=False):
        yield _FakeCursor(current["rows"])

    monkeypatch.setattr(face_engine, "db_cursor", fake_db_cursor)
    engine.reload_known_users()

    stop = threading.Event()

    def reloader():
        while not stop.is_set():
            current["rows"] = state_b if current["rows"] is state_a else state_a
            engine.reload_known_users()

    t = threading.Thread(target=reloader)
    t.start()

    probe = _vec(11, 0)  # exactly member 11's sample in state A; matches nobody in state B
    outcomes = set()
    try:
        for _ in range(20000):
            user_id, distance = engine._match_known([probe])
            if user_id is None:
                outcomes.add("no-match")
                assert distance is not None and distance > config.KNOWN_USER_TOLERANCE
            else:
                # the ONLY member this exact vector may ever be attributed to
                assert user_id == 11, f"wrong member {user_id}: encodings/user_ids were mismatched"
                outcomes.add("match")
    finally:
        stop.set()
        t.join()

    assert outcomes  # sanity: the loop actually ran


def test_known_cache_empty_states(engine):
    assert engine._match_known([np.zeros(128)]) == (None, None)
    assert engine._known.encodings is None


# --------------------------------------------------------------------------- #
# Fix 2b: a reload must never drop a freshly enqueued pending face
# --------------------------------------------------------------------------- #
def test_reload_pending_never_drops_a_freshly_enqueued_face(engine, monkeypatch, fast_thread_switching):
    monkeypatch.setattr(engine, "_save_face_crop", lambda frame, loc: Path("face.jpg"))

    # The unfixed race needs the INSERT to land inside the tiny gap between a
    # reload's SELECT and its cache swap. A real fsync'd commit takes
    # milliseconds and would almost always miss that gap, so make commits
    # ~free here (safe: throwaway test DB) to give the race a fair chance.
    raw = database.get_raw_connection()
    raw.execute("PRAGMA synchronous=OFF")
    raw.execute("PRAGMA journal_mode=MEMORY")

    stop = threading.Event()

    def reloader():
        while not stop.is_set():
            engine.reload_pending_queue()

    N = 2000
    t = threading.Thread(target=reloader)
    t.start()
    try:
        for i in range(N):
            engine._enqueue_new_face(np.full(128, float(i)), (0, 0, 0, 0), None)
    finally:
        stop.set()
        t.join()

    with database.db_cursor() as cur:
        cur.execute("SELECT id FROM pending_queue WHERE status='pending'")
        db_ids = {r["id"] for r in cur.fetchall()}
    cache_ids = [p["id"] for p in engine._pending_cache]

    assert len(db_ids) == N
    assert set(cache_ids) == db_ids, f"{len(db_ids - set(cache_ids))} faces missing from the cache"
    assert len(cache_ids) == len(set(cache_ids))  # and no duplicates either


# --------------------------------------------------------------------------- #
# End-to-end through _process_frame (normal path must still work after the refactor)
# --------------------------------------------------------------------------- #
def _fake_one_face(monkeypatch, encoding):
    monkeypatch.setattr(face_engine.face_recognition, "face_locations",
                        lambda img, *a, **k: [(20, 60, 60, 20)])
    monkeypatch.setattr(face_engine.face_recognition, "face_encodings",
                        lambda img, locs=None, *a, **k: [encoding])


def test_enrolled_member_is_recognized_and_attendance_logged(engine, monkeypatch):
    member_enc = _vec(7)
    with database.db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO registered_users (membership_code, full_name, encoding) VALUES ('111111', 'Test Member', ?)",
            (face_engine.encoding_to_blob(member_enc),),
        )
        uid = cur.lastrowid
    face_engine.add_face_sample(uid, member_enc, source="registration")
    engine.reload_known_users()

    _fake_one_face(monkeypatch, member_enc)
    engine._process_frame(np.zeros((480, 640, 3), np.uint8))

    with database.db_cursor() as cur:
        cur.execute("SELECT user_id, event_type FROM attendance_log")
        rows = [tuple(r) for r in cur.fetchall()]
        cur.execute("SELECT COUNT(*) AS n FROM pending_queue")
        pending = cur.fetchone()["n"]
    assert rows == [(uid, "in")]
    assert pending == 0


def test_unknown_face_is_queued_once_with_a_photo(engine, monkeypatch):
    _fake_one_face(monkeypatch, _vec(99))
    frame = np.full((480, 640, 3), 90, np.uint8)

    engine._process_frame(frame)
    engine._process_frame(frame)  # same person still standing there -> anti-spam, no 2nd row

    with database.db_cursor() as cur:
        cur.execute("SELECT photo_path FROM pending_queue WHERE status='pending'")
        rows = cur.fetchall()
    assert len(rows) == 1
    assert Path(rows[0]["photo_path"]).exists()
