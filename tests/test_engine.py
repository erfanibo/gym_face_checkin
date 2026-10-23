"""FaceEngine's decisions (known / still-waiting unknown / new unknown), driven by FakeRecognizer."""
import asyncio

import numpy as np
import pytest

from app.database import db_cursor
from app.face_engine import FaceEngine
from app.face_samples import add_face_sample, encoding_to_blob
from app.recognizer.base import DetectedFace
from tests.fakes import FakeRecognizer, unit

FRAME = np.full((480, 640, 3), 90, np.uint8)
BOX = (100, 300, 260, 140)  # top, right, bottom, left -- full-frame pixels


@pytest.fixture
def make_engine(db):
    loop = asyncio.new_event_loop()
    created = []

    def factory(recognizer=None):
        engine = FaceEngine(loop, recognizer or FakeRecognizer())
        engine.events = []
        engine._broadcast_soon = engine.events.append
        engine._send_webhook_soon = lambda payload: None
        created.append(engine)
        return engine

    yield factory
    loop.close()


def add_member(name="Ali"):
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO registered_users (full_name, membership_code) VALUES (?, ?)", (name, name[:3] + "123"))
        return cur.lastrowid


def pending_rows():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM pending_queue ORDER BY id")
        return cur.fetchall()


def attendance_count():
    with db_cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM attendance_log")
        return cur.fetchone()["n"]


def test_known_member_is_recognised_and_checked_in(make_engine):
    uid = add_member()
    add_face_sample(uid, unit(1), model="fake/1")
    engine = make_engine()
    engine.reload_known_users()

    engine._handle_face([unit(1)], BOX, FRAME)

    assert attendance_count() == 1
    assert pending_rows() == []
    assert any(e["event"] == "checkin" and e["user_id"] == uid for e in engine.events)


def test_best_of_several_frames_wins_not_the_average(make_engine):
    uid = add_member()
    add_face_sample(uid, unit(1), model="fake/1")
    engine = make_engine()
    engine.reload_known_users()
    # first frame is garbage (someone else), second is a perfect match
    engine._handle_face([unit(5), unit(1)], BOX, FRAME)
    assert attendance_count() == 1


def test_samples_of_another_model_are_never_compared(make_engine):
    uid = add_member()
    add_face_sample(uid, unit(1), model="dlib")           # same numbers, different vector space
    engine = make_engine(FakeRecognizer("fake/1"))
    engine.reload_known_users()

    engine._handle_face([unit(1)], BOX, FRAME)

    assert attendance_count() == 0                          # NOT recognised
    assert len(pending_rows()) == 1                         # treated as a new unknown face


def test_new_unknown_face_is_queued_with_its_model_and_a_full_frame_crop(make_engine, db):
    from app import config
    engine = make_engine()
    engine._handle_face([unit(2)], BOX, FRAME)

    [row] = pending_rows()
    assert row["model"] == "fake/1" and row["status"] == "pending"
    crop = np.asarray(__import__("cv2").imread(row["photo_path"]))
    # box is 160 px tall, 160 px wide, +20 px padding on each side, in FULL-frame pixels
    assert crop.shape[:2] == (200, 200)
    assert any(e["event"] == "new_pending" for e in engine.events)


def test_same_unknown_person_is_not_queued_twice(make_engine):
    engine = make_engine()
    engine._handle_face([unit(2)], BOX, FRAME)
    engine._handle_face([unit(2)], BOX, FRAME)
    assert len(pending_rows()) == 1
    engine._handle_face([unit(3)], BOX, FRAME)               # a different person
    assert len(pending_rows()) == 2


def test_pending_rows_of_another_model_are_not_used_for_dedup(make_engine, tmp_path):
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at) VALUES (?, 'dlib', 'x.jpg', ?, ?)",
            (encoding_to_blob(unit(2)), "2999-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00"),
        )
    engine = make_engine(FakeRecognizer("fake/1"))
    engine._handle_face([unit(2)], BOX, FRAME)
    assert len(pending_rows()) == 2                          # the dlib row did not swallow it


def test_process_frame_folds_burst_frames_into_one_sighting(make_engine):
    class Video:
        def read(self):
            return True, FRAME

    rec = FakeRecognizer()
    rec.faces = [DetectedFace(BOX, unit(4))]
    engine = make_engine(rec)
    engine._video = Video()
    engine._process_frame(FRAME)
    assert len(pending_rows()) == 1                          # one person, not three


def test_known_user_cache_loads_only_the_active_model_and_survives_empty(make_engine):
    uid = add_member()
    add_face_sample(uid, unit(1), model="dlib")
    engine = make_engine(FakeRecognizer("fake/1"))
    engine.reload_known_users()
    assert engine._known_matrix is None and engine._known_sample_user_ids == []
    engine._handle_face([unit(1)], BOX, FRAME)               # must not crash with no known vectors
    add_face_sample(uid, unit(1), model="fake/1")
    engine.reload_known_users()
    assert engine._known_matrix.shape == (1, 8) and engine._known_sample_user_ids == [uid]


def test_prepare_recognizer_warms_up_reembeds_and_reports_ready(make_engine):
    import cv2
    from app import config
    from tests.fakes import image_for
    cv2.imwrite(str(config.PENDING_PHOTOS_DIR / "m.png"), image_for(6))
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO registered_users (full_name, photo_path) VALUES ('Ali', 'm.png')")
    rec = FakeRecognizer()
    engine = make_engine(rec)
    engine._running = True

    assert engine._prepare_recognizer() is True

    assert rec.warm_up_calls == 1 and engine.recognizer_ready and engine.recognizer_error is None
    assert engine._known_matrix is not None and engine._known_matrix.shape[0] == 1   # reloaded after re-embedding
    assert {"event": "recognizer_status", "ready": True, "error": None} in engine.events


def test_prepare_recognizer_retries_after_a_failed_warm_up_and_stops_on_request(make_engine, monkeypatch):
    from app import config
    monkeypatch.setattr(config, "CAMERA_RETRY_SECONDS", 0.05)
    rec = FakeRecognizer()
    rec.warm_up_error = RuntimeError("weights could not be downloaded")
    engine = make_engine(rec)
    engine._running = True

    import threading
    result = []
    t = threading.Thread(target=lambda: result.append(engine._prepare_recognizer()))
    t.start()
    for _ in range(100):
        if rec.warm_up_calls >= 2:
            break
        threading.Event().wait(0.02)
    assert rec.warm_up_calls >= 2 and not engine.recognizer_ready
    assert "weights could not be downloaded" in engine.recognizer_error
    engine._running = False                                   # stop() while still failing
    t.join(timeout=2)
    assert result == [False]
