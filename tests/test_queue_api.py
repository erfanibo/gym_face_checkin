"""Registering / assigning a queued face when vectors come from different models."""
import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth, config
from app.database import db_cursor
from app.face_samples import add_face_sample, encoding_to_blob
from app.routers import queue
from tests.fakes import image_for, unit


class StubEngine:
    camera_connected = True
    camera_problem = None
    recognizer_ready = False
    recognizer_error = "weights missing"

    def __init__(self):
        self.reloads = 0

    def reload_known_users(self):
        self.reloads += 1

    def reload_pending_queue(self):
        pass

    def forget_unknown_debounce(self, pending_id):
        pass


@pytest.fixture
def client(db, fake):
    app = FastAPI()
    app.include_router(queue.router)
    app.dependency_overrides[auth.require_manager] = lambda: None
    app.state.face_engine = StubEngine()
    return TestClient(app)


def queue_face(model, vector, image=None, photo="p.png"):
    if image is not None:
        cv2.imwrite(str(config.PENDING_PHOTOS_DIR / photo), image)
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at) VALUES (?, ?, ?, 't', 't')",
            (encoding_to_blob(vector), model, str(config.PENDING_PHOTOS_DIR / photo)),
        )
        return cur.lastrowid


def member_samples(user_id):
    with db_cursor() as cur:
        cur.execute("SELECT model, source FROM member_face_samples WHERE user_id=?", (user_id,))
        return [(r["model"], r["source"]) for r in cur.fetchall()]


def test_camera_status_reports_recognizer_state(client):
    body = client.get("/api/queue/camera-status").json()
    assert body == {"connected": True, "problem": None, "recognizer_ready": False, "recognizer_error": "weights missing"}


def test_register_a_face_made_by_the_active_model(client, fake):
    pid = queue_face(fake.model_id, unit(2), image_for(2))
    r = client.post(f"/api/queue/{pid}/register", json={"full_name": "Ali Rezaei"})
    assert r.status_code == 200, r.text
    uid = r.json()["id"]
    assert member_samples(uid) == [(fake.model_id, "registration")]
    with db_cursor() as cur:
        cur.execute("SELECT encoding FROM registered_users WHERE id=?", (uid,))
        assert cur.fetchone()["encoding"] is None             # legacy column is dlib-only


def test_register_a_face_queued_by_the_old_model_rebuilds_the_vector_from_its_photo(client, fake):
    pid = queue_face("dlib", unit(7), image_for(2))           # old vector is useless; the photo says "face 2"
    r = client.post(f"/api/queue/{pid}/register", json={"full_name": "Sara Ahmadi"})
    assert r.status_code == 200, r.text
    uid = r.json()["id"]
    assert member_samples(uid) == [(fake.model_id, "registration")]
    with db_cursor() as cur:
        cur.execute("SELECT encoding FROM member_face_samples WHERE user_id=?", (uid,))
        vec = np.frombuffer(cur.fetchone()["encoding"], dtype=np.float64)
    assert vec == pytest.approx(unit(2))


def test_old_model_face_with_unreadable_photo_gives_a_clear_error(client, fake):
    pid = queue_face("dlib", unit(7), image=None, photo="missing.png")
    r = client.post(f"/api/queue/{pid}/register", json={"full_name": "Sara Ahmadi"})
    assert r.status_code == 422 and "حذف" in r.json()["detail"]


def test_duplicate_registration_is_caught_only_against_same_model_samples(client, fake):
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO registered_users (full_name, membership_code) VALUES ('Existing', '100001')")
        existing = cur.lastrowid
    add_face_sample(existing, unit(3), model=fake.model_id)

    dup = queue_face(fake.model_id, unit(3), image_for(3), photo="d.png")
    r = client.post(f"/api/queue/{dup}/register", json={"full_name": "Someone Else"})
    assert r.status_code == 409 and "Existing" in r.json()["detail"]

    # an old-model sample with the very same numbers is NOT evidence of anything
    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM member_face_samples")
    add_face_sample(existing, unit(3), model="dlib")
    fresh = queue_face(fake.model_id, unit(3), image_for(3), photo="e.png")
    assert client.post(f"/api/queue/{fresh}/register", json={"full_name": "Someone Else"}).status_code == 200


def test_assign_adds_a_sample_of_the_active_model_and_counts_only_those(client, fake):
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO registered_users (full_name, membership_code) VALUES ('Ali', '100002')")
        uid = cur.lastrowid
    add_face_sample(uid, unit(0), model="dlib")
    pid = queue_face("dlib", unit(7), image_for(5), photo="a.png")

    r = client.post(f"/api/queue/{pid}/assign", json={"user_id": uid})

    assert r.status_code == 200, r.text
    assert r.json()["sample_count"] == 1                      # the dlib sample is not counted
    assert (fake.model_id, "assigned") in member_samples(uid)
    assert client.app.state.face_engine.reloads == 1
