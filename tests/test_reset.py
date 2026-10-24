"""Settings > بازنشانی اطلاعات: what each category deletes, what it must NOT touch, and the safeguards."""
import cv2
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import auth, config, data_reset
from app.database import db_cursor, get_setting, set_setting
from app.face_samples import add_face_sample
from app.routers import reset as reset_router
from app.routers import settings as settings_router
from tests.fakes import image_for, unit


class StubEngine:
    def __init__(self):
        self.reloads = 0

    def reload_all(self):
        self.reloads += 1


@pytest.fixture
def client(db, fake):
    auth.ensure_manager_password_seeded()
    app = FastAPI()
    app.include_router(reset_router.router)
    app.dependency_overrides[auth.require_manager] = lambda: None
    app.state.face_engine = StubEngine()
    return TestClient(app)


def photo(name, folder=None, i=1):
    path = (folder or config.PENDING_PHOTOS_DIR) / name
    cv2.imwrite(str(path), image_for(i))
    return path


def seed():
    """2 members (Ali has a photo, face sample, attendance, plan purchase, workout) + queue + log + plan."""
    ali_photo = photo("ali.png", i=1)
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO membership_plans (name, price, duration_days) VALUES ('ماهانه', 100, 30)")
        plan = cur.lastrowid
        cur.execute(
            "INSERT INTO registered_users (full_name, membership_code, photo_path, encoding, membership_expires_at) "
            "VALUES ('Ali', '111111', ?, x'00', '2999-01-01T00:00:00+00:00')", (str(ali_photo),))
        ali = cur.lastrowid
        cur.execute("INSERT INTO registered_users (full_name, membership_code) VALUES ('Sara', '222222')")
        sara = cur.lastrowid
        cur.execute("INSERT INTO attendance_log (user_id, event_type) VALUES (?, 'in')", (ali,))
        cur.execute("INSERT INTO workout_items (user_id, position, exercise_name, sets, reps) VALUES (?, 0, 'squat', 3, '10')", (ali,))
        cur.execute(
            "INSERT INTO membership_purchases (user_id, plan_id, plan_name, price, duration_days, starts_at, expires_at, created_at) "
            "VALUES (?, ?, 'ماهانه', 100, 30, 't', 't', 't')", (ali, plan))
        cur.execute("INSERT INTO recognition_log (user_id, full_name, distance) VALUES (?, 'Ali', 0.1)", (ali,))
        cur.execute("INSERT INTO recognition_log (user_id, full_name, distance) VALUES (NULL, 'چهرهٔ ناشناس #1', NULL)")
        # queue: one registered (history of Ali), one waiting, one rejected
        cur.execute("INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at, status, registered_user_id) "
                    "VALUES (x'00', 'fake/1', ?, 't', 't', 'registered', ?)", (str(ali_photo), ali))
    waiting = photo("waiting.png", i=2)
    rejected = photo("rejected.png", i=3)
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at, status) VALUES (x'00','fake/1',?, 't','t','pending')", (str(waiting),))
        cur.execute("INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at, status) VALUES (x'00','fake/1',?, 't','t','rejected')", (str(rejected),))
    add_face_sample(ali, unit(1), model="fake/1")
    return {"ali": ali, "sara": sara, "plan": plan, "ali_photo": ali_photo, "waiting": waiting, "rejected": rejected}


def n(table, where=""):
    with db_cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} {where}")
        return cur.fetchone()[0]


def post(client, cats, confirm="حذف"):
    return client.post("/api/reset", json={"categories": cats, "confirm": confirm})


# ------------------------------------------------------------------ preview
def test_preview_counts_and_dependencies(client):
    seed()
    body = client.get("/api/reset/preview").json()
    by_id = {c["id"]: c for c in body["categories"]}
    assert [c["id"] for c in body["categories"]] == data_reset.CATEGORY_IDS
    assert by_id["members"]["count"] == 2 and by_id["attendance"]["count"] == 1 and by_id["plans"]["count"] == 1
    assert by_id["pending"]["count"] == 2                      # waiting + rejected, not the registered one
    assert by_id["member_photos"]["count"] == 1
    assert set(by_id["members"]["implies"]) == {"attendance", "face_samples", "workouts", "purchases", "member_photos", "recognition_log"}
    assert body["confirm_selected"] == "حذف" and body["confirm_all"] == "حذف همه"


# ------------------------------------------------------------------ single categories
def test_each_simple_category_touches_only_its_own_data(client):
    ids = seed()
    assert post(client, ["recognition_log"]).status_code == 200
    assert n("recognition_log") == 0 and n("attendance_log") == 1 and n("registered_users") == 2

    assert post(client, ["attendance"]).status_code == 200 and n("attendance_log") == 0
    assert post(client, ["workouts"]).status_code == 200 and n("workout_items") == 0
    assert n("membership_purchases") == 1 and n("member_face_samples") == 1 and n("registered_users") == 2


def test_pending_deletes_waiting_and_rejected_with_their_photos_but_keeps_member_history(client):
    ids = seed()
    r = post(client, ["pending"])
    assert r.status_code == 200
    assert n("pending_queue") == 1 and n("pending_queue", "WHERE status='registered'") == 1
    assert not ids["waiting"].exists() and not ids["rejected"].exists()
    assert ids["ali_photo"].exists()                           # member's photo is shared with the registered row: untouched
    assert r.json()["photos_removed"] == 2


def test_purchases_also_clears_every_membership_end_date(client):
    seed()
    assert post(client, ["purchases"]).status_code == 200
    assert n("membership_purchases") == 0
    assert n("registered_users", "WHERE membership_expires_at IS NOT NULL") == 0
    assert n("membership_plans") == 1                          # plans untouched


def test_face_samples_keep_the_members_and_their_photos(client):
    ids = seed()
    assert post(client, ["face_samples"]).status_code == 200
    assert n("member_face_samples") == 0 and n("registered_users") == 2
    assert ids["ali_photo"].exists()
    assert n("registered_users", "WHERE encoding IS NOT NULL") == 0


def test_member_photos_removes_files_and_clears_the_path_only(client):
    ids = seed()
    assert post(client, ["member_photos"]).status_code == 200
    assert not ids["ali_photo"].exists()
    assert n("registered_users", "WHERE photo_path IS NOT NULL") == 0 and n("registered_users") == 2
    assert ids["waiting"].exists()                             # the waiting queue photo is not a member photo


def test_plans_deleted_but_purchase_history_survives_with_its_snapshot(client):
    seed()
    assert post(client, ["plans"]).status_code == 200
    assert n("membership_plans") == 0 and n("membership_purchases") == 1
    with db_cursor() as cur:
        cur.execute("SELECT plan_id, plan_name FROM membership_purchases")
        row = cur.fetchone()
    assert row["plan_id"] is None and row["plan_name"] == "ماهانه"


# ------------------------------------------------------------------ members + dependents
def test_deleting_members_drags_everything_that_points_at_them(client):
    ids = seed()
    r = post(client, ["members"])
    assert r.status_code == 200, r.text
    assert set(r.json()["applied"]) >= {"members", "attendance", "face_samples", "workouts", "purchases", "member_photos", "recognition_log"}
    for table in ("registered_users", "attendance_log", "member_face_samples", "workout_items", "membership_purchases", "recognition_log"):
        assert n(table) == 0, table
    assert not ids["ali_photo"].exists()
    assert n("pending_queue", "WHERE registered_user_id IS NOT NULL") == 0
    assert n("membership_plans") == 1                          # plans and the waiting queue are not members' data
    assert ids["waiting"].exists() and n("pending_queue", "WHERE status='pending'") == 1


# ------------------------------------------------------------------ everything
def test_delete_everything_empties_all_but_keeps_the_manager_password(client):
    ids = seed()
    set_setting("club_info_name", "Gym X")
    set_setting("behavior_attendance_cooldown_seconds", "60")
    config.ATTENDANCE_COOLDOWN_SECONDS = 60
    original_default = settings_router._BEHAVIOR_DEFAULTS["ATTENDANCE_COOLDOWN_SECONDS"]
    password_hash = get_setting(auth.MANAGER_PASSWORD_SETTING_KEY)
    assert password_hash

    r = client.post("/api/reset/all", json={"confirm": "حذف همه", "password": config.MANAGER_INITIAL_PASSWORD})

    assert r.status_code == 200, r.text
    for table in ("registered_users", "pending_queue", "attendance_log", "member_face_samples", "workout_items",
                  "membership_plans", "membership_purchases", "recognition_log"):
        assert n(table) == 0, table
    assert get_setting("club_info_name") is None and get_setting("behavior_attendance_cooldown_seconds") is None
    assert get_setting(auth.MANAGER_PASSWORD_SETTING_KEY) == password_hash        # still able to log in
    assert config.ATTENDANCE_COOLDOWN_SECONDS == original_default                  # live config back to defaults too
    assert not any(p.exists() for p in (ids["ali_photo"], ids["waiting"], ids["rejected"]))
    assert client.app.state.face_engine.reloads == 1


# ------------------------------------------------------------------ safeguards
def test_wrong_or_missing_confirmation_deletes_nothing(client):
    seed()
    for confirm in ("", "نه", "حذف همه", "delete"):
        assert post(client, ["attendance"], confirm=confirm).status_code == 400
    assert n("attendance_log") == 1 and client.app.state.face_engine.reloads == 0


def test_confirmation_ignores_invisible_characters_and_extra_spaces(client):
    seed()
    assert post(client, ["attendance"], confirm="  ح‌ذف ‏").status_code == 200


def test_empty_or_unknown_selection_is_rejected(client):
    seed()
    assert post(client, []).status_code == 400
    assert post(client, ["attendance", "everything_please"]).status_code == 422
    assert n("attendance_log") == 1


def test_delete_everything_needs_the_right_password_and_phrase(client):
    seed()
    bad_pw = client.post("/api/reset/all", json={"confirm": "حذف همه", "password": "wrong"})
    assert bad_pw.status_code == 403
    no_pw = client.post("/api/reset/all", json={"confirm": "حذف همه"})
    assert no_pw.status_code == 403
    short_phrase = client.post("/api/reset/all", json={"confirm": "حذف", "password": config.MANAGER_INITIAL_PASSWORD})
    assert short_phrase.status_code == 400
    assert n("registered_users") == 2 and client.app.state.face_engine.reloads == 0


def test_requires_a_manager_session(db, fake):
    app = FastAPI()
    app.include_router(reset_router.router)           # no dependency override: real auth
    app.state.face_engine = StubEngine()
    from starlette.middleware.sessions import SessionMiddleware
    app.add_middleware(SessionMiddleware, secret_key="x")
    c = TestClient(app)
    assert c.get("/api/reset/preview").status_code == 401
    assert c.post("/api/reset", json={"categories": ["attendance"], "confirm": "حذف"}).status_code == 401
    assert c.post("/api/reset/all", json={"confirm": "حذف همه", "password": "x"}).status_code == 401


def test_photo_deletion_never_leaves_the_photo_folders(client, tmp_path):
    outside = tmp_path / "precious.txt"
    outside.write_text("keep me")
    with db_cursor(commit=True) as cur:
        cur.execute("INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at, status) "
                    "VALUES (x'00','fake/1',?, 't','t','pending')", (str(outside),))
    # the stored directory is ignored: only the file NAME is looked up inside the two photo folders
    assert post(client, ["pending"]).status_code == 200
    assert outside.exists()
