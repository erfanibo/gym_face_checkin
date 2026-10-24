import cv2
import pytest

from app import config
from app.database import db_cursor
from app.face_samples import add_face_sample, blob_to_encoding
from app.reembed import reembed_missing_members, resolve_photo
from tests.fakes import FakeRecognizer, image_for, unit


def add_member(name, photo_name=None, image=None, folder=None):
    if photo_name and image is not None:
        cv2.imwrite(str((folder or config.PENDING_PHOTOS_DIR) / photo_name), image)
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO registered_users (full_name, photo_path) VALUES (?, ?)",
            (name, f"/old/machine/path/{photo_name}" if photo_name else None),
        )
        return cur.lastrowid


def samples(user_id):
    with db_cursor() as cur:
        cur.execute("SELECT model, source, encoding FROM member_face_samples WHERE user_id=? ORDER BY id", (user_id,))
        return [(r["model"], r["source"], blob_to_encoding(r["encoding"])) for r in cur.fetchall()]


def test_member_with_only_old_model_samples_gets_a_new_one_from_the_photo(db):
    uid = add_member("Ali", "a.png", image_for(3))
    add_face_sample(uid, unit(0), model="dlib")
    rec = FakeRecognizer("fake/1")

    report = reembed_missing_members(rec)

    assert report.migrated == [uid] and not report.needs_attention
    rows = samples(uid)
    assert [(m, s) for m, s, _ in rows] == [("dlib", "assigned"), ("fake/1", "migration")]
    assert rows[1][2] == pytest.approx(unit(3))          # vector came from the photo, via the new model
    assert rows[0][2] == pytest.approx(unit(0))          # the old sample is untouched (way back to dlib)


def test_is_idempotent(db):
    uid = add_member("Ali", "a.png", image_for(1))
    rec = FakeRecognizer()
    assert reembed_missing_members(rec).migrated == [uid]
    assert reembed_missing_members(rec).migrated == []
    assert len(samples(uid)) == 1


def test_missing_file_and_unreadable_face_are_reported_not_fatal(db):
    gone = add_member("Gone", "x.png", image_for(1))
    (config.PENDING_PHOTOS_DIR / "x.png").unlink()
    blank = add_member("Blank", "b.png", image_for(2))
    ok = add_member("Ok", "c.png", image_for(4))

    class PickyRecognizer(FakeRecognizer):
        def embed_image(self, image_bgr):
            return None if int(image_bgr[0, 0, 0]) == 12 else super().embed_image(image_bgr)

    report = reembed_missing_members(PickyRecognizer())
    assert report.migrated == [ok]
    assert report.no_photo_file == [gone] and report.no_face == [blank]
    assert sorted(report.needs_attention) == sorted([gone, blank])


def test_one_exploding_photo_does_not_stop_the_rest(db):
    bad = add_member("Bad", "bad.png", image_for(1))
    good = add_member("Good", "good.png", image_for(2))

    class Exploding(FakeRecognizer):
        def embed_image(self, image_bgr):
            if int(image_bgr[0, 0, 0]) == 11:
                raise RuntimeError("boom")
            return super().embed_image(image_bgr)

    report = reembed_missing_members(Exploding())
    assert report.migrated == [good] and report.failed[0][0] == bad and "boom" in report.failed[0][1]


def test_dry_run_writes_nothing(db):
    uid = add_member("Ali", "a.png", image_for(1))
    report = reembed_missing_members(FakeRecognizer(), dry_run=True)
    assert report.pending == [uid] and report.migrated == [] and samples(uid) == []


def test_members_without_a_photo_are_not_the_jobs_business(db):
    uid = add_member("Manual only")           # added without camera: never had a face
    report = reembed_missing_members(FakeRecognizer())
    assert report.migrated == [] and report.needs_attention == [] and samples(uid) == []


def test_photo_is_found_by_file_name_even_if_the_stored_directory_is_stale(db):
    add_member("Ali", "p.png", image_for(1), folder=config.USER_PHOTOS_DIR)
    assert resolve_photo("/some/other/dir/p.png") == config.USER_PHOTOS_DIR / "p.png"
    assert resolve_photo("/some/other/dir/none.png") is None and resolve_photo(None) is None


def test_stop_request_ends_the_job_early(db):
    add_member("A", "a.png", image_for(1))
    add_member("B", "b.png", image_for(2))
    report = reembed_missing_members(FakeRecognizer(), should_stop=lambda: True)
    assert report.migrated == []


def test_sample_cap_applies_per_model(db, monkeypatch):
    monkeypatch.setattr(config, "MAX_FACE_SAMPLES_PER_MEMBER", 2)
    uid = add_member("Ali")
    for i in range(3):
        add_face_sample(uid, unit(i), model="dlib")
    for i in range(3):
        add_face_sample(uid, unit(i), model="fake/1")
    models = [m for m, _, _ in samples(uid)]
    assert models.count("dlib") == 2 and models.count("fake/1") == 2
