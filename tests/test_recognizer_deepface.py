"""DeepFaceRecognizer's parsing/filtering, with deepface.DeepFace.represent replaced by a fake that
returns what the real one is documented to return (list of {embedding, facial_area, face_confidence})."""
import numpy as np
import pytest

from app import config
from app.recognizer.deepface_backend import DeepFaceRecognizer, _cosine_threshold


class FakeDeepFace:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def represent(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if kwargs["detector_backend"] == "skip":
            return [{"embedding": [3.0, 4.0], "facial_area": {"x": 0, "y": 0, "w": 10, "h": 10}, "face_confidence": 0}]
        return self.results


def face(x, y, w, h, conf=0.99, emb=(1.0, 0.0)):
    return {"embedding": list(emb), "facial_area": {"x": x, "y": y, "w": w, "h": h}, "face_confidence": conf}


@pytest.fixture
def rec():
    r = DeepFaceRecognizer(model_name="SFace", detector="yunet")
    return r


def frame(h=480, w=640):
    return np.zeros((h, w, 3), np.uint8)


def test_model_id_and_threshold(rec):
    assert rec.model_id == "deepface/SFace"
    assert rec.verification_threshold == pytest.approx(0.593)
    assert _cosine_threshold("Facenet512") == pytest.approx(0.30)
    with pytest.raises(ValueError):
        _cosine_threshold("NoSuchModel")


def test_default_tolerances_use_the_projects_existing_ratios(rec, monkeypatch):
    for name in ("KNOWN_USER_TOLERANCE", "PENDING_DEDUP_TOLERANCE", "REGISTRATION_DUPLICATE_FACE_TOLERANCE"):
        monkeypatch.setattr(config, name, None)
    t = rec.tolerances()
    assert t.known == pytest.approx(0.593 * 0.5 / 0.6, abs=1e-3)
    assert t.pending == pytest.approx(0.593 * 0.58 / 0.6, abs=1e-3)
    assert t.registration_duplicate == t.known
    assert t.known < t.pending < rec.verification_threshold


def test_config_overrides_win(rec, monkeypatch):
    monkeypatch.setattr(config, "KNOWN_USER_TOLERANCE", 0.31)
    monkeypatch.setattr(config, "PENDING_DEDUP_TOLERANCE", None)
    monkeypatch.setattr(config, "REGISTRATION_DUPLICATE_FACE_TOLERANCE", None)
    t = rec.tolerances()
    assert t.known == 0.31 and t.registration_duplicate == 0.31 and t.pending > 0.31


def test_detect_returns_boxes_as_top_right_bottom_left_and_unit_vectors(rec):
    rec._df = FakeDeepFace([face(100, 50, 80, 90, emb=(3.0, 4.0))])
    [f] = rec.detect_and_embed(frame())
    assert f.box == (50, 180, 140, 100)               # top=y, right=x+w, bottom=y+h, left=x
    assert np.linalg.norm(f.embedding) == pytest.approx(1.0)
    call = rec._df.calls[0]
    assert call["model_name"] == "SFace" and call["detector_backend"] == "yunet"
    assert call["enforce_detection"] is True


def test_big_frames_are_downscaled_and_boxes_scaled_back(rec, monkeypatch):
    monkeypatch.setattr(config, "DEEPFACE_MAX_FRAME_SIDE", 960)
    fd = FakeDeepFace([face(100, 60, 120, 120)])
    rec._df = fd
    [f] = rec.detect_and_embed(frame(1080, 1920))    # shrunk by 0.5 before detection
    sent = fd.calls[0]["img_path"]
    assert sent.shape[:2] == (540, 960)
    assert f.box == (120, 440, 360, 200)             # coordinates of the ORIGINAL 1920x1080 frame


def test_small_frames_are_not_resized(rec):
    fd = FakeDeepFace([face(10, 10, 100, 100)])
    rec._df = fd
    rec.detect_and_embed(frame(480, 640))
    assert fd.calls[0]["img_path"].shape[:2] == (480, 640)


def test_low_confidence_and_tiny_faces_are_ignored(rec, monkeypatch):
    monkeypatch.setattr(config, "DEEPFACE_MIN_FACE_CONFIDENCE", 0.85)
    monkeypatch.setattr(config, "DEEPFACE_MIN_FACE_PIXELS", 60)
    rec._df = FakeDeepFace([face(0, 0, 100, 100, conf=0.5), face(0, 0, 30, 30), face(200, 100, 100, 100)])
    faces = rec.detect_and_embed(frame())
    assert [f.box for f in faces] == [(100, 300, 200, 200)]


def test_missing_confidence_counts_as_confident(rec):
    item = face(10, 10, 100, 100)
    item["face_confidence"] = None
    rec._df = FakeDeepFace([item])
    assert len(rec.detect_and_embed(frame())) == 1


def test_no_face_is_an_empty_list_not_an_error(rec):
    from deepface.modules.exceptions import FaceNotDetected
    rec._df = FakeDeepFace(error=FaceNotDetected("Face could not be detected."))
    assert rec.detect_and_embed(frame()) == []


def test_real_problems_are_not_mistaken_for_no_face(rec):
    rec._df = FakeDeepFace(error=ValueError("An exception occurred while downloading face_recognition_sface_2021dec.onnx"))
    with pytest.raises(ValueError, match="downloading"):
        rec.detect_and_embed(frame())


def test_embed_image_picks_the_largest_face(rec):
    rec._df = FakeDeepFace([face(0, 0, 20, 20, emb=(1.0, 0.0)), face(0, 0, 90, 90, emb=(0.0, 1.0))])
    v = rec.embed_image(frame(100, 100))
    assert v == pytest.approx([0.0, 1.0])


def test_embed_image_falls_back_to_the_whole_crop_when_no_face_is_detected(rec):
    from deepface.modules.exceptions import FaceNotDetected

    class NoFaceThenSkip(FakeDeepFace):
        def represent(self, **kw):
            if kw["detector_backend"] != "skip":
                raise FaceNotDetected("Face could not be detected.")
            return super().represent(**kw)

    fd = NoFaceThenSkip()
    rec._df = fd
    v = rec.embed_image(frame(100, 100))
    assert v == pytest.approx([0.6, 0.8])                     # (3,4) normalised
    assert fd.calls[-1]["detector_backend"] == "skip" and fd.calls[-1]["enforce_detection"] is False


def test_embed_image_rejects_empty_input(rec):
    assert rec.embed_image(None) is None
    assert rec.embed_image(np.zeros((0, 0, 3), np.uint8)) is None


def test_cosine_distances(rec):
    known = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])    # un-normalised on purpose
    d = rec.distances(known, np.array([1.0, 0.0]))
    assert d == pytest.approx([0.0, 1.0, 1 - 1 / np.sqrt(2)])


def test_unknown_backend_name_is_rejected():
    from app.recognizer import create_recognizer
    with pytest.raises(ValueError, match="RECOGNIZER_BACKEND"):
        create_recognizer("nope")
