import numpy as np
import pytest

from scripts.benchmark_deepface import evaluate, load_labelled_images
from tests.fakes import unit


def euclid(matrix, vec):
    return np.linalg.norm(np.asarray(matrix) - np.asarray(vec), axis=1)


def noisy(i, eps):
    v = unit(i).copy()
    v[(i + 1) % 8] += eps
    return v


def test_perfect_separation():
    gallery = {"ali": [unit(0), noisy(0, 0.05), noisy(0, 0.1)], "sara": [unit(1), noisy(1, 0.05)]}
    r = evaluate(gallery, euclid, known_tolerance=0.5)
    assert r["photos"] == 5 and r["people"] == 2
    assert r["genuine_pairs"] == 4 and r["impostor_pairs"] == 6
    assert r["false_accept_rate"] == 0 and r["false_reject_rate"] == 0
    assert r["best_balanced_accuracy"] == 1.0
    assert r["identify"] == {"queries": 5, "correct": 5, "wrong_person": 0, "not_recognised": 0}


def test_too_strict_tolerance_means_members_are_not_recognised():
    gallery = {"ali": [unit(0), noisy(0, 0.4)], "sara": [unit(1), noisy(1, 0.4)]}
    r = evaluate(gallery, euclid, known_tolerance=0.1)
    assert r["false_reject_rate"] == 1.0 and r["false_accept_rate"] == 0
    assert r["identify"]["not_recognised"] == 4 and r["identify"]["correct"] == 0


def test_too_loose_tolerance_lets_strangers_in():
    gallery = {"ali": [unit(0), unit(0)], "sara": [unit(1), unit(1)]}
    r = evaluate(gallery, euclid, known_tolerance=5.0)
    assert r["false_accept_rate"] == 1.0


def test_wrong_person_is_counted():
    # sara's second photo looks exactly like ali's: identification picks ali for it
    gallery = {"ali": [unit(0), unit(0)], "sara": [unit(0), unit(2)]}
    r = evaluate(gallery, euclid, known_tolerance=0.5)
    assert r["identify"]["wrong_person"] >= 1


def test_a_person_with_one_photo_is_not_queried_but_still_a_stranger_to_others():
    gallery = {"ali": [unit(0), noisy(0, 0.05)], "bob": [unit(3)]}
    r = evaluate(gallery, euclid, known_tolerance=0.5)
    assert r["identify"]["queries"] == 2 and r["impostor_pairs"] == 2


def test_not_enough_data_is_an_error():
    with pytest.raises(ValueError):
        evaluate({"ali": [unit(0), unit(0)]}, euclid, 0.5)
    with pytest.raises(ValueError):
        evaluate({"ali": [unit(0)], "sara": [unit(1)]}, euclid, 0.5)


def test_load_labelled_images(tmp_path):
    (tmp_path / "ali").mkdir()
    (tmp_path / "ali" / "1.jpg").write_bytes(b"x")
    (tmp_path / "ali" / "notes.txt").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    (tmp_path / "loose.jpg").write_bytes(b"x")
    found = load_labelled_images(tmp_path)
    assert list(found) == ["ali"] and [p.name for p in found["ali"]] == ["1.jpg"]
