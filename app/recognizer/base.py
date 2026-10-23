"""
The one interface the rest of the app uses for "find faces in a picture and turn
each into a vector". face_engine.py and routers/queue.py talk ONLY to this --
never to face_recognition / DeepFace directly -- so swapping the library (or
the model inside it) is a change inside app/recognizer/, nowhere else.

Conventions every backend follows:
  * Frames/images come in as OpenCV BGR uint8 arrays.
  * Face boxes go out as (top, right, bottom, left) in pixels of the ORIGINAL
    full-size image (backends that shrink the picture internally scale back).
  * Vectors are 1-D float64 numpy arrays; "distance" is lower = more similar.
  * `model_id` names the vector space. Vectors with different model_ids are not
    comparable and must never be matched against each other (the database tags
    every stored vector with it; see database.py).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .. import config

# The previous (dlib) setup used 0.5 for "this is that member" and 0.58 for the anti-spam check,
# on a library whose own default is 0.6. Those two ratios (0.5/0.6 and 0.58/0.6) are the
# tuning the project already lived with, so every other model gets its default tolerances by
# applying the same ratios to ITS standard "same person" threshold.
KNOWN_RATIO = 0.5 / 0.6
PENDING_RATIO = 0.58 / 0.6


@dataclass(frozen=True)
class DetectedFace:
    box: tuple[int, int, int, int]  # (top, right, bottom, left), full-image pixels
    embedding: np.ndarray           # 1-D float64
    confidence: float = 1.0         # detector's confidence, 0..1 (1.0 if the backend has none)


@dataclass(frozen=True)
class Tolerances:
    known: float                    # closer than this to a member sample => IS that member
    pending: float                  # closer than this to a queued face => same unknown person
    registration_duplicate: float   # closer than this to a member at signup => duplicate registration


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class Recognizer(ABC):
    #: names the vector space, e.g. "dlib" or "deepface/SFace"
    model_id: str
    #: the model's own standard "same person" distance threshold
    verification_threshold: float

    def tolerances(self) -> Tolerances:
        """Defaults derived from the model's threshold, overridden by the optional values in
        config (KNOWN_USER_TOLERANCE, PENDING_DEDUP_TOLERANCE, REGISTRATION_DUPLICATE_FACE_TOLERANCE)."""
        known = config.KNOWN_USER_TOLERANCE
        if known is None:
            known = round(self.verification_threshold * KNOWN_RATIO, 4)
        pending = config.PENDING_DEDUP_TOLERANCE
        if pending is None:
            pending = round(self.verification_threshold * PENDING_RATIO, 4)
        duplicate = config.REGISTRATION_DUPLICATE_FACE_TOLERANCE
        if duplicate is None:
            duplicate = known
        return Tolerances(known=known, pending=pending, registration_duplicate=duplicate)

    @abstractmethod
    def warm_up(self) -> None:
        """Load models now (first use may download weights). Raises if that is impossible."""

    @abstractmethod
    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        """All faces found in a camera frame. Empty list when there are none."""

    @abstractmethod
    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """
        Vector for the main face of an already-cropped face photo (used to rebuild vectors from
        stored member photos). The largest face wins; if the detector finds none in a tight crop
        the whole image is treated as the face. None only if the image is unusable.
        """

    @abstractmethod
    def distances(self, known: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Distance from `vec` to each row of `known` (an N x D matrix)."""
