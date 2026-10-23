"""A deterministic stand-in for a real face library, so tests never need model weights or a camera."""
from __future__ import annotations

import numpy as np

from app.recognizer.base import DetectedFace, Recognizer

DIM = 8


def unit(i: int) -> np.ndarray:
    """Orthogonal 'faces': distance 0 to itself, sqrt(2) to any other (euclidean)."""
    v = np.zeros(DIM)
    v[i % DIM] = 1.0
    return v


def image_for(i: int, size: int = 40) -> np.ndarray:
    """A flat-colour BGR image that FakeRecognizer.embed_image maps back to unit(i)."""
    return np.full((size, size, 3), 10 + i, dtype=np.uint8)


class FakeRecognizer(Recognizer):
    verification_threshold = 0.6  # euclidean, like dlib -> known 0.5, pending 0.58

    def __init__(self, model_id: str = "fake/1"):
        self.model_id = model_id
        self.faces: list[DetectedFace] = []        # what detect_and_embed returns next
        self.unreadable_images = False             # embed_image returns None
        self.warm_up_calls = 0
        self.warm_up_error: Exception | None = None

    def warm_up(self) -> None:
        self.warm_up_calls += 1
        if self.warm_up_error is not None:
            raise self.warm_up_error

    def detect_and_embed(self, frame_bgr):
        return list(self.faces)

    def embed_image(self, image_bgr):
        if self.unreadable_images or image_bgr is None:
            return None
        return unit(int(image_bgr[0, 0, 0]) - 10)

    def distances(self, known, vec):
        return np.linalg.norm(np.asarray(known) - np.asarray(vec), axis=1)
