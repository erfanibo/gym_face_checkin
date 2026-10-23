"""
The previous engine (face_recognition / dlib HOG + 128-d ResNet vectors), moved
behind the Recognizer interface unchanged. Kept for A/B comparison and as a way
back; it needs `pip install -r requirements-dlib.txt` (dlib compiles from source).
"""
from __future__ import annotations

import cv2
import numpy as np

from .. import config
from .base import DetectedFace, Recognizer


class DlibRecognizer(Recognizer):
    model_id = "dlib"
    verification_threshold = 0.6  # face_recognition's own default tolerance

    def __init__(self):
        try:
            import face_recognition  # imported lazily: heavy, and absent when DeepFace is the backend
        except ImportError as exc:
            raise RuntimeError(
                "RECOGNIZER_BACKEND=dlib needs the face_recognition package: "
                "pip install -r requirements-dlib.txt"
            ) from exc
        self._fr = face_recognition

    def warm_up(self) -> None:
        pass  # models ship inside the face_recognition_models package; nothing to load

    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        scale = config.FRAME_RESIZE_SCALE
        small = cv2.resize(frame_bgr, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locations = self._fr.face_locations(rgb_small)
        if not locations:
            return []
        encodings = self._fr.face_encodings(rgb_small, locations)
        inv = 1.0 / scale
        return [
            DetectedFace(
                box=tuple(int(round(v * inv)) for v in loc),  # type: ignore[arg-type]
                embedding=np.asarray(enc, dtype=np.float64),
            )
            for loc, enc in zip(locations, encodings)
        ]

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        locations = self._fr.face_locations(rgb)
        if not locations:
            h, w = rgb.shape[:2]
            locations = [(0, w, h, 0)]  # a tight face crop: the whole image is the face
        top, right, bottom, left = max(locations, key=lambda l: (l[2] - l[0]) * (l[1] - l[3]))
        encodings = self._fr.face_encodings(rgb, [(top, right, bottom, left)])
        return np.asarray(encodings[0], dtype=np.float64) if encodings else None

    def distances(self, known: np.ndarray, vec: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.asarray(known, dtype=np.float64) - np.asarray(vec, dtype=np.float64), axis=1)
