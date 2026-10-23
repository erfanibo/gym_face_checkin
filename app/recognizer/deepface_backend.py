"""
DeepFace backend (https://github.com/serengil/deepface).

Defaults are chosen for a normal PC without a GPU: SFace (embedding) + YuNet
(detector), both run by onnxruntime/OpenCV, so TensorFlow is not needed at all.
DEEPFACE_MODEL / DEEPFACE_DETECTOR switch them; scripts/benchmark_deepface.py
measures speed and accuracy of the candidates on the real machine.

One DeepFace.represent() call both finds the faces and embeds each of them, so a
frame is not run through a detector twice.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from .. import config
from .base import DetectedFace, Recognizer, l2_normalize

# DeepFace's standard cosine "same person" thresholds (deepface/config/threshold.py). Used only
# if that module cannot be read; the installed library's own table wins when it can.
_FALLBACK_COSINE_THRESHOLDS = {
    "VGG-Face": 0.68, "Facenet": 0.40, "Facenet512": 0.30, "ArcFace": 0.68, "Dlib": 0.07,
    "SFace": 0.593, "OpenFace": 0.10, "DeepFace": 0.23, "DeepID": 0.015, "GhostFaceNet": 0.65,
    "Buffalo_L": 0.55,
}


def _cosine_threshold(model_name: str) -> float:
    try:
        from deepface.config.threshold import thresholds
        return float(thresholds[model_name]["cosine"])
    except Exception:
        pass
    if model_name in _FALLBACK_COSINE_THRESHOLDS:
        return _FALLBACK_COSINE_THRESHOLDS[model_name]
    raise ValueError(f"unknown DeepFace model {model_name!r} (no cosine threshold known for it)")


def _is_face_not_detected(exc: Exception) -> bool:
    """DeepFace raises a ValueError subclass (FaceNotDetected) when a picture has no face. Other
    ValueErrors (weights that could not be downloaded, bad model name, ...) are REAL problems and
    must not be mistaken for 'nobody in front of the camera'."""
    try:
        from deepface.modules.exceptions import FaceNotDetected
        if isinstance(exc, FaceNotDetected):
            return True
    except ImportError:
        pass
    return isinstance(exc, ValueError) and "Face could not be detected" in str(exc)


class DeepFaceRecognizer(Recognizer):
    def __init__(self, model_name: str | None = None, detector: str | None = None):
        self.model_name = model_name or config.DEEPFACE_MODEL
        self.detector = detector or config.DEEPFACE_DETECTOR
        self.model_id = f"deepface/{self.model_name}"
        self.verification_threshold = _cosine_threshold(self.model_name)

        if config.DEEPFACE_BACKEND_ENGINE:
            # must be set before deepface is imported/first used; setdefault keeps an explicit
            # environment value (e.g. from docker-compose) in charge
            os.environ.setdefault("DEEPFACE_BACKEND_ENGINE", config.DEEPFACE_BACKEND_ENGINE)
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise RuntimeError(
                "RECOGNIZER_BACKEND=deepface needs the deepface package: pip install -r requirements.txt"
            ) from exc
        self._df = DeepFace
        # The camera thread and request threads (registration re-embeds a stored photo) can both
        # call in; onnxruntime/OpenCV sessions are not guaranteed re-entrant, so calls take turns.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def warm_up(self) -> None:
        from deepface.modules import modeling
        with self._lock:
            modeling.build_model(task="facial_recognition", model_name=self.model_name)
            modeling.build_model(task="face_detector", model_name=self.detector)

    def _represent(self, image: np.ndarray, detector: str) -> list[dict]:
        with self._lock:
            try:
                results = self._df.represent(
                    img_path=image,
                    model_name=self.model_name,
                    detector_backend=detector,
                    enforce_detection=detector != "skip",
                    align=True,
                    l2_normalize=True,
                )
            except Exception as exc:
                if _is_face_not_detected(exc):
                    return []
                raise
        return list(results)

    # ------------------------------------------------------------------ #
    def detect_and_embed(self, frame_bgr: np.ndarray) -> list[DetectedFace]:
        height, width = frame_bgr.shape[:2]
        scale = 1.0
        image = frame_bgr
        cap = config.DEEPFACE_MAX_FRAME_SIDE
        longest = max(height, width)
        if cap > 0 and longest > cap:
            scale = cap / longest
            image = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)

        faces: list[DetectedFace] = []
        for item in self._represent(image, self.detector):
            confidence = item.get("face_confidence")
            confidence = 1.0 if confidence is None else float(confidence)
            if confidence < config.DEEPFACE_MIN_FACE_CONFIDENCE:
                continue
            area = item["facial_area"]
            x, y, w, h = (float(area[k]) for k in ("x", "y", "w", "h"))
            if config.DEEPFACE_MIN_FACE_PIXELS > 0 and w / scale < config.DEEPFACE_MIN_FACE_PIXELS:
                continue
            top = max(0, int(round(y / scale)))
            left = max(0, int(round(x / scale)))
            bottom = min(height, int(round((y + h) / scale)))
            right = min(width, int(round((x + w) / scale)))
            if bottom <= top or right <= left:
                continue
            faces.append(DetectedFace((top, right, bottom, left), l2_normalize(item["embedding"]), confidence))
        return faces

    def embed_image(self, image_bgr: np.ndarray) -> np.ndarray | None:
        if image_bgr is None or image_bgr.size == 0:
            return None
        items = self._represent(image_bgr, self.detector)
        if items:
            best = max(items, key=lambda it: it["facial_area"]["w"] * it["facial_area"]["h"])
        else:
            # a tight face crop can defeat the detector (no margin around the face); the crop IS
            # the face, so embed it as it is
            skipped = self._represent(image_bgr, "skip")
            if not skipped:
                return None
            best = skipped[0]
        return l2_normalize(best["embedding"])

    def distances(self, known: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Cosine distance (1 - cosine similarity); vectors are L2-normalised first so the
        result does not depend on whether a stored vector was normalised."""
        matrix = np.asarray(known, dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1.0
        return 1.0 - (matrix / norms[:, None]) @ l2_normalize(vec)
