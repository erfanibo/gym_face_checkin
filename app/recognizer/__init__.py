"""
Face recognition backends behind one interface. See base.py for the contract.

    from app.recognizer import get_recognizer
    faces = get_recognizer().detect_and_embed(frame)

The backend is chosen once, from config.RECOGNIZER_BACKEND ("deepface" | "dlib").
"""
from __future__ import annotations

import threading

from .. import config
from .base import DetectedFace, Recognizer, Tolerances

__all__ = ["DetectedFace", "Recognizer", "Tolerances", "get_recognizer", "set_recognizer", "create_recognizer"]

_instance: Recognizer | None = None
_lock = threading.Lock()


def create_recognizer(backend: str | None = None) -> Recognizer:
    backend = (backend or config.RECOGNIZER_BACKEND).strip().lower()
    if backend == "deepface":
        from .deepface_backend import DeepFaceRecognizer
        return DeepFaceRecognizer()
    if backend == "dlib":
        from .dlib_backend import DlibRecognizer
        return DlibRecognizer()
    raise ValueError(f"unknown RECOGNIZER_BACKEND {backend!r} (use 'deepface' or 'dlib')")


def get_recognizer() -> Recognizer:
    """The process-wide recognizer, created on first use."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = create_recognizer()
        return _instance


def set_recognizer(recognizer: Recognizer | None) -> None:
    """Replace (or with None, reset) the process-wide recognizer -- used by tests."""
    global _instance
    with _lock:
        _instance = recognizer
