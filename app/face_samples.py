"""
Storing face vectors ("samples") for members. Kept out of face_engine.py so both
the camera engine and the re-embedding job (reembed.py) can use it without
importing each other.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from . import config
from .database import db_cursor


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoding_to_blob(encoding: np.ndarray) -> bytes:
    """Face vector (any length: 128 for dlib/SFace, 512 for Facenet512/ArcFace, ...) as float64
    bytes, for a BLOB column. The length is implied by the byte count."""
    return np.asarray(encoding, dtype=np.float64).reshape(-1).tobytes()


def blob_to_encoding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)


def add_face_sample(user_id: int, encoding: np.ndarray, source: str = "assigned", model: str | None = None) -> None:
    """
    Adds one more recognition sample for an EXISTING member. Used right after
    registration (source='registration', routers/queue.py:register_pending),
    whenever an operator says "this pending-queue face is actually this same
    member, just a different angle/light" (source='assigned',
    routers/queue.py:assign_pending_to_member), and when vectors are rebuilt from
    a member's stored photo after the recognizer model changed (source='migration',
    reembed.py).

    `model` names the vector space the encoding belongs to (defaults to the active
    recognizer's model_id); a sample is only ever compared against vectors of its
    own model.

    Caps each member at config.MAX_FACE_SAMPLES_PER_MEMBER samples PER MODEL: once
    the cap is hit, the OLDEST sample of that model is dropped to make room for the
    new one. This keeps the per-frame comparison cost bounded (see
    FaceEngine._match_known) and naturally rotates a member's profile toward their
    most recent appearances instead of growing forever.

    Callers are responsible for calling FaceEngine.reload_known_users()
    afterwards so the background camera thread's in-memory cache picks up
    the new sample immediately, rather than waiting for the next full reload.
    """
    if model is None:
        from .recognizer import get_recognizer
        model = get_recognizer().model_id
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO member_face_samples (user_id, encoding, model, source, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, encoding_to_blob(encoding), model, source, now_iso()),
        )
        cur.execute(
            "SELECT id FROM member_face_samples WHERE user_id=? AND model=? ORDER BY id ASC",
            (user_id, model),
        )
        sample_ids = [r["id"] for r in cur.fetchall()]
        overflow = len(sample_ids) - config.MAX_FACE_SAMPLES_PER_MEMBER
        if overflow > 0:
            oldest = [(i,) for i in sample_ids[:overflow]]
            cur.executemany("DELETE FROM member_face_samples WHERE id=?", oldest)
