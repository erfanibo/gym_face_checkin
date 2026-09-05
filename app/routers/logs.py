"""
Read-only endpoint backing the "لاگ زنده" (live log) panel in the reception
UI: every time a known member's face is matched by the camera, independent
of attendance_log's 5-minute cooldown. Rows are written by
FaceEngine._maybe_log_recognition (see face_engine.py) and pushed live over
the existing /ws/queue WebSocket as a "recognition_seen" event -- this
endpoint only serves the initial/backfill list when the panel is opened.
"""
from fastapi import APIRouter

from .. import config
from ..database import db_cursor
from ..models import RecognitionLogEntry

router = APIRouter(prefix="/api/recognition-log", tags=["recognition-log"])


@router.get("", response_model=list[RecognitionLogEntry])
def get_recognition_log(limit: int = 200):
    limit = max(1, min(limit, config.RECOGNITION_LOG_MAX_LIMIT))
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, user_id, full_name, distance, created_at "
            "FROM recognition_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]
