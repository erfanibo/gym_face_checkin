from pathlib import Path

from fastapi import APIRouter

from ..database import db_cursor
from ..models import AttendanceEvent, RegisteredUserOut

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/users", response_model=list[RegisteredUserOut])
def list_users():
    with db_cursor() as cur:
        cur.execute(
            """SELECT u.*, la.checkin_at AS last_checkin_at, la.event_type AS last_event_type
               FROM registered_users u
               LEFT JOIN (
                   SELECT a1.user_id, a1.checkin_at, a1.event_type
                   FROM attendance_log a1
                   JOIN (
                       SELECT user_id, MAX(id) AS max_id
                       FROM attendance_log GROUP BY user_id
                   ) a2 ON a2.user_id = a1.user_id AND a2.max_id = a1.id
               ) la ON la.user_id = u.id
               ORDER BY u.created_at DESC"""
        )
        rows = cur.fetchall()

    result = []
    for r in rows:
        photo_url = f"/static/pending_faces/{Path(r['photo_path']).name}" if r["photo_path"] else None
        result.append(
            RegisteredUserOut(
                id=r["id"],
                membership_code=r["membership_code"],
                full_name=r["full_name"],
                phone=r["phone"],
                photo_url=photo_url,
                created_at=r["created_at"],
                last_checkin_at=r["last_checkin_at"],
                last_event_type=r["last_event_type"],
            )
        )
    return result


@router.get("/attendance", response_model=list[AttendanceEvent])
def list_attendance(limit: int = 50):
    with db_cursor() as cur:
        cur.execute(
            """SELECT a.user_id, u.full_name, a.event_type, a.checkin_at
               FROM attendance_log a
               JOIN registered_users u ON u.id = a.user_id
               ORDER BY a.checkin_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        AttendanceEvent(
            user_id=r["user_id"],
            full_name=r["full_name"],
            event_type=r["event_type"],
            checkin_at=r["checkin_at"],
        )
        for r in rows
    ]
