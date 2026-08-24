from pathlib import Path

from fastapi import APIRouter

from ..database import db_cursor
from ..models import AttendanceEvent, RegisteredUserOut

router = APIRouter(prefix="/api", tags=["users"])


@router.get("/users", response_model=list[RegisteredUserOut])
def list_users():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM registered_users ORDER BY created_at DESC")
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
            )
        )
    return result


@router.get("/attendance", response_model=list[AttendanceEvent])
def list_attendance(limit: int = 50):
    with db_cursor() as cur:
        cur.execute(
            """SELECT a.user_id, u.full_name, a.checkin_at
               FROM attendance_log a
               JOIN registered_users u ON u.id = a.user_id
               ORDER BY a.checkin_at DESC
               LIMIT ?""",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        AttendanceEvent(user_id=r["user_id"], full_name=r["full_name"], checkin_at=r["checkin_at"])
        for r in rows
    ]
