from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from ..database import db_cursor
from ..models import AttendanceEvent, ManualAttendanceRequest, RegisteredUserOut, UpdateUserRequest
from ..ws_manager import manager

router = APIRouter(prefix="/api", tags=["users"])

_USER_WITH_ATTENDANCE_SELECT = """
    SELECT u.*, la.checkin_at AS last_checkin_at, la.event_type AS last_event_type
    FROM registered_users u
    LEFT JOIN (
        SELECT a1.user_id, a1.checkin_at, a1.event_type
        FROM attendance_log a1
        JOIN (
            SELECT user_id, MAX(id) AS max_id
            FROM attendance_log GROUP BY user_id
        ) a2 ON a2.user_id = a1.user_id AND a2.max_id = a1.id
    ) la ON la.user_id = u.id
"""


def _row_to_user_out(r) -> RegisteredUserOut:
    photo_url = f"/static/pending_faces/{Path(r['photo_path']).name}" if r["photo_path"] else None
    return RegisteredUserOut(
        id=r["id"],
        membership_code=r["membership_code"],
        full_name=r["full_name"],
        phone=r["phone"],
        photo_url=photo_url,
        created_at=r["created_at"],
        last_checkin_at=r["last_checkin_at"],
        last_event_type=r["last_event_type"],
    )


@router.get("/users", response_model=list[RegisteredUserOut])
def list_users():
    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " ORDER BY u.created_at DESC")
        rows = cur.fetchall()
    return [_row_to_user_out(r) for r in rows]


@router.get("/attendance", response_model=list[AttendanceEvent])
def list_attendance(limit: int = 50, since_id: Optional[int] = None):
    """
    Two modes:
      - default (no since_id): most recent `limit` events, newest first —
        used by the reception panel's live feed.
      - since_id given: events with id > since_id, OLDEST first, capped at
        `limit` — the polling-fallback shape an external server should use:
        remember the highest `id` you've seen, pass it back next call, and
        you'll never miss or double-process an event even if a webhook
        delivery was missed.
    """
    with db_cursor() as cur:
        if since_id is not None:
            cur.execute(
                """SELECT a.id, a.user_id, u.full_name, u.membership_code, a.event_type, a.checkin_at
                   FROM attendance_log a
                   JOIN registered_users u ON u.id = a.user_id
                   WHERE a.id > ?
                   ORDER BY a.id ASC
                   LIMIT ?""",
                (since_id, limit),
            )
        else:
            cur.execute(
                """SELECT a.id, a.user_id, u.full_name, u.membership_code, a.event_type, a.checkin_at
                   FROM attendance_log a
                   JOIN registered_users u ON u.id = a.user_id
                   ORDER BY a.id DESC
                   LIMIT ?""",
                (limit,),
            )
        rows = cur.fetchall()
    return [
        AttendanceEvent(
            id=r["id"],
            user_id=r["user_id"],
            membership_code=r["membership_code"],
            full_name=r["full_name"],
            event_type=r["event_type"],
            checkin_at=r["checkin_at"],
        )
        for r in rows
    ]


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request):
    """
    Permanently removes a member: their registration row, their attendance
    history, and (if any) the historical pending_queue row that points at
    them — all three are deleted in one go because attendance_log.user_id and
    pending_queue.registered_user_id are foreign keys, and foreign_keys=ON is
    enabled, so the parent row can't be deleted while children still
    reference it.
    """
    with db_cursor() as cur:
        cur.execute("SELECT full_name, photo_path FROM registered_users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM attendance_log WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM pending_queue WHERE registered_user_id=?", (user_id,))
        cur.execute("DELETE FROM registered_users WHERE id=?", (user_id,))

    # best-effort cleanup of the stored photo file; a missing/locked file
    # shouldn't fail the whole delete
    if row["photo_path"]:
        try:
            Path(row["photo_path"]).unlink(missing_ok=True)
        except OSError:
            pass

    # the camera thread must stop matching this face immediately, or the next
    # sighting would try to INSERT an attendance row for a user_id that no
    # longer exists (foreign key violation)
    engine = request.app.state.face_engine
    engine.reload_known_users()

    await manager.broadcast({"event": "member_deleted", "id": user_id})

    return {"ok": True, "full_name": row["full_name"]}


@router.patch("/users/{user_id}", response_model=RegisteredUserOut)
async def update_user(user_id: int, payload: UpdateUserRequest, request: Request):
    """Edit a member's name and/or phone. Their face data is untouched."""
    with db_cursor() as cur:
        cur.execute("SELECT full_name, phone FROM registered_users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    new_name = payload.full_name.strip() if payload.full_name is not None else row["full_name"]
    new_phone = payload.phone if payload.phone is not None else row["phone"]

    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE registered_users SET full_name=?, phone=? WHERE id=?", (new_name, new_phone, user_id))

    # the camera's known-face cache keeps a copy of each member's name for
    # the live feed/webhook payloads, so it needs to see the new name too
    engine = request.app.state.face_engine
    engine.reload_known_users()

    await manager.broadcast({"event": "member_updated", "id": user_id, "full_name": new_name, "phone": new_phone})

    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " WHERE u.id = ?", (user_id,))
        updated_row = cur.fetchone()
    return _row_to_user_out(updated_row)


@router.post("/users/{user_id}/attendance", response_model=AttendanceEvent)
async def set_attendance(user_id: int, payload: ManualAttendanceRequest, request: Request):
    """
    Manually record an entry/exit for a member from the reception panel —
    for correcting a missed camera detection, or logging someone who came in
    through a side door, etc. Not throttled by the usual 5-minute cooldown
    and doesn't require alternating in/out: the operator is always right.
    """
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM registered_users WHERE id=?", (user_id,))
        if cur.fetchone() is None:
            raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    if payload.event_type is not None and payload.event_type not in ("in", "out"):
        raise HTTPException(400, "event_type باید 'in' یا 'out' باشد")

    engine = request.app.state.face_engine
    try:
        result = engine.log_manual_attendance(user_id, payload.event_type)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    return AttendanceEvent(
        id=result["event_id"],
        user_id=user_id,
        membership_code=result["membership_code"],
        full_name=result["full_name"],
        event_type=result["event_type"],
        checkin_at=result["checkin_at"],
    )
