from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from .. import auth
from ..database import db_cursor, generate_unique_membership_code
from ..membership import apply_manual_days, apply_plan, get_active_plan
from ..models import (
    AssignPlanRequest,
    AttendanceEvent,
    ExtendMembershipRequest,
    ManualAttendanceRequest,
    ManualRegisterRequest,
    RegisteredUserOut,
    UpdateUserRequest,
)
from ..ws_manager import manager

# Every endpoint here is manager-only (member data management) -- the member
# panel gets its own separate, read-only "my own data" endpoints later.
router = APIRouter(prefix="/api", tags=["users"], dependencies=[Depends(auth.require_manager)])

_USER_WITH_ATTENDANCE_SELECT = """
    SELECT u.*, la.checkin_at AS last_checkin_at, la.event_type AS last_event_type,
           EXISTS(SELECT 1 FROM member_face_samples fs WHERE fs.user_id = u.id) AS has_face_sample,
           (SELECT pu.plan_name FROM membership_purchases pu WHERE pu.user_id = u.id AND pu.plan_id IS NOT NULL ORDER BY pu.id DESC LIMIT 1) AS last_plan_name
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
        membership_expires_at=r["membership_expires_at"],
        has_face_sample=bool(r["has_face_sample"]),
        last_plan_name=r["last_plan_name"],
    )


@router.get("/users", response_model=list[RegisteredUserOut])
def list_users():
    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " ORDER BY u.created_at DESC")
        rows = cur.fetchall()
    return [_row_to_user_out(r) for r in rows]


@router.post("/users", response_model=RegisteredUserOut)
async def create_user_manual(payload: ManualRegisterRequest, request: Request):
    """
    Adds a brand-new member directly from the "اعضا > افزودن عضو" page,
    with NO captured face -- for data entry when the camera is
    disconnected/not installed yet, or the operator simply doesn't want to
    stand the person in front of it right now. Unlike /api/queue/{id}/register
    (the normal camera-driven flow), there's no pending-queue row and no
    face to dedupe against, so this never raises a "already registered"
    conflict -- the operator is trusted here, same as manual attendance.

    The member is created with encoding=NULL and zero member_face_samples
    rows, so FaceEngine.reload_known_users() simply never sees them --
    recognition for this member stays off until an operator later sees them
    in the pending queue and uses "این چهره متعلق به کدام عضو است" to link a
    captured face to this same row (routers/queue.py assign_pending_to_member).
    That existing flow is the only thing needed to "finish" this member's
    registration once a camera is available -- no new endpoint for it.
    """
    full_name = payload.full_name.strip()
    phone = payload.phone.strip() if payload.phone else None
    now = datetime.now(timezone.utc).isoformat()
    membership_code = generate_unique_membership_code()

    # اگر پلن خواسته شده، قبل از ساختن عضو چک می‌شود تا یک پلن نامعتبر عضو نیمه‌کاره نسازد.
    plan = None
    if payload.plan_id is not None:
        with db_cursor() as cur:
            plan = get_active_plan(cur, payload.plan_id)
        if plan is None:
            raise HTTPException(400, "تعرفه‌ی انتخاب‌شده پیدا نشد یا حذف شده است.")

    with db_cursor(commit=True) as cur:
        cur.execute(
            """INSERT INTO registered_users
                   (membership_code, full_name, phone, encoding, photo_path, created_at)
               VALUES (?, ?, ?, NULL, NULL, ?)""",
            (membership_code, full_name, phone, now),
        )
        new_user_id = cur.lastrowid
        if plan is not None:
            apply_plan(cur, new_user_id, plan)  # همان تراکنش: یا عضو+پلن هر دو، یا هیچ‌کدام

    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " WHERE u.id = ?", (new_user_id,))
        row = cur.fetchone()
    result = _row_to_user_out(row)

    await manager.broadcast({"event": "member_registered", "member": result.model_dump()})

    return result


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
    history, all of their stored face samples (member_face_samples), all of
    their recognition-log rows, their workout plan (workout_items), their
    membership purchase history (membership_purchases), and (if
    any) the historical pending_queue row that points at them — all deleted
    in one go because attendance_log.user_id, member_face_samples.user_id,
    recognition_log.user_id, workout_items.user_id, and
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
        cur.execute("DELETE FROM member_face_samples WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM recognition_log WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM workout_items WHERE user_id=?", (user_id,))
        cur.execute("DELETE FROM membership_purchases WHERE user_id=?", (user_id,))
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


@router.post("/users/{user_id}/membership/extend", response_model=RegisteredUserOut)
async def extend_membership(user_id: int, payload: ExtendMembershipRequest, request: Request):
    """
    Manual extension WITHOUT a plan (gift, compensation for downtime, correcting a mistake):
    adds `days` to the expiry -- on top of the current expiry when it is still in the future,
    otherwise from now. It is recorded in the purchase history as "تمدید دستی" with price 0 so
    the history stays complete and financial reports are not distorted. Normal renewals go
    through a plan (POST .../membership/plan).
    """
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM registered_users WHERE id=?", (user_id,))
        if cur.fetchone() is None:
            raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    with db_cursor(commit=True) as cur:
        new_expiry_iso = apply_manual_days(cur, user_id, payload.days)

    await manager.broadcast({"event": "membership_updated", "id": user_id, "membership_expires_at": new_expiry_iso})

    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " WHERE u.id = ?", (user_id,))
        updated_row = cur.fetchone()
    return _row_to_user_out(updated_row)


@router.post("/users/{user_id}/membership/plan", response_model=RegisteredUserOut)
async def assign_plan(user_id: int, payload: AssignPlanRequest, request: Request):
    """
    Registers one of the manager-defined plans (مالی و شهریه > تعرفه‌ها) for a member:
    the plan's number of days is added to their expiry (on top of the current expiry when
    it's still in the future, otherwise from now -- same rule as the manual extension) and a purchase snapshot is recorded (membership_purchases).
    """
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM registered_users WHERE id=?", (user_id,))
        user_exists = cur.fetchone() is not None
        plan = get_active_plan(cur, payload.plan_id)
    if not user_exists:
        raise HTTPException(404, "کاربری با این شناسه پیدا نشد")
    if plan is None:
        raise HTTPException(404, "تعرفه‌ی انتخاب‌شده پیدا نشد یا حذف شده است.")

    with db_cursor(commit=True) as cur:
        new_expiry_iso = apply_plan(cur, user_id, plan)

    await manager.broadcast({
        "event": "membership_updated", "id": user_id,
        "membership_expires_at": new_expiry_iso, "last_plan_name": plan["name"],
    })

    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " WHERE u.id = ?", (user_id,))
        updated_row = cur.fetchone()
    return _row_to_user_out(updated_row)


@router.patch("/users/{user_id}/membership/end", response_model=RegisteredUserOut)
async def end_membership(user_id: int, request: Request):
    """
    Manually ends a membership immediately (sets the expiry to right now),
    for example when someone cancels. This does NOT block their check-in --
    reception still only warns on an expired membership, never refuses it
    (the manager decided that's the desired behavior) -- and it's fully
    reversible by renewing again later.
    """
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM registered_users WHERE id=?", (user_id,))
        if cur.fetchone() is None:
            raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    now_iso = datetime.now(timezone.utc).isoformat()
    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE registered_users SET membership_expires_at=? WHERE id=?", (now_iso, user_id))

    await manager.broadcast({"event": "membership_updated", "id": user_id, "membership_expires_at": now_iso})

    with db_cursor() as cur:
        cur.execute(_USER_WITH_ATTENDANCE_SELECT + " WHERE u.id = ?", (user_id,))
        updated_row = cur.fetchone()
    return _row_to_user_out(updated_row)
