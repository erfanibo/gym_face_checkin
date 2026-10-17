"""
The logged-in MEMBER's own account (member panel): who am I, and set/change my password.
Every route takes the user id from the signed session (auth.require_member), never from the request.
"""
import math
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query

from .. import auth, ratelimit
from ..database import db_cursor
from ..models import (
    MemberAttendanceEventOut,
    MemberAttendanceOut,
    MemberMembershipOut,
    MemberProfile,
    MemberPurchaseOut,
    SetMemberPasswordRequest,
)
from ..phone import phone_key
from ..ws_manager import manager

router = APIRouter(prefix="/api/me", tags=["member-account"])


def _load_member(user_id: int):
    with db_cursor() as cur:
        cur.execute("SELECT full_name, phone, member_password_hash FROM registered_users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(401, "این عضویت دیگر وجود ندارد؛ لطفاً دوباره وارد شوید.")
    return row


@router.get("/profile", response_model=MemberProfile)
def my_profile(user_id: int = Depends(auth.require_member)):
    row = _load_member(user_id)
    return MemberProfile(full_name=row["full_name"], has_password=row["member_password_hash"] is not None)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@router.get("/membership", response_model=MemberMembershipOut)
def my_membership(user_id: int = Depends(auth.require_member)):
    """Own membership status + own purchase history (newest first, last 50)."""
    with db_cursor() as cur:
        cur.execute("SELECT membership_expires_at FROM registered_users WHERE id=?", (user_id,))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(401, "این عضویت دیگر وجود ندارد؛ لطفاً دوباره وارد شوید.")
        cur.execute(
            """SELECT plan_id, plan_name, price, duration_days, starts_at, expires_at, created_at
               FROM membership_purchases WHERE user_id=? ORDER BY id DESC LIMIT 50""",
            (user_id,),
        )
        purchases = cur.fetchall()

    expiry = _parse_utc(row["membership_expires_at"])
    now = datetime.now(timezone.utc)
    if expiry is None:
        status, days_left = "none", None
    elif expiry > now:
        status, days_left = "active", max(1, math.ceil((expiry - now).total_seconds() / 86400))
    else:
        status, days_left = "expired", None

    return MemberMembershipOut(
        membership_expires_at=row["membership_expires_at"],
        status=status,
        days_left=days_left,
        # تمدید دستی (بدون تعرفه) «آخرین تعرفه» حساب نمی‌شود
        last_plan_name=next((p["plan_name"] for p in purchases if p["plan_id"] is not None), None),
        purchases=[
            MemberPurchaseOut(
                plan_name=p["plan_name"], price=p["price"], duration_days=p["duration_days"],
                starts_at=p["starts_at"], expires_at=p["expires_at"], created_at=p["created_at"],
            )
            for p in purchases
        ],
    )


@router.get("/attendance", response_model=MemberAttendanceOut)
def my_attendance(limit: int = Query(30, ge=1, le=100), user_id: int = Depends(auth.require_member)):
    """Own recent entries/exits (newest first) + number of visits ('in' events) in the last 30 days."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT event_type, checkin_at FROM attendance_log WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        events = cur.fetchall()
        cur.execute(
            "SELECT checkin_at FROM attendance_log WHERE user_id=? AND event_type='in' ORDER BY id DESC LIMIT 500",
            (user_id,),
        )
        entries = cur.fetchall()

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    visits = 0
    for e in entries:
        t = _parse_utc(e["checkin_at"])
        if t is not None and t >= cutoff:
            visits += 1
    return MemberAttendanceOut(
        events=[MemberAttendanceEventOut(event_type=e["event_type"], checkin_at=e["checkin_at"]) for e in events],
        visits_last_30_days=visits,
        last_visit_at=entries[0]["checkin_at"] if entries else None,
    )


@router.post("/password")
async def set_my_password(payload: SetMemberPasswordRequest, user_id: int = Depends(auth.require_member)):
    """First password: just the new one. Changing an existing one also needs the current one."""
    row = _load_member(user_id)
    stored = row["member_password_hash"]

    if stored is not None:
        limiter_key = f"uid:{user_id}"
        if ratelimit.member_password_change.blocked(limiter_key):
            raise HTTPException(429, ratelimit.TOO_MANY_MESSAGE)
        if not payload.current_password or not auth.verify_password(payload.current_password, stored):
            ratelimit.member_password_change.record(limiter_key)
            raise HTTPException(401, "رمز عبور فعلی درست نیست.")
        ratelimit.member_password_change.clear(limiter_key)

    # the phone number is what an outsider is most likely to guess
    if phone_key(payload.new_password) is not None and phone_key(payload.new_password) == phone_key(row["phone"]):
        raise HTTPException(400, "رمز عبور نباید با شماره تماس یکسان باشد.")

    if not auth.set_member_password(user_id, payload.new_password):
        raise HTTPException(401, "این عضویت دیگر وجود ندارد؛ لطفاً دوباره وارد شوید.")
    # the manager's member list shows who has a password (and offers "remove password")
    await manager.broadcast({"event": "member_password_changed", "id": user_id, "has_member_password": True})
    return {"ok": True, "has_password": True}
