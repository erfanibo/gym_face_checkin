"""
The logged-in MEMBER's own account (member panel): who am I, and set/change my password.
Every route takes the user id from the signed session (auth.require_member), never from the request.
"""
from fastapi import APIRouter, Depends, HTTPException

from .. import auth, ratelimit
from ..database import db_cursor
from ..models import MemberProfile, SetMemberPasswordRequest
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
