from fastapi import APIRouter, Depends, HTTPException, Request

from .. import auth, ratelimit
from ..phone import phone_key
from ..models import (
    AuthIdentity,
    ChangeManagerPasswordRequest,
    ManagerLoginRequest,
    MemberLoginRequest,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/manager/login", response_model=AuthIdentity)
def manager_login(payload: ManagerLoginRequest, request: Request):
    if not auth.verify_manager_password(payload.password):
        raise HTTPException(401, "رمز عبور اشتباه است")
    request.session.clear()
    request.session["role"] = "manager"
    return AuthIdentity(role="manager")


@router.post("/member/login", response_model=AuthIdentity)
def member_login(payload: MemberLoginRequest, request: Request):
    """Phone number (+ the member's own password, only if they set one)."""
    bad_login = HTTPException(401, "شماره تماس یا رمز عبور درست نیست.")
    key = phone_key(payload.phone)
    ip = request.client.host if request.client else "unknown"

    if ratelimit.member_login_by_ip.blocked(ip) or (key and ratelimit.member_login_by_phone.blocked(key)):
        raise HTTPException(429, ratelimit.TOO_MANY_MESSAGE)

    try:
        row = auth.login_member(key, payload.password) if key else None
    except auth.AmbiguousPhone:
        raise HTTPException(409, "این شماره برای بیش از یک عضو ثبت شده است؛ لطفاً به پذیرش مراجعه کنید.")

    if row is None:
        ratelimit.member_login_by_ip.record(ip)
        if key:
            ratelimit.member_login_by_phone.record(key)
        raise bad_login

    ratelimit.member_login_by_phone.clear(key)
    request.session.clear()
    request.session["role"] = "member"
    request.session["user_id"] = row["id"]
    return AuthIdentity(role="member", user_id=row["id"])


@router.post("/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/me", response_model=AuthIdentity)
def me(request: Request):
    return AuthIdentity(**auth.current_identity(request))


@router.post("/manager/change-password", dependencies=[Depends(auth.require_manager)])
def change_manager_password(payload: ChangeManagerPasswordRequest):
    if not auth.verify_manager_password(payload.current_password):
        raise HTTPException(401, "رمز فعلی اشتباه است")
    auth.set_manager_password(payload.new_password)
    return {"ok": True}
