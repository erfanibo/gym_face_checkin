from fastapi import APIRouter, Depends, HTTPException, Request

from .. import auth
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
    row = auth.find_member_for_login(payload.phone, payload.membership_code)
    if row is None:
        raise HTTPException(401, "شماره تلفن یا کد عضویت اشتباه است")
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
