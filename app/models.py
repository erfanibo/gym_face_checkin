from typing import Optional
from pydantic import BaseModel, Field


class PendingItem(BaseModel):
    id: int
    photo_url: str
    first_seen_at: str
    last_seen_at: str


class RegisterPendingRequest(BaseModel):
    full_name: str = Field(..., min_length=2, description="نام و نام خانوادگی مشتری")
    phone: Optional[str] = Field(None, description="شماره تماس")
    # membership_code is intentionally NOT here — it's generated automatically
    # by the server at registration time (see routers/queue.py), never typed
    # by the operator.


class AssignPendingRequest(BaseModel):
    user_id: int = Field(..., description="شناسه‌ی عضو موجودی که این چهره در صف انتظار متعلق به اوست")


class AssignPendingResponse(BaseModel):
    ok: bool
    user_id: int
    full_name: str
    sample_count: int


class RegisteredUserOut(BaseModel):
    id: int
    membership_code: Optional[str]
    full_name: str
    phone: Optional[str]
    photo_url: Optional[str]
    created_at: str
    last_checkin_at: Optional[str] = None
    last_event_type: Optional[str] = None  # 'in' | 'out' | None (هرگز تردد ثبت نشده)


class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, description="نام جدید (اختیاری)")
    phone: Optional[str] = Field(None, description="شماره تماس جدید (اختیاری)")


class ManualAttendanceRequest(BaseModel):
    event_type: Optional[str] = Field(
        None, description="'in' یا 'out'؛ اگر خالی باشد، از وضعیت آخر toggle می‌شود"
    )


class AttendanceEvent(BaseModel):
    id: int
    user_id: int
    membership_code: Optional[str] = None
    full_name: str
    event_type: str
    checkin_at: str
