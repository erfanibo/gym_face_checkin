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


class RegisteredUserOut(BaseModel):
    id: int
    membership_code: Optional[str]
    full_name: str
    phone: Optional[str]
    photo_url: Optional[str]
    created_at: str
    last_checkin_at: Optional[str] = None
    last_event_type: Optional[str] = None  # 'in' | 'out' | None (هرگز تردد ثبت نشده)


class AttendanceEvent(BaseModel):
    id: int
    user_id: int
    membership_code: Optional[str] = None
    full_name: str
    event_type: str
    checkin_at: str
