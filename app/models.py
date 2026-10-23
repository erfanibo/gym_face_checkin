from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PendingItem(BaseModel):
    id: int
    photo_url: str
    first_seen_at: str
    last_seen_at: str


class RegisterPendingRequest(BaseModel):
    full_name: str = Field(..., min_length=2, description="نام و نام خانوادگی مشتری")
    phone: Optional[str] = Field(None, description="شماره تماس")
    plan_id: Optional[int] = Field(None, description="پلن عضویت (اختیاری) که همان لحظه برای عضو ثبت می‌شود")
    # membership_code is intentionally NOT here — it's generated automatically
    # by the server at registration time (see routers/queue.py), never typed
    # by the operator.


class ManualRegisterRequest(BaseModel):
    """
    Add a member directly from the admin panel, with no captured face --
    for when the camera is disconnected/not installed yet, or the operator
    just doesn't want to stand the person in front of it right now (see
    routers/users.py create_user_manual). Same membership_code auto-
    generation rule as RegisterPendingRequest above.
    """

    full_name: str = Field(..., min_length=2, description="نام و نام خانوادگی عضو")
    phone: Optional[str] = Field(None, description="شماره تماس")
    plan_id: Optional[int] = Field(None, description="پلن عضویت (اختیاری) که همان لحظه برای عضو ثبت می‌شود")


class CameraStatus(BaseModel):
    connected: bool = Field(..., description="آیا FaceEngine همین الان دوربین را باز و در حال خواندن فریم دارد")
    problem: Optional[str] = Field(
        None,
        description="علت وصل‌نبودن: device_missing | no_permission | open_failed | no_frames (None وقتی وصل است)",
    )
    recognizer_ready: bool = Field(
        True,
        description="آیا مدل‌های تشخیص چهره بارگذاری شده‌اند (بار اول ممکن است وزن‌ها دانلود شوند)",
    )
    recognizer_error: Optional[str] = Field(None, description="علت آماده‌نبودن موتور تشخیص چهره (None وقتی آماده است)")


class AssignPendingRequest(BaseModel):
    user_id: int = Field(..., description="شناسه‌ی عضو موجودی که این چهره در صف انتظار متعلق به اوست")


class AssignPendingResponse(BaseModel):
    ok: bool
    user_id: int
    full_name: str
    sample_count: int


class RecognitionLogEntry(BaseModel):
    id: int
    user_id: int | None
    full_name: str
    distance: float | None
    created_at: str


class RegisteredUserOut(BaseModel):
    id: int
    membership_code: Optional[str]
    full_name: str
    phone: Optional[str]
    photo_url: Optional[str]
    created_at: str
    last_checkin_at: Optional[str] = None
    last_event_type: Optional[str] = None  # 'in' | 'out' | None (هرگز تردد ثبت نشده)
    membership_expires_at: Optional[str] = None  # None = تاریخ پایانی هنوز برایش ثبت نشده (نه «منقضی»)
    # False for a member added manually with no face sample yet (see
    # ManualRegisterRequest) -- recognition stays inactive for them until an
    # operator links a captured face via the pending-queue "assign" flow.
    has_face_sample: bool = True
    # نام آخرین پلنی که برای این عضو ثبت شده (None = هرگز پلنی ثبت نشده، مثلاً فقط با «۱ ماهه»
    # تمدید شده). فقط یک نشانه‌ی اطلاعاتی است، نه وضعیت قطعی عضویت فعلی.
    last_plan_name: Optional[str] = None
    # آیا عضو برای ورود به پنل عضو رمز تعیین کرده؟ (خودِ رمز/هش هرگز در پاسخ‌ها نیست)
    has_member_password: bool = False


class ExtendMembershipRequest(BaseModel):
    # strict=True: فقط عدد صحیح JSON؛ نه رشته‌ی "30" و نه true/1.5
    days: int = Field(..., ge=1, le=3650, strict=True, description="تعداد روزِ تمدید دستی (بدون تعرفه)")


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


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class ManagerLoginRequest(BaseModel):
    password: str = Field(..., min_length=1)


class MemberLoginRequest(BaseModel):
    phone: str = Field(..., min_length=3, max_length=40, description="همان شماره‌ای که هنگام ثبت‌نام ثبت شده")
    # فقط اگر عضو خودش رمز تعیین کرده باشد لازم است
    password: Optional[str] = Field(None, max_length=128, description="رمز عبور (اگر تعیین کرده)")


class SetMemberPasswordRequest(BaseModel):
    new_password: str = Field(..., min_length=6, max_length=128, description="حداقل ۶ کاراکتر")
    current_password: Optional[str] = Field(None, max_length=128, description="فقط هنگام «تغییر» رمزِ موجود لازم است")

    @field_validator("new_password")
    @classmethod
    def _not_blank(cls, v):
        if not v.strip():
            raise ValueError("رمز نمی‌تواند فقط فاصله باشد")
        return v


class MemberProfile(BaseModel):
    full_name: str
    has_password: bool


class MemberPurchaseOut(BaseModel):
    plan_name: str
    price: int
    duration_days: int
    starts_at: str
    expires_at: str
    created_at: str


class MemberMembershipOut(BaseModel):
    """وضعیت عضویت و سابقهٔ تمدیدهای خودِ عضو (بخش «شهریه و عضویت» پنل عضو)."""
    membership_expires_at: Optional[str] = None
    status: Literal["active", "expired", "none"]  # none = تاریخ پایانی هرگز ثبت نشده
    days_left: Optional[int] = None  # فقط وقتی status == "active"
    last_plan_name: Optional[str] = None
    purchases: list[MemberPurchaseOut] = []


class MemberAttendanceEventOut(BaseModel):
    event_type: str  # 'in' | 'out'
    checkin_at: str


class MemberAttendanceOut(BaseModel):
    events: list[MemberAttendanceEventOut] = []
    visits_last_30_days: int = 0
    last_visit_at: Optional[str] = None  # آخرین «ورود»


class ChangeManagerPasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=4, description="حداقل ۴ کاراکتر")


class AuthIdentity(BaseModel):
    role: Optional[str] = None  # 'manager' | 'member' | None  (manager wins if both are logged in)
    user_id: Optional[int] = None
    manager: bool = False       # logged in as manager (independent of the member login)
    member: bool = False        # logged in as a member (independent of the manager login)


# ---------------------------------------------------------------------------
# Workout plans
# ---------------------------------------------------------------------------
class WorkoutItemIn(BaseModel):
    exercise_name: str = Field(..., min_length=1, description="نام حرکت")
    sets: int = Field(..., ge=1, description="تعداد ست")
    reps: str = Field(..., min_length=1, description="تعداد تکرار؛ متن است چون می‌تواند رِنج هم باشد، مثل «8-12»")
    notes: Optional[str] = Field(None, description="توضیح اختیاری (وزن، نکته‌ی اجرا و ...)")


class SaveWorkoutPlanRequest(BaseModel):
    # The admin panel always sends the WHOLE list, in the order it should be
    # shown — there's no per-item edit endpoint, see routers/workouts.py.
    items: list[WorkoutItemIn] = Field(default_factory=list)


class WorkoutItemOut(BaseModel):
    id: int
    position: int
    exercise_name: str
    sets: int
    reps: str
    notes: Optional[str]
    is_done: bool


class WorkoutPlanOut(BaseModel):
    user_id: int
    full_name: Optional[str] = None  # only filled in on the manager's view
    updated_at: Optional[str] = None
    items: list[WorkoutItemOut]


# ---------------------------------------------------------------------------
# تنظیمات پذیرش > رفتار ثبت تردد (routers/settings.py)
# ---------------------------------------------------------------------------
class BehaviorSettings(BaseModel):
    attendance_cooldown_minutes: int = Field(
        ..., ge=1, le=120, description="فاصله‌ی ثبت ورود/خروج یکسان برای یک عضو، به دقیقه"
    )
    recognition_log_debounce_seconds: int = Field(
        ..., ge=1, le=60, description="حداقل فاصله بین دو خط لاگ زنده برای همان عضو، به ثانیه"
    )
    pending_dedup_window_minutes: int = Field(
        ..., ge=1, le=60, description="پنجره‌ی آنتی‌اسپم صف انتظار، به دقیقه"
    )
    max_face_samples_per_member: int = Field(
        ..., ge=1, le=100, description="حداکثر نمونه‌ی چهره به ازای هر عضو"
    )


# ---------------------------------------------------------------------------
# تنظیمات > اطلاعات باشگاه (routers/settings.py)
# ---------------------------------------------------------------------------
class ClubInfo(BaseModel):
    # strip قبل از اعتبارسنجی: «  ا  » (فقط فاصله) نباید به‌عنوان نام معتبر ذخیره شود.
    club_name: str = Field(..., min_length=2, max_length=80, description="نام باشگاه")
    phone: Optional[str] = Field(None, description="شماره تماس باشگاه")
    address: Optional[str] = Field(None, description="آدرس باشگاه")

    @field_validator("club_name", mode="before")
    @classmethod
    def _strip_club_name(cls, v):
        return v.strip() if isinstance(v, str) else v


class ClubPublic(BaseModel):
    """فقط چیزی که قبل از لاگین هم نمایش داده می‌شود (GET /api/public/club)."""
    club_name: str


# ---------------------------------------------------------------------------
# تنظیمات > به‌روزرسانی سیستم (routers/system.py)
# ---------------------------------------------------------------------------
class VersionInfo(BaseModel):
    version: str
    update_check_enabled: bool


class UpdateCheckResult(BaseModel):
    # همیشه با HTTP 200 برمی‌گردد؛ نتیجه‌ی بررسی (حتی «اینترنت نیست») یک status است
    # نه خطای سرور، تا فرانت همه‌ی حالت‌ها را یکدست نشان بدهد.
    status: Literal[
        "up_to_date", "update_available", "unknown",
        "not_found", "rate_limited", "unreachable", "disabled",
    ]
    current_version: str
    latest_version: Optional[str] = None
    release_name: Optional[str] = None
    release_notes: Optional[str] = None
    release_url: Optional[str] = None
    published_at: Optional[str] = None


# ---------------------------------------------------------------------------
# مالی و شهریه > تعرفه‌ها و پلن‌ها (routers/plans.py)
# ---------------------------------------------------------------------------
class PlanCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=60, description="نام تعرفه، مثلاً «ماهانه»")
    # strict=True: فقط عدد صحیح JSON؛ نه رشته‌ی "100" و نه true/1.5
    price: int = Field(..., ge=0, le=100_000_000_000, strict=True, description="قیمت به تومان")
    duration_days: int = Field(..., ge=1, le=3650, strict=True, description="مدت اعتبار به روز")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_name(cls, v):
        return v.strip() if isinstance(v, str) else v


class PlanOut(BaseModel):
    id: int
    name: str
    price: int
    duration_days: int
    active_members: int = Field(0, description="تعداد اعضایی که آخرین پلنشان همین است و عضویتشان هنوز منقضی نشده")
    created_at: str


class AssignPlanRequest(BaseModel):
    plan_id: int
