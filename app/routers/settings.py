"""
تنظیمات پذیرش > رفتار ثبت تردد -- اولین زیربخشِ "تنظیمات پذیرش" که از
placeholder به بک‌اند واقعی وصل شد (نگاه کن به UPDATE.md).

این چهار مقدار قبلاً فقط در app/config.py به‌صورت ثابت (常量) تعریف شده بودند؛
از این مسیر هم قابل خواندن و هم از پنل مدیر قابل تغییرند. مقدار جدید همزمان
دو جا ذخیره می‌شود:
  ۱. در app_settings (دیتابیس) تا بعد از ری‌استارت سرویس هم باقی بماند.
  ۲. مستقیم روی خودِ ماژول config (`config.ATTENDANCE_COOLDOWN_SECONDS = ...`)
     تا همون لحظه، بدون نیاز به ری‌استارت، روی FaceEngine اثر بگذارد -- چون
     face_engine.py همیشه با `config.XXX` (نه یک مقدار کپی‌شده‌ی زمان import)
     به این متغیرها ارجاع می‌دهد (چک شده: چهار خط مربوطه در face_engine.py).

نکته‌ی مهم: صفحه‌ی «دوربین و دستگاه‌ها» عمداً اینجا نیست. تغییر CAMERA_INDEX
نیاز به ری‌استارت ترد دوربین (cv2.VideoCapture) دارد و بدون دسترسی به یک
دوربین واقعی برای تست، ریسکش بیشتر از فایده‌شه -- همونطور که در UPDATE.md
یادداشت شده، اون یکی گام بعدی جداست.
"""
from fastapi import APIRouter, Depends

from .. import auth, config
from ..database import get_setting, set_setting
from ..models import BehaviorSettings, ClubInfo
from ..ws_manager import manager

router = APIRouter(prefix="/api/settings", tags=["settings"], dependencies=[Depends(auth.require_manager)])

# کلید app_settings برای هرکدوم -- همیشه در همون واحدی که config.py استفاده
# می‌کنه ذخیره می‌شه (ثانیه برای سه‌تای اول، یک عدد ساده برای آخری)؛ تبدیل
# دقیقه/ثانیه فقط در مرز API (BehaviorSettings <-> config) اتفاق می‌افتد، نه
# در storage -- که یک‌جا تبدیل واحد اشتباه کل زنجیره رو خراب نکنه.
_ATTENDANCE_COOLDOWN_KEY = "behavior_attendance_cooldown_seconds"
_RECOGNITION_DEBOUNCE_KEY = "behavior_recognition_log_debounce_seconds"
_PENDING_DEDUP_WINDOW_KEY = "behavior_pending_dedup_window_seconds"
_MAX_FACE_SAMPLES_KEY = "behavior_max_face_samples_per_member"


def apply_persisted_overrides() -> None:
    """فقط یک‌بار، موقع استارت سرور صدا زده می‌شود (main.py's lifespan)،
    حتماً قبل از `engine.start()` -- تا اگر قبلاً مقداری از پنل ذخیره شده،
    FaceEngine از همون اول با مقدار درست کار کنه، نه دیفالت config.py."""
    cooldown = get_setting(_ATTENDANCE_COOLDOWN_KEY)
    if cooldown is not None:
        config.ATTENDANCE_COOLDOWN_SECONDS = int(cooldown)

    debounce = get_setting(_RECOGNITION_DEBOUNCE_KEY)
    if debounce is not None:
        config.RECOGNITION_LOG_DEBOUNCE_SECONDS = int(debounce)

    dedup_window = get_setting(_PENDING_DEDUP_WINDOW_KEY)
    if dedup_window is not None:
        config.PENDING_DEDUP_WINDOW_SECONDS = int(dedup_window)

    max_samples = get_setting(_MAX_FACE_SAMPLES_KEY)
    if max_samples is not None:
        config.MAX_FACE_SAMPLES_PER_MEMBER = int(max_samples)


def _current_settings() -> BehaviorSettings:
    return BehaviorSettings(
        attendance_cooldown_minutes=round(config.ATTENDANCE_COOLDOWN_SECONDS / 60),
        recognition_log_debounce_seconds=config.RECOGNITION_LOG_DEBOUNCE_SECONDS,
        pending_dedup_window_minutes=round(config.PENDING_DEDUP_WINDOW_SECONDS / 60),
        max_face_samples_per_member=config.MAX_FACE_SAMPLES_PER_MEMBER,
    )


@router.get("/behavior", response_model=BehaviorSettings)
def get_behavior_settings():
    return _current_settings()


@router.put("/behavior", response_model=BehaviorSettings)
def update_behavior_settings(body: BehaviorSettings):
    config.ATTENDANCE_COOLDOWN_SECONDS = body.attendance_cooldown_minutes * 60
    config.RECOGNITION_LOG_DEBOUNCE_SECONDS = body.recognition_log_debounce_seconds
    config.PENDING_DEDUP_WINDOW_SECONDS = body.pending_dedup_window_minutes * 60
    config.MAX_FACE_SAMPLES_PER_MEMBER = body.max_face_samples_per_member

    set_setting(_ATTENDANCE_COOLDOWN_KEY, str(config.ATTENDANCE_COOLDOWN_SECONDS))
    set_setting(_RECOGNITION_DEBOUNCE_KEY, str(config.RECOGNITION_LOG_DEBOUNCE_SECONDS))
    set_setting(_PENDING_DEDUP_WINDOW_KEY, str(config.PENDING_DEDUP_WINDOW_SECONDS))
    set_setting(_MAX_FACE_SAMPLES_KEY, str(config.MAX_FACE_SAMPLES_PER_MEMBER))

    return _current_settings()


# ---------------------------------------------------------------------------
# تنظیمات > اطلاعات باشگاه.
# نام باشگاه «منبع واقعیِ» برندینگ کل سیستم است (پروژه قرار است برای هر باشگاهی
# قابل استفاده باشد، نه فقط یک باشگاه): عنوان تب مرورگر، صفحه‌ی ورود و نوار کناریِ
# پنل مدیر، و صفحه‌ی ورود/هدر پنل عضو همه از همین مقدار می‌خوانند. چون صفحه‌های
# ورود قبل از لاگین رندر می‌شوند و این مسیر (/api/settings/...) پشت require_manager
# است، خودِ نام از طریق یک مسیر عمومیِ فقط-خواندنی جدا (routers/public.py ->
# GET /api/public/club) هم در دسترس است؛ تلفن و آدرس عمداً فقط پشت لاگین می‌مانند.
# ---------------------------------------------------------------------------
_CLUB_NAME_KEY = "club_info_name"
_CLUB_PHONE_KEY = "club_info_phone"
_CLUB_ADDRESS_KEY = "club_info_address"

# نام پیش‌فرضِ خنثی تا وقتی مدیر اولین بار نام واقعی را از «تنظیمات > اطلاعات
# باشگاه» وارد کند. عمداً اسم هیچ باشگاه خاصی نیست.
DEFAULT_CLUB_NAME = "باشگاه من"


def get_club_name() -> str:
    """نام فعلی باشگاه (یا پیش‌فرض خنثی). هم برای پنل مدیر و هم مسیر عمومی."""
    return (get_setting(_CLUB_NAME_KEY) or "").strip() or DEFAULT_CLUB_NAME


def _current_club_info() -> ClubInfo:
    # "" و None یکی‌ان از نظر معنایی (یعنی هنوز چیزی ثبت نشده) -- ذخیره‌شده
    # به‌صورت "" است (چون app_settings.value ستونش NOT NULL)، اینجا برای
    # خروجی API دوباره به None تبدیل می‌شود تا فرانت یکدست رفتار کند.
    phone = get_setting(_CLUB_PHONE_KEY) or None
    address = get_setting(_CLUB_ADDRESS_KEY) or None
    return ClubInfo(
        club_name=get_club_name(),
        phone=phone,
        address=address,
    )


@router.get("/club-info", response_model=ClubInfo)
def get_club_info():
    return _current_club_info()


@router.put("/club-info", response_model=ClubInfo)
async def update_club_info(body: ClubInfo):
    set_setting(_CLUB_NAME_KEY, body.club_name)
    set_setting(_CLUB_PHONE_KEY, body.phone or "")
    set_setting(_CLUB_ADDRESS_KEY, body.address or "")
    await manager.broadcast({"event": "club_updated"})
    return _current_club_info()
