"""
مسیرهای عمومی (بدون لاگین) -- فقط-خواندنی و عمداً حداقلی.

تنها کاربردش این است که صفحه‌های ورود (پنل مدیر و پنل عضو) که قبل از احراز هویت
رندر می‌شوند بتوانند نام باشگاه را نشان دهند. تلفن/آدرس باشگاه و هر چیز دیگری
عمداً اینجا نیست؛ آن‌ها همچنان فقط پشت `/api/settings/club-info` (مدیر) هستند.
"""
from fastapi import APIRouter, Response

from ..models import ClubPublic
from .settings import get_club_name

router = APIRouter(prefix="/api/public", tags=["public"])


@router.get("/club", response_model=ClubPublic)
def get_public_club(response: Response):
    # no-store: بعد از تغییر نام از تنظیمات، رفرش بعدی هر دو پنل باید نام جدید را
    # ببیند، نه یک نسخه‌ی کش‌شده‌ی مرورگر/پراکسی.
    response.headers["Cache-Control"] = "no-store"
    return ClubPublic(club_name=get_club_name())
