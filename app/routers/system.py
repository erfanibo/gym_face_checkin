"""
تنظیمات > به‌روزرسانی سیستم -- نمایش نسخه‌ی نصب‌شده + بررسی وجود نسخه‌ی جدیدتر.

عمداً فقط «بررسی» است، نه «نصب»: این برنامه داخل یک کانتینر اجرا می‌شود که نه git
دارد و نه به داکرِ سرور دسترسی؛ نصب خودکار از داخل پنل یا نیاز به دادن docker.sock
(= دسترسی root به سرور باشگاه) دارد یا ریسک خراب‌شدن وسط کار بدون راه برگشت. پس
اینجا فقط آخرین Release منتشرشده‌ی GitHub پرسیده و با نسخه‌ی فعلی مقایسه می‌شود؛
خودِ آپدیت همچنان دستی است (دستورها در README و در همین صفحه‌ی پنل).

درخواست به GitHub فقط وقتی مدیر دکمه را می‌زند ارسال می‌شود -- هیچ تماس پس‌زمینه‌ای
وجود ندارد، پس ویژگی «کاملاً لوکال» سیستم دست‌نخورده می‌ماند.
"""
import re

import httpx
from fastapi import APIRouter, Depends

from .. import auth, config
from ..models import UpdateCheckResult, VersionInfo
from ..version import __version__

router = APIRouter(prefix="/api/system", tags=["system"], dependencies=[Depends(auth.require_manager)])

# owner: حروف/عدد/خط‌تیره (در GitHub نقطه ندارد)؛ repo: حروف/عدد/_ . - ولی نه «.» یا «..»
# (وگرنه «../x» از مسیر /repos/ بیرون می‌زد).
_REPO_RE = re.compile(r"^[A-Za-z0-9-]+/(?!\.{1,2}$)[A-Za-z0-9_.-]+$")
# 1 تا 4 بخش عددی، با «v» اختیاری و پسوند اختیاری pre-release/build (-beta.1 / +abc).
# هر چیز دیگری (مثلاً «nightly» یا «1.0.0abc») عمداً نامعتبر است.
_VERSION_RE = re.compile(r"^[vV]?(\d+(?:\.\d+){0,3})(?:[-+].*)?$")
_NOTES_MAX_CHARS = 4000
_SHORT_MAX_CHARS = 100


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """'v1.2.10' -> (1, 2, 10). None اگر قابل‌فهم نباشد."""
    m = _VERSION_RE.match((text or "").strip())
    return tuple(int(part) for part in m.group(1).split(".")) if m else None


def _is_newer(latest: tuple[int, ...], current: tuple[int, ...]) -> bool:
    # طول‌ها را با صفر برابر می‌کنیم تا (1, 0) با (1, 0, 0) برابر حساب شود.
    n = max(len(latest), len(current))
    pad = lambda t: t + (0,) * (n - len(t))
    return pad(latest) > pad(current)


def _clip(value, limit: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value if len(value) <= limit else value[:limit].rstrip() + "…"


@router.get("/version", response_model=VersionInfo)
def get_version():
    repo = config.UPDATE_CHECK_REPO
    return VersionInfo(version=__version__, update_check_enabled=bool(repo and _REPO_RE.match(repo)))


@router.post("/check-update", response_model=UpdateCheckResult)
async def check_update():
    current = __version__
    repo = config.UPDATE_CHECK_REPO
    if not repo or not _REPO_RE.match(repo):
        return UpdateCheckResult(status="disabled", current_version=current)

    url = f"{config.UPDATE_CHECK_API_BASE}/repos/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"gym-face-checkin/{current}",
    }
    try:
        async with httpx.AsyncClient(timeout=config.UPDATE_CHECK_TIMEOUT_SECONDS, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers)
    except httpx.HTTPError:
        return UpdateCheckResult(status="unreachable", current_version=current)

    if resp.status_code == 404:
        # یا هنوز هیچ Release منتشر نشده، یا ریپو private است -- از بیرون قابل تشخیص نیست.
        return UpdateCheckResult(status="not_found", current_version=current)
    if resp.status_code in (403, 429):
        return UpdateCheckResult(status="rate_limited", current_version=current)
    if resp.status_code != 200:
        return UpdateCheckResult(status="unreachable", current_version=current)

    try:
        data = resp.json()
    except ValueError:
        data = None
    tag = _clip(data.get("tag_name"), _SHORT_MAX_CHARS) if isinstance(data, dict) else None
    if not tag:
        return UpdateCheckResult(status="unreachable", current_version=current)

    # لینک فقط وقتی نشان داده می‌شود که واقعاً به همین ریپو در github.com اشاره کند؛
    # محتوای پاسخ داده‌ی بیرونی است و نباید بتواند یک لینک دلخواه (مثلاً javascript:) تزریق کند.
    html_url = data.get("html_url")
    release_url = (
        html_url if isinstance(html_url, str)
        and html_url.lower().startswith(f"https://github.com/{repo.lower()}/")
        else None
    )

    latest_parsed, current_parsed = parse_version(tag), parse_version(current)
    if latest_parsed is None or current_parsed is None:
        status = "unknown"
    elif _is_newer(latest_parsed, current_parsed):
        status = "update_available"
    else:
        status = "up_to_date"

    return UpdateCheckResult(
        status=status,
        current_version=current,
        latest_version=tag,
        release_name=_clip(data.get("name"), _SHORT_MAX_CHARS),
        release_notes=_clip(data.get("body"), _NOTES_MAX_CHARS),
        release_url=release_url,
        published_at=_clip(data.get("published_at"), 40),
    )
