"""
Settings > بازنشانی اطلاعات (reset / delete data).

  GET  /api/reset/preview  what can be reset and how much of it exists right now
  POST /api/reset          delete the selected categories      (must send confirm = "حذف")
  POST /api/reset/all      delete everything                   (confirm = "حذف همه" + the manager's password)

Irreversible by design (the backup page is where to save a copy first), so the server re-checks the
confirmation itself -- the panel's typed-word dialog is a convenience, not the safeguard. The
manager password is never deleted, so the manager cannot lock themselves out.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .. import auth, data_reset
from ..ws_manager import manager
from . import settings as settings_router

router = APIRouter(prefix="/api/reset", tags=["reset"], dependencies=[Depends(auth.require_manager)])

CONFIRM_SELECTED = "حذف"
CONFIRM_ALL = "حذف همه"


def _normalise(text: str) -> str:
    """Typed confirmation, forgiving about invisible characters and extra spaces."""
    cleaned = "".join(ch for ch in text if ch not in "‌‍‎‏‪‫‬")
    return " ".join(cleaned.split())


class ResetSelectedRequest(BaseModel):
    categories: list[str] = Field(..., description="شناسه‌ی موارد انتخاب‌شده (از GET /api/reset/preview)")
    confirm: str = Field("", description=f"باید «{CONFIRM_SELECTED}» باشد")


class ResetAllRequest(BaseModel):
    confirm: str = Field("", description=f"باید «{CONFIRM_ALL}» باشد")
    password: str = Field("", description="رمز مدیر")


@router.get("/preview")
def preview():
    existing = data_reset.counts()
    return {
        "categories": [
            {
                "id": c.id,
                "label": c.label,
                "hint": c.hint,
                "count": existing[c.id],
                "implies": sorted(data_reset.IMPLIES.get(c.id, set())),
            }
            for c in data_reset.CATEGORIES
        ],
        "confirm_selected": CONFIRM_SELECTED,
        "confirm_all": CONFIRM_ALL,
    }


async def _finish(request: Request, result: dict, everything: bool) -> dict:
    # the camera thread's in-memory copies (known faces, queue, last in/out state) now point at rows
    # that are gone -- rebuild them before anything else can match against a deleted member
    engine = request.app.state.face_engine
    engine.reload_all()
    if everything:
        settings_router.restore_defaults()  # club info is gone from the DB; behaviour values back to defaults
    await manager.broadcast({"event": "data_reset", "applied": result["applied"]})
    return result


@router.post("")
async def reset_selected(payload: ResetSelectedRequest, request: Request):
    if _normalise(payload.confirm) != CONFIRM_SELECTED:
        raise HTTPException(400, f"برای تأیید، کلمه‌ی «{CONFIRM_SELECTED}» را تایپ کنید.")
    if not payload.categories:
        raise HTTPException(400, "هیچ موردی انتخاب نشده است.")
    unknown = sorted(set(payload.categories) - set(data_reset.CATEGORY_IDS))
    if unknown:
        raise HTTPException(422, f"مورد ناشناخته: {', '.join(unknown)}")
    result = data_reset.perform_reset(set(payload.categories))
    return await _finish(request, result, everything=False)


@router.post("/all")
async def reset_all(payload: ResetAllRequest, request: Request):
    if _normalise(payload.confirm) != CONFIRM_ALL:
        raise HTTPException(400, f"برای تأیید، عبارت «{CONFIRM_ALL}» را تایپ کنید.")
    if not auth.verify_manager_password(payload.password):
        # 403, not 401: the panel treats 401 as "logged out"
        raise HTTPException(403, "رمز مدیر اشتباه است.")
    result = data_reset.perform_reset(set(), everything=True)
    return await _finish(request, result, everything=True)
