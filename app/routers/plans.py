"""
مالی و شهریه > تعرفه‌ها و پلن‌ها -- تعریف تعرفه‌های عضویت (نام، قیمت، تعداد روز).

مدیر هر تعرفه را یک‌بار تعریف می‌کند و بعد هنگام ثبت‌نام یا تمدید یک عضو، همان تعرفه را
برای او «ثبت» می‌کند (routers/users.py، routers/queue.py، membership.py): تعداد روزِ تعرفه به
تاریخ پایان عضویتش اضافه می‌شود و یک رکورد خرید (با نام/قیمت/روزِ همان لحظه) ذخیره می‌شود.

«حذف» یک تعرفه آن را فقط از لیست و از ثبت برای عضوهای جدید برمی‌دارد (is_active=0)؛ تاریخچه‌ی
خریدهای قبلی دست‌نخورده می‌ماند.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from .. import auth
from ..database import db_cursor
from ..models import PlanCreate, PlanOut

router = APIRouter(prefix="/api/plans", tags=["plans"], dependencies=[Depends(auth.require_manager)])


def _list_plans() -> list[PlanOut]:
    now = datetime.now(timezone.utc)
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM membership_plans WHERE is_active=1 ORDER BY duration_days ASC, price ASC, id ASC"
        )
        plans = cur.fetchall()
        # «عضو فعال» = آخرین پلنِ ثبت‌شده برای او همین پلن است و عضویتش هنوز منقضی نشده.
        cur.execute(
            """SELECT u.membership_expires_at AS expires_at,
                      (SELECT pu.plan_id FROM membership_purchases pu
                        WHERE pu.user_id = u.id ORDER BY pu.id DESC LIMIT 1) AS plan_id
               FROM registered_users u
               WHERE u.membership_expires_at IS NOT NULL"""
        )
        members = cur.fetchall()

    active_by_plan: dict[int, int] = {}
    for m in members:
        if m["plan_id"] is None:
            continue
        try:
            expires = datetime.fromisoformat(m["expires_at"])
        except ValueError:
            continue
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires > now:
            active_by_plan[m["plan_id"]] = active_by_plan.get(m["plan_id"], 0) + 1

    return [
        PlanOut(
            id=p["id"], name=p["name"], price=p["price"], duration_days=p["duration_days"],
            active_members=active_by_plan.get(p["id"], 0), created_at=p["created_at"],
        )
        for p in plans
    ]


@router.get("", response_model=list[PlanOut])
def list_plans():
    return _list_plans()


@router.post("", response_model=PlanOut)
def create_plan(payload: PlanCreate):
    name = payload.name
    with db_cursor() as cur:
        cur.execute("SELECT name FROM membership_plans WHERE is_active=1")
        existing = [r["name"].strip().casefold() for r in cur.fetchall()]
    if name.casefold() in existing:
        raise HTTPException(409, f"تعرفه‌ای با نام «{name}» از قبل وجود دارد.")

    now = datetime.now(timezone.utc).isoformat()
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO membership_plans (name, price, duration_days, created_at) VALUES (?, ?, ?, ?)",
            (name, payload.price, payload.duration_days, now),
        )
        new_id = cur.lastrowid
    return PlanOut(id=new_id, name=name, price=payload.price, duration_days=payload.duration_days,
                   active_members=0, created_at=now)


@router.delete("/{plan_id}")
def delete_plan(plan_id: int):
    with db_cursor() as cur:
        cur.execute("SELECT name FROM membership_plans WHERE id=? AND is_active=1", (plan_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "این تعرفه پیدا نشد یا قبلاً حذف شده است.")
    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE membership_plans SET is_active=0 WHERE id=?", (plan_id,))
    return {"ok": True, "name": row["name"]}
