"""
Selective / total data reset (Settings > بازنشانی اطلاعات).

Everything happens in ONE database transaction (so the camera thread, which writes through the
same lock, can never see or add a half-deleted state), photo files are removed only after that
transaction committed, and nothing here talks HTTP -- routers/reset.py does the confirmation
checks and cache reloads, this module only knows what each category means.

Categories (ids are the API contract):
  pending        queued unknown faces that were never registered (+ their photos)
  recognition_log live recognition log
  attendance     entry/exit history
  workouts       workout plans
  purchases      membership purchase history (+ every member's membership end date)
  face_samples   members' face vectors (members stay, but are no longer recognised; note: the
                 vectors are rebuilt from the member photos at the next start unless
                 `member_photos` is deleted too)
  member_photos  members' photo files (+ photo_path cleared)
  plans          membership plans (purchase history keeps its own snapshot of the plan)
  members        members and, necessarily, everything that hangs off them (IMPLIES below)

`everything=True` additionally empties the whole queue and all settings (club info, behaviour) --
but never the manager password.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import config
from .auth import MANAGER_PASSWORD_SETTING_KEY
from .database import db_cursor


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    hint: str = ""


CATEGORIES: list[Category] = [
    Category("pending", "صف انتظار (چهره‌های ناشناس)", "چهره‌هایی که هنوز ثبت‌نام یا رد نشده‌اند، همراه با عکسشان"),
    Category("recognition_log", "لاگ زنده‌ی تشخیص"),
    Category("attendance", "تاریخچه‌ی ورود و خروج"),
    Category("workouts", "برنامه‌های تمرینی"),
    Category("purchases", "تاریخچه‌ی خرید و تمدید", "تاریخ پایان عضویت همه‌ی اعضا هم خالی می‌شود"),
    Category(
        "face_samples",
        "نمونه‌های چهره‌ی اعضا",
        "اعضا می‌مانند ولی دیگر شناخته نمی‌شوند. تا وقتی عکس اعضا هست، با ری‌استارت بعدی دوباره از روی عکس ساخته می‌شوند",
    ),
    Category("member_photos", "عکس‌های اعضا", "فایل عکس‌ها پاک می‌شود"),
    Category("plans", "تعرفه‌ها", "تاریخچه‌ی خریدهای قبلی دست‌نخورده می‌ماند"),
    Category("members", "اعضا", "همراه با همه‌ی اطلاعاتشان؛ موارد وابسته خودکار انتخاب می‌شوند"),
]
CATEGORY_IDS = [c.id for c in CATEGORIES]

# deleting members forces deleting everything that points at them
IMPLIES: dict[str, set[str]] = {
    "members": {"attendance", "face_samples", "workouts", "purchases", "member_photos", "recognition_log"},
}


def expand(selected: set[str]) -> set[str]:
    """The selection plus everything it necessarily drags along."""
    result = set(selected)
    for cat in list(result):
        result |= IMPLIES.get(cat, set())
    return result


def counts() -> dict[str, int]:
    """How many rows/files each category would remove (for the confirmation screen)."""
    queries = {
        "pending": "SELECT COUNT(*) FROM pending_queue WHERE status != 'registered'",
        "recognition_log": "SELECT COUNT(*) FROM recognition_log",
        "attendance": "SELECT COUNT(*) FROM attendance_log",
        "workouts": "SELECT COUNT(*) FROM workout_items",
        "purchases": "SELECT COUNT(*) FROM membership_purchases",
        "face_samples": "SELECT COUNT(*) FROM member_face_samples",
        "member_photos": "SELECT COUNT(*) FROM registered_users WHERE photo_path IS NOT NULL",
        "plans": "SELECT COUNT(*) FROM membership_plans",
        "members": "SELECT COUNT(*) FROM registered_users",
    }
    with db_cursor() as cur:
        out = {}
        for key, sql in queries.items():
            cur.execute(sql)
            out[key] = int(cur.fetchone()[0])
    return out


def _photo_names(rows) -> set[str]:
    return {Path(r[0]).name for r in rows if r[0]}


def perform_reset(selected: set[str], everything: bool = False) -> dict:
    """
    Deletes the selected categories (or everything). Returns
    {"applied": [category ids, in canonical order], "deleted": {id: rows removed}, "photos_removed": n}.
    """
    unknown = set(selected) - set(CATEGORY_IDS)
    if unknown:
        raise ValueError(f"unknown categories: {sorted(unknown)}")
    cats = set(CATEGORY_IDS) if everything else expand(set(selected))
    deleted: dict[str, int] = {}
    doomed_photos: set[str] = set()

    with db_cursor(commit=True) as cur:
        def run(sql: str, params: tuple = ()) -> int:
            cur.execute(sql, params)
            return cur.rowcount

        # children first, parents last (foreign_keys=ON)
        if "recognition_log" in cats:
            deleted["recognition_log"] = run("DELETE FROM recognition_log")
        if "attendance" in cats:
            deleted["attendance"] = run("DELETE FROM attendance_log")
        if "workouts" in cats:
            deleted["workouts"] = run("DELETE FROM workout_items")
        if "purchases" in cats:
            deleted["purchases"] = run("DELETE FROM membership_purchases")
            run("UPDATE registered_users SET membership_expires_at = NULL")
        if "face_samples" in cats:
            deleted["face_samples"] = run("DELETE FROM member_face_samples")
            run("UPDATE registered_users SET encoding = NULL")  # legacy single-vector column

        if "member_photos" in cats or "members" in cats:
            cur.execute("SELECT photo_path FROM registered_users WHERE photo_path IS NOT NULL")
            member_photos = _photo_names(cur.fetchall())
            doomed_photos |= member_photos
            if "member_photos" in cats:
                deleted["member_photos"] = run("UPDATE registered_users SET photo_path = NULL WHERE photo_path IS NOT NULL")

        if "members" in cats:
            run("DELETE FROM pending_queue WHERE registered_user_id IS NOT NULL")
            deleted["members"] = run("DELETE FROM registered_users")

        if "plans" in cats:
            run("UPDATE membership_purchases SET plan_id = NULL WHERE plan_id IS NOT NULL")
            deleted["plans"] = run("DELETE FROM membership_plans")

        if "pending" in cats or everything:
            where = "" if everything else " WHERE status != 'registered'"
            cur.execute(f"SELECT photo_path FROM pending_queue{where}")
            queue_photos = _photo_names(cur.fetchall())
            deleted["pending"] = run(f"DELETE FROM pending_queue{where}")
            # a queue photo can be the very same file a (remaining) member uses as their photo
            cur.execute("SELECT photo_path FROM registered_users WHERE photo_path IS NOT NULL")
            doomed_photos |= queue_photos - _photo_names(cur.fetchall())

        if everything:
            run("DELETE FROM app_settings WHERE key != ?", (MANAGER_PASSWORD_SETTING_KEY,))

    photos_removed = _remove_photo_files(doomed_photos)
    return {
        "applied": [c for c in CATEGORY_IDS if c in cats],
        "deleted": deleted,
        "photos_removed": photos_removed,
    }


def _remove_photo_files(names: set[str]) -> int:
    """Delete photo files by NAME from the two photo folders (the stored directory can be stale and
    must never be trusted to point outside them). Best effort: a locked file doesn't fail the reset."""
    removed = 0
    for name in names:
        if not name or name != Path(name).name:
            continue
        for folder in (config.PENDING_PHOTOS_DIR, config.USER_PHOTOS_DIR):
            target = folder / name
            try:
                if target.is_file():
                    target.unlink()
                    removed += 1
            except OSError:
                pass
    return removed
