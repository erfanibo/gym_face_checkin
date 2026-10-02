from fastapi import APIRouter, Depends, HTTPException

from .. import auth
from ..database import db_cursor
from ..models import SaveWorkoutPlanRequest, WorkoutItemOut, WorkoutPlanOut

# No router-wide `dependencies=` here (unlike users.py) -- this file mixes
# manager-only routes (edit ANY member's plan) and member-only routes (view/
# tick off only YOUR OWN plan), so each route declares its own auth below.
router = APIRouter(prefix="/api", tags=["workouts"])


def _load_items(user_id: int) -> list[WorkoutItemOut]:
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, position, exercise_name, sets, reps, notes, is_done "
            "FROM workout_items WHERE user_id=? ORDER BY position",
            (user_id,),
        )
        rows = cur.fetchall()
    return [
        WorkoutItemOut(
            id=r["id"],
            position=r["position"],
            exercise_name=r["exercise_name"],
            sets=r["sets"],
            reps=r["reps"],
            notes=r["notes"],
            is_done=bool(r["is_done"]),
        )
        for r in rows
    ]


def _load_updated_at(user_id: int) -> str | None:
    with db_cursor() as cur:
        cur.execute("SELECT MAX(updated_at) AS u FROM workout_items WHERE user_id=?", (user_id,))
        return cur.fetchone()["u"]


# ---------------------------------------------------------------------------
# Manager: view/replace a specific member's plan
# ---------------------------------------------------------------------------
@router.get(
    "/users/{user_id}/workout",
    response_model=WorkoutPlanOut,
    dependencies=[Depends(auth.require_manager)],
)
def get_member_workout(user_id: int):
    with db_cursor() as cur:
        cur.execute("SELECT full_name FROM registered_users WHERE id=?", (user_id,))
        row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "کاربری با این شناسه پیدا نشد")
    return WorkoutPlanOut(
        user_id=user_id,
        full_name=row["full_name"],
        updated_at=_load_updated_at(user_id),
        items=_load_items(user_id),
    )


@router.put(
    "/users/{user_id}/workout",
    response_model=WorkoutPlanOut,
    dependencies=[Depends(auth.require_manager)],
)
def save_member_workout(user_id: int, payload: SaveWorkoutPlanRequest):
    """
    Replaces this member's ENTIRE plan with the given list, in the given
    order. There's no per-item add/remove/reorder endpoint on purpose -- the
    admin panel is a todo-list-style editor that always sends the whole
    current list, and this deletes+reinserts rather than diffing. Matches
    "only one current plan, overwritten on update" from the agreed design.
    """
    with db_cursor() as cur:
        cur.execute("SELECT 1 FROM registered_users WHERE id=?", (user_id,))
        if cur.fetchone() is None:
            raise HTTPException(404, "کاربری با این شناسه پیدا نشد")

    with db_cursor(commit=True) as cur:
        cur.execute("DELETE FROM workout_items WHERE user_id=?", (user_id,))
        for position, item in enumerate(payload.items):
            cur.execute(
                "INSERT INTO workout_items (user_id, position, exercise_name, sets, reps, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, position, item.exercise_name, item.sets, item.reps, item.notes),
            )

    return get_member_workout(user_id)


# ---------------------------------------------------------------------------
# Member: view/tick off your OWN plan only
# ---------------------------------------------------------------------------
@router.get("/me/workout", response_model=WorkoutPlanOut)
def get_my_workout(user_id: int = Depends(auth.require_member)):
    return WorkoutPlanOut(user_id=user_id, updated_at=_load_updated_at(user_id), items=_load_items(user_id))


@router.patch("/me/workout/{item_id}/toggle", response_model=WorkoutItemOut)
def toggle_my_item(item_id: int, user_id: int = Depends(auth.require_member)):
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, user_id, position, exercise_name, sets, reps, notes, is_done "
            "FROM workout_items WHERE id=?",
            (item_id,),
        )
        row = cur.fetchone()

    # Same 404 whether the item doesn't exist OR belongs to someone else --
    # never confirm/deny another member's item_id exists.
    if row is None or row["user_id"] != user_id:
        raise HTTPException(404, "موردی با این شناسه پیدا نشد")

    new_done = 0 if row["is_done"] else 1
    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE workout_items SET is_done=? WHERE id=?", (new_done, item_id))

    return WorkoutItemOut(
        id=row["id"],
        position=row["position"],
        exercise_name=row["exercise_name"],
        sets=row["sets"],
        reps=row["reps"],
        notes=row["notes"],
        is_done=bool(new_done),
    )
