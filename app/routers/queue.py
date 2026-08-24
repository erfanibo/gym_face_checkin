from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from ..database import db_cursor
from ..models import PendingItem, RegisterPendingRequest, RegisteredUserOut
from ..ws_manager import manager

router = APIRouter(prefix="/api/queue", tags=["queue"])


def _pending_row_to_item(row) -> PendingItem:
    return PendingItem(
        id=row["id"],
        photo_url=f"/static/pending_faces/{Path(row['photo_path']).name}",
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
    )


@router.get("", response_model=list[PendingItem])
def list_pending():
    """Live snapshot of unresolved faces waiting at reception."""
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM pending_queue WHERE status='pending' ORDER BY first_seen_at DESC"
        )
        rows = cur.fetchall()
    return [_pending_row_to_item(r) for r in rows]


@router.post("/{pending_id}/register", response_model=RegisteredUserOut)
async def register_pending(pending_id: int, payload: RegisterPendingRequest, request: Request):
    """
    Operator fills in the sign-up form for a face in the queue.
    Moves the row (encoding + photo) from pending_queue into registered_users.
    """
    with db_cursor() as cur:
        cur.execute("SELECT * FROM pending_queue WHERE id=? AND status='pending'", (pending_id,))
        pending_row = cur.fetchone()
    if pending_row is None:
        raise HTTPException(404, "این مورد در صف انتظار پیدا نشد یا قبلاً پردازش شده است")

    now = datetime.now(timezone.utc).isoformat()
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO registered_users
                       (membership_code, full_name, phone, encoding, photo_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    payload.membership_code,
                    payload.full_name,
                    payload.phone,
                    pending_row["encoding"],
                    pending_row["photo_path"],
                    now,
                ),
            )
            new_user_id = cur.lastrowid
            cur.execute(
                "UPDATE pending_queue SET status='registered', registered_user_id=? WHERE id=?",
                (new_user_id, pending_id),
            )
    except Exception as exc:
        # most likely a UNIQUE constraint violation on membership_code
        raise HTTPException(400, f"ثبت‌نام ناموفق بود: {exc}")

    # keep the in-memory recognition caches in sync with the DB
    engine = request.app.state.face_engine
    engine.reload_known_users()
    engine.reload_pending_queue()

    await manager.broadcast({"event": "pending_resolved", "id": pending_id})

    return RegisteredUserOut(
        id=new_user_id,
        membership_code=payload.membership_code,
        full_name=payload.full_name,
        phone=payload.phone,
        photo_url=f"/static/pending_faces/{Path(pending_row['photo_path']).name}",
        created_at=now,
    )


@router.delete("/{pending_id}")
async def reject_pending(pending_id: int, request: Request):
    """Operator dismisses a queue entry (e.g. false detection, passerby)."""
    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE pending_queue SET status='rejected' WHERE id=? AND status='pending'",
            (pending_id,),
        )
        affected = cur.rowcount
    if affected == 0:
        raise HTTPException(404, "این مورد در صف انتظار پیدا نشد یا قبلاً پردازش شده است")

    engine = request.app.state.face_engine
    engine.reload_pending_queue()

    await manager.broadcast({"event": "pending_resolved", "id": pending_id})
    return {"ok": True}
