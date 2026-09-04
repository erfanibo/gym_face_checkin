from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import face_recognition
from fastapi import APIRouter, HTTPException, Request

from .. import config
from ..database import db_cursor, generate_unique_membership_code
from ..face_engine import add_face_sample, blob_to_encoding
from ..models import (
    AssignPendingRequest,
    AssignPendingResponse,
    PendingItem,
    RegisterPendingRequest,
    RegisteredUserOut,
)
from ..ws_manager import manager

router = APIRouter(prefix="/api/queue", tags=["queue"])


def _pending_row_to_item(row) -> PendingItem:
    return PendingItem(
        id=row["id"],
        photo_url=f"/static/pending_faces/{Path(row['photo_path']).name}",
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
    )


def _find_duplicate_registered_face(pending_encoding: np.ndarray):
    """
    Checks the face being registered against EVERY sample of EVERY
    already-registered member (member_face_samples, not just each member's
    original single encoding -- a member registered from one angle should
    still be caught as a duplicate even if a later sample looks different).
    Returns (existing_user_id, existing_full_name) if it's close enough to be
    considered the same physical person, otherwise None.
    """
    with db_cursor() as cur:
        cur.execute(
            """SELECT s.user_id, s.encoding, u.full_name
               FROM member_face_samples s
               JOIN registered_users u ON u.id = s.user_id"""
        )
        rows = cur.fetchall()
    if not rows:
        return None

    encodings = [blob_to_encoding(r["encoding"]) for r in rows]
    distances = face_recognition.face_distance(encodings, pending_encoding)
    best_idx = int(np.argmin(distances))
    if distances[best_idx] <= config.REGISTRATION_DUPLICATE_FACE_TOLERANCE:
        row = rows[best_idx]
        return row["user_id"], row["full_name"]
    return None


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

    Guards against double registration: the face is compared against every
    already-registered member first (name is NOT the primary key here — the
    face itself is, since that's what actually prevents someone getting a
    second membership record for the same physical person).
    """
    with db_cursor() as cur:
        cur.execute("SELECT * FROM pending_queue WHERE id=? AND status='pending'", (pending_id,))
        pending_row = cur.fetchone()
    if pending_row is None:
        raise HTTPException(404, "این مورد در صف انتظار پیدا نشد یا قبلاً پردازش شده است")

    pending_encoding = blob_to_encoding(pending_row["encoding"])
    duplicate = _find_duplicate_registered_face(pending_encoding)
    if duplicate is not None:
        existing_id, existing_name = duplicate
        if payload.full_name.strip().casefold() == existing_name.strip().casefold():
            raise HTTPException(
                409,
                f"«{existing_name}» با همین چهره قبلاً ثبت‌نام کرده است — دوباره ثبت‌نام لازم نیست.",
            )
        raise HTTPException(
            409,
            f"این چهره قبلاً تحت نام «{existing_name}» ثبت شده است. اگر نام واقعاً تغییر کرده، "
            f"لطفاً پروفایل موجود (شناسه {existing_id}) را ویرایش کنید نه یک ثبت‌نام جدید.",
        )

    now = datetime.now(timezone.utc).isoformat()
    membership_code = generate_unique_membership_code()
    try:
        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO registered_users
                       (membership_code, full_name, phone, encoding, photo_path, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    membership_code,
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

    # the pending face becomes this member's first recognition sample
    add_face_sample(new_user_id, pending_encoding, source="registration")

    # keep the in-memory recognition caches in sync with the DB
    engine = request.app.state.face_engine
    engine.reload_known_users()
    engine.reload_pending_queue()

    await manager.broadcast({"event": "pending_resolved", "id": pending_id})

    return RegisteredUserOut(
        id=new_user_id,
        membership_code=membership_code,
        full_name=payload.full_name,
        phone=payload.phone,
        photo_url=f"/static/pending_faces/{Path(pending_row['photo_path']).name}",
        created_at=now,
    )


@router.post("/{pending_id}/assign", response_model=AssignPendingResponse)
async def assign_pending_to_member(pending_id: int, payload: AssignPendingRequest, request: Request):
    """
    Operator says "this pending face isn't a new person -- it's an existing
    member, just caught from an angle/lighting the system doesn't recognize
    yet." Unlike /register, this does NOT create a new registered_users row:
    it adds the pending face as one more recognition sample (member_face_
    samples) for the member the operator picked, then resolves the queue
    entry exactly like a normal registration would.

    Deliberately manual (operator picks the member, one click at a time)
    rather than automatic: if the system silently "learned" new samples on
    its own whenever it *thought* it recognized someone, a single wrong
    match would reinforce itself into more wrong matches. Requiring a human
    to confirm each new sample keeps that risk out of the loop.
    """
    with db_cursor() as cur:
        cur.execute("SELECT * FROM pending_queue WHERE id=? AND status='pending'", (pending_id,))
        pending_row = cur.fetchone()
    if pending_row is None:
        raise HTTPException(404, "این مورد در صف انتظار پیدا نشد یا قبلاً پردازش شده است")

    with db_cursor() as cur:
        cur.execute("SELECT id, full_name FROM registered_users WHERE id=?", (payload.user_id,))
        member_row = cur.fetchone()
    if member_row is None:
        raise HTTPException(404, "عضوی با این شناسه پیدا نشد")

    pending_encoding = blob_to_encoding(pending_row["encoding"])
    add_face_sample(payload.user_id, pending_encoding, source="assigned")

    with db_cursor(commit=True) as cur:
        cur.execute(
            "UPDATE pending_queue SET status='registered', registered_user_id=? WHERE id=?",
            (payload.user_id, pending_id),
        )
        cur.execute("SELECT COUNT(*) AS n FROM member_face_samples WHERE user_id=?", (payload.user_id,))
        sample_count = cur.fetchone()["n"]

    # keep the in-memory recognition caches in sync with the DB
    engine = request.app.state.face_engine
    engine.reload_known_users()
    engine.reload_pending_queue()

    await manager.broadcast({"event": "pending_resolved", "id": pending_id})

    return AssignPendingResponse(
        ok=True,
        user_id=payload.user_id,
        full_name=member_row["full_name"],
        sample_count=sample_count,
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
