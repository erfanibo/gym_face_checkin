"""
Core recognition engine.

Runs entirely in a background `threading.Thread` (NOT an asyncio task) because
cv2.VideoCapture().read() and face_recognition's detection/encoding calls are
blocking, CPU-bound calls -- running them inside the asyncio event loop would
freeze the whole FastAPI server (including the WebSocket updates) while a
frame is being processed.

Flow per processed frame:
  1. Grab frame from the webcam, downscale, find faces + 128-d encodings.
  2. For each face:
       a. Compare against every ENROLLED member (registered_users).
          -> match: log attendance (throttled, alternates ورود/خروج) and stop.
       b. Compare against faces already waiting in pending_queue that were
          seen within the last PENDING_DEDUP_WINDOW_SECONDS.
          -> match: just bump last_seen_at (anti-spam, no new row/photo).
          NOTE: each pending entry keeps a small ROLLING WINDOW of its most
          recent encodings (not just the first one it was created with), so
          natural pose/expression drift while someone stands at the camera
          doesn't cause them to "stop matching" and get re-enqueued as a
          brand-new face.
       c. Otherwise: this is a brand-new unknown face -> crop + save a photo,
          insert a pending_queue row, and push a "new_pending" WebSocket event
          so the reception panel updates live.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import face_recognition
import httpx

from . import config
from .database import db_cursor
from .ws_manager import manager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def encoding_to_blob(encoding: np.ndarray) -> bytes:
    """128 x float64 numpy array -> raw bytes, for storing in a BLOB column."""
    return np.asarray(encoding, dtype=np.float64).tobytes()


def blob_to_encoding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float64)


def add_face_sample(user_id: int, encoding: np.ndarray, source: str = "assigned") -> None:
    """
    Adds one more recognition sample for an EXISTING member. Used both right
    after registration (source='registration', one call from
    routers/queue.py:register_pending) and whenever an operator later says
    "this pending-queue face is actually this same member, just a different
    angle/light" (source='assigned', routers/queue.py:assign_pending_to_member).

    Caps each member at config.MAX_FACE_SAMPLES_PER_MEMBER samples: once the
    cap is hit, the OLDEST sample is dropped to make room for the new one.
    This keeps the per-frame comparison cost bounded (see FaceEngine.
    _match_known) and naturally rotates a member's profile toward their most
    recent appearances instead of growing forever.

    Callers are responsible for calling FaceEngine.reload_known_users()
    afterwards so the background camera thread's in-memory cache picks up
    the new sample immediately, rather than waiting for the next full reload.
    """
    with db_cursor(commit=True) as cur:
        cur.execute(
            "INSERT INTO member_face_samples (user_id, encoding, source, created_at) VALUES (?, ?, ?, ?)",
            (user_id, encoding_to_blob(encoding), source, _now_iso()),
        )
        cur.execute(
            "SELECT id FROM member_face_samples WHERE user_id=? ORDER BY id ASC",
            (user_id,),
        )
        sample_ids = [r["id"] for r in cur.fetchall()]
        overflow = len(sample_ids) - config.MAX_FACE_SAMPLES_PER_MEMBER
        if overflow > 0:
            oldest = [(i,) for i in sample_ids[:overflow]]
            cur.executemany("DELETE FROM member_face_samples WHERE id=?", oldest)


class FaceEngine:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._thread: threading.Thread | None = None
        self._running = False

        # in-memory caches mirrored from SQLite, refreshed on demand so the
        # hot per-frame loop never has to hit the DB just to compare vectors.
        # Each member can have MULTIPLE samples (member_face_samples table),
        # so these are flat, parallel lists: _known_sample_user_ids[i] is the
        # owner of _known_sample_encodings[i]. A member with 8 samples simply
        # appears 8 times, once per sample -- face_distance() compares the
        # incoming frame against all of them at once and we take whichever
        # single sample is closest, from whichever member.
        self._known_sample_user_ids: list[int] = []
        self._known_sample_encodings: list[np.ndarray] = []

        self._pending_cache: list[dict] = []  # {id, encodings: [..], last_seen}

        # user_id -> {"at": datetime, "type": "in"|"out"} for the LAST logged
        # attendance event. Loaded from the DB (not just kept in memory) so a
        # service restart doesn't reset the in/out toggle or the cooldown.
        self._last_attendance: dict[int, dict] = {}

        # user_id -> datetime of the last row written to recognition_log.
        # Purely an in-memory rate limiter (see _maybe_log_recognition) --
        # unlike _last_attendance, it's fine for this to reset on restart:
        # worst case is one extra log line right after a restart, not a
        # correctness issue like losing the in/out toggle would be.
        self._last_recognition_log: dict[int, datetime] = {}

        self.reload_known_users()
        self.reload_pending_queue()
        self._load_last_attendance()

    # ------------------------------------------------------------------ #
    # cache management (call these after any write to the relevant tables)
    # ------------------------------------------------------------------ #
    def reload_known_users(self):
        """
        Loads EVERY sample for EVERY member (not one row per member anymore --
        see member_face_samples). Names aren't cached here: the hot path only
        ever needs a user_id (see _match_known), and every place that reports
        a name back to the operator/webhook already re-reads it fresh from
        registered_users (e.g. _record_attendance), so there's no separate
        name cache to keep in sync here.
        """
        with db_cursor() as cur:
            cur.execute("SELECT user_id, encoding FROM member_face_samples")
            rows = cur.fetchall()
        self._known_sample_user_ids = [r["user_id"] for r in rows]
        self._known_sample_encodings = [blob_to_encoding(r["encoding"]) for r in rows]

    def reload_pending_queue(self):
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, encoding, last_seen_at FROM pending_queue WHERE status='pending'"
            )
            rows = cur.fetchall()
        self._pending_cache = [
            {"id": r["id"], "encodings": [blob_to_encoding(r["encoding"])], "last_seen": r["last_seen_at"]}
            for r in rows
        ]

    def _load_last_attendance(self):
        """Restore the last known in/out state per member from attendance_log,
        so the toggle and the 5-minute cooldown survive a service restart."""
        with db_cursor() as cur:
            cur.execute(
                """SELECT a.user_id, a.event_type, a.checkin_at
                   FROM attendance_log a
                   JOIN (
                       SELECT user_id, MAX(id) AS max_id
                       FROM attendance_log GROUP BY user_id
                   ) latest ON latest.user_id = a.user_id AND latest.max_id = a.id"""
            )
            rows = cur.fetchall()
        self._last_attendance = {
            r["user_id"]: {"at": datetime.fromisoformat(r["checkin_at"]), "type": r["event_type"]}
            for r in rows
        }

    def reload_all(self):
        """
        Public entry point used by routers/backup.py right after a database
        restore swaps out the .db file from under this running process: every
        in-memory cache this class keeps (known-face samples, pending-queue
        samples, last in/out attendance state) gets rebuilt from whatever is
        now in the (freshly restored) database, and the small per-member
        recognition-log debounce timer is cleared since it no longer reflects
        reality anyway.
        """
        self.reload_known_users()
        self.reload_pending_queue()
        self._load_last_attendance()
        self._last_recognition_log = {}

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="face-engine")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def _run_loop(self):
        video = cv2.VideoCapture(config.CAMERA_INDEX)
        if not video.isOpened():
            print(f"[face_engine] ERROR: could not open camera index {config.CAMERA_INDEX}")
            self._running = False
            return

        frame_count = 0
        try:
            while self._running:
                ok, frame = video.read()
                if not ok:
                    time.sleep(0.05)
                    continue

                frame_count += 1
                if frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                try:
                    self._process_frame(frame)
                except Exception as exc:  # never let one bad frame kill the thread
                    print(f"[face_engine] frame processing error: {exc}")
        finally:
            video.release()

    def _process_frame(self, frame: np.ndarray):
        small = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_SCALE, fy=config.FRAME_RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb_small)
        if not locations:
            return
        encodings = face_recognition.face_encodings(rgb_small, locations)

        for location, encoding in zip(locations, encodings):
            self._handle_face(encoding, location, frame)

    # ------------------------------------------------------------------ #
    # per-face decision tree
    # ------------------------------------------------------------------ #
    def _handle_face(self, encoding: np.ndarray, location_small: tuple, full_frame: np.ndarray):
        match = self._match_known(encoding)
        if match is not None:
            user_id, distance = match
            self._maybe_log_attendance(user_id)
            self._maybe_log_recognition(user_id, distance)
            return

        pending_id = self._match_pending(encoding)
        if pending_id is not None:
            self._touch_pending(pending_id, encoding)
            return

        self._enqueue_new_face(encoding, location_small, full_frame)

    def _match_known(self, encoding: np.ndarray) -> tuple[int, float] | None:
        """
        Compares against every sample of every member at once (a few thousand
        vector comparisons even at ~200 members x 8 samples -- negligible
        with numpy, well under the per-frame budget) and returns
        (user_id, distance) for whichever member owns the single closest
        sample, provided it's within tolerance. The distance is only used
        for the recognition log below (nice for debugging borderline
        matches); attendance logic never needed it before, so it's a plain
        tuple return rather than something that couldn't ever return before.
        """
        if not self._known_sample_encodings:
            return None
        distances = face_recognition.face_distance(self._known_sample_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= config.KNOWN_USER_TOLERANCE:
            return self._known_sample_user_ids[best_idx], float(distances[best_idx])
        return None

    def _match_pending(self, encoding: np.ndarray) -> int | None:
        """
        Anti-spam check: is this the same unknown person already waiting in
        the queue? Compares against EVERY recent sample kept for each pending
        entry (see PENDING_DEDUP_MAX_SAMPLES), not just its original encoding,
        so gradual pose/expression drift doesn't break the match.
        """
        if not self._pending_cache:
            return None
        now = datetime.now(timezone.utc)

        flat_ids: list[int] = []
        flat_encodings: list[np.ndarray] = []
        for p in self._pending_cache:
            if (now - datetime.fromisoformat(p["last_seen"])).total_seconds() > config.PENDING_DEDUP_WINDOW_SECONDS:
                continue
            for sample in p["encodings"]:
                flat_ids.append(p["id"])
                flat_encodings.append(sample)

        if not flat_encodings:
            return None

        distances = face_recognition.face_distance(flat_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= config.PENDING_DEDUP_TOLERANCE:
            return flat_ids[best_idx]
        return None

    def _touch_pending(self, pending_id: int, encoding: np.ndarray):
        now = _now_iso()
        # also refresh the DB row's stored encoding to this latest (likely
        # better-framed) sample, so registration later matches off the best
        # available snapshot rather than whatever the very first glance looked like
        blob = encoding_to_blob(encoding)
        with db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE pending_queue SET last_seen_at=?, encoding=? WHERE id=?",
                (now, blob, pending_id),
            )
        for p in self._pending_cache:
            if p["id"] == pending_id:
                p["last_seen"] = now
                p["encodings"].append(encoding)
                if len(p["encodings"]) > config.PENDING_DEDUP_MAX_SAMPLES:
                    p["encodings"] = p["encodings"][-config.PENDING_DEDUP_MAX_SAMPLES:]
                break

    def _enqueue_new_face(self, encoding: np.ndarray, location_small: tuple, full_frame: np.ndarray):
        now = _now_iso()
        photo_path = self._save_face_crop(full_frame, location_small)

        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO pending_queue (encoding, photo_path, first_seen_at, last_seen_at, status)
                   VALUES (?, ?, ?, ?, 'pending')""",
                (encoding_to_blob(encoding), str(photo_path), now, now),
            )
            new_id = cur.lastrowid

        self._pending_cache.append({"id": new_id, "encodings": [encoding], "last_seen": now})

        self._broadcast_soon({
            "event": "new_pending",
            "id": new_id,
            "photo_url": f"/static/pending_faces/{photo_path.name}",
            "first_seen_at": now,
        })

    def _save_face_crop(self, full_frame: np.ndarray, location_small: tuple):
        top, right, bottom, left = location_small
        scale = int(round(1 / config.FRAME_RESIZE_SCALE))
        top, right, bottom, left = top * scale, right * scale, bottom * scale, left * scale

        pad = 20
        h, w = full_frame.shape[:2]
        top, left = max(0, top - pad), max(0, left - pad)
        bottom, right = min(h, bottom + pad), min(w, right + pad)
        crop = full_frame[top:bottom, left:right]

        filename = f"{uuid.uuid4().hex}.jpg"
        path = config.PENDING_PHOTOS_DIR / filename
        cv2.imwrite(str(path), crop)
        return path

    def _maybe_log_attendance(self, user_id: int):
        """
        CAMERA path — throttled. Alternates between 'in' (ورود) و 'out' (خروج):
        the first time a member is seen it's an entry; if the same member is
        seen again after ATTENDANCE_COOLDOWN_SECONDS (5 min), it's logged as
        the opposite of their last event. Sightings within the cooldown
        window are ignored entirely (still considered "the same visit").
        """
        now = datetime.now(timezone.utc)
        last = self._last_attendance.get(user_id)
        if last is not None and (now - last["at"]).total_seconds() < config.ATTENDANCE_COOLDOWN_SECONDS:
            return  # same visit, too soon to log another event

        next_type = "out" if (last is not None and last["type"] == "in") else "in"
        self._record_attendance(user_id, next_type)

    def _maybe_log_recognition(self, user_id: int, distance: float):
        """
        Powers the "لاگ زنده" (live log) panel — a raw, near-real-time
        "X شناسایی شد" feed, completely separate from attendance_log's
        5-minute-cooldown/in-out logic above. Debounced by a much smaller
        interval (config.RECOGNITION_LOG_DEBOUNCE_SECONDS, a few seconds) so
        one person standing at the camera doesn't write + broadcast dozens of
        near-identical rows per minute (see PROCESS_EVERY_N_FRAMES).
        """
        now = datetime.now(timezone.utc)
        last_at = self._last_recognition_log.get(user_id)
        if last_at is not None and (now - last_at).total_seconds() < config.RECOGNITION_LOG_DEBOUNCE_SECONDS:
            return
        self._last_recognition_log[user_id] = now

        with db_cursor(commit=True) as cur:
            cur.execute("SELECT full_name FROM registered_users WHERE id=?", (user_id,))
            row = cur.fetchone()
            full_name = row["full_name"] if row else "?"
            cur.execute(
                "INSERT INTO recognition_log (user_id, full_name, distance, created_at) VALUES (?, ?, ?, ?)",
                (user_id, full_name, distance, now.isoformat()),
            )

        self._broadcast_soon({
            "event": "recognition_seen",
            "user_id": user_id,
            "full_name": full_name,
            "distance": distance,
            "created_at": now.isoformat(),
        })

    def log_manual_attendance(self, user_id: int, event_type: str | None = None) -> dict:
        """
        ADMIN path — used by the reception panel's "ثبت ورود" / "ثبت خروج"
        buttons. Unlike the camera path, this is NOT throttled by the cooldown
        and does NOT require alternation: an operator correcting a mistake
        (e.g. the camera missed someone's exit) needs to be able to force the
        correct state regardless of what was last recorded. If event_type is
        omitted, it toggles from the last known state instead (kept for API
        flexibility, though the panel's two explicit buttons never rely on this).
        """
        if event_type is None:
            last = self._last_attendance.get(user_id)
            event_type = "out" if (last is not None and last["type"] == "in") else "in"
        if event_type not in ("in", "out"):
            raise ValueError("event_type must be 'in' or 'out'")
        return self._record_attendance(user_id, event_type)

    def _record_attendance(self, user_id: int, event_type: str) -> dict:
        """Shared by both paths above: writes the row, refreshes the in-memory
        toggle state, and fires the WebSocket broadcast + webhook."""
        now = datetime.now(timezone.utc)

        with db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO attendance_log (user_id, event_type, checkin_at) VALUES (?, ?, ?)",
                (user_id, event_type, now.isoformat()),
            )
            event_id = cur.lastrowid
            cur.execute("SELECT full_name, membership_code FROM registered_users WHERE id=?", (user_id,))
            row = cur.fetchone()

        self._last_attendance[user_id] = {"at": now, "type": event_type}

        full_name = row["full_name"] if row else "?"
        membership_code = row["membership_code"] if row else None
        checkin_at_iso = now.isoformat()

        self._broadcast_soon({
            "event": "checkin",
            "user_id": user_id,
            "full_name": full_name,
            "event_type": event_type,
            "checkin_at": checkin_at_iso,
        })

        self._send_webhook_soon({
            "event_id": event_id,
            "user_id": user_id,
            "membership_code": membership_code,
            "full_name": full_name,
            "event_type": event_type,
            "checkin_at": checkin_at_iso,
        })

        return {
            "event_id": event_id,
            "event_type": event_type,
            "checkin_at": checkin_at_iso,
            "full_name": full_name,
            "membership_code": membership_code,
        }

    # ------------------------------------------------------------------ #
    # bridge: background thread -> asyncio event loop
    # ------------------------------------------------------------------ #
    def _broadcast_soon(self, message: dict):
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), self._loop)

    def _send_webhook_soon(self, payload: dict):
        if not config.WEBHOOK_URL:
            return
        asyncio.run_coroutine_threadsafe(self._send_webhook(payload), self._loop)

    async def _send_webhook(self, payload: dict):
        """
        POSTs an attendance event to WEBHOOK_URL, signed with WEBHOOK_SECRET
        (if set) so the receiving server can verify it really came from this
        system. Retries a few times with backoff if the target is briefly
        unreachable; if it still fails, the event is NOT lost — it's already
        safely in attendance_log and can be picked up via
        GET /api/attendance?since_id=... as a polling fallback.
        """
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if config.WEBHOOK_SECRET:
            signature = hmac.new(config.WEBHOOK_SECRET.encode("utf-8"), body, hashlib.sha256).hexdigest()
            headers["X-Webhook-Signature"] = f"sha256={signature}"

        delay = config.WEBHOOK_RETRY_DELAY_SECONDS
        async with httpx.AsyncClient(timeout=config.WEBHOOK_TIMEOUT_SECONDS) as client:
            for attempt in range(1, config.WEBHOOK_MAX_RETRIES + 1):
                try:
                    resp = await client.post(config.WEBHOOK_URL, content=body, headers=headers)
                    if resp.status_code < 300:
                        return
                    print(f"[webhook] attempt {attempt}/{config.WEBHOOK_MAX_RETRIES} got HTTP {resp.status_code}")
                except Exception as exc:
                    print(f"[webhook] attempt {attempt}/{config.WEBHOOK_MAX_RETRIES} failed: {exc}")

                if attempt < config.WEBHOOK_MAX_RETRIES:
                    await asyncio.sleep(delay)
                    delay *= 2  # exponential backoff

        print(f"[webhook] giving up after {config.WEBHOOK_MAX_RETRIES} attempts for event_id={payload.get('event_id')} "
              f"— it's still in attendance_log, safe to pick up via polling")
