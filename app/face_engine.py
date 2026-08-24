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
          -> match: log attendance (throttled) and stop.
       b. Compare against faces already waiting in pending_queue that were
          seen within the last PENDING_DEDUP_WINDOW_SECONDS.
          -> match: just bump last_seen_at (anti-spam, no new row/photo).
       c. Otherwise: this is a brand-new unknown face -> crop + save a photo,
          insert a pending_queue row, and push a "new_pending" WebSocket event
          so the reception panel updates live.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import face_recognition

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


class FaceEngine:
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self._thread: threading.Thread | None = None
        self._running = False

        # in-memory caches mirrored from SQLite, refreshed on demand so the
        # hot per-frame loop never has to hit the DB just to compare vectors
        self._known_ids: list[int] = []
        self._known_names: list[str] = []
        self._known_encodings: list[np.ndarray] = []

        self._pending_cache: list[dict] = []          # {id, encoding, last_seen}
        self._last_attendance: dict[int, float] = {}   # user_id -> monotonic() timestamp

        self.reload_known_users()
        self.reload_pending_queue()

    # ------------------------------------------------------------------ #
    # cache management (call these after any write to the two tables)
    # ------------------------------------------------------------------ #
    def reload_known_users(self):
        with db_cursor() as cur:
            cur.execute("SELECT id, full_name, encoding FROM registered_users")
            rows = cur.fetchall()
        self._known_ids = [r["id"] for r in rows]
        self._known_names = [r["full_name"] for r in rows]
        self._known_encodings = [blob_to_encoding(r["encoding"]) for r in rows]

    def reload_pending_queue(self):
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, encoding, last_seen_at FROM pending_queue WHERE status='pending'"
            )
            rows = cur.fetchall()
        self._pending_cache = [
            {"id": r["id"], "encoding": blob_to_encoding(r["encoding"]), "last_seen": r["last_seen_at"]}
            for r in rows
        ]

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
        user_id = self._match_known(encoding)
        if user_id is not None:
            self._maybe_log_attendance(user_id)
            return

        pending_id = self._match_pending(encoding)
        if pending_id is not None:
            self._touch_pending(pending_id)
            return

        self._enqueue_new_face(encoding, location_small, full_frame)

    def _match_known(self, encoding: np.ndarray) -> int | None:
        if not self._known_encodings:
            return None
        distances = face_recognition.face_distance(self._known_encodings, encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= config.KNOWN_USER_TOLERANCE:
            return self._known_ids[best_idx]
        return None

    def _match_pending(self, encoding: np.ndarray) -> int | None:
        if not self._pending_cache:
            return None
        now = datetime.now(timezone.utc)
        candidates = [
            p for p in self._pending_cache
            if (now - datetime.fromisoformat(p["last_seen"])).total_seconds()
            <= config.PENDING_DEDUP_WINDOW_SECONDS
        ]
        if not candidates:
            return None
        distances = face_recognition.face_distance([c["encoding"] for c in candidates], encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= config.PENDING_DEDUP_TOLERANCE:
            return candidates[best_idx]["id"]
        return None

    def _touch_pending(self, pending_id: int):
        now = _now_iso()
        with db_cursor(commit=True) as cur:
            cur.execute("UPDATE pending_queue SET last_seen_at=? WHERE id=?", (now, pending_id))
        for p in self._pending_cache:
            if p["id"] == pending_id:
                p["last_seen"] = now
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

        self._pending_cache.append({"id": new_id, "encoding": encoding, "last_seen": now})

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
        now_mono = time.monotonic()
        last = self._last_attendance.get(user_id)
        if last is not None and (now_mono - last) < config.ATTENDANCE_COOLDOWN_SECONDS:
            return  # throttled: same member seen again too soon
        self._last_attendance[user_id] = now_mono

        with db_cursor(commit=True) as cur:
            cur.execute("INSERT INTO attendance_log (user_id) VALUES (?)", (user_id,))
            cur.execute("SELECT full_name FROM registered_users WHERE id=?", (user_id,))
            row = cur.fetchone()

        full_name = row["full_name"] if row else "?"
        self._broadcast_soon({
            "event": "checkin",
            "user_id": user_id,
            "full_name": full_name,
            "checkin_at": _now_iso(),
        })

    # ------------------------------------------------------------------ #
    # bridge: background thread -> asyncio event loop
    # ------------------------------------------------------------------ #
    def _broadcast_soon(self, message: dict):
        asyncio.run_coroutine_threadsafe(manager.broadcast(message), self._loop)
