"""
Core recognition engine.

Runs entirely in a background `threading.Thread` (NOT an asyncio task) because
cv2.VideoCapture().read() and the recognizer's detection/embedding calls are
blocking, CPU-bound calls -- running them inside the asyncio event loop would
freeze the whole FastAPI server (including the WebSocket updates) while a
frame is being processed.

The face library itself (DeepFace, or the older dlib) lives behind
app/recognizer/: this file only asks it "which faces are in this frame, and what
is each one's vector" and compares vectors. Swapping the library or model never
touches the logic below.

Flow per processed frame:
  1. Grab frame from the webcam, find faces + their vectors (recognizer).
  2. If at least one face was found, immediately grab RECOGNITION_BURST_FRAMES-1
     more frames (right then, not waiting for the next scheduled frame) and
     detect+embed faces in those too, folding each one into whichever
     "sighting" (from step 1) its face-box position is closest to.
  3. For each sighting (now backed by up to RECOGNITION_BURST_FRAMES vectors):
       a. Compare against every ENROLLED member (registered_users) -- taking
          whichever single vector gave the closest match, not an average.
          -> match: log attendance (throttled, alternates ورود/خروج), write
             a "لاگ زنده" line, and stop.
       b. Compare against faces already waiting in pending_queue that were
          seen within the last PENDING_DEDUP_WINDOW_SECONDS.
          -> match: just bump last_seen_at (anti-spam, no new row/photo),
             and write a "لاگ زنده" line for this unknown face too (debounced
             per pending_id, same cadence as a known match).
          NOTE: each pending entry keeps a small ROLLING WINDOW of its most
          recent vectors (not just the first one it was created with), so
          natural pose/expression drift while someone stands at the camera
          doesn't cause them to "stop matching" and get re-enqueued as a
          brand-new face.
       c. Otherwise: this is a brand-new unknown face -> crop + save a photo,
          insert a pending_queue row, push a "new_pending" WebSocket event so
          the reception panel updates live, AND write a "لاگ زنده" line.

Only vectors made by the ACTIVE recognizer model are ever compared (every stored
vector is tagged with its model); see app/recognizer/base.py.
"""
from __future__ import annotations

import asyncio
import hashlib
import glob
import hmac
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import httpx

from . import config
from .database import db_cursor
from .face_samples import add_face_sample, blob_to_encoding, encoding_to_blob, now_iso as _now_iso  # noqa: F401  (re-exported: routers import these from here)
from .recognizer import Recognizer, get_recognizer
from .reembed import describe as _describe_reembed, reembed_missing_members
from .ws_manager import manager


class FaceEngine:
    def __init__(self, loop: asyncio.AbstractEventLoop, recognizer: Recognizer | None = None):
        self._loop = loop
        self._thread: threading.Thread | None = None
        self._running = False
        self._video: cv2.VideoCapture | None = None  # set in _run_loop; read by _capture_burst_frames too

        # The face library (DeepFace / dlib). Injectable so tests can use a fake.
        self._recognizer: Recognizer = recognizer if recognizer is not None else get_recognizer()
        self._tolerances = self._recognizer.tolerances()

        # False until the recognizer's models are loaded (first start may download weights) and
        # member vectors are rebuilt for the active model; recognizer_error says why not, if it failed.
        self.recognizer_ready: bool = False
        self.recognizer_error: str | None = None

        # in-memory caches mirrored from SQLite, refreshed on demand so the
        # hot per-frame loop never has to hit the DB just to compare vectors.
        # Each member can have MULTIPLE samples (member_face_samples table),
        # so these are parallel: _known_sample_user_ids[i] is the owner of row i
        # of _known_matrix. A member with 8 samples simply appears 8 times, once
        # per sample -- the recognizer compares the incoming face against all of
        # them at once and we take whichever single sample is closest, from
        # whichever member. Only samples of the ACTIVE model are loaded.
        self._known_sample_user_ids: list[int] = []
        self._known_matrix: np.ndarray | None = None

        # True once cv2.VideoCapture is actually open and giving frames.
        # Plain bool, no lock -- same as `_running`/`_video` below, read from
        # request-handling threads but only ever WRITTEN from this class's own
        # background thread, and CPython's GIL makes a single attribute
        # read/write atomic enough for a status flag like this.
        self.camera_connected: bool = False
        # Why the camera is not connected right now (None while connected / not yet tried):
        #   device_missing | no_permission | open_failed | no_frames
        # Shown in the panel's banner so the manager does not have to read logs.
        self.camera_problem: str | None = None

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

        # same idea, but keyed by pending_id for UNKNOWN-face log entries
        # (see _maybe_log_unknown_recognition) -- kept separate from the dict
        # above so a member id and a pending id sharing the same integer can
        # never suppress each other's debounce timer.
        self._last_unknown_recognition_log: dict[int, datetime] = {}

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
            cur.execute(
                "SELECT user_id, encoding FROM member_face_samples WHERE model=?",
                (self._recognizer.model_id,),
            )
            rows = cur.fetchall()
        vectors = [blob_to_encoding(r["encoding"]) for r in rows]
        # a corrupt/odd-length blob must not make np.vstack (and with it the whole camera loop) fail
        dim = max((len(v) for v in vectors), default=0)
        keep = [i for i, v in enumerate(vectors) if len(v) == dim]
        self._known_sample_user_ids = [rows[i]["user_id"] for i in keep]
        self._known_matrix = np.vstack([vectors[i] for i in keep]) if keep else None

    def reload_pending_queue(self):
        with db_cursor() as cur:
            cur.execute(
                "SELECT id, encoding, last_seen_at FROM pending_queue WHERE status='pending' AND model=?",
                (self._recognizer.model_id,),
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
        self._last_unknown_recognition_log = {}

    def forget_unknown_debounce(self, pending_id: int):
        """
        Called by routers/queue.py whenever a pending_queue row leaves
        'pending' status (registered, assigned, or rejected). Without this,
        _last_unknown_recognition_log would keep one entry PER PENDING_ID
        EVER CREATED for the lifetime of the process -- harmless for
        matching (it's never read after the row is resolved), but pure
        memory growth over a long uptime. Safe to call even if the id was
        never in there.
        """
        self._last_unknown_recognition_log.pop(pending_id, None)

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
        """
        Outer loop owns "is a camera connected at all"; never exits just
        because one isn't, right now. No camera at startup (unplugged,
        wrong CAMERA_INDEX, `docker compose up` without the camera override --
        see docker-compose.camera.yml) used to permanently kill this thread
        (self._running = False) -- the rest of the app still ran fine, but
        recognition was silently dead forever with no indication why. Now it
        just keeps retrying every config.CAMERA_RETRY_SECONDS and reports its
        state via self.camera_connected (see _set_camera_connected), which
        the panel shows as a banner on the reception pages.
        """
        video = None
        frame_count = 0
        read_failures = 0
        try:
            if not self._prepare_recognizer():
                return  # stop() was called while still loading
            while self._running:
                if video is None:
                    video = self._try_open_camera()
                    if video is None:
                        time.sleep(config.CAMERA_RETRY_SECONDS)
                        continue
                    self._video = video  # so _capture_burst_frames (via _process_frame) can read more

                ok, frame = video.read()
                if not ok:
                    # A single dropped frame happens occasionally even on a
                    # healthy camera (skip it, keep going). But if the device
                    # is no longer open (unplugged mid-run), OR it stays open
                    # while delivering nothing for a while (e.g. another program
                    # grabbed it), drop it and go back to the retry branch above
                    # instead of spinning on a dead handle forever -- and without
                    # still claiming "connected".
                    read_failures += 1
                    if not video.isOpened() or read_failures >= config.CAMERA_MAX_READ_FAILURES:
                        video.release()
                        video = None
                        self._video = None
                        read_failures = 0
                        self._set_camera_connected(False, "no_frames")
                        time.sleep(config.CAMERA_RETRY_SECONDS)
                    else:
                        time.sleep(0.05)
                    continue
                read_failures = 0

                frame_count += 1
                if frame_count % config.PROCESS_EVERY_N_FRAMES != 0:
                    continue

                try:
                    self._process_frame(frame)
                except Exception as exc:  # never let one bad frame kill the thread
                    print(f"[face_engine] frame processing error: {exc}")
        finally:
            self._video = None
            if video is not None:
                video.release()
            self._set_camera_connected(False)

    def _sleep_while_running(self, seconds: float):
        """Sleep, but wake up promptly when stop() is called."""
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            time.sleep(0.2)

    def _prepare_recognizer(self) -> bool:
        """
        Loads the recognizer's models (the very first start may download their weights, which needs
        internet once), then rebuilds member vectors for the active model from their stored photos
        (reembed.py; a no-op when nothing is missing). Retries every CAMERA_RETRY_SECONDS if the
        models cannot be loaded -- e.g. first start while offline -- instead of dying silently, and
        reports the reason in self.recognizer_error / the "recognizer_status" WebSocket event.
        Returns False only if stop() was called before it finished.
        """
        while self._running:
            try:
                self._recognizer.warm_up()
                break
            except Exception as exc:
                self._set_recognizer_state(False, f"{type(exc).__name__}: {exc}")
                self._sleep_while_running(config.CAMERA_RETRY_SECONDS)
        else:
            return False

        try:
            report = reembed_missing_members(self._recognizer, should_stop=lambda: not self._running)
            if report.migrated or report.needs_attention:
                print(f"[face_engine] {_describe_reembed(report, self._recognizer.model_id)}")
            if report.migrated:
                self.reload_known_users()
        except Exception as exc:  # recognition itself still works for everyone who has samples
            print(f"[face_engine] re-embedding member photos failed: {type(exc).__name__}: {exc}")

        self._set_recognizer_state(True, None)
        return self._running

    def _set_recognizer_state(self, ready: bool, error: str | None):
        """Edge-triggered like _set_camera_connected: logs/broadcasts only on a real change."""
        if ready == self.recognizer_ready and error == self.recognizer_error:
            return
        self.recognizer_ready = ready
        self.recognizer_error = error
        if ready:
            print(f"[face_engine] recognizer {self._recognizer.model_id} ready")
        else:
            print(f"[face_engine] recognizer {self._recognizer.model_id} not ready, will retry every "
                  f"{config.CAMERA_RETRY_SECONDS}s: {error}")
        self._broadcast_soon({"event": "recognizer_status", "ready": ready, "error": error})

    def _diagnose_open_failure(self) -> str:
        """Best guess at WHY the camera could not be opened, as a problem code (see camera_problem)."""
        if sys.platform.startswith("linux"):
            path = f"/dev/video{config.CAMERA_INDEX}"
            if not os.path.exists(path):
                return "device_missing"
            if not os.access(path, os.R_OK | os.W_OK):
                return "no_permission"
        return "open_failed"

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Opens config.CAMERA_INDEX. On Linux the V4L2 backend is asked for explicitly first (OpenCV's
        automatic choice can pick a backend that cannot read some webcams), then OpenCV's default."""
        backends: list = [None]
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            backends = [cv2.CAP_V4L2, None]
        for backend in backends:
            video = cv2.VideoCapture(config.CAMERA_INDEX) if backend is None else cv2.VideoCapture(config.CAMERA_INDEX, backend)
            if video.isOpened():
                if config.CAMERA_FOURCC:
                    video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.CAMERA_FOURCC))
                if config.CAMERA_WIDTH > 0:
                    video.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                if config.CAMERA_HEIGHT > 0:
                    video.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
                return video
            video.release()
        return None

    def _try_open_camera(self) -> cv2.VideoCapture | None:
        """One attempt to (re)open config.CAMERA_INDEX. Returns the opened
        capture, or None (and flips camera_connected to False) if it's not
        available right now -- caller just waits and retries. "Opened" is not
        enough: the camera must also hand over a real frame, otherwise (device
        busy, unsupported format, ...) it would be reported as connected while
        nothing ever arrives."""
        video = self._open_capture()
        if video is None:
            self._set_camera_connected(False, self._diagnose_open_failure())
            return None
        for _ in range(config.CAMERA_OPEN_READ_ATTEMPTS):
            ok, _frame = video.read()
            if ok:
                self._set_camera_connected(True)
                return video
            time.sleep(0.1)
        video.release()
        self._set_camera_connected(False, "no_frames")
        return None

    _PROBLEM_HINTS = {
        "device_missing": "the device does not exist in this process. In Docker, check that /dev/video* exists on the host "
                          "(ls /dev/video*) and that the /dev bind in docker-compose.yml is applied; fallback: docker-compose.camera.yml",
        "no_permission": "the device exists but this user cannot read/write it (check the 'video' group / permissions)",
        "open_failed": "the device could not be opened (another program may be using it, or CAMERA_INDEX is wrong)",
        "no_frames": "the device opened but delivers no frames (busy -- close ffplay/browser tabs using the camera -- "
                     "or an unsupported format; try CAMERA_FOURCC=MJPG)",
    }

    def _set_camera_connected(self, connected: bool, problem: str | None = None):
        """Edge-triggered: only logs/broadcasts on an actual change (of the connected flag OR of the
        reason), so the retry loop polling every CAMERA_RETRY_SECONDS doesn't spam the log or the
        WebSocket while the camera stays missing."""
        problem = None if connected else problem
        if connected == self.camera_connected and problem == self.camera_problem:
            return
        self.camera_connected = connected
        self.camera_problem = problem
        if connected:
            print(f"[face_engine] camera index {config.CAMERA_INDEX} connected")
        else:
            seen = sorted(glob.glob("/dev/video*")) if sys.platform.startswith("linux") else []
            print(f"[face_engine] camera index {config.CAMERA_INDEX} not available "
                  f"-- will keep retrying every {config.CAMERA_RETRY_SECONDS}s")
            if problem:
                print(f"[face_engine]   reason ({problem}): {self._PROBLEM_HINTS.get(problem, '')}"
                      + (f" | video devices visible here: {', '.join(seen) or 'none'}" if sys.platform.startswith("linux") else ""))
        self._broadcast_soon({"event": "camera_status", "connected": connected, "problem": problem})

    def _process_frame(self, frame: np.ndarray):
        faces = self._recognizer.detect_and_embed(frame)
        if not faces:
            return

        # one "sighting" per face found in THIS frame; extra burst frames
        # below just add more vectors onto whichever sighting they're
        # closest to, they never start new ones
        sightings = [{"location": f.box, "encodings": [f.embedding]} for f in faces]

        for _ in range(config.RECOGNITION_BURST_FRAMES - 1):
            if self._video is None:
                break
            ok, extra_frame = self._video.read()
            if not ok:
                continue
            for face in self._recognizer.detect_and_embed(extra_frame):
                sighting = self._closest_sighting(sightings, face.box)
                if sighting is not None:
                    sighting["encodings"].append(face.embedding)
                # if nothing is close enough, this extra face is dropped --
                # it's either someone else who just stepped into frame, or
                # the original person moved too much; safer to ignore it
                # than to risk folding a different face into this sighting

        for sighting in sightings:
            self._handle_face(sighting["encodings"], sighting["location"], frame)

    @staticmethod
    def _closest_sighting(sightings: list[dict], location: tuple, max_relative_distance: float = 0.6):
        """
        Finds whichever existing sighting's face-box center is nearest to
        `location`, so a face seen a moment later in a burst frame gets
        folded into "the same person's" encoding list instead of starting a
        new one. The allowed radius scales with face size (max_relative_distance
        x box width) rather than being a fixed pixel count, so it behaves the
        same whether someone is standing close to the camera (big box) or far
        (small box). Returns None if nothing is close enough -- e.g. the
        original person stepped out of frame, or someone else walked in.
        """
        top, right, bottom, left = location
        cx, cy = (left + right) / 2, (top + bottom) / 2
        width = max(right - left, 1)

        best, best_dist = None, width * max_relative_distance
        for sighting in sightings:
            st, sr, sb, sl = sighting["location"]
            scx, scy = (sl + sr) / 2, (st + sb) / 2
            dist = ((cx - scx) ** 2 + (cy - scy) ** 2) ** 0.5
            if dist < best_dist:
                best, best_dist = sighting, dist
        return best

    # ------------------------------------------------------------------ #
    # per-face decision tree
    # ------------------------------------------------------------------ #
    def _handle_face(self, encodings: list[np.ndarray], box: tuple, full_frame: np.ndarray):
        user_id, distance = self._match_known(encodings)
        if user_id is not None:
            self._maybe_log_attendance(user_id)
            self._maybe_log_recognition(user_id, distance)
            return

        # the unknown/pending path only ever needed one representative
        # encoding -- use the ORIGINAL (first) frame's, for consistency with
        # the face crop that gets saved from `full_frame` below
        encoding = encodings[0]
        pending_id = self._match_pending(encoding)
        if pending_id is not None:
            self._touch_pending(pending_id, encoding)
            self._maybe_log_unknown_recognition(pending_id, distance)
            return

        pending_id = self._enqueue_new_face(encoding, box, full_frame)
        self._maybe_log_unknown_recognition(pending_id, distance)

    def _match_known(self, encodings: list[np.ndarray]) -> tuple[int | None, float | None]:
        """
        Compares EVERY supplied encoding (1 to config.RECOGNITION_BURST_FRAMES
        of them, all of the same sighting) against every sample of every
        member, and returns (user_id, distance) for whichever single
        (encoding, stored-sample) pairing was closest overall. Deliberately
        "best of N", not an average: a good frame should be able to rescue a
        match even if another frame in the same burst was blurry/off-angle.

        ALWAYS returns a distance when there's at least one enrolled member,
        even if it didn't clear tolerance (user_id is then None) -- this is
        used to log "closest known match was 0.58 away" on unknown-face log
        entries, which is handy for tuning KNOWN_USER_TOLERANCE later.
        """
        known = self._known_matrix  # one reference, so a concurrent reload can't swap it mid-loop
        user_ids = self._known_sample_user_ids
        if known is None or len(user_ids) != len(known):
            return None, None

        best_user_id, best_distance = None, None
        for encoding in encodings:
            distances = self._recognizer.distances(known, encoding)
            idx = int(np.argmin(distances))
            distance = float(distances[idx])
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_user_id = user_ids[idx]

        if best_distance is not None and best_distance <= self._tolerances.known:
            return best_user_id, best_distance
        return None, best_distance

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
        dim = len(encoding)
        for p in self._pending_cache:
            if (now - datetime.fromisoformat(p["last_seen"])).total_seconds() > config.PENDING_DEDUP_WINDOW_SECONDS:
                continue
            for sample in p["encodings"]:
                if len(sample) != dim:
                    continue
                flat_ids.append(p["id"])
                flat_encodings.append(sample)

        if not flat_encodings:
            return None

        distances = self._recognizer.distances(np.vstack(flat_encodings), encoding)
        best_idx = int(np.argmin(distances))
        if distances[best_idx] <= self._tolerances.pending:
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

    def _enqueue_new_face(self, encoding: np.ndarray, box: tuple, full_frame: np.ndarray) -> int:
        now = _now_iso()
        photo_path = self._save_face_crop(full_frame, box)

        with db_cursor(commit=True) as cur:
            cur.execute(
                """INSERT INTO pending_queue (encoding, model, photo_path, first_seen_at, last_seen_at, status)
                   VALUES (?, ?, ?, ?, ?, 'pending')""",
                (encoding_to_blob(encoding), self._recognizer.model_id, str(photo_path), now, now),
            )
            new_id = cur.lastrowid

        self._pending_cache.append({"id": new_id, "encodings": [encoding], "last_seen": now})

        self._broadcast_soon({
            "event": "new_pending",
            "id": new_id,
            "photo_url": f"/static/pending_faces/{photo_path.name}",
            "first_seen_at": now,
        })

        return new_id

    def _save_face_crop(self, full_frame: np.ndarray, box: tuple):
        # `box` is (top, right, bottom, left) in pixels of the FULL frame (the recognizer already
        # scaled back whatever shrinking it did internally)
        top, right, bottom, left = box

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

    def _maybe_log_unknown_recognition(self, pending_id: int, distance: float | None):
        """
        Same live-log feed as _maybe_log_recognition above, but for a face
        that did NOT match any enrolled member -- either a brand-new pending
        entry or one still waiting in the queue. Kept in a SEPARATE debounce
        dict keyed by pending_id (not user_id) so a member id and a pending
        id that happen to share the same integer can never collide/suppress
        each other's log lines.

        `distance` is the nearest known-member distance found anyway (see
        _match_known's ALWAYS-return-a-distance behavior), so an operator
        can see e.g. "closest known member was 0.58 away" -- useful for
        judging whether KNOWN_USER_TOLERANCE needs adjusting -- and is simply
        omitted (NULL) if there are no enrolled members at all yet.
        """
        now = datetime.now(timezone.utc)
        last_at = self._last_unknown_recognition_log.get(pending_id)
        if last_at is not None and (now - last_at).total_seconds() < config.RECOGNITION_LOG_DEBOUNCE_SECONDS:
            return
        self._last_unknown_recognition_log[pending_id] = now

        label = f"چهرهٔ ناشناس #{pending_id}"
        with db_cursor(commit=True) as cur:
            cur.execute(
                "INSERT INTO recognition_log (user_id, full_name, distance, created_at) VALUES (NULL, ?, ?, ?)",
                (label, distance, now.isoformat()),
            )

        self._broadcast_soon({
            "event": "recognition_seen",
            "user_id": None,
            "full_name": label,
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
