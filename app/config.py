"""
Central configuration for the gym face check-in system.
Tune these values once the camera and lighting are in place; everything
else in the codebase reads from here so there is a single source of truth.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where persistent data (SQLite db + saved face photos) lives. Defaults to the
# project folder for local/dev use. In Docker this is overridden to /app/data
# and mounted as a volume, so data survives `docker compose down` / rebuilds
# and never gets baked into the image.
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))

# ---------------------------------------------------------------------------
# Camera / capture
# ---------------------------------------------------------------------------
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))  # index passed to cv2.VideoCapture
PROCESS_EVERY_N_FRAMES = 5       # only run face detection on 1 out of every N grabbed
                                  # frames -> keeps CPU usage sane on a normal PC
FRAME_RESIZE_SCALE = 0.25        # frame is downscaled by this factor before detection
                                  # (4x smaller frame ~= 16x faster HOG/CNN detection)

# ---------------------------------------------------------------------------
# Face matching
# ---------------------------------------------------------------------------
# face_recognition returns a "distance" between two 128-d encodings; lower = more similar.
# The library's own default tolerance is 0.6. We default a bit stricter (0.5) to reduce
# false positives at an entrance door. Adjust after testing with real users/lighting.
KNOWN_USER_TOLERANCE = 0.5

# Threshold used ONLY to decide "is this the same unknown person still standing
# in front of the camera" (anti-spam), not for real identification. Slightly
# looser than KNOWN_USER_TOLERANCE on purpose: a false positive here just means
# skipping one duplicate photo (harmless), while a false negative means the
# same person gets re-photographed and re-queued repeatedly (the bug we saw).
PENDING_DEDUP_TOLERANCE = 0.58

# How many of the most recent encodings to keep per pending queue entry for
# the comparison above. Comparing against several recent samples (not just
# the very first one) absorbs natural pose/expression drift while someone
# stands at the camera for a while.
PENDING_DEDUP_MAX_SAMPLES = 5

# Anti-spam window from the spec: ignore repeated captures of the same unknown
# face for this many seconds after it was first/last seen.
PENDING_DEDUP_WINDOW_SECONDS = 120  # 2 minutes

# Used at REGISTRATION time: if the face being registered already matches an
# existing member this closely, block the registration instead of creating a
# second membership record for the same physical person. Kept equal to
# KNOWN_USER_TOLERANCE since it's asking the same question ("is this really
# the same face as an existing member?"), just named separately so the two
# can be tuned independently later if needed.
REGISTRATION_DUPLICATE_FACE_TOLERANCE = KNOWN_USER_TOLERANCE

# Extra (not explicitly requested, but recommended) throttle so a *known* member
# standing near the camera doesn't spam attendance_log with a row per second.
ATTENDANCE_COOLDOWN_SECONDS = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Live recognition log ("لاگ زنده") — separate from attendance_log above.
# ---------------------------------------------------------------------------
# Every time a KNOWN member's face is matched in a camera frame, a row is
# written here immediately (no 5-minute wait) so an operator can open the
# live-log panel and see "X شناسایی شد" in near real time. Still debounced by
# a SMALL interval per member -- not to hide events, but because the camera
# re-matches the same standing person several times per second
# (see PROCESS_EVERY_N_FRAMES above), and a DB row + WebSocket broadcast for
# every single one of those would flood both the log and the database for no
# benefit. 4 seconds was chosen as a middle ground between "feels instant"
# and "doesn't spam the panel while someone just stands there".
RECOGNITION_LOG_DEBOUNCE_SECONDS = 4

# Upper bound on how many rows GET /api/recognition-log will ever return in
# one request, regardless of what the caller asks for.
RECOGNITION_LOG_MAX_LIMIT = 500

# ---------------------------------------------------------------------------
# Multi-sample recognition per member
# ---------------------------------------------------------------------------
# Each member can have several stored face vectors (member_face_samples table)
# instead of just one, so recognition still works from angles/lighting not
# seen at signup. Once a member hits this many samples, the OLDEST one is
# dropped whenever a new one is added (see face_engine.add_face_sample) --
# keeps the per-frame comparison cost bounded and rotates toward more recent
# appearances. Start conservative; raise later if 8 turns out to be too few
# for real-world angle/lighting variation.
MAX_FACE_SAMPLES_PER_MEMBER = 8

# ---------------------------------------------------------------------------
# Membership codes — auto-generated by the system, never typed by the operator
# ---------------------------------------------------------------------------
MEMBERSHIP_CODE_DIGITS = 6  # e.g. "482913"; gives 900,000 possible codes, plenty for <200 members

# ---------------------------------------------------------------------------
# Webhook — push each entry/exit event to another local server in real time
# ---------------------------------------------------------------------------
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "").strip() or None
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "").strip() or None
WEBHOOK_TIMEOUT_SECONDS = float(os.environ.get("WEBHOOK_TIMEOUT_SECONDS", "5"))
WEBHOOK_MAX_RETRIES = int(os.environ.get("WEBHOOK_MAX_RETRIES", "3"))
WEBHOOK_RETRY_DELAY_SECONDS = float(os.environ.get("WEBHOOK_RETRY_DELAY_SECONDS", "2"))

# ---------------------------------------------------------------------------
# Storage (all under DATA_DIR so a single volume mount persists everything)
# ---------------------------------------------------------------------------
STATIC_DIR = DATA_DIR / "static"
PENDING_PHOTOS_DIR = STATIC_DIR / "pending_faces"
USER_PHOTOS_DIR = STATIC_DIR / "user_photos"

for _dir in (PENDING_PHOTOS_DIR, USER_PHOTOS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "gym_face.db"

# frontend/ stays with the code (not user data), so it's addressed via BASE_DIR
FRONTEND_DIR = BASE_DIR / "frontend"
