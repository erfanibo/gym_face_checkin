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
FRAME_RESIZE_SCALE = 0.25        # dlib backend ONLY: frame is downscaled by this factor before detection
                                  # (4x smaller frame ~= 16x faster HOG/CNN detection). The DeepFace
                                  # backend uses DEEPFACE_MAX_FRAME_SIDE instead (see below).

# If no camera is found at CAMERA_INDEX (not plugged in, container started
# without the device passed through, wrong index, ...), FaceEngine does NOT
# crash or give up -- it just sits in a retry loop, trying to (re)open the
# camera every this-many seconds, and reports camera_connected=False in the
# meantime (surfaced to the panel -- see app/routers/queue.py's
# /camera-status and the "camera_status" WebSocket event). This means the
# whole app (and `docker compose up`, once docker-compose.yml's camera
# `devices:` passthrough is also made optional -- see docker-compose.camera.yml)
# runs fine on a machine with no camera at all; recognition simply stays
# paused until one is connected, with no restart needed.
CAMERA_RETRY_SECONDS = 5

# Optional tuning for cameras that OpenCV does not read well with its defaults. All off by default.
#   CAMERA_FOURCC="MJPG"  -> ask the camera for MJPG (fixes many USB cameras that open but give no frames
#                            at the default format, or are too slow in YUYV). "" = leave the camera's own default.
#   CAMERA_WIDTH / CAMERA_HEIGHT -> requested resolution (0 = camera default).
CAMERA_FOURCC = os.environ.get("CAMERA_FOURCC", "").strip()[:4]
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "0") or 0)
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "0") or 0)
# A freshly opened camera must deliver a real frame within this many reads (0.1 s apart) to count as connected;
# and a connected camera that fails this many reads IN A ROW (~50 ms apart) is treated as lost and reopened.
CAMERA_OPEN_READ_ATTEMPTS = 20
CAMERA_MAX_READ_FAILURES = 50

# ---------------------------------------------------------------------------
# Recognition backend (see app/recognizer/)
# ---------------------------------------------------------------------------
# Which library turns a camera frame into "faces + 1-D vectors":
#   "deepface" (default) -- DeepFace. SFace + YuNet by default: both run on
#                           onnxruntime/OpenCV (no TensorFlow, no GPU needed).
#   "dlib"               -- the previous engine (face_recognition). Kept for A/B comparison and
#                           as a way back; needs `pip install -r requirements-dlib.txt`.
# Vectors from different models are NOT comparable, so every stored vector is tagged with the
# model that made it and the engine only loads vectors of the active backend.
RECOGNIZER_BACKEND = os.environ.get("RECOGNIZER_BACKEND", "deepface").strip().lower()

# DeepFace settings (ignored by the dlib backend).
#   DEEPFACE_MODEL    embedding model: SFace (light, default), Facenet512, ArcFace, Buffalo_L, ...
#                     Pick with scripts/benchmark_deepface.py on the real machine.
#   DEEPFACE_DETECTOR face detector: yunet (light, default), opencv, retinaface (slow), ...
DEEPFACE_MODEL = os.environ.get("DEEPFACE_MODEL", "SFace").strip()
DEEPFACE_DETECTOR = os.environ.get("DEEPFACE_DETECTOR", "yunet").strip()
# Runtime DeepFace uses to run the models: "onnx" (default: onnxruntime, light, fast on CPU, no
# TensorFlow needed) | "tensorflow" | "pytorch" (those need their own packages installed).
# Without this DeepFace silently switches to TensorFlow the moment it happens to be installed.
DEEPFACE_BACKEND_ENGINE = os.environ.get("DEEPFACE_BACKEND_ENGINE", "onnx").strip().lower()
# Faces the detector is less sure about than this (0..1) are ignored -- stops hands/posters/
# background blobs from becoming "unknown people" in the queue.
DEEPFACE_MIN_FACE_CONFIDENCE = float(os.environ.get("DEEPFACE_MIN_FACE_CONFIDENCE", "0.85"))
# Faces narrower than this many pixels (in the full frame) are ignored: too small to recognise
# reliably, someone far behind the door. 0 disables.
DEEPFACE_MIN_FACE_PIXELS = int(os.environ.get("DEEPFACE_MIN_FACE_PIXELS", "60"))
# Frames whose longer side exceeds this are downscaled before detection (keeps CPU use bounded
# on 1080p cameras). Unlike the dlib path this is NOT a 4x shrink: the embedding is cut out of
# this image, so shrinking too much throws away exactly the detail recognition needs.
DEEPFACE_MAX_FRAME_SIDE = int(os.environ.get("DEEPFACE_MAX_FRAME_SIDE", "960"))


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else None


# ---------------------------------------------------------------------------
# Face matching
# ---------------------------------------------------------------------------
# "Distance" between two face vectors; lower = more similar. The scale depends on the
# backend/model (dlib: euclidean, ~0.6 is its own default; DeepFace: cosine, each model has
# its own threshold), so the three values below are None by default = "use the backend's own
# defaults" (app/recognizer/base.py, derived from the model's standard threshold). Set them in
# .env (e.g. KNOWN_USER_TOLERANCE=0.45) only after testing with real users/lighting, and
# remember a value tuned for one model is meaningless for another.
#
# KNOWN_USER_TOLERANCE: a face closer than this to a member's sample IS that member. Stricter
#   than the library default on purpose: a false positive at an entrance door is worse.
KNOWN_USER_TOLERANCE = _optional_float("KNOWN_USER_TOLERANCE")

# Threshold used ONLY to decide "is this the same unknown person still standing
# in front of the camera" (anti-spam), not for real identification. Slightly
# looser than KNOWN_USER_TOLERANCE on purpose: a false positive here just means
# skipping one duplicate photo (harmless), while a false negative means the
# same person gets re-photographed and re-queued repeatedly (the bug we saw).
PENDING_DEDUP_TOLERANCE = _optional_float("PENDING_DEDUP_TOLERANCE")

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
# second membership record for the same physical person. Defaults to the same
# value as KNOWN_USER_TOLERANCE since it's asking the same question ("is this
# really the same face as an existing member?"), just named separately so the
# two can be tuned independently later if needed.
REGISTRATION_DUPLICATE_FACE_TOLERANCE = _optional_float("REGISTRATION_DUPLICATE_FACE_TOLERANCE")

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
MAX_FACE_SAMPLES_PER_MEMBER = 15

# ---------------------------------------------------------------------------
# Multi-frame ("burst") recognition
# ---------------------------------------------------------------------------
# The moment a face is found in a processed frame, this many consecutive
# frames (including that one) are captured RIGHT THEN and each compared
# against every known member. Whichever single (frame, stored-sample) pairing
# comes back closest wins -- this is a "best of N attempts" strategy, not an
# average: one bad frame (blink, motion blur, an odd angle for a split
# second) can't drag down two good ones. Only kicks in once a face is
# actually detected, so it adds zero overhead while the camera sees nobody.
# Set to 1 to fully disable and fall back to single-frame matching.
RECOGNITION_BURST_FRAMES = 3

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

# ---------------------------------------------------------------------------
# Update check (Settings > System update). Only runs when the manager presses the
# "check for updates" button -- nothing is ever sent in the background, so the
# system stays fully local otherwise. It asks GitHub for the latest *published
# Release* of this repo; it never downloads or installs anything.
#   UPDATE_CHECK_REPO     "owner/repo" to check; set to an empty string to disable.
#   UPDATE_CHECK_API_BASE GitHub API root (change only for GitHub Enterprise / tests).
# ---------------------------------------------------------------------------
UPDATE_CHECK_REPO = os.environ.get("UPDATE_CHECK_REPO", "erfanibo/gym_face_checkin").strip() or None
UPDATE_CHECK_API_BASE = os.environ.get("UPDATE_CHECK_API_BASE", "https://api.github.com").strip().rstrip("/")
UPDATE_CHECK_TIMEOUT_SECONDS = float(os.environ.get("UPDATE_CHECK_TIMEOUT_SECONDS", "8"))
WEBHOOK_RETRY_DELAY_SECONDS = float(os.environ.get("WEBHOOK_RETRY_DELAY_SECONDS", "2"))

# ---------------------------------------------------------------------------
# Auth / sessions
# ---------------------------------------------------------------------------
# Signs the session cookie (see main.py's SessionMiddleware). MUST be set to a
# fixed value via env in Docker/production -- if left empty, a random key is
# generated per process start, which silently logs everyone out on every
# restart/redeploy. Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
SESSION_SECRET_KEY = os.environ.get("SESSION_SECRET_KEY", "").strip()
if not SESSION_SECRET_KEY:
    import secrets as _secrets
    SESSION_SECRET_KEY = _secrets.token_hex(32)

# How long a login (manager or member) stays valid before the cookie expires
# and the person has to log in again.
SESSION_MAX_AGE_SECONDS = int(os.environ.get("SESSION_MAX_AGE_SECONDS", str(60 * 60 * 24 * 30)))  # 30 days

# Seeded as the manager's password ONLY on first-ever run (when no password
# has been set yet in app_settings) -- change it from the Settings tab right
# after first login. Overriding this env var later has NO effect once a
# password already exists in the database.
MANAGER_INITIAL_PASSWORD = os.environ.get("MANAGER_INITIAL_PASSWORD", "").strip() or "changeme"

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
MEMBER_FRONTEND_DIR = BASE_DIR / "frontend_member"  # separate, member-only panel -- see routers/workouts.py
