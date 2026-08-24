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
# in front of the camera" (anti-spam), not for real identification.
PENDING_DEDUP_TOLERANCE = 0.5

# Anti-spam window from the spec: ignore repeated captures of the same unknown
# face for this many seconds after it was first/last seen.
PENDING_DEDUP_WINDOW_SECONDS = 120  # 2 minutes

# Extra (not explicitly requested, but recommended) throttle so a *known* member
# standing near the camera doesn't spam attendance_log with a row per second.
ATTENDANCE_COOLDOWN_SECONDS = 300  # 5 minutes

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
