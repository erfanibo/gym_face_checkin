"""
Authentication & authorization.

Only two roles exist, on purpose (see PROJECT_OVERVIEW.md -- the gym owner
handles both reception and management, so there's no separate "reception"
role):

  - "manager": one shared account. Full access to every endpoint (the merged
    reception+admin panel).
  - "member":  one identity per row in registered_users, used only by the
    member panel to view/act on THEIR OWN data. Can never touch another
    member's rows -- routes that use require_member get back a user_id taken
    from the signed session, never from anything the client sends.

Sessions, not JWT: Starlette's SessionMiddleware (added in main.py) stores
role/user_id in a signed-and-tamper-proof cookie. No extra crypto dependency,
no server-side session table to manage -- more than enough for one gym on
one local network.
"""
import hashlib
import hmac
import secrets

from fastapi import HTTPException, Request

from . import config
from .database import db_cursor, get_setting, set_setting

MANAGER_PASSWORD_SETTING_KEY = "manager_password_hash"

# OWASP's 2023-ish recommendation for PBKDF2-HMAC-SHA256. Pure stdlib
# (hashlib), so no bcrypt/argon2/passlib dependency needed for this project's
# scale (one manager account + a couple hundred members).
_PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    """Returns 'salt_hex$digest_hex'."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest_hex = stored.split("$", 1)
    except ValueError:
        return False  # malformed/legacy value -- fail closed, not open
    expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS)
    # constant-time compare so a timing attack can't leak the hash byte-by-byte
    return hmac.compare_digest(expected.hex(), digest_hex)


# ---------------------------------------------------------------------------
# Manager account (single, stored in app_settings)
# ---------------------------------------------------------------------------
def ensure_manager_password_seeded() -> None:
    """
    Called once at startup (see main.py's lifespan). If no manager password
    has EVER been set, seeds one from config.MANAGER_INITIAL_PASSWORD so a
    fresh install always has something to log in with. Never overwrites an
    existing hash -- if the manager already changed their password, this is
    a no-op forever after.
    """
    if get_setting(MANAGER_PASSWORD_SETTING_KEY) is None:
        set_setting(MANAGER_PASSWORD_SETTING_KEY, hash_password(config.MANAGER_INITIAL_PASSWORD))


def verify_manager_password(password: str) -> bool:
    stored = get_setting(MANAGER_PASSWORD_SETTING_KEY)
    if stored is None:
        return False
    return verify_password(password, stored)


def set_manager_password(new_password: str) -> None:
    set_setting(MANAGER_PASSWORD_SETTING_KEY, hash_password(new_password))


# ---------------------------------------------------------------------------
# Member login (phone + membership_code -- both already exist per member,
# no new column needed for this first version)
# ---------------------------------------------------------------------------
def find_member_for_login(phone: str, membership_code: str):
    """
    Returns the registered_users row if phone + membership_code match an
    existing member, else None. Both are required: membership_code alone
    isn't a secret (an operator can see it on-screen at signup), so phone
    acts as the second factor.
    """
    phone = (phone or "").strip()
    membership_code = (membership_code or "").strip()
    if not phone or not membership_code:
        return None
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM registered_users WHERE phone = ? AND membership_code = ?",
            (phone, membership_code),
        )
        return cur.fetchone()


# ---------------------------------------------------------------------------
# FastAPI dependencies -- add to a router/route to require a role.
# Usage:  router = APIRouter(..., dependencies=[Depends(auth.require_manager)])
#         @router.get("/me/plan")
#         def my_plan(user_id: int = Depends(auth.require_member)): ...
# ---------------------------------------------------------------------------
def require_manager(request: Request) -> None:
    if request.session.get("role") != "manager":
        raise HTTPException(401, "ورود مدیر لازم است")


def require_member(request: Request) -> int:
    """Returns the logged-in member's user_id straight from the signed
    session -- never trust a user_id passed in the request body/query for
    "which member is this" checks."""
    if request.session.get("role") != "member":
        raise HTTPException(401, "ورود عضو لازم است")
    return request.session["user_id"]


def current_identity(request: Request) -> dict:
    """Backing GET /api/auth/me. Never raises -- just reports what's there so
    the frontend can decide which panel/login-screen to show on page load."""
    role = request.session.get("role")
    if role == "manager":
        return {"role": "manager"}
    if role == "member":
        return {"role": "member", "user_id": request.session.get("user_id")}
    return {"role": None}
