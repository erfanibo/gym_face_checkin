"""
Authentication & authorization.

Only two roles exist, on purpose (see PROJECT_OVERVIEW.md -- the gym owner
handles both reception and management, so there's no separate "reception"
role):

  - "manager": one shared account. Full access to every endpoint (the merged
    reception+admin panel).
  - "member":  one identity per row in registered_users (phone + optional own password), used only by the
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
from .phone import members_with_phone

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
# Member login: phone number, plus the member's own password IF they set one.
#
# The system runs on the gym's local network, so only people connected to the gym router can even
# open the member panel. Because of that a member without a password can log in with just their
# phone number; the panel nags them to set a password, and once one exists it is always required.
# ---------------------------------------------------------------------------
class AmbiguousPhone(Exception):
    """The same phone number belongs to more than one member (only possible with old data)."""


_dummy_hash: str | None = None


def login_member(key: str, password: str | None):
    """
    `key` is a phone_key(). Returns the registered_users row on success, else None.
    A member with a password must supply it; a member without one gets in with the phone alone.
    """
    global _dummy_hash
    with db_cursor() as cur:
        candidates = members_with_phone(cur, key)
        rows = []
        for c in candidates:
            cur.execute("SELECT * FROM registered_users WHERE id=?", (c["id"],))
            rows.append(cur.fetchone())

    if not rows:
        # same work as a real password check, so "no such phone" and "wrong password" look alike
        if password:
            if _dummy_hash is None:
                _dummy_hash = hash_password(secrets.token_hex(8))
            verify_password(password, _dummy_hash)
        return None
    if len(rows) > 1:
        raise AmbiguousPhone()

    row = rows[0]
    stored = row["member_password_hash"]
    if stored is None:
        return row
    if not password or not verify_password(password, stored):
        return None
    return row


def set_member_password(user_id: int, new_password: str | None) -> bool:
    """Sets (or, with None, removes) a member's password. False if the member does not exist."""
    value = hash_password(new_password) if new_password is not None else None
    with db_cursor(commit=True) as cur:
        cur.execute("UPDATE registered_users SET member_password_hash=? WHERE id=?", (value, user_id))
        return cur.rowcount > 0


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
