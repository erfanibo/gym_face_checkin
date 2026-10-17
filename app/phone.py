"""
Phone-number helpers. The member panel logs in with the phone number, so two spellings of the
same number ("09123456789", "۰۹۱۲۳۴۵۶۷۸۹", "+98 912 345 6789") must count as the SAME number.

The stored value stays exactly as the operator typed it (nothing is rewritten); comparison always
goes through phone_key().
"""
import re
from typing import Optional

from fastapi import HTTPException

from .database import db_cursor

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def phone_key(raw) -> str | None:
    """Canonical form used to compare phone numbers, or None if there is no usable number."""
    if not raw:
        return None
    digits = re.sub(r"\D", "", str(raw).translate(_DIGITS))
    if not digits:
        return None
    if digits.startswith("0098") and len(digits) == 14:
        digits = "0" + digits[4:]
    elif digits.startswith("98") and len(digits) == 12:
        digits = "0" + digits[2:]
    elif len(digits) == 10 and digits.startswith("9"):
        digits = "0" + digits
    return digits


def members_with_phone(cur, key: str, exclude_id: int | None = None) -> list:
    """Every registered member whose stored phone is the same number as `key` (a phone_key())."""
    cur.execute("SELECT id, full_name, phone FROM registered_users WHERE phone IS NOT NULL AND phone != ''")
    return [r for r in cur.fetchall() if r["id"] != exclude_id and phone_key(r["phone"]) == key]


def ensure_phone_free(phone: Optional[str], exclude_id: Optional[int] = None) -> None:
    """The member panel logs in by phone number, so one number can belong to only one member."""
    key = phone_key(phone)
    if key is None:
        return
    with db_cursor() as cur:
        owners = members_with_phone(cur, key, exclude_id=exclude_id)
    if owners:
        raise HTTPException(409, f"این شماره تماس قبلاً برای «{owners[0]['full_name']}» ثبت شده است.")
