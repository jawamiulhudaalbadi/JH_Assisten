"""Pure helpers for admin authentication and chat policy checks."""

from __future__ import annotations

from datetime import datetime
import hashlib
import hmac
import re
from zoneinfo import ZoneInfo

_WIB = ZoneInfo("Asia/Jakarta")
_PBKDF2_RE = re.compile(r"^pbkdf2_sha256\$(\d+)\$([^$]+)\$([0-9a-fA-F]+)$")
_SCRYPT_RE = re.compile(r"^scrypt\$([^$]+)\$([0-9a-fA-F]+)$")


def verify_admin_code(code: str, encoded_hash: str) -> bool:
    """Verify `pbkdf2_sha256$rounds$salt$hex` or `scrypt$salt$hex` secrets."""
    try:
        pbkdf2_match = _PBKDF2_RE.fullmatch(encoded_hash)
        if pbkdf2_match:
            iterations = int(pbkdf2_match.group(1))
            if not 1 <= iterations <= 1_000_000:
                return False
            salt = pbkdf2_match.group(2).encode("utf-8")
            expected = bytes.fromhex(pbkdf2_match.group(3))
            actual = hashlib.pbkdf2_hmac(
                "sha256", code.encode("utf-8"), salt, iterations
            )
        else:
            scrypt_match = _SCRYPT_RE.fullmatch(encoded_hash)
            if not scrypt_match:
                return False
            salt = scrypt_match.group(1).encode("utf-8")
            expected = bytes.fromhex(scrypt_match.group(2))
            actual = hashlib.scrypt(
                code.encode("utf-8"),
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                dklen=len(expected),
            )
        return len(expected) > 0 and hmac.compare_digest(actual, expected)
    except (ValueError, OverflowError):
        return False


def question_is_allowed(
    user_data: dict[str, object],
    daily_question_limit: int,
    now: datetime | None = None,
) -> tuple[bool, str]:
    """Check cached per-session block and WIB daily-limit fields."""
    if bool(user_data.get("blocked", False)):
        return False, "Perangkat ini diblokir oleh administrator."
    if daily_question_limit <= 0:
        return True, ""

    current_time = now or datetime.now(_WIB)
    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=_WIB)
    current_day = current_time.astimezone(_WIB).strftime("%Y%m%d")
    stored_day = str(user_data.get("day", ""))
    try:
        questions_today = (
            int(user_data.get("day_count", 0)) if stored_day == current_day else 0
        )
    except (TypeError, ValueError):
        questions_today = 0
    if questions_today >= daily_question_limit:
        return (
            False,
            f"Batas pertanyaan hari ini ({daily_question_limit}) telah tercapai.",
        )
    return True, ""
