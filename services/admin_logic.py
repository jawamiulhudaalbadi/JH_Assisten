"""Pure helpers for admin authentication and chat policy checks."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime
import hashlib
import hmac
from math import ceil
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


def process_admin_login(
    code: str,
    encoded_hash: str,
    session_state: MutableMapping[str, object],
    now: float,
    empty_code_notice: str = "Kode admin belum diisi.",
) -> bool:
    """Validate an admin login and update its session lockout state."""
    locked_until = float(session_state.get("admin_locked_until", 0))
    if now < locked_until:
        remaining = ceil(locked_until - now)
        session_state["_admin_login_notice"] = (
            f"Login admin terkunci. Coba lagi dalam {remaining} detik."
        )
        return False
    if locked_until:
        session_state["admin_failed_attempts"] = 0
        session_state.pop("admin_locked_until", None)

    if not code:
        session_state["_admin_login_notice"] = empty_code_notice
        return False
    if not encoded_hash:
        session_state["_admin_login_notice"] = (
            "Dashboard admin terkunci: secret ADMIN_CODE_HASH belum diatur."
        )
        return False
    if verify_admin_code(code, encoded_hash):
        session_state["admin_failed_attempts"] = 0
        session_state.pop("admin_locked_until", None)
        session_state["_admin_login_prompt_open"] = False
        session_state["_admin_button_clicks"] = 0
        session_state["admin_authenticated"] = True
        session_state.pop("_admin_login_notice", None)
        return True

    attempts = int(session_state.get("admin_failed_attempts", 0)) + 1
    if attempts >= 5:
        session_state["admin_failed_attempts"] = 5
        session_state["admin_locked_until"] = now + 15 * 60
        session_state["_admin_login_notice"] = (
            "Lima percobaan gagal. Login admin terkunci selama 15 menit."
        )
    else:
        session_state["admin_failed_attempts"] = attempts
        session_state["_admin_login_notice"] = (
            f"Kode admin salah ({attempts}/5 percobaan)."
        )
    return False


def register_admin_button_click(
    session_state: MutableMapping[str, object],
) -> bool:
    """Open the login prompt after three sidebar button clicks."""
    clicks = int(session_state.get("_admin_button_clicks", 0)) + 1
    if clicks >= 3:
        session_state["_admin_button_clicks"] = 0
        session_state["_admin_login_prompt_open"] = True
        return True

    session_state["_admin_button_clicks"] = clicks
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
