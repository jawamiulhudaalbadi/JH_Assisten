"""Pure helpers for admin authentication and chat policy checks."""

from __future__ import annotations

from collections.abc import MutableMapping
from datetime import datetime
import hashlib
import hmac
from math import ceil
import re
import secrets
from threading import Lock
from zoneinfo import ZoneInfo

_WIB = ZoneInfo("Asia/Jakarta")
_PBKDF2_RE = re.compile(r"^pbkdf2_sha256\$(\d+)\$([^$]+)\$([0-9a-fA-F]+)$")
_SCRYPT_RE = re.compile(r"^scrypt\$([^$]+)\$([0-9a-fA-F]+)$")
_PROCESS_ADMIN_LOCK = Lock()
_PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID: dict[str, int] = {}
_PROCESS_ADMIN_LOCKED_UNTIL_BY_UID: dict[str, float] = {}
_PROCESS_ADMIN_FAILED_ATTEMPTS = 0
_PROCESS_ADMIN_LOCKED_UNTIL = 0.0
_PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID: dict[str, int] = {}
_PROCESS_GUEST_LOCKED_UNTIL_BY_UID: dict[str, float] = {}
_PROCESS_GUEST_FAILED_ATTEMPTS = 0
_PROCESS_GUEST_LOCKED_UNTIL = 0.0


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


def hash_settings_password(password: str) -> str:
    """Hash a settings password entered by an authenticated administrator."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()
    return f"pbkdf2_sha256$200000${salt}${digest}"


def process_admin_login(
    code: str,
    encoded_hash: str,
    session_state: MutableMapping[str, object],
    now: float,
    uid: str,
    empty_code_notice: str = "Kode admin belum diisi.",
) -> bool:
    """Validate an admin login with session and process-wide lockout state."""
    global _PROCESS_ADMIN_FAILED_ATTEMPTS, _PROCESS_ADMIN_LOCKED_UNTIL

    with _PROCESS_ADMIN_LOCK:
        session_locked_until = float(session_state.get("admin_locked_until", 0))
        uid_locked_until = _PROCESS_ADMIN_LOCKED_UNTIL_BY_UID.get(uid, 0.0)
        if now >= uid_locked_until and uid_locked_until:
            _PROCESS_ADMIN_LOCKED_UNTIL_BY_UID.pop(uid, None)
            _PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID.pop(uid, None)
            uid_locked_until = 0.0
        if now >= _PROCESS_ADMIN_LOCKED_UNTIL and _PROCESS_ADMIN_LOCKED_UNTIL:
            _PROCESS_ADMIN_LOCKED_UNTIL = 0.0
            _PROCESS_ADMIN_FAILED_ATTEMPTS = 0
        if now >= session_locked_until and session_locked_until:
            session_state["admin_failed_attempts"] = 0
            session_state.pop("admin_locked_until", None)
            session_locked_until = 0.0

        locked_until = max(
            session_locked_until,
            uid_locked_until,
            _PROCESS_ADMIN_LOCKED_UNTIL,
        )
        if now < locked_until:
            remaining = ceil(locked_until - now)
            session_state["_admin_login_notice"] = (
                f"Login admin terkunci. Coba lagi dalam {remaining} detik."
            )
            return False

        if not code:
            session_state["_admin_login_notice"] = empty_code_notice
            return False
        if not encoded_hash:
            session_state["_admin_login_notice"] = (
                "Dashboard admin terkunci: secret ADMIN_CODE_HASH belum diatur."
            )
            return False
        if verify_admin_code(code, encoded_hash):
            _PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID.pop(uid, None)
            _PROCESS_ADMIN_LOCKED_UNTIL_BY_UID.pop(uid, None)
            _PROCESS_ADMIN_FAILED_ATTEMPTS = 0
            _PROCESS_ADMIN_LOCKED_UNTIL = 0.0
            session_state["admin_failed_attempts"] = 0
            session_state.pop("admin_locked_until", None)
            session_state["_admin_login_prompt_open"] = False
            session_state["_admin_button_clicks"] = 0
            session_state["admin_authenticated"] = True
            session_state.pop("_admin_login_notice", None)
            return True

        uid_attempts = _PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID.get(uid, 0) + 1
        global_attempts = _PROCESS_ADMIN_FAILED_ATTEMPTS + 1
        _PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID[uid] = uid_attempts
        _PROCESS_ADMIN_FAILED_ATTEMPTS = global_attempts
        attempts = max(
            int(session_state.get("admin_failed_attempts", 0)) + 1,
            uid_attempts,
            global_attempts,
        )
        if attempts >= 5:
            lockout_until = now + 15 * 60
            session_state["admin_failed_attempts"] = 5
            session_state["admin_locked_until"] = lockout_until
            _PROCESS_ADMIN_LOCKED_UNTIL_BY_UID[uid] = lockout_until
            _PROCESS_ADMIN_LOCKED_UNTIL = lockout_until
            session_state["_admin_login_notice"] = (
                "Lima percobaan gagal. Login admin terkunci selama 15 menit."
            )
        else:
            session_state["admin_failed_attempts"] = attempts
            session_state["_admin_login_notice"] = (
                f"Kode admin salah ({attempts}/5 percobaan)."
            )
        return False


def process_guest_login(
    code: str,
    encoded_hash: str,
    session_state: MutableMapping[str, object],
    now: float,
    uid: str,
) -> bool:
    """Validate guest access with lockout counters isolated from admin login."""
    global _PROCESS_GUEST_FAILED_ATTEMPTS, _PROCESS_GUEST_LOCKED_UNTIL

    with _PROCESS_ADMIN_LOCK:
        session_locked_until = float(session_state.get("guest_locked_until", 0))
        uid_locked_until = _PROCESS_GUEST_LOCKED_UNTIL_BY_UID.get(uid, 0.0)
        if now >= uid_locked_until and uid_locked_until:
            _PROCESS_GUEST_LOCKED_UNTIL_BY_UID.pop(uid, None)
            _PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID.pop(uid, None)
            uid_locked_until = 0.0
        if now >= _PROCESS_GUEST_LOCKED_UNTIL and _PROCESS_GUEST_LOCKED_UNTIL:
            _PROCESS_GUEST_LOCKED_UNTIL = 0.0
            _PROCESS_GUEST_FAILED_ATTEMPTS = 0
        if now >= session_locked_until and session_locked_until:
            session_state["guest_failed_attempts"] = 0
            session_state.pop("guest_locked_until", None)
            session_locked_until = 0.0

        locked_until = max(
            session_locked_until,
            uid_locked_until,
            _PROCESS_GUEST_LOCKED_UNTIL,
        )
        if now < locked_until:
            remaining = ceil(locked_until - now)
            session_state["_guest_login_notice"] = (
                f"Mode tamu terkunci. Coba deui dina {remaining} detik."
            )
            return False
        if not code or not encoded_hash:
            session_state["_guest_login_notice"] = (
                "Sandi mode tamu teu leres."
            )
            return False
        if verify_admin_code(code, encoded_hash):
            _PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID.pop(uid, None)
            _PROCESS_GUEST_LOCKED_UNTIL_BY_UID.pop(uid, None)
            _PROCESS_GUEST_FAILED_ATTEMPTS = 0
            _PROCESS_GUEST_LOCKED_UNTIL = 0.0
            session_state["guest_failed_attempts"] = 0
            session_state.pop("guest_locked_until", None)
            session_state.pop("_guest_login_notice", None)
            return True

        uid_attempts = _PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID.get(uid, 0) + 1
        global_attempts = _PROCESS_GUEST_FAILED_ATTEMPTS + 1
        _PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID[uid] = uid_attempts
        _PROCESS_GUEST_FAILED_ATTEMPTS = global_attempts
        attempts = max(
            int(session_state.get("guest_failed_attempts", 0)) + 1,
            uid_attempts,
            global_attempts,
        )
        if attempts >= 5:
            lockout_until = now + 15 * 60
            session_state["guest_failed_attempts"] = 5
            session_state["guest_locked_until"] = lockout_until
            _PROCESS_GUEST_LOCKED_UNTIL_BY_UID[uid] = lockout_until
            _PROCESS_GUEST_LOCKED_UNTIL = lockout_until
            session_state["_guest_login_notice"] = (
                "Lima percobaan gagal. Mode tamu terkunci salila 15 menit."
            )
        else:
            session_state["guest_failed_attempts"] = attempts
            session_state["_guest_login_notice"] = (
                f"Sandi mode tamu teu leres ({attempts}/5)."
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
    try:
        user_limit = int(user_data.get("daily_question_limit", 0))
    except (TypeError, ValueError):
        user_limit = 0
    effective_limit = user_limit if user_limit > 0 else daily_question_limit
    if effective_limit <= 0:
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
    if questions_today >= effective_limit:
        return (
            False,
            f"Batas pertanyaan hari ini ({effective_limit}) telah tercapai.",
        )
    return True, ""
