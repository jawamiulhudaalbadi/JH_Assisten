"""Signed, revocable per-device admin sessions."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import re
import threading
import time
from collections.abc import Mapping
from typing import Any

from services.firestore_store import _bounded_read

logger = logging.getLogger(__name__)

ADMIN_SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
ADMIN_SESSION_EPOCH_CACHE_SECONDS = 300.0
ADMIN_SESSION_COOKIE_NAME = "jh_adm"
ADMIN_SESSION_STORAGE_KEY = "jh_adm"
_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_EPOCH_CACHE_LOCK = threading.Lock()
_EPOCH_CACHE: dict[int, tuple[float, int | None]] = {}


def _session_signing_key(secrets: Mapping[str, Any]) -> bytes:
    secret = str(
        secrets.get("ADMIN_SESSION_SECRET", "")
        or secrets.get("ENCRYPTION_KEY", "")
        or ""
    ).strip()
    if not secret:
        return b""
    return hmac.new(
        secret.encode("utf-8"),
        b"JH_Assisten/admin-session/v1",
        hashlib.sha256,
    ).digest()


def create_admin_session_token(
    uid: str,
    epoch: int,
    secrets: Mapping[str, Any],
    *,
    now: int | None = None,
) -> str:
    """Create a base64url `uid|expiry|signature` token without padding."""
    if not _UID_RE.fullmatch(uid):
        raise ValueError("ID perangkat admin henteu valid.")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Epoch sesi admin henteu valid.")
    signing_key = _session_signing_key(secrets)
    if not signing_key:
        raise ValueError("ADMIN_SESSION_SECRET atanapi ENCRYPTION_KEY teu acan diatur.")
    issued_at = int(time.time()) if now is None else int(now)
    expires_at = issued_at + ADMIN_SESSION_TTL_SECONDS
    signed_data = f"{uid}|{expires_at}|{epoch}".encode("utf-8")
    signature = hmac.new(signing_key, signed_data, hashlib.sha256).hexdigest()
    token_bytes = f"{uid}|{expires_at}|{signature}".encode("ascii")
    return base64.urlsafe_b64encode(token_bytes).rstrip(b"=").decode("ascii")


def verify_admin_session_token(
    token: str,
    uid: str,
    epoch: int,
    secrets: Mapping[str, Any],
    *,
    now: int | None = None,
) -> bool:
    """Validate token integrity, expiry, device UID, and the current epoch."""
    if (
        not isinstance(token, str)
        or not token
        or not _TOKEN_RE.fullmatch(token)
        or not _UID_RE.fullmatch(uid)
        or isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        return False
    signing_key = _session_signing_key(secrets)
    if not signing_key:
        return False
    try:
        decoded = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        token_uid, raw_expiry, supplied_signature = decoded.decode("ascii").split("|")
        if not raw_expiry.isdecimal() or not re.fullmatch(
            r"[0-9a-f]{64}", supplied_signature
        ):
            return False
        expires_at = int(raw_expiry)
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return False
    current_time = int(time.time()) if now is None else int(now)
    if token_uid != uid or expires_at <= current_time:
        return False
    signed_data = f"{token_uid}|{expires_at}|{epoch}".encode("utf-8")
    expected_signature = hmac.new(
        signing_key, signed_data, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(supplied_signature, expected_signature)


def build_admin_session_store_script(token: str) -> str:
    """Return a browser script that stores the token in cookie and localStorage."""
    if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
        raise ValueError("Token sesi admin henteu valid.")
    token_literal = json.dumps(token)
    cookie_name = json.dumps(ADMIN_SESSION_COOKIE_NAME)
    storage_key = json.dumps(ADMIN_SESSION_STORAGE_KEY)
    return f"""(() => {{
  try {{
    const token = {token_literal};
    const secure = location.protocol === "https:" ? "; Secure" : "";
    document.cookie = {cookie_name} + "=" + token
      + "; Max-Age={ADMIN_SESSION_TTL_SECONDS}; Path=/; SameSite=Lax" + secure;
    localStorage.setItem({storage_key}, token);
    return "stored";
  }} catch (error) {{
    return "error";
  }}
}})()"""


def build_admin_session_clear_script() -> str:
    """Return a browser script that removes both admin-session copies."""
    cookie_name = json.dumps(ADMIN_SESSION_COOKIE_NAME)
    storage_key = json.dumps(ADMIN_SESSION_STORAGE_KEY)
    return f"""(() => {{
  try {{
    document.cookie = {cookie_name}
      + "=; Max-Age=0; Path=/; SameSite=Lax"
      + (location.protocol === "https:" ? "; Secure" : "");
  }} catch (error) {{
    return "error";
  }}
  if (document.cookie.split("; ").some(value => value.startsWith({cookie_name} + "="))) {{
    return "error";
  }}
  try {{
    localStorage.removeItem({storage_key});
    return "cleared";
  }} catch (error) {{
    return "cookie-cleared";
  }}
}})()"""


def get_admin_session_epoch(
    client: Any | None,
    *,
    force_refresh: bool = False,
) -> int | None:
    """Read the revocation epoch, caching successful and failed reads for 5 minutes."""
    if client is None:
        return None
    cache_key = id(client)
    now = time.monotonic()
    with _EPOCH_CACHE_LOCK:
        cached = _EPOCH_CACHE.get(cache_key)
        if (
            not force_refresh
            and cached is not None
            and now - cached[0] < ADMIN_SESSION_EPOCH_CACHE_SECONDS
        ):
            return cached[1]

    def read_epoch() -> int:
        snapshot = (
            client.collection("config")
            .document("admin_session")
            .get(timeout=2.0, retry=None)
        )
        data = snapshot.to_dict() or {}
        epoch = data.get("epoch", 0)
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch sesi admin tidak valid.")
        return epoch

    epoch = _bounded_read(
        client,
        read_epoch,
        timeout=2.5,
        context="pembacaan epoch sesi admin",
        fallback=None,
    )
    with _EPOCH_CACHE_LOCK:
        _EPOCH_CACHE[cache_key] = (time.monotonic(), epoch)
    return epoch


def revoke_all_admin_sessions(client: Any | None) -> None:
    """Atomically increase the epoch, invalidating every previously issued token."""
    if client is None:
        raise RuntimeError("Firestore tidak tersedia untuk mencabut sesi admin.")
    try:
        from google.cloud import firestore

        batch = client.batch()
        batch.set(
            client.collection("config").document("admin_session"),
            {"epoch": firestore.Increment(1)},
            merge=True,
        )
        batch.commit(timeout=10.0, retry=None)
    except Exception as error:
        logger.warning(
            "Admin session revocation failed error_class=%s",
            type(error).__name__,
        )
        raise RuntimeError(
            "Sesi admin tidak berhasil dicabut "
            f"({type(error).__name__})."
        ) from error
    with _EPOCH_CACHE_LOCK:
        _EPOCH_CACHE.pop(id(client), None)


def clear_admin_session_epoch_cache() -> None:
    """Reset epoch cache for isolated tests."""
    with _EPOCH_CACHE_LOCK:
        _EPOCH_CACHE.clear()
