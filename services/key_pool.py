"""Encrypted, isolated API key groups with short-lived in-memory status."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import os
import re
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

from cryptography.fernet import Fernet, InvalidToken

from services.firestore_store import _trip_circuit, firestore_status

logger = logging.getLogger(__name__)

KEY_GROUPS = ("chat", "indexing", "admin")
KEY_DOCUMENTS = {
    "chat": "keys_chat",
    "indexing": "keys_index",
    "admin": "keys_admin",
}
KEY_SEEDS = {
    "chat": "GOOGLE_API_KEY",
    "indexing": "GOOGLE_API_KEY_INDEX",
    "admin": "GOOGLE_API_KEY_ADMIN",
}
FIRESTORE_KEY_TIMEOUT_SECONDS = 2.0
FIRESTORE_ADMIN_TIMEOUT_SECONDS = 10.0
KEY_CACHE_SECONDS = 60.0
_PACIFIC = ZoneInfo("America/Los_Angeles")
_CACHE_LOCK = threading.RLock()
_KEY_CACHE: dict[str, tuple[float, tuple["ManagedKey", ...]]] = {}
_MODEL_ORDER_CACHE: tuple[float, dict[str, Any]] | None = None
_SEED_DISABLED: dict[str, bool] = {group: False for group in KEY_GROUPS}


class KeyPoolError(ValueError):
    """A safe, user-facing key-pool configuration error."""


@dataclass(frozen=True)
class ManagedKey:
    sid: str
    label: str
    enabled: bool
    cooldown_until: datetime | None
    last_error: str
    key: str = field(repr=False)
    model_cooldowns: dict[str, datetime] = field(default_factory=dict, repr=False)


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


def get_seed_key(group: str, secrets: Mapping[str, Any]) -> str:
    _validate_group(group)
    return str(
        secrets.get(KEY_SEEDS[group], "")
        or os.getenv(KEY_SEEDS[group], "")
        or ""
    ).strip()


def get_keys(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    *,
    refresh: bool = False,
    admin: bool = False,
) -> tuple[ManagedKey, ...]:
    """Return persisted keys or the group-specific seed, falling back offline."""
    _validate_group(group)
    now = time.monotonic()
    with _CACHE_LOCK:
        cached = _KEY_CACHE.get(group)
        if cached and not refresh and now - cached[0] < KEY_CACHE_SECONDS:
            return cached[1]
    cached_records = cached[1] if cached else ()

    seed = _seed_record(group, get_seed_key(group, secrets))
    records: tuple[ManagedKey, ...] = ()
    seed_disabled = False
    loaded = False
    if client is not None and (
        admin or firestore_status()["connected"]
    ):
        try:
            timeout = (
                FIRESTORE_ADMIN_TIMEOUT_SECONDS
                if admin
                else FIRESTORE_KEY_TIMEOUT_SECONDS
            )
            snapshot = (
                client.collection("config")
                .document(KEY_DOCUMENTS[group])
                .get(timeout=timeout, retry=None)
            )
            data = snapshot.to_dict() or {}
            stored_records = data.get("keys", [])
            records = _decode_records(stored_records, secrets)
            seed_disabled = bool(data.get("seed_disabled", False)) or bool(
                stored_records
            )
            loaded = True
        except Exception as error:
            if admin:
                raise KeyPoolError(
                    "Pembacaan kelompok API key gagal "
                    f"({type(error).__name__}): "
                    f"{_safe_admin_error(error, group, secrets)}"
                ) from error
            _trip_circuit(error, f"pembacaan kelompok key {group}")
            logger.warning(
                "API key group read failed group=%s error_class=%s",
                group,
                type(error).__name__,
            )
    if not loaded:
        records = cached_records
        with _CACHE_LOCK:
            seed_disabled = _SEED_DISABLED[group]
    if not records and not seed_disabled:
        records = (seed,) if seed is not None else ()
    with _CACHE_LOCK:
        _KEY_CACHE[group] = (time.monotonic(), records)
        _SEED_DISABLED[group] = seed_disabled
    return records


def get_available_keys(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    *,
    exclude: Sequence[str] = (),
) -> tuple[ManagedKey, ...]:
    now = datetime.now(timezone.utc)
    excluded = set(exclude)
    available = [
        item
        for item in get_keys(group, client, secrets)
        if item.enabled
        and item.sid not in excluded
        and (item.cooldown_until is None or item.cooldown_until <= now)
    ]
    return tuple(available)


def models_for_key(key: ManagedKey, models: Sequence[str]) -> list[str]:
    now = datetime.now(timezone.utc)
    return [
        model
        for model in models
        if key.model_cooldowns.get(model) is None
        or key.model_cooldowns[model] <= now
    ]


def add_key(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    key: str,
    label: str,
) -> ManagedKey:
    clean_key = key.strip()
    clean_label = label.strip()[:60]
    if not clean_key:
        raise KeyPoolError("API key henteu kénging kosong.")
    if not clean_label:
        clean_label = f"Key {key_fingerprint(clean_key)}"
    if clean_key in clean_label:
        clean_label = f"Key {key_fingerprint(clean_key)}"

    current = list(get_keys(group, client, secrets, refresh=True, admin=True))
    sid = key_fingerprint(clean_key)
    if any(item.sid == sid for item in current):
        raise KeyPoolError("API key ieu parantos aya dina kelompokna.")
    new_key = ManagedKey(sid, clean_label, True, None, "", clean_key)
    current.append(new_key)
    _save_group(group, client, secrets, current, admin=True)
    return new_key


def remove_key(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    sid: str,
) -> None:
    current = list(get_keys(group, client, secrets, refresh=True, admin=True))
    updated = [item for item in current if item.sid != sid]
    if len(updated) == len(current):
        raise KeyPoolError("API key henteu kapendak.")
    seed_disabled = any(
        item.sid == sid and item.label == f"Seed {group}" for item in current
    )
    _save_group(
        group,
        client,
        secrets,
        updated,
        seed_disabled=seed_disabled,
        admin=True,
    )


def set_key_enabled(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    sid: str,
    enabled: bool,
) -> None:
    current = list(get_keys(group, client, secrets, refresh=True, admin=True))
    updated = [
        replace(item, enabled=enabled) if item.sid == sid else item
        for item in current
    ]
    if updated == current:
        if not any(item.sid == sid for item in current):
            raise KeyPoolError("API key henteu kapendak.")
        return
    _save_group(group, client, secrets, updated, admin=True)


def record_quota(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    key: str,
    error: BaseException,
    *,
    model: str | None = None,
    daily: bool | None = None,
) -> None:
    """Record only safe status markers; never retain provider error text."""
    records = list(get_keys(group, client, secrets, refresh=True))
    sid = key_fingerprint(key)
    is_daily = _is_daily_quota(error) if daily is None else daily
    cooldown = _next_pacific_reset() if is_daily else _short_quota_cooldown(error)
    safe_error = "daily_quota" if is_daily else "rate_limit"
    updated: list[ManagedKey] = []
    changed = False
    for item in records:
        if item.sid != sid:
            updated.append(item)
            continue
        if group == "chat" and model:
            cooldowns = dict(item.model_cooldowns)
            previous = cooldowns.get(model)
            if previous is None or previous < cooldown:
                cooldowns[model] = cooldown
                changed = True
            changed = changed or item.last_error != safe_error
            updated.append(
                replace(item, model_cooldowns=cooldowns, last_error=safe_error)
            )
        else:
            changed = (
                item.cooldown_until != cooldown or item.last_error != safe_error
            )
            updated.append(
                replace(item, cooldown_until=cooldown, last_error=safe_error)
            )
    if changed:
        _save_group(group, client, secrets, updated)


def get_key_status(
    key: ManagedKey,
    chat_models: Sequence[str] = (),
) -> str:
    if not key.enabled:
        return "nonaktif"
    now = datetime.now(timezone.utc)
    if key.cooldown_until and key.cooldown_until > now:
        return f"kena limit sampai {key.cooldown_until.astimezone(_PACIFIC):%H:%M} PT"
    if key.model_cooldowns and chat_models:
        cooldowns = [
            key.model_cooldowns.get(model)
            for model in chat_models
        ]
        if cooldowns and all(
            cooldown is not None and cooldown > now for cooldown in cooldowns
        ):
            next_available = min(cooldown for cooldown in cooldowns if cooldown)
            return (
                "kena limit sampai "
                f"{next_available.astimezone(_PACIFIC):%H:%M} PT"
            )
    return "aktif"


def get_model_order(
    client: Any | None,
    secrets: Mapping[str, Any],
    *,
    admin: bool = False,
) -> dict[str, Any]:
    global _MODEL_ORDER_CACHE
    now = time.monotonic()
    with _CACHE_LOCK:
        if _MODEL_ORDER_CACHE and now - _MODEL_ORDER_CACHE[0] < KEY_CACHE_SECONDS:
            return dict(_MODEL_ORDER_CACHE[1])

    result: dict[str, Any] = {
        "mode": "automatic",
        "models": [],
        "secrets_models": _parse_model_list(
            secrets.get("GEMINI_CHAT_MODELS", "")
            or os.getenv("GEMINI_CHAT_MODELS", "")
        ),
    }
    if client is not None and (admin or firestore_status()["connected"]):
        try:
            snapshot = (
                client.collection("config")
                .document("model_order")
                .get(
                    timeout=(
                        FIRESTORE_ADMIN_TIMEOUT_SECONDS
                        if admin
                        else FIRESTORE_KEY_TIMEOUT_SECONDS
                    ),
                    retry=None,
                )
            )
            data = snapshot.to_dict() or {}
            mode = data.get("mode")
            models = data.get("models")
            if mode in {"automatic", "manual"} and isinstance(models, list):
                result.update(
                    mode=mode,
                    models=[str(model) for model in models if str(model).strip()],
                )
        except Exception as error:
            if admin:
                raise KeyPoolError(
                    "Pembacaan urutan model gagal "
                    f"({type(error).__name__}): "
                    f"{_safe_admin_error(error, 'model_order', secrets)}"
                ) from error
            _trip_circuit(error, "pembacaan urutan model")
            logger.warning(
                "Model order read failed error_class=%s",
                type(error).__name__,
            )
    with _CACHE_LOCK:
        _MODEL_ORDER_CACHE = (time.monotonic(), dict(result))
    return result


def save_model_order(
    client: Any | None,
    mode: str,
    models: Sequence[str],
) -> None:
    global _MODEL_ORDER_CACHE
    if client is None:
        raise KeyPoolError("Firestore teu sayogi pikeun nyimpen urutan model.")
    if mode not in {"automatic", "manual"}:
        raise KeyPoolError("Mode urutan model henteu valid.")
    data = {"mode": mode, "models": list(dict.fromkeys(models))}
    try:
        client.collection("config").document("model_order").set(
            data,
            timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS,
        )
    except Exception as error:
        raise KeyPoolError(
            "Penyimpanan urutan model gagal "
            f"({type(error).__name__}): "
            f"{_safe_admin_error(error, 'model_order', {})}"
        ) from error
    with _CACHE_LOCK:
        _MODEL_ORDER_CACHE = (time.monotonic(), data)


def ordered_models(
    available: Sequence[str],
    settings: Mapping[str, Any],
) -> list[str]:
    available_models = list(dict.fromkeys(available))
    if settings.get("mode") == "manual":
        configured = [str(model) for model in settings.get("models", [])]
    elif settings.get("secrets_models"):
        configured = [str(model) for model in settings["secrets_models"]]
    else:
        return available_models
    return (
        [model for model in configured if model in available_models]
        or available_models
    )


def redact_secrets(text: object, keys: Sequence[str]) -> str:
    result = str(text)
    for key in keys:
        if key:
            result = result.replace(key, mask_key(key))
    return result


def set_session_chat_override(session_state: Any, key: str) -> None:
    clean_key = key.strip()
    if not clean_key:
        raise KeyPoolError("API key henteu kénging kosong.")
    session_state["chat_api_key_override"] = clean_key


def clear_session_chat_override(session_state: Any) -> None:
    session_state.pop("chat_api_key_override", None)


def invalidate_cache(group: str | None = None) -> None:
    global _MODEL_ORDER_CACHE
    with _CACHE_LOCK:
        if group is None:
            _KEY_CACHE.clear()
            _MODEL_ORDER_CACHE = None
        else:
            _validate_group(group)
            _KEY_CACHE.pop(group, None)


def _save_group(
    group: str,
    client: Any | None,
    secrets: Mapping[str, Any],
    records: Sequence[ManagedKey],
    *,
    seed_disabled: bool | None = None,
    admin: bool = False,
) -> None:
    if client is None or (not admin and not firestore_status()["connected"]):
        raise KeyPoolError("Firestore teu sayogi pikeun nyimpen API key.")
    cipher = _get_cipher(secrets)
    encrypted_records = []
    for item in records:
        encrypted_records.append(
            {
                "id": item.sid,
                "label": item.label,
                "enabled": item.enabled,
                "encrypted_key": cipher.encrypt(item.key.encode("utf-8")).decode("ascii"),
                "cooldown_until": _format_datetime(item.cooldown_until),
                "last_error": item.last_error,
                "model_cooldowns": {
                    model: timestamp.isoformat()
                    for model, timestamp in item.model_cooldowns.items()
                },
            }
        )
    with _CACHE_LOCK:
        disabled = (
            _SEED_DISABLED[group]
            if seed_disabled is None
            else seed_disabled
        )
    try:
        client.collection("config").document(KEY_DOCUMENTS[group]).set(
            {"keys": encrypted_records, "seed_disabled": disabled},
            timeout=(
                FIRESTORE_ADMIN_TIMEOUT_SECONDS
                if admin
                else FIRESTORE_KEY_TIMEOUT_SECONDS
            ),
        )
    except Exception as error:
        sensitive_keys = [item.key for item in records]
        if not admin:
            _trip_circuit(error, f"penyimpanan kelompok key {group}")
            raise KeyPoolError("API key teu tiasa disimpen.") from error
        raise KeyPoolError(
            "Penyimpanan API key gagal "
            f"({type(error).__name__}): "
            f"{_safe_admin_error(error, group, secrets, *sensitive_keys)}"
        ) from error
    with _CACHE_LOCK:
        _KEY_CACHE[group] = (time.monotonic(), tuple(records))
        _SEED_DISABLED[group] = disabled


def _decode_records(value: Any, secrets: Mapping[str, Any]) -> tuple[ManagedKey, ...]:
    if not isinstance(value, list) or not value:
        return ()
    try:
        cipher = _get_cipher(secrets)
    except KeyPoolError:
        return ()
    result: list[ManagedKey] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id", ""))
        try:
            key = cipher.decrypt(str(row["encrypted_key"]).encode("ascii")).decode("utf-8")
        except (KeyError, ValueError, InvalidToken):
            logger.warning("Stored API key could not be decrypted sid=%s", sid[:10])
            continue
        model_cooldowns = row.get("model_cooldowns", {})
        if not isinstance(model_cooldowns, dict):
            model_cooldowns = {}
        result.append(
            ManagedKey(
                sid=sid or key_fingerprint(key),
                label=_safe_label(str(row.get("label", "")), key),
                enabled=bool(row.get("enabled", True)),
                cooldown_until=_parse_datetime(row.get("cooldown_until")),
                last_error=str(row.get("last_error", ""))[:40],
                key=key,
                model_cooldowns={
                    str(model): timestamp
                    for model, value in model_cooldowns.items()
                    if (timestamp := _parse_datetime(value)) is not None
                },
            )
        )
    return tuple(result)


def _seed_record(group: str, key: str) -> ManagedKey | None:
    if not key:
        return None
    return ManagedKey(
        sid=key_fingerprint(key),
        label=f"Seed {group}",
        enabled=True,
        cooldown_until=None,
        last_error="",
        key=key,
    )


def _get_cipher(secrets: Mapping[str, Any]) -> Fernet:
    key = str(secrets.get("ENCRYPTION_KEY", "") or "").strip()
    if not key:
        raise KeyPoolError("ENCRYPTION_KEY teu acan diatur dina Secrets.")
    try:
        return Fernet(key.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        raise KeyPoolError("ENCRYPTION_KEY henteu valid.") from None


def _safe_label(label: str, key: str) -> str:
    cleaned = label.strip()[:60]
    if key and key in cleaned:
        return f"Key {key_fingerprint(key)}"
    return cleaned or f"Key {key_fingerprint(key)}"


def _validate_group(group: str) -> None:
    if group not in KEY_GROUPS:
        raise KeyPoolError("Kelompok API key henteu valid.")


def _safe_admin_error(
    error: BaseException,
    context: str,
    secrets: Mapping[str, Any],
    *sensitive_values: str,
) -> str:
    message = str(error)
    seed = str(secrets.get(KEY_SEEDS.get(context, ""), "") or "")
    for value in (seed, *sensitive_values):
        if value:
            message = message.replace(value, "[disamarkan]")
    message = re.sub(r"\b[a-f0-9]{16,32}\b", "[UID]", message, flags=re.IGNORECASE)
    return message[:200]


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _next_pacific_reset() -> datetime:
    now = datetime.now(_PACIFIC)
    return (now + timedelta(days=1)).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    ).astimezone(timezone.utc)


def _short_quota_cooldown(error: BaseException) -> datetime:
    try:
        from services.ai_service import get_quota_retry_delay_seconds

        delay = get_quota_retry_delay_seconds(error)
    except Exception:
        delay = None
    return datetime.now(timezone.utc) + timedelta(
        seconds=min(max(delay or 60.0, 5.0), 60.0)
    )


def _is_daily_quota(error: BaseException) -> bool:
    compact = "".join(char for char in str(error).casefold() if char.isalnum())
    return any(
        marker in compact
        for marker in ("perday", "daily", "requestsperday", "tokensperday")
    )


def _parse_model_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [
            model.strip().removeprefix("models/")
            for model in value.split(",")
            if model.strip()
        ]
    if isinstance(value, Sequence):
        return [
            str(model).strip().removeprefix("models/")
            for model in value
            if str(model).strip()
        ]
    return []
