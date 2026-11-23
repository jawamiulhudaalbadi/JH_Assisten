"""Penyimpanan riwayat dan statistik pengguna di Firestore."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta
import hashlib
import json
import logging
from collections.abc import Mapping
import re
import threading
import time
import unicodedata
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st

logger = logging.getLogger(__name__)
MAX_MESSAGES_PER_CHUNK = 20
HISTORY_WARNING_MESSAGES = 200
HISTORY_WARNING_CHUNKS = 10
FIRESTORE_READ_TIMEOUT_SECONDS = 5.0
FIRESTORE_WRITE_TIMEOUT_SECONDS = 10.0
HISTORY_LOAD_TIMEOUT_SECONDS = 5.0
FIRESTORE_CIRCUIT_SECONDS = 120.0
RUNTIME_CONFIG_CACHE_SECONDS = 300.0
RUNTIME_CONFIG_READ_WAIT_SECONDS = 0.1
_WIB = ZoneInfo("Asia/Jakarta")
_WRITE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firestore-write")
_READ_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="firestore-read")
_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_OPEN_UNTIL = 0.0
_CIRCUIT_FAILURE_LOGGED = False
_RUNTIME_CONFIG_LOCK = threading.Lock()
_RUNTIME_CONFIG: dict[str, Any] = {
    "maintenance": False,
    "announcement": "",
    "daily_question_limit": 0,
    "settings_password_hash": "",
}
SAVED_CHAT_LIMIT = 5
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_SAVED_CHAT_NAME_LIMIT = 60
_RUNTIME_CONFIG_LOADED_AT = 0.0
_RECENT_ERRORS: list[dict[str, str]] = []
_RECENT_ERRORS_LOCK = threading.Lock()
_ERRORS_FLUSHED_AT = 0.0


def credentials_json_from_secrets(secrets: Mapping[str, Any]) -> str | None:
    """Validasi kredensial service account dari Streamlit Secrets."""
    raw = secrets.get("FIREBASE_SERVICE_ACCOUNT")
    if isinstance(raw, Mapping):
        return json.dumps(dict(raw), sort_keys=True)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("FIREBASE_SERVICE_ACCOUNT bukan JSON yang valid.")
            return None
        if isinstance(parsed, dict):
            return json.dumps(parsed, sort_keys=True)
    return None


@st.cache_resource(show_spinner=False)
def get_firestore_client(credentials_json: str | None) -> Any | None:
    """Buat klien database default satu kali; impor Firestore dilakukan malas."""
    if not credentials_json:
        return None
    try:
        from google.cloud import firestore
        from google.oauth2 import service_account

        service_account_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )
        return firestore.Client(
            project=service_account_info["project_id"],
            credentials=credentials,
        )
    except Exception:
        logger.exception("Klien Firestore tidak dapat dibuat.")
        return None


def _now() -> datetime:
    return datetime.now(_WIB)


def _empty_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {
        "total_messages": 0,
        "total_chunks": 0,
        "active_chunk_count": 0,
        "active_chunk_number": 0,
        "blocked": False,
        "day": "",
        "day_count": 0,
    }


def firestore_status() -> dict[str, Any]:
    """Report the in-process Firestore circuit state without performing I/O."""
    with _CIRCUIT_LOCK:
        remaining = max(0.0, _CIRCUIT_OPEN_UNTIL - time.monotonic())
    return {
        "connected": remaining == 0,
        "circuit_open_seconds": round(remaining),
    }


def _record_recent_error(error: BaseException, context: str) -> None:
    entry = {
        "timestamp": _now().isoformat(),
        "context": context[:80],
        "error": str(error)[:300],
    }
    with _RECENT_ERRORS_LOCK:
        _RECENT_ERRORS.append(entry)
        del _RECENT_ERRORS[:-20]


def _trip_circuit(error: BaseException, context: str) -> None:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURE_LOGGED
    _record_recent_error(error, context)
    with _CIRCUIT_LOCK:
        now = time.monotonic()
        if _CIRCUIT_OPEN_UNTIL <= now:
            _CIRCUIT_OPEN_UNTIL = now + FIRESTORE_CIRCUIT_SECONDS
            _CIRCUIT_FAILURE_LOGGED = False
        should_log = not _CIRCUIT_FAILURE_LOGGED
        _CIRCUIT_FAILURE_LOGGED = True
    if should_log:
        logger.error("Firestore dinonaktifkan sementara (%s): %s", context, error)


def _circuit_is_open() -> bool:
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURE_LOGGED
    with _CIRCUIT_LOCK:
        now = time.monotonic()
        if _CIRCUIT_OPEN_UNTIL <= now:
            _CIRCUIT_OPEN_UNTIL = 0.0
            _CIRCUIT_FAILURE_LOGGED = False
            return False
        return True


def _bounded_read(
    client: Any | None,
    operation: Any,
    *,
    timeout: float = FIRESTORE_READ_TIMEOUT_SECONDS,
    context: str,
    fallback: Any,
) -> Any:
    if client is None or _circuit_is_open():
        return fallback
    future = _READ_EXECUTOR.submit(operation)
    try:
        return future.result(timeout=max(0.01, timeout))
    except FutureTimeoutError:
        future.cancel()
        _trip_circuit(
            TimeoutError(f"Pembacaan melampaui batas {timeout:g} detik."),
            context,
        )
    except Exception as error:
        _trip_circuit(error, context)
    return fallback


def _remaining_timeout(deadline: float, maximum: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Batas waktu operasi Firestore terlampaui.")
    return min(maximum, remaining)


def reset_firestore_circuit() -> None:
    """Reset process-local state for isolated tests."""
    global _CIRCUIT_OPEN_UNTIL, _CIRCUIT_FAILURE_LOGGED
    global _RUNTIME_CONFIG_LOADED_AT, _ERRORS_FLUSHED_AT
    with _CIRCUIT_LOCK:
        _CIRCUIT_OPEN_UNTIL = 0.0
        _CIRCUIT_FAILURE_LOGGED = False
    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG_LOADED_AT = 0.0
    with _RECENT_ERRORS_LOCK:
        _RECENT_ERRORS.clear()
        _ERRORS_FLUSHED_AT = 0.0


def normalize_question(question: str) -> str:
    """Normalize equivalent question text for anonymous FAQ aggregation."""
    normalized = unicodedata.normalize("NFKC", question).casefold()
    normalized = "".join(
        char
        if char.isspace() or char.isalnum() or unicodedata.category(char).startswith("M")
        else " "
        for char in normalized
    )
    return " ".join(normalized.split())


def _snapshots(query: Any, deadline: float) -> list[Any]:
    return list(
        query.stream(
            timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
            retry=None,
        )
    )


def get_admin_statistics(client: Any | None) -> dict[str, Any]:
    """Read the bounded daily/hourly summary documents used by the dashboard."""
    current_time = _now()
    today = current_time.date()
    day_keys = [
        (today - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(14)
    ]
    hour_keys = [
        (
            current_time.replace(minute=0, second=0, microsecond=0)
            - timedelta(hours=offset)
        ).strftime("%Y%m%d%H")
        for offset in range(2)
    ]

    def read_statistics() -> dict[str, Any]:
        deadline = time.monotonic() + HISTORY_LOAD_TIMEOUT_SECONDS
        stats_ref = client.collection("stats")
        global_data = (
            stats_ref.document("global")
            .get(
                timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
                retry=None,
            )
            .to_dict()
            or {}
        )
        daily_questions: dict[str, int] = {}
        daily_unique: dict[str, int] = {}
        daily_uids: dict[str, set[str]] = {}
        for day_key in day_keys:
            data = (
                stats_ref.document(f"daily_{day_key}")
                .get(
                    timeout=_remaining_timeout(
                        deadline, FIRESTORE_READ_TIMEOUT_SECONDS
                    ),
                    retry=None,
                )
                .to_dict()
                or {}
            )
            daily_questions[day_key] = int(data.get("questions", 0))
            uids = {str(value) for value in data.get("uids", [])}
            daily_uids[day_key] = uids
            daily_unique[day_key] = len(uids)
        active_uids: set[str] = set()
        fallback_active_uids: set[str] = set()
        active_since = current_time - timedelta(hours=1)
        for hour_key in hour_keys:
            data = (
                stats_ref.document(f"hourly_{hour_key}")
                .get(
                    timeout=_remaining_timeout(
                        deadline, FIRESTORE_READ_TIMEOUT_SECONDS
                    ),
                    retry=None,
                )
                .to_dict()
                or {}
            )
            active_at = data.get("active_at", {})
            if isinstance(active_at, dict):
                for uid, timestamp in active_at.items():
                    if isinstance(timestamp, datetime) and timestamp.tzinfo is None:
                        timestamp = timestamp.replace(tzinfo=_WIB)
                    if isinstance(timestamp, datetime) and timestamp >= active_since:
                        active_uids.add(str(uid))
            else:
                fallback_active_uids.update(
                    str(value) for value in data.get("uids", [])
                )
        return {
            "total_devices": int(global_data.get("total_devices", 0)),
            "active_last_hour": len(active_uids | fallback_active_uids),
            "active_window_label": (
                "perkiraan untuk ringkasan jam lama tanpa cap waktu"
                if fallback_active_uids
                else "dari cap waktu aktivitas dalam ringkasan per jam"
            ),
            "unique_today": daily_unique.get(today.strftime("%Y%m%d"), 0),
            "unique_7_days": len(
                {
                    uid
                    for offset, day_key in enumerate(day_keys)
                    if offset < 7
                    for uid in daily_uids[day_key]
                }
            ),
            "daily_questions": [
                {"day": day_key, "questions": daily_questions[day_key]}
                for day_key in reversed(day_keys)
            ],
        }

    return _bounded_read(
        client,
        read_statistics,
        timeout=HISTORY_LOAD_TIMEOUT_SECONDS,
        context="pembacaan ringkasan statistik",
        fallback={
            "total_devices": 0,
            "active_last_hour": 0,
            "active_window_label": "data tidak tersedia",
            "unique_today": 0,
            "unique_7_days": 0,
            "daily_questions": [],
        },
    )
def get_admin_users(client: Any | None) -> list[dict[str, Any]]:
    """Return a bounded page source for the dashboard's abbreviated user list."""

    def read_users() -> list[dict[str, Any]]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        snapshots = _snapshots(client.collection("users").limit(500), deadline)
        users = []
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            users.append(
                {
                    "uid": snapshot.id,
                    "short_id": snapshot.id[:8],
                    "total_questions": int(data.get("total_questions", 0)),
                    "day_count": int(data.get("day_count", 0)),
                    "day": str(data.get("day", "")),
                    "blocked": bool(data.get("blocked", False)),
                    "last_active": data.get("last_active"),
                }
            )
        return sorted(users, key=lambda user: user["uid"])

    return _bounded_read(
        client,
        read_users,
        context="pembacaan daftar pengguna admin",
        fallback=[],
    )


def get_admin_user_chats(client: Any | None, uid: str) -> list[dict[str, str]]:
    def known_chat_ids(user_data: dict[str, Any]) -> list[str]:
        values = user_data.get("chat_ids", [])
        if not isinstance(values, list):
            return []
        return [value for value in values if isinstance(value, str)]

    def read_chats() -> list[dict[str, str]]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        user_ref = client.collection("users").document(uid)
        user_data = (
            user_ref.get(
                timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
                retry=None,
            ).to_dict()
            or {}
        )
        chat_ids = set(known_chat_ids(user_data))
        active_chat_id = user_data.get("active_chat_id")
        if isinstance(active_chat_id, str) and active_chat_id:
            chat_ids.add(active_chat_id)
        snapshots = _snapshots(user_ref.collection("chats").limit(100), deadline)
        chat_ids.update(snapshot.id for snapshot in snapshots)
        return [
            {"chat_id": chat_id, "short_id": chat_id[:8]}
            for chat_id in sorted(chat_ids)
        ]

    return _bounded_read(
        client,
        read_chats,
        context="pembacaan daftar chat admin",
        fallback=[],
    )


def get_admin_chat_history(
    client: Any | None,
    uid: str,
    chat_id: str | None = None,
    keyword: str = "",
) -> list[dict[str, str]]:
    def read_history() -> list[dict[str, str]]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        user_ref = client.collection("users").document(uid)
        chats_ref = user_ref.collection("chats")
        if chat_id:
            chat_ids = [chat_id]
        else:
            user_data = (
                user_ref.get(
                    timeout=_remaining_timeout(
                        deadline, FIRESTORE_READ_TIMEOUT_SECONDS
                    ),
                    retry=None,
                ).to_dict()
                or {}
            )
            chat_ids = user_data.get("chat_ids", [])
            chat_ids = (
                [item for item in chat_ids if isinstance(item, str)]
                if isinstance(chat_ids, list)
                else []
            )
            active_chat_id = user_data.get("active_chat_id")
            if isinstance(active_chat_id, str) and active_chat_id:
                chat_ids.append(active_chat_id)
            chat_ids.extend(
                snapshot.id
                for snapshot in _snapshots(chats_ref.limit(100), deadline)
            )
            chat_ids = list(dict.fromkeys(chat_ids))[:100]
        query = keyword.casefold().strip()
        results: list[dict[str, str]] = []
        for selected_chat_id in chat_ids:
            chunks = _snapshots(
                chats_ref.document(selected_chat_id)
                .collection("chunks")
                .order_by("__name__"),
                deadline,
            )
            for chunk in chunks:
                data = chunk.to_dict() or {}
                for message in data.get("messages", []):
                    if not isinstance(message, dict):
                        continue
                    content = str(message.get("content", ""))
                    if query and query not in content.casefold():
                        continue
                    results.append(
                        {
                            "chat_id": selected_chat_id,
                            "role": str(message.get("role", "")),
                            "content": content,
                            "timestamp": str(message.get("timestamp", "")),
                        }
                    )
        return results[-500:]

    return _bounded_read(
        client,
        read_history,
        context="pencarian riwayat chat admin",
        fallback=[],
    )


def get_admin_faq(client: Any | None) -> list[dict[str, Any]]:
    def read_faq() -> list[dict[str, Any]]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        snapshots = _snapshots(
            client.collection("faq").order_by("count", direction="DESCENDING").limit(50),
            deadline,
        )
        return [
            {
                "question": str((snapshot.to_dict() or {}).get("display_question", "")),
                "count": int((snapshot.to_dict() or {}).get("count", 0)),
            }
            for snapshot in snapshots
        ]

    return _bounded_read(
        client,
        read_faq,
        context="pembacaan FAQ admin",
        fallback=[],
    )


def get_runtime_config(client: Any | None) -> dict[str, Any]:
    """Load maintenance and announcement settings at most every five minutes."""
    global _RUNTIME_CONFIG_LOADED_AT
    now = time.monotonic()
    with _RUNTIME_CONFIG_LOCK:
        if now - _RUNTIME_CONFIG_LOADED_AT < RUNTIME_CONFIG_CACHE_SECONDS:
            return dict(_RUNTIME_CONFIG)

    def read_config() -> dict[str, Any]:
        return (
            client.collection("settings")
            .document("runtime")
            .get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
            .to_dict()
            or {}
        )

    loaded = _bounded_read(
        client,
        read_config,
        timeout=RUNTIME_CONFIG_READ_WAIT_SECONDS,
        context="pembacaan konfigurasi runtime",
        fallback=None,
    )
    with _RUNTIME_CONFIG_LOCK:
        if loaded is not None:
            try:
                daily_question_limit = max(
                    0, int(loaded.get("daily_question_limit", 0))
                )
            except (TypeError, ValueError):
                daily_question_limit = 0
            _RUNTIME_CONFIG.update(
                {
                    "maintenance": bool(loaded.get("maintenance", False)),
                    "announcement": str(loaded.get("announcement", ""))[:1000],
                    "daily_question_limit": daily_question_limit,
                    "settings_password_hash": str(
                        loaded.get("settings_password_hash", "")
                    ),
                }
            )
        _RUNTIME_CONFIG_LOADED_AT = time.monotonic()
        return dict(_RUNTIME_CONFIG)


def update_runtime_config(client: Any | None, **updates: Any) -> None:
    global _RUNTIME_CONFIG_LOADED_AT
    allowed = {"maintenance", "announcement", "daily_question_limit"}
    sanitized = {key: value for key, value in updates.items() if key in allowed}
    if "announcement" in sanitized:
        sanitized["announcement"] = str(sanitized["announcement"])[:1000]
    if "daily_question_limit" in sanitized:
        sanitized["daily_question_limit"] = max(
            0, int(sanitized["daily_question_limit"])
        )
    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG.update(sanitized)
        _RUNTIME_CONFIG_LOADED_AT = time.monotonic()
    if client is None or _circuit_is_open():
        return
    _WRITE_EXECUTOR.submit(_write_runtime_config, client, sanitized)


def save_settings_password_hash(client: Any | None, encoded_hash: str) -> None:
    """Persist an admin-entered settings password as a hash."""
    global _RUNTIME_CONFIG_LOADED_AT
    if not encoded_hash.startswith("pbkdf2_sha256$"):
        raise ValueError("Format hash sandi Pengaturan tidak valid.")
    if client is None:
        raise RuntimeError("Firestore tidak tersedia untuk menyimpan sandi.")
    if _circuit_is_open():
        raise RuntimeError("Firestore sedang tidak tersedia; coba lagi nanti.")

    future = _WRITE_EXECUTOR.submit(_write_settings_password_hash, client, encoded_hash)
    try:
        future.result(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS)
    except FutureTimeoutError as error:
        future.cancel()
        timeout_error = TimeoutError(
            "Penyimpanan sandi Pengaturan melewati batas waktu."
        )
        _trip_circuit(timeout_error, "penyimpanan sandi Pengaturan")
        raise RuntimeError(str(timeout_error)) from error
    except Exception as error:
        _trip_circuit(error, "penyimpanan sandi Pengaturan")
        raise RuntimeError("Sandi Pengaturan tidak berhasil disimpan.") from error

    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG["settings_password_hash"] = encoded_hash
        _RUNTIME_CONFIG_LOADED_AT = time.monotonic()


def _write_settings_password_hash(client: Any, encoded_hash: str) -> None:
    batch = client.batch()
    batch.set(
        client.collection("settings").document("runtime"),
        {"settings_password_hash": encoded_hash},
        merge=True,
    )
    batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)


def _write_runtime_config(client: Any, config: dict[str, Any]) -> None:
    try:
        batch = client.batch()
        batch.set(client.collection("settings").document("runtime"), config, merge=True)
        batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        _trip_circuit(error, "penyimpanan konfigurasi runtime")


def set_user_blocked(client: Any | None, uid: str, blocked: bool) -> None:
    if client is None or _circuit_is_open():
        return
    _WRITE_EXECUTOR.submit(_write_user_blocked, client, uid, blocked)


def _write_user_blocked(client: Any, uid: str, blocked: bool) -> None:
    try:
        batch = client.batch()
        batch.set(
            client.collection("users").document(uid),
            {"blocked": blocked},
            merge=True,
        )
        batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        _trip_circuit(error, "perubahan status blokir pengguna")


def get_admin_recent_errors(client: Any | None) -> list[dict[str, str]]:
    with _RECENT_ERRORS_LOCK:
        local_errors = list(reversed(_RECENT_ERRORS))

    def read_errors() -> list[dict[str, str]]:
        data = (
            client.collection("system_health")
            .document("errors")
            .get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
            .to_dict()
            or {}
        )
        persisted = data.get("errors", [])
        return [
            item for item in persisted if isinstance(item, dict)
        ][-20:]

    persisted = _bounded_read(
        client,
        read_errors,
        context="pembacaan ringkasan error sistem",
        fallback=[],
    )
    return (local_errors + list(reversed(persisted)))[-20:]


def flush_recent_errors(client: Any | None) -> None:
    """Persist the in-memory error ring in one bounded Firestore document."""
    global _ERRORS_FLUSHED_AT
    if client is None or _circuit_is_open():
        return
    with _RECENT_ERRORS_LOCK:
        if not _RECENT_ERRORS or time.monotonic() - _ERRORS_FLUSHED_AT < 300:
            return
        pending = list(_RECENT_ERRORS)
        _ERRORS_FLUSHED_AT = time.monotonic()
    _WRITE_EXECUTOR.submit(_write_recent_errors, client, pending)


def _write_recent_errors(client: Any, pending: list[dict[str, str]]) -> None:
    try:
        reference = client.collection("system_health").document("errors")
        previous = (
            reference.get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None).to_dict()
            or {}
        ).get("errors", [])
        combined = [
            item for item in previous if isinstance(item, dict)
        ] + pending
        unique = {
            (item.get("timestamp"), item.get("context"), item.get("error")): item
            for item in combined
        }
        batch = client.batch()
        batch.set(reference, {"errors": list(unique.values())[-20:]})
        batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        _trip_circuit(error, "penyimpanan ringkasan error sistem")


def export_admin_users_csv(users: list[dict[str, Any]]) -> str:
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=(
            "short_id",
            "total_questions",
            "day_count",
            "day",
            "blocked",
        ),
    )
    writer.writeheader()
    writer.writerows(
        {key: user.get(key, "") for key in writer.fieldnames} for user in users
    )
    return output.getvalue()


def _chunk_number(document_id: str) -> int:
    try:
        return int(document_id)
    except ValueError:
        return 0


def _stats_uid(uid: str) -> str:
    """Ringkas UID agar dokumen statistik tetap kecil dan anonim."""
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]


def _transaction_snapshot(
    transaction: Any,
    document_ref: Any,
    timeout: float = FIRESTORE_READ_TIMEOUT_SECONDS,
) -> Any:
    """Ambil snapshot tunggal dari hasil iterator transaction.get()."""
    snapshot = next(
        iter(transaction.get(document_ref, timeout=timeout, retry=None)),
        None,
    )
    if snapshot is None:
        raise RuntimeError("Pembacaan dokumen Firestore tidak mengembalikan snapshot.")
    return snapshot


def append_message_pair(
    current_messages: list[dict[str, Any]],
    question: dict[str, Any],
    answer: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Tambahkan satu pasang pesan tanpa membuat chunk melebihi batas."""
    if len(current_messages) + 2 > MAX_MESSAGES_PER_CHUNK:
        return [question, answer], True
    return [*current_messages, question, answer], False


def load_chat_history(
    client: Any | None, uid: str, chat_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Muat maksimal dua chunk terakhir dan ringkasan hitungan pengguna."""
    deadline = time.monotonic() + HISTORY_LOAD_TIMEOUT_SECONDS

    def read_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        user_ref = client.collection("users").document(uid)
        user_data = (
            user_ref.get(
                timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
                retry=None,
            ).to_dict()
            or {}
        )
        chunks = (
            user_ref.collection("chats")
            .document(chat_id)
            .collection("chunks")
            .order_by("__name__", direction="DESCENDING")
            .limit(2)
            .stream(
                timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
                retry=None,
            )
        )
        recent_chunks = list(chunks)
        recent_chunks.sort(key=lambda item: _chunk_number(item.id))
        messages: list[dict[str, Any]] = []
        for chunk in recent_chunks:
            chunk_data = chunk.to_dict() or {}
            chunk_messages = chunk_data.get("messages", [])
            if isinstance(chunk_messages, list):
                messages.extend(
                    message for message in chunk_messages if isinstance(message, dict)
                )
        active_chunk = recent_chunks[-1].to_dict() or {} if recent_chunks else {}
        return messages[-30:], {
            "total_messages": int(user_data.get("total_messages", 0)),
            "total_chunks": int(user_data.get("total_chunks", 0)),
            "active_chunk_count": int(active_chunk.get("count", 0)),
            "active_chunk_number": (
                _chunk_number(recent_chunks[-1].id) if recent_chunks else 0
            ),
            "blocked": bool(user_data.get("blocked", False)),
            "day": str(user_data.get("day", "")),
            "day_count": int(user_data.get("day_count", 0)),
        }

    return _bounded_read(
        client,
        read_history,
        timeout=HISTORY_LOAD_TIMEOUT_SECONDS,
        context="pemuatan riwayat chat",
        fallback=_empty_history(),
    )


def get_active_chat_id(client: Any | None, uid: str) -> str | None:
    """Baca obrolan terakhir dari dokumen perangkat."""

    def read_active_chat_id() -> str | None:
        user_data = (
            client.collection("users")
            .document(uid)
            .get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
            .to_dict()
            or {}
        )
        chat_id = user_data.get("active_chat_id")
        return str(chat_id) if chat_id else None

    return _bounded_read(
        client,
        read_active_chat_id,
        timeout=0.2,
        context="pemuatan chat aktif",
        fallback=None,
    )


class SavedChatLimitError(ValueError):
    """Raised when a user tries to save more than the supported chat limit."""


class ChatStoreError(RuntimeError):
    """Raised when a named-chat operation cannot be persisted."""


def _run_chat_mutation(
    client: Any | None,
    operation: Any,
    *args: Any,
    context: str,
) -> Any:
    if client is None:
        raise ChatStoreError("Penyimpanan riwayat tidak tersedia.")
    if _circuit_is_open():
        raise ChatStoreError("Firestore sedang tidak tersedia; coba lagi nanti.")
    future = _WRITE_EXECUTOR.submit(operation, client, *args)
    try:
        return future.result(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS)
    except SavedChatLimitError:
        raise
    except FutureTimeoutError as error:
        future.cancel()
        timeout_error = TimeoutError(
            f"Operasi {context} melewati batas waktu."
        )
        _trip_circuit(timeout_error, context)
        raise ChatStoreError(str(timeout_error)) from error
    except Exception as error:
        _trip_circuit(error, context)
        raise ChatStoreError(f"Operasi {context} tidak berhasil.") from error


def _valid_saved_chats(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    saved: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        chat_id = str(item.get("chat_id", ""))
        name = str(item.get("name", "")).strip()
        if _CHAT_ID_RE.fullmatch(chat_id) and name:
            saved.append({"chat_id": chat_id, "name": name[:_SAVED_CHAT_NAME_LIMIT]})
    return saved


def _upsert_saved_chat(
    saved_chats: list[dict[str, str]], chat_id: str, name: str
) -> list[dict[str, str]]:
    existing = any(item["chat_id"] == chat_id for item in saved_chats)
    if not existing and len(saved_chats) >= SAVED_CHAT_LIMIT:
        raise SavedChatLimitError(
            f"Maksimal {SAVED_CHAT_LIMIT} paguneman tiasa disimpen. "
            "Pupus heula salah sahiji paguneman."
        )
    updated = [item for item in saved_chats if item["chat_id"] != chat_id]
    updated.insert(0, {"chat_id": chat_id, "name": name})
    return updated


def get_saved_chats(client: Any | None, uid: str) -> list[dict[str, str]]:
    """Read the user's named chat list."""
    if client is None:
        raise ChatStoreError("Penyimpanan riwayat tidak tersedia.")

    def read_saved_chats() -> list[dict[str, str]]:
        user_data = (
            client.collection("users")
            .document(uid)
            .get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
            .to_dict()
            or {}
        )
        return _valid_saved_chats(user_data.get("saved_chats", []))

    failed = object()
    result = _bounded_read(
        client,
        read_saved_chats,
        timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        context="pembacaan daftar chat simpan",
        fallback=failed,
    )
    if result is failed:
        raise ChatStoreError("Daftar chat simpan tidak berhasil dimuat.")
    return result


def save_named_chat(
    client: Any | None,
    uid: str,
    chat_id: str,
    name: str,
) -> list[dict[str, str]]:
    """Save or rename a chat, enforcing the per-device chat limit."""
    clean_name = name.strip()
    if not _CHAT_ID_RE.fullmatch(chat_id):
        raise ValueError("ID paguneman henteu valid.")
    if not clean_name:
        raise ValueError("Lebetkeun heula nami paguneman.")
    if len(clean_name) > _SAVED_CHAT_NAME_LIMIT:
        raise ValueError(
            f"Nami paguneman maksimal {_SAVED_CHAT_NAME_LIMIT} karakter."
        )
    return _run_chat_mutation(
        client,
        _save_named_chat,
        uid,
        chat_id,
        clean_name,
        context="nyimpen paguneman",
    )


def _save_named_chat(
    client: Any,
    uid: str,
    chat_id: str,
    name: str,
) -> list[dict[str, str]]:
    from google.cloud import firestore

    user_ref = client.collection("users").document(uid)
    transaction = client.transaction()

    @firestore.transactional
    def save(transaction: Any) -> list[dict[str, str]]:
        snapshot = _transaction_snapshot(transaction, user_ref)
        user_data = snapshot.to_dict() or {}
        saved = _valid_saved_chats(user_data.get("saved_chats", []))
        saved = _upsert_saved_chat(saved, chat_id, name)
        transaction.set(
            user_ref,
            {
                "saved_chats": saved,
                "chat_ids": firestore.ArrayUnion([chat_id]),
            },
            merge=True,
        )
        return saved

    return save(transaction)


def delete_user_chat(
    client: Any | None, uid: str, chat_id: str
) -> dict[str, int]:
    """Delete one chat's chunks and remove its saved/list metadata."""
    if not _CHAT_ID_RE.fullmatch(chat_id):
        raise ValueError("ID paguneman henteu valid.")
    return _run_chat_mutation(
        client,
        _delete_user_chat,
        uid,
        chat_id,
        context="ngahapus paguneman",
    )


def _delete_chat_contents(
    client: Any, user_ref: Any, chat_id: str
) -> tuple[int, int, int]:
    chat_ref = user_ref.collection("chats").document(chat_id)
    chunks = chat_ref.collection("chunks").stream(
        timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        retry=None,
    )
    batch = client.batch()
    pending = 0
    message_count = 0
    question_count = 0
    chunk_count = 0
    for chunk in chunks:
        chunk_data = chunk.to_dict() or {}
        messages = chunk_data.get("messages", [])
        if isinstance(messages, list):
            message_count += len(messages)
            question_count += sum(
                isinstance(message, dict) and message.get("role") == "user"
                for message in messages
            )
        batch.delete(chunk.reference)
        pending += 1
        chunk_count += 1
        if pending >= 450:
            batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
            batch = client.batch()
            pending = 0
    batch.delete(chat_ref)
    batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    return message_count, question_count, chunk_count


def _delete_user_chat(client: Any, uid: str, chat_id: str) -> dict[str, int]:
    user_ref = client.collection("users").document(uid)
    user_snapshot = user_ref.get(
        timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        retry=None,
    )
    user_data = user_snapshot.to_dict() or {}
    message_count, question_count, chunk_count = _delete_chat_contents(
        client, user_ref, chat_id
    )
    chat_ids = user_data.get("chat_ids", [])
    chat_ids = (
        [value for value in chat_ids if isinstance(value, str) and value != chat_id]
        if isinstance(chat_ids, list)
        else []
    )
    active_chat_id = user_data.get("active_chat_id")
    saved = [
        item
        for item in _valid_saved_chats(user_data.get("saved_chats", []))
        if item["chat_id"] != chat_id
    ]
    updated_stats = {
        "total_messages": max(
            0, int(user_data.get("total_messages", 0)) - message_count
        ),
        "total_questions": max(
            0, int(user_data.get("total_questions", 0)) - question_count
        ),
        "total_chunks": max(
            0, int(user_data.get("total_chunks", 0)) - chunk_count
        ),
    }
    user_ref.set(
        {
            "chat_ids": chat_ids,
            "saved_chats": saved,
            "active_chat_id": None if active_chat_id == chat_id else active_chat_id,
            **updated_stats,
        },
        merge=True,
        timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS,
    )
    return updated_stats


def delete_all_user_chats(client: Any | None, uid: str) -> None:
    """Delete every chat and clear saved-chat metadata for one device."""
    _run_chat_mutation(
        client,
        _delete_all_user_chats,
        uid,
        context="ngahapus sadaya paguneman",
    )


def _delete_all_user_chats(client: Any, uid: str) -> None:
    user_ref = client.collection("users").document(uid)
    user_snapshot = user_ref.get(
        timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        retry=None,
    )
    user_data = user_snapshot.to_dict() or {}
    raw_ids = user_data.get("chat_ids", [])
    chat_ids = (
        [item for item in raw_ids if isinstance(item, str)]
        if isinstance(raw_ids, list)
        else []
    )
    active_chat_id = user_data.get("active_chat_id")
    if isinstance(active_chat_id, str) and active_chat_id:
        chat_ids.append(active_chat_id)
    chat_ids.extend(
        item["chat_id"]
        for item in _valid_saved_chats(user_data.get("saved_chats", []))
    )
    chat_ids.extend(
        snapshot.id
        for snapshot in user_ref.collection("chats").stream(
            timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
            retry=None,
        )
    )

    for chat_id in dict.fromkeys(chat_ids):
        if _CHAT_ID_RE.fullmatch(chat_id):
            _delete_chat_contents(client, user_ref, chat_id)

    user_ref.set(
        {
            "chat_ids": [],
            "saved_chats": [],
            "active_chat_id": None,
            "total_questions": 0,
            "total_messages": 0,
            "total_chunks": 0,
        },
        merge=True,
        timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS,
    )


def enqueue_question_answer(
    client: Any | None,
    uid: str,
    chat_id: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
) -> None:
    """Jadwalkan satu penulisan pasangan pesan di thread latar."""
    if client is None or _circuit_is_open():
        return
    _WRITE_EXECUTOR.submit(
        _persist_question_answer,
        client,
        uid,
        chat_id,
        question_text,
        answer_text,
        sources,
    )


def _persist_question_answer(
    client: Any,
    uid: str,
    chat_id: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
) -> None:
    try:
        from google.cloud import firestore

        user_ref = client.collection("users").document(uid)
        chunks_ref = (
            user_ref.collection("chats").document(chat_id).collection("chunks")
        )
        now = _now()
        stats_uid = _stats_uid(uid)
        question = {
            "role": "user",
            "content": question_text,
            "timestamp": now,
            "model_used": None,
            "sources": [],
        }
        answer = {
            "role": "assistant",
            "content": answer_text,
            "timestamp": now,
            "model_used": None,
            "sources": sources,
        }
        user_snapshot = user_ref.get(
            timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
            retry=None,
        )
        user_data = user_snapshot.to_dict() or {}
        latest_chunks = list(
            chunks_ref.order_by(
                "__name__", direction=firestore.Query.DESCENDING
            )
            .limit(1)
            .stream(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
        )
        current_chunk = latest_chunks[0] if latest_chunks else None
        chunk_number = (
            _chunk_number(current_chunk.id) if current_chunk is not None else 0
        )
        current_messages = (
            (current_chunk.to_dict() or {}).get("messages", [])
            if current_chunk is not None
            else []
        )
        if not isinstance(current_messages, list):
            current_messages = []
        next_messages, needs_new_chunk = append_message_pair(
            current_messages, question, answer
        )
        if needs_new_chunk or current_chunk is None:
            chunk_number += 1
        chunk_ref = chunks_ref.document(f"{chunk_number:06d}")
        day = now.strftime("%Y%m%d")
        user_fields: dict[str, Any] = {
            "last_active": now,
            "active_chat_id": chat_id,
            "chat_ids": firestore.ArrayUnion([chat_id]),
            "active_chunk_number": chunk_number,
            "active_chunk_count": len(next_messages),
            "total_questions": firestore.Increment(1),
            "total_messages": firestore.Increment(2),
            "total_chunks": firestore.Increment(
                1 if needs_new_chunk or current_chunk is None else 0
            ),
            "day": day,
            "day_count": (
                int(user_data.get("day_count", 0)) + 1
                if user_data.get("day") == day
                else 1
            ),
        }
        if not user_snapshot.exists:
            user_fields.update({"first_seen": now, "blocked": False})

        normalized_question = normalize_question(question_text)
        batch = client.batch()
        batch.set(chunk_ref, {"messages": next_messages, "count": len(next_messages)})
        batch.set(user_ref, user_fields, merge=True)
        batch.set(
            client.collection("stats").document(f"daily_{day}"),
            {
                "questions": firestore.Increment(1),
                "uids": firestore.ArrayUnion([stats_uid]),
            },
            merge=True,
        )
        if normalized_question:
            faq_id = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
            batch.set(
                client.collection("faq").document(faq_id),
                {
                    "question": normalized_question,
                    "display_question": question_text[:500],
                    "count": firestore.Increment(1),
                    "last_seen": now,
                },
                merge=True,
            )
        batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        _trip_circuit(error, f"penyimpanan riwayat perangkat {uid[:6]}")


def mark_active(client: Any | None, uid: str) -> None:
    """Catat aktivitas paling banyak sekali per jam per sesi."""
    if client is None or _circuit_is_open():
        return
    now = _now()
    previous = st.session_state.get("last_active_written_at")
    hour_key = now.strftime("%Y%m%d%H")
    if (
        isinstance(previous, datetime)
        and previous.strftime("%Y%m%d%H") == hour_key
    ):
        return
    st.session_state.last_active_written_at = now
    _WRITE_EXECUTOR.submit(_mark_active, client, uid, now, hour_key)


def _mark_active(client: Any, uid: str, now: datetime, hour_key: str) -> None:
    try:
        from google.cloud import firestore

        user_ref = client.collection("users").document(uid)
        user_snapshot = user_ref.get(
            timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
            retry=None,
        )
        fields: dict[str, Any] = {"last_active": now}
        if not user_snapshot.exists:
            fields.update(
                {
                    "first_seen": now,
                    "total_questions": 0,
                    "total_messages": 0,
                    "total_chunks": 0,
                    "blocked": False,
                    "day": now.strftime("%Y%m%d"),
                    "day_count": 0,
                }
            )

        batch = client.batch()
        batch.set(user_ref, fields, merge=True)
        batch.set(
            client.collection("stats").document(f"hourly_{hour_key}"),
            {
                "uids": firestore.ArrayUnion([_stats_uid(uid)]),
                "active_at": {_stats_uid(uid): now},
            },
            merge=True,
        )
        batch.set(
            client.collection("stats").document(f"daily_{now:%Y%m%d}"),
            {"uids": firestore.ArrayUnion([_stats_uid(uid)])},
            merge=True,
        )
        if not user_snapshot.exists:
            batch.set(
                client.collection("stats").document("global"),
                {"total_devices": firestore.Increment(1)},
                merge=True,
            )
        batch.commit(timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        _trip_circuit(error, f"pencatatan aktivitas perangkat {uid[:6]}")


def clear_user_history(client: Any | None, uid: str) -> None:
    """Compatibility wrapper for callers that clear all chats."""
    delete_all_user_chats(client, uid)
