"""Penyimpanan riwayat dan statistik pengguna di Firestore."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
from collections.abc import Mapping, Sequence
import re
import threading
import time
import unicodedata
from typing import Any
from zoneinfo import ZoneInfo

import streamlit as st
from services.user_profile import validate_user_name

logger = logging.getLogger(__name__)
MAX_MESSAGES_PER_CHUNK = 20
HISTORY_WARNING_MESSAGES = 200
HISTORY_WARNING_CHUNKS = 10
FIRESTORE_READ_TIMEOUT_SECONDS = 5.0
FIRESTORE_WRITE_TIMEOUT_SECONDS = 10.0
FIRESTORE_ADMIN_TIMEOUT_SECONDS = 10.0
FIRESTORE_WRITE_QUEUE_LIMIT = 200
FIRESTORE_WRITE_RETRY_DELAYS = (1.0, 3.0, 8.0)
HISTORY_LOAD_TIMEOUT_SECONDS = 5.0
FIRESTORE_CIRCUIT_SECONDS = 120.0
RUNTIME_CONFIG_CACHE_SECONDS = 300.0
RUNTIME_CONFIG_READ_TIMEOUT_SECONDS = 5.0
RUNTIME_CONFIG_RETRY_SECONDS = 5.0
_WIB = ZoneInfo("Asia/Jakarta")
_WRITE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="firestore-write")
_READ_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="firestore-read")
_DEVICE_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_OPEN_UNTIL = 0.0
_CIRCUIT_FAILURE_LOGGED = False
_RUNTIME_CONFIG_LOCK = threading.Lock()
_RUNTIME_CONFIG_DEFAULTS: dict[str, Any] = {
    "maintenance": False,
    "announcement": "",
    "daily_question_limit": 0,
    "guest_daily_question_limit": 10,
    "guest_emergency_enabled": False,
    "guest_emergency_expires_at": "",
    "settings_password_hash": "",
}
_RUNTIME_CONFIG: dict[str, Any] = dict(_RUNTIME_CONFIG_DEFAULTS)
SAVED_CHAT_LIMIT = 5
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_SAVED_CHAT_NAME_LIMIT = 60
_RUNTIME_CONFIG_LOADED_AT = 0.0
_RUNTIME_CONFIG_LAST_ATTEMPT_AT = 0.0
_RUNTIME_CONFIG_REFRESH_ID = 0
_RUNTIME_CONFIG_REFRESHING = False
_RUNTIME_CONFIG_SKIPPED_TIMEOUTS = 0
_RUNTIME_CONFIG_VERSION = 0
_RECENT_ERRORS: list[dict[str, str]] = []
_RECENT_ERRORS_LOCK = threading.Lock()
_ERRORS_FLUSHED_AT = 0.0


@dataclass
class _QueuedWrite:
    client: Any | None
    kind: str
    operation: Any
    args: tuple[Any, ...]
    retries: int = 0
    ready_at: float = 0.0


_WRITE_QUEUE: deque[_QueuedWrite] = deque()
_WRITE_CONDITION = threading.Condition()
_WRITE_THREAD: threading.Thread | None = None
_LATEST_WRITE_CLIENT: Any | None = None
_WRITE_HEALTH_LOCK = threading.Lock()
_WRITE_HEALTH: dict[str, Any] = {
    "writes_ok": 0,
    "writes_failed": 0,
    "writes_retried": 0,
    "writes_dropped": 0,
    "last_error": "",
    "last_error_by_kind": {},
}


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


def get_device_firestore_client(client: Any | None) -> Any | None:
    """Expose the client to device-scoped operations only after UID resolution."""
    if not st.session_state.get("_device_uid_browser_resolved"):
        return None
    return client


def _now() -> datetime:
    return datetime.now(_WIB)


def _empty_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return [], {
        "name": "",
        "total_messages": 0,
        "total_chunks": 0,
        "active_chunk_count": 0,
        "active_chunk_number": 0,
        "blocked": False,
        "day": "",
        "day_count": 0,
        "daily_question_limit": 0,
    }


def firestore_status() -> dict[str, Any]:
    """Report the in-process Firestore circuit state without performing I/O."""
    with _CIRCUIT_LOCK:
        remaining = max(0.0, _CIRCUIT_OPEN_UNTIL - time.monotonic())
    return {
        "connected": remaining == 0,
        "circuit_open": remaining > 0,
        "circuit_open_seconds": round(remaining),
    }


def firestore_write_health() -> dict[str, Any]:
    """Return bounded write-queue and delivery health counters."""
    with _WRITE_HEALTH_LOCK:
        health = dict(_WRITE_HEALTH)
    with _WRITE_CONDITION:
        health["writes_queued"] = len(_WRITE_QUEUE)
    with _CIRCUIT_LOCK:
        remaining = max(0.0, _CIRCUIT_OPEN_UNTIL - time.monotonic())
    health["breaker_open_until"] = (
        (_now() + timedelta(seconds=remaining)).isoformat()
        if remaining
        else None
    )
    with _RUNTIME_CONFIG_LOCK:
        health["runtime_config_skipped_timeouts"] = (
            _RUNTIME_CONFIG_SKIPPED_TIMEOUTS
        )
    return health


def _safe_error_message(error: BaseException, *sensitive_values: str) -> str:
    message = str(error)
    for value in sensitive_values:
        if value:
            message = message.replace(value, "[disamarkan]")
    message = re.sub(r"\b[a-f0-9]{16,32}\b", "[UID]", message, flags=re.IGNORECASE)
    return message[:240]


def _firestore_failure_kind(error: BaseException) -> str | None:
    """Return a circuit-worthy RPC failure, excluding local timeout budgets."""
    code = getattr(error, "code", None)
    if callable(code):
        try:
            code = code()
        except Exception:
            code = None
    code_name = str(getattr(code, "name", None) or code or "").upper()
    error_name = type(error).__name__.upper().replace("_", "")
    if "UNAVAILABLE" in code_name or error_name in {
        "UNAVAILABLE",
        "SERVICEUNAVAILABLE",
    }:
        return "UNAVAILABLE"
    if "DEADLINE_EXCEEDED" in code_name or error_name == "DEADLINEEXCEEDED":
        return "DEADLINE_EXCEEDED"
    if "PERMISSION_DENIED" in code_name or error_name == "PERMISSIONDENIED":
        return "PERMISSION_DENIED"
    return None


def _is_local_timeout(error: BaseException) -> bool:
    return type(error) is TimeoutError or isinstance(error, FutureTimeoutError)


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
    failure_kind = _firestore_failure_kind(error)
    if failure_kind is None:
        return
    _record_recent_error(error, context)
    with _CIRCUIT_LOCK:
        now = time.monotonic()
        if _CIRCUIT_OPEN_UNTIL <= now:
            _CIRCUIT_OPEN_UNTIL = now + FIRESTORE_CIRCUIT_SECONDS
            _CIRCUIT_FAILURE_LOGGED = False
        should_log = not _CIRCUIT_FAILURE_LOGGED
        _CIRCUIT_FAILURE_LOGGED = True
    if should_log:
        logger.error(
            "Firestore dinonaktifkan sementara (%s) kind=%s error_class=%s: %s",
            context,
            failure_kind,
            type(error).__name__,
            _safe_error_message(error),
        )


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
        _record_recent_error(
            TimeoutError(f"Pembacaan melampaui batas {timeout:g} detik."),
            f"{context} (anggaran waktu lokal)",
        )
    except Exception as error:
        if _firestore_failure_kind(error) is None:
            _record_recent_error(
                RuntimeError(
                    f"{type(error).__name__}: {_safe_error_message(error)}"
                ),
                context,
            )
            logger.warning(
                "Firestore read failed context=%s error_class=%s: %s",
                context,
                type(error).__name__,
                _safe_error_message(error),
            )
        else:
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
    global _RUNTIME_CONFIG_LOADED_AT, _RUNTIME_CONFIG_LAST_ATTEMPT_AT
    global _RUNTIME_CONFIG_REFRESH_ID, _RUNTIME_CONFIG_REFRESHING
    global _RUNTIME_CONFIG_SKIPPED_TIMEOUTS, _RUNTIME_CONFIG_VERSION
    global _ERRORS_FLUSHED_AT
    global _LATEST_WRITE_CLIENT
    with _CIRCUIT_LOCK:
        _CIRCUIT_OPEN_UNTIL = 0.0
        _CIRCUIT_FAILURE_LOGGED = False
    with _WRITE_CONDITION:
        _WRITE_QUEUE.clear()
        _LATEST_WRITE_CLIENT = None
        _WRITE_CONDITION.notify_all()
    with _WRITE_HEALTH_LOCK:
        _WRITE_HEALTH.update(
            writes_ok=0,
            writes_failed=0,
            writes_retried=0,
            writes_dropped=0,
            last_error="",
            last_error_by_kind={},
        )
    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG_LOADED_AT = 0.0
        _RUNTIME_CONFIG_LAST_ATTEMPT_AT = 0.0
        _RUNTIME_CONFIG_REFRESH_ID += 1
        _RUNTIME_CONFIG_REFRESHING = False
        _RUNTIME_CONFIG_SKIPPED_TIMEOUTS = 0
        _RUNTIME_CONFIG_VERSION += 1
        _RUNTIME_CONFIG.clear()
        _RUNTIME_CONFIG.update(_RUNTIME_CONFIG_DEFAULTS)
    with _RECENT_ERRORS_LOCK:
        _RECENT_ERRORS.clear()
        _ERRORS_FLUSHED_AT = 0.0


def _enqueue_write(
    client: Any | None,
    kind: str,
    operation: Any,
    *args: Any,
) -> None:
    """Queue a bounded write; retries are scheduled without blocking the worker."""
    global _LATEST_WRITE_CLIENT, _WRITE_THREAD
    with _WRITE_CONDITION:
        if client is not None:
            _LATEST_WRITE_CLIENT = client
        _append_write_locked(
            _QueuedWrite(client=client, kind=kind, operation=operation, args=args)
        )
        if _WRITE_THREAD is None or not _WRITE_THREAD.is_alive():
            _WRITE_THREAD = threading.Thread(
                target=_write_queue_worker,
                name="firestore-write-queue",
                daemon=True,
            )
            _WRITE_THREAD.start()
        _WRITE_CONDITION.notify_all()


def _append_write_locked(job: _QueuedWrite) -> None:
    if len(_WRITE_QUEUE) >= FIRESTORE_WRITE_QUEUE_LIMIT:
        dropped = _WRITE_QUEUE.popleft()
        with _WRITE_HEALTH_LOCK:
            _WRITE_HEALTH["writes_dropped"] += 1
        logger.warning(
            "Firestore write dropped kind=%s reason=queue_full",
            dropped.kind,
        )
    _WRITE_QUEUE.append(job)


def _write_queue_worker() -> None:
    while True:
        with _WRITE_CONDITION:
            while True:
                now = time.monotonic()
                client_available = _LATEST_WRITE_CLIENT is not None
                circuit_wait = _circuit_remaining()
                ready_index = next(
                    (
                        index
                        for index, item in enumerate(_WRITE_QUEUE)
                        if item.ready_at <= now
                    ),
                    None,
                )
                if (
                    ready_index is not None
                    and client_available
                    and circuit_wait <= 0
                ):
                    job = _WRITE_QUEUE[ready_index]
                    del _WRITE_QUEUE[ready_index]
                    client = (
                        _LATEST_WRITE_CLIENT
                        if _LATEST_WRITE_CLIENT is not None
                        else job.client
                    )
                    break

                waits = [
                    item.ready_at - now
                    for item in _WRITE_QUEUE
                    if item.ready_at > now
                ]
                if circuit_wait > 0:
                    waits.append(circuit_wait)
                _WRITE_CONDITION.wait(timeout=max(0.01, min(waits)) if waits else None)

        try:
            _run_write_with_timeout(job, client)
        except Exception as error:
            _record_write_failure(job, error)
            if _is_retryable_write_error(error):
                if job.retries < len(FIRESTORE_WRITE_RETRY_DELAYS):
                    delay = FIRESTORE_WRITE_RETRY_DELAYS[job.retries]
                    job.retries += 1
                    job.ready_at = time.monotonic() + delay
                    with _WRITE_HEALTH_LOCK:
                        _WRITE_HEALTH["writes_retried"] += 1
                    with _WRITE_CONDITION:
                        _append_write_locked(job)
                        _WRITE_CONDITION.notify_all()
                else:
                    _open_write_circuit(error, f"penulisan {job.kind}")
                    job.retries = 0
                    job.ready_at = time.monotonic() + FIRESTORE_CIRCUIT_SECONDS
                    with _WRITE_CONDITION:
                        _append_write_locked(job)
                        _WRITE_CONDITION.notify_all()
            elif _firestore_failure_kind(error) is not None:
                _trip_circuit(error, f"penulisan {job.kind}")
        else:
            with _WRITE_HEALTH_LOCK:
                _WRITE_HEALTH["writes_ok"] += 1


def _circuit_remaining() -> float:
    with _CIRCUIT_LOCK:
        return max(0.0, _CIRCUIT_OPEN_UNTIL - time.monotonic())


def _run_write_with_timeout(job: _QueuedWrite, client: Any) -> None:
    finished = threading.Event()
    outcome: list[Exception | None] = []
    deadline = time.monotonic() + FIRESTORE_WRITE_TIMEOUT_SECONDS

    def run() -> None:
        try:
            job.operation(client, deadline, *job.args)
        except Exception as error:
            outcome.append(error)
        else:
            outcome.append(None)
        finally:
            finished.set()

    threading.Thread(
        target=run,
        name=f"firestore-write-{job.kind}",
        daemon=True,
    ).start()
    if not finished.wait(FIRESTORE_WRITE_TIMEOUT_SECONDS):
        raise TimeoutError("Firestore write deadline exceeded.")
    if outcome[0] is not None:
        raise outcome[0]


def _write_error_class(error: BaseException) -> str:
    if isinstance(error, ValueError):
        return "ValueError"
    return type(error).__name__


def _record_write_failure(job: _QueuedWrite, error: BaseException) -> None:
    error_class = _write_error_class(error)
    field_name = getattr(error, "field_name", "")
    error_message = _safe_error_message(error)
    details = f"{error_class}: {error_message}"
    if field_name:
        details = f"{details} (field={field_name})"
    with _WRITE_HEALTH_LOCK:
        _WRITE_HEALTH["writes_failed"] += 1
        _WRITE_HEALTH["last_error"] = f"{job.kind}: {details}"
        _WRITE_HEALTH["last_error_by_kind"][job.kind] = details
    _record_recent_error(
        RuntimeError(details),
        f"Firestore write {job.kind}",
    )
    logger.error(
        "Firestore write failed kind=%s error_class=%s field=%s error_message=%s",
        job.kind,
        error_class,
        field_name or "-",
        error_message,
    )


def _is_retryable_write_error(error: BaseException) -> bool:
    retryable = {
        "UNAVAILABLE",
        "SERVICEUNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "DEADLINEEXCEEDED",
        "ABORTED",
    }
    error_name = type(error).__name__.upper().replace("_", "")
    if error_name in {name.replace("_", "") for name in retryable} or isinstance(
        error, TimeoutError
    ):
        return True
    code = getattr(error, "code", None)
    if callable(code):
        code = code()
    code_name = getattr(code, "name", None) or str(code)
    return any(name in code_name.upper() for name in retryable)


def _open_write_circuit(error: BaseException, context: str) -> None:
    _trip_circuit(error, context)


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
        daily_guest_questions: dict[str, int] = {}
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
            daily_guest_questions[day_key] = int(data.get("guest_questions", 0))
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
                {"day": day_key, "questions": daily_questions[day_key], "guest_questions": daily_guest_questions[day_key]}
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
        today_key = _now().strftime("%Y%m%d")
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            name = str(data.get("name", "") or "").strip()
            users.append(
                {
                    "uid": snapshot.id,
                    "short_id": snapshot.id[:8],
                    "name": name,
                    "name_lower": str(
                        data.get("name_lower", "") or name.casefold()
                    ),
                    "total_questions": int(data.get("total_questions", 0)),
                    "day_count": int(data.get("day_count", 0)),
                    "day": str(data.get("day", "")),
                    "blocked": bool(data.get("blocked", False)),
                    "first_seen": data.get("first_seen"),
                    "last_active": data.get("last_active"),
                    "daily_question_limit": max(
                        0, int(data.get("daily_question_limit", 0))
                    ),
                }
            )
        name_counts = Counter(
            user["name_lower"] for user in users if user["name_lower"]
        )
        for user in users:
            user["duplicate_name"] = bool(
                user["name_lower"] and name_counts[user["name_lower"]] > 1
            )
        return sorted(users, key=lambda user: user["uid"])

    return _bounded_read(
        client,
        read_users,
        context="pembacaan daftar pengguna admin",
        fallback=[],
    )


def get_users_without_name(client: Any | None) -> list[str]:
    """List users without a valid name (empty, null, or whitespace only)."""
    if client is None:
        raise RuntimeError("Penyimpanan Firestore tidak tersedia.")
    from google.api_core.exceptions import GoogleAPICallError

    try:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        users_ref = client.collection("users")
        uids_without_name: list[str] = []
        for user_snapshot in _snapshots(users_ref, deadline):
            user_data = user_snapshot.to_dict() or {}
            name = user_data.get("name")
            if not isinstance(name, str) or not name.strip():
                uids_without_name.append(user_snapshot.id)
        return uids_without_name
    except (
        GoogleAPICallError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "Daftar pengguna tanpa nama gagal dimuat "
            f"({type(error).__name__}): {_safe_error_message(error)}"
        ) from error


def delete_users_without_name_and_recount(
    client: Any | None, candidate_uids: list[str]
) -> tuple[int, int]:
    """Delete only the previewed users without name and recalculate device count."""
    if client is None:
        raise RuntimeError("Penyimpanan Firestore tidak tersedia.")
    from google.api_core.exceptions import GoogleAPICallError

    try:
        deadline = time.monotonic() + FIRESTORE_ADMIN_TIMEOUT_SECONDS
        candidates = set(candidate_uids)
        still_without_name = set(get_users_without_name(client)) & candidates
        users_ref = client.collection("users")
        batch = client.batch()
        pending = 0
        for uid in sorted(still_without_name):
            user_ref = users_ref.document(uid)
            for chat_snapshot in _snapshots(user_ref.collection("chats"), deadline):
                batch.delete(user_ref.collection("chats").document(chat_snapshot.id))
                pending += 1
            batch.delete(user_ref)
            pending += 1
            if pending >= 450:
                batch.commit(
                    timeout=_remaining_timeout(
                        deadline, FIRESTORE_ADMIN_TIMEOUT_SECONDS
                    ),
                    retry=None,
                )
                batch = client.batch()
                pending = 0
        if pending:
            batch.commit(
                timeout=_remaining_timeout(
                    deadline, FIRESTORE_ADMIN_TIMEOUT_SECONDS
                ),
                retry=None,
            )

        remaining = len(_snapshots(users_ref, deadline))
        recount_batch = client.batch()
        recount_batch.set(
            client.collection("stats").document("global"),
            {"total_devices": remaining},
            merge=True,
        )
        recount_batch.commit(
            timeout=_remaining_timeout(deadline, FIRESTORE_ADMIN_TIMEOUT_SECONDS),
            retry=None,
        )
        return len(still_without_name), remaining
    except (
        GoogleAPICallError,
        OSError,
        RuntimeError,
        TimeoutError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "Penghapusan pengguna tanpa nama atau hitung ulang total perangkat gagal "
            f"({type(error).__name__}): {_safe_error_message(error)}"
        ) from error


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
        chat_ids.update(
            item["chat_id"]
            for item in _valid_saved_chats(user_data.get("saved_chats", []))
        )
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
            chunks_query = chats_ref.document(selected_chat_id).collection("chunks")
            if chat_id:
                chunks_query = (
                    chunks_query.select(["messages"])
                    .order_by("__name__", direction="DESCENDING")
                    .limit(2)
                )
            chunks = _snapshots(chunks_query, deadline)
            chunks.sort(key=lambda item: _chunk_number(item.id))
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


def _quality_summary_snapshot(client: Any) -> dict[str, Any]:
    return (
        client.collection("stats")
        .document("quality")
        .get(timeout=FIRESTORE_READ_TIMEOUT_SECONDS, retry=None)
        .to_dict()
        or {}
    )


def get_answer_quality_summary(client: Any | None) -> dict[str, Any]:
    """Read quality counters from the bounded summary document, never scan records."""
    return _bounded_read(
        client,
        lambda: _quality_summary_snapshot(client),
        context="pembacaan ringkasan kualitas jawaban",
        fallback={},
    )


def submit_answer_feedback(
    client: Any | None,
    *,
    uid: str,
    name: str,
    chat_id: str,
    answer_id: str,
    question: str,
    answer: str,
    rating: str,
    reason: str | None,
    quality_metrics: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
) -> bool:
    """Store one feedback record and update its aggregate in a single transaction."""
    if client is None:
        raise RuntimeError("Panyimpenan eupan balik teu sayogi.")
    if (
        not _DEVICE_UID_RE.fullmatch(uid)
        or not _CHAT_ID_RE.fullmatch(chat_id)
        or not re.fullmatch(r"[a-f0-9]{32}", answer_id)
    ):
        raise ValueError("ID eupan balik henteu valid.")
    if rating not in {"cocok", "Teu cocok"}:
        raise ValueError("Peunteun eupan balik henteu valid.")
    allowed_reasons = {"Kaluar tina kontéks", "Salah", "Teu lengkep"}
    if reason is not None and (rating != "Teu cocok" or reason not in allowed_reasons):
        raise ValueError("Alesan eupan balik henteu valid.")
    try:
        latency_ms = max(
            0, min(600_000, int(quality_metrics.get("latency_ms", 0)))
        )
        source_count = max(
            0, min(100, int(quality_metrics.get("source_count", len(sources))))
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Métrik kualitas jawaban henteu valid.") from error
    fallback = bool(quality_metrics.get("fallback", False))
    model_used = str(quality_metrics.get("model_used") or "")[:100]
    source_rows = [
        {
            "name": str(item.get("name", "Dokumen"))[:100],
            "excerpt": str(item.get("excerpt", ""))[:350],
        }
        for item in sources[:3]
        if isinstance(item, Mapping)
    ]
    now = _now()
    feedback_ref = client.collection("feedback").document(answer_id)
    summary_ref = client.collection("stats").document("quality")
    model_key = hashlib.sha256(model_used.casefold().encode("utf-8")).hexdigest()[:16]
    is_positive = rating == "cocok"
    feedback = {
        "answer_id": answer_id,
        "uid": uid,
        "name": str(name).strip()[:30],
        "name_lower": str(name).strip().casefold()[:60],
        "chat_id": chat_id,
        "created_at": now,
        "rating": rating,
        "reason": reason,
        "question": str(question)[:120],
        "answer": str(answer)[:120],
        "source_count": source_count,
        "fallback": fallback,
        "model_used": model_used,
        "sources": source_rows,
        "latency_ms": latency_ms,
    }

    try:
        from google.cloud import firestore

        @firestore.transactional
        def persist(transaction: Any) -> bool:
            existing = _transaction_snapshot(
                transaction,
                feedback_ref,
                timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
            )
            if existing.exists:
                return False
            summary_update: dict[str, Any] = {
                "feedback_count": firestore.Increment(1),
                "positive_count": firestore.Increment(1 if is_positive else 0),
                "negative_count": firestore.Increment(0 if is_positive else 1),
                "without_sources_count": firestore.Increment(
                    1 if source_count == 0 else 0
                ),
                "fallback_count": firestore.Increment(1 if fallback else 0),
                "latency_total_ms": firestore.Increment(latency_ms),
                "latency_count": firestore.Increment(1),
                "updated_at": now,
            }
            if model_used:
                summary_update["models"] = {
                    model_key: {
                        "name": model_used,
                        "feedback_count": firestore.Increment(1),
                        "latency_total_ms": firestore.Increment(latency_ms),
                        "latency_count": firestore.Increment(1),
                    }
                }
            transaction.set(feedback_ref, feedback)
            transaction.set(summary_ref, summary_update, merge=True)
            return True

        transaction = client.transaction()
        return persist(transaction)
    except Exception as error:
        raise RuntimeError(
            "Eupan balik teu hasil disimpen "
            f"({type(error).__name__})."
        ) from error


def get_answer_quality_page(
    client: Any | None,
    *,
    filter_type: str = "Sadayana",
    filter_value: str = "",
    start_after: Any | None = None,
    page_size: int = 25,
) -> dict[str, Any]:
    """Read one projected page of answer feedback with one optional filter."""
    if page_size < 1 or page_size > 25:
        raise ValueError("Ukuran kaca eupan balik kudu 1 dugi 25.")

    def read_page() -> dict[str, Any]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        query = client.collection("feedback").select(
            [
                "answer_id",
                "uid",
                "name",
                "name_lower",
                "chat_id",
                "created_at",
                "rating",
                "reason",
                "question",
                "answer",
                "source_count",
                "fallback",
                "model_used",
                "sources",
                "latency_ms",
            ]
        )
        filters = {
            "Teu cocok": ("rating", "Teu cocok"),
            "Tanpa sumber": ("source_count", 0),
            "Fallback tanpa AI": ("fallback", True),
        }
        if filter_type in filters:
            field_name, expected = filters[filter_type]
            query = query.where(field_name, "==", expected)
        elif filter_type == "Per model" and filter_value:
            query = query.where("model_used", "==", filter_value)
        elif filter_type == "Per nama" and filter_value:
            normalized_name = filter_value.casefold().strip()
            query = query.where("name_lower", ">=", normalized_name).where(
                "name_lower", "<=", normalized_name + "\uf8ff"
            )
            query = query.order_by("name_lower").order_by("__name__")
        else:
            query = query.order_by("__name__", direction="DESCENDING")
        query = query.limit(page_size)
        if start_after is not None:
            if filter_type == "Per nama" and isinstance(start_after, tuple):
                name_lower, answer_id = start_after
                cursor = {
                    "name_lower": name_lower,
                    "__name__": client.collection("feedback").document(answer_id),
                }
            else:
                answer_id = (
                    start_after[1]
                    if isinstance(start_after, tuple)
                    else start_after
                )
                cursor = {
                    "__name__": client.collection("feedback").document(answer_id)
                }
            query = query.start_after(cursor)
        snapshots = _snapshots(query, deadline)
        rows = []
        for snapshot in snapshots:
            row = snapshot.to_dict() or {}
            row["answer_id"] = str(row.get("answer_id") or snapshot.id)
            rows.append(row)
        return {
            "items": rows,
            "next_cursor": (
                (
                    (
                        str((snapshots[-1].to_dict() or {}).get("name_lower", "")),
                        str((snapshots[-1].to_dict() or {}).get("answer_id") or snapshots[-1].id),
                    )
                    if filter_type == "Per nama"
                    else str(
                        (snapshots[-1].to_dict() or {}).get("answer_id")
                        or snapshots[-1].id
                    )
                )
                if len(snapshots) == page_size
                else None
            ),
        }

    return _bounded_read(
        client,
        read_page,
        context="pembacaan kaca kualitas jawaban",
        fallback={"items": [], "next_cursor": None},
    )


def get_answer_reviews(
    client: Any | None,
    answer_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Fetch review labels for up to 25 answer IDs in a single query."""
    clean_ids = [
        value for value in dict.fromkeys(answer_ids[:25])
        if re.fullmatch(r"[a-f0-9]{32}", value)
    ]
    if not clean_ids:
        return {}

    def read_reviews() -> dict[str, dict[str, Any]]:
        query = (
            client.collection("reviews")
            .where("answer_id", "in", clean_ids)
            .select(["answer_id", "label", "ai_label", "ai_rationale", "updated_at"])
        )
        return {
            str(data["answer_id"]): data
            for snapshot in _snapshots(
                query, time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
            )
            if (data := snapshot.to_dict() or {}).get("answer_id")
        }

    return _bounded_read(
        client,
        read_reviews,
        context="pembacaan label tinjauan kualitas",
        fallback={},
    )


def save_answer_reviews(
    client: Any | None,
    reviews: Sequence[Mapping[str, Any]],
) -> None:
    """Write up to twenty human or AI review labels in one Firestore batch."""
    if client is None:
        raise RuntimeError("Panyimpenan tinjauan teu sayogi.")
    if len(reviews) > 20:
        raise ValueError("Maksimal 20 jawaban tiasa ditinjau sakaligus.")
    valid_labels = {"Sesuai", "Kaluar kontéks", "Salah", "Perlu sumber"}
    normalized_reviews: list[tuple[str, str | None, str | None, str]] = []
    for review in reviews:
        answer_id = str(review.get("answer_id", ""))
        raw_label = review.get("label")
        raw_ai_label = review.get("ai_label")
        label = raw_label if isinstance(raw_label, str) else None
        ai_label = raw_ai_label if isinstance(raw_ai_label, str) else None
        if (
            not re.fullmatch(r"[a-f0-9]{32}", answer_id)
            or (label not in valid_labels and ai_label not in valid_labels)
        ):
            raise ValueError("Label tinjauan kualitas henteu valid.")
        rationale = str(review.get("ai_rationale", ""))[:300]
        normalized_reviews.append((answer_id, label, ai_label, rationale))
    try:
        batch = client.batch()
        for answer_id, label, ai_label, rationale in normalized_reviews:
            payload: dict[str, Any] = {
                "answer_id": answer_id,
                "updated_at": _now(),
            }
            if label in valid_labels:
                payload["label"] = label
            if ai_label in valid_labels:
                payload["ai_label"] = ai_label
                payload["ai_rationale"] = rationale
            batch.set(
                client.collection("reviews").document(answer_id),
                payload,
                merge=True,
            )
        batch.commit(timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        raise RuntimeError(
            "Label tinjauan teu hasil disimpen "
            f"({type(error).__name__})."
        ) from error


def get_admin_user_page(
    client: Any | None,
    *,
    name_query: str = "",
    start_after: tuple[str, str] | None = None,
    page_size: int = 25,
) -> dict[str, Any]:
    """Read a projected, paginated user page and detect duplicate names in one query."""
    if page_size < 1 or page_size > 25:
        raise ValueError("Ukuran kaca pangguna kudu 1 dugi 25.")

    def read_page() -> dict[str, Any]:
        deadline = time.monotonic() + FIRESTORE_READ_TIMEOUT_SECONDS
        today_key = _now().strftime("%Y%m%d")
        query = client.collection("users").select(
            [
                "name",
                "name_lower",
                "first_seen",
                "last_active",
                "total_questions",
                "blocked",
                "day_count",
                "day",
                "daily_question_limit",
            ]
        )
        normalized_query = name_query.casefold().strip()
        if normalized_query:
            query = query.where("name_lower", ">=", normalized_query).where(
                "name_lower", "<=", normalized_query + "\uf8ff"
            )
            query = query.order_by("name_lower").order_by("__name__")
        else:
            query = query.order_by("__name__")
        query = query.limit(page_size)
        if start_after is not None:
            name_lower, uid = start_after
            cursor = {"__name__": client.collection("users").document(uid)}
            if normalized_query:
                cursor["name_lower"] = name_lower
            query = query.start_after(cursor)
        snapshots = _snapshots(query, deadline)
        names = list(
            dict.fromkeys(
                str((snapshot.to_dict() or {}).get("name_lower", "")).casefold()
                for snapshot in snapshots
                if (snapshot.to_dict() or {}).get("name_lower")
            )
        )
        duplicate_names: set[str] = set()
        if names:
            duplicate_query = (
                client.collection("users")
                .where("name_lower", "in", names[:25])
                .select(["name_lower"])
                .limit(100)
            )
            counts = Counter(
                str((snapshot.to_dict() or {}).get("name_lower", "")).casefold()
                for snapshot in _snapshots(duplicate_query, deadline)
            )
            duplicate_names = {
                name for name, count in counts.items() if count > 1
            }
        users = []
        for snapshot in snapshots:
            data = snapshot.to_dict() or {}
            try:
                day_count = max(0, int(data.get("day_count", 0)))
                question_count = max(0, int(data.get("total_questions", 0)))
                daily_limit = max(0, int(data.get("daily_question_limit", 0)))
            except (TypeError, ValueError, OverflowError):
                day_count = question_count = daily_limit = 0
            name_lower = str(data.get("name_lower", "")).casefold()
            day = str(data.get("day", ""))
            users.append(
                {
                    "uid": snapshot.id,
                    "short_id": snapshot.id[:6],
                    "name": str(data.get("name", "") or ""),
                    "first_seen": data.get("first_seen"),
                    "last_active": data.get("last_active"),
                    "total_questions": question_count,
                    "day_count": day_count,
                    "day": day,
                    "blocked": bool(data.get("blocked", False)),
                    "daily_question_limit": daily_limit,
                    "duplicate_name": name_lower in duplicate_names,
                    "no_questions": question_count == 0,
                    "question_spike": day == today_key and day_count >= 20,
                }
            )
        return {
            "items": users,
            "next_cursor": (
                (
                    (
                        str((snapshots[-1].to_dict() or {}).get("name_lower", "")).casefold()
                        if normalized_query
                        else ""
                    ),
                    snapshots[-1].id,
                )
                if len(snapshots) == page_size
                else None
            ),
        }

    return _bounded_read(
        client,
        read_page,
        context="pembacaan kaca pangguna admin",
        fallback={"items": [], "next_cursor": None},
    )


def update_admin_user_policy(
    client: Any | None,
    uid: str,
    *,
    blocked: bool,
    daily_question_limit: int,
) -> None:
    """Update a single user's block and daily-limit fields with one write."""
    if client is None:
        raise RuntimeError("Firestore teu sayogi pikeun ngarobah kawijakan pangguna.")
    if not _DEVICE_UID_RE.fullmatch(uid):
        raise ValueError("ID pangguna henteu valid.")
    if isinstance(daily_question_limit, bool) or not 0 <= daily_question_limit <= 10_000:
        raise ValueError("Batas harian kudu 0 dugi 10.000.")
    try:
        client.collection("users").document(uid).set(
            {
                "blocked": bool(blocked),
                "daily_question_limit": daily_question_limit,
            },
            merge=True,
            timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS,
        )
    except Exception as error:
        raise RuntimeError(
            "Kawijakan pangguna teu hasil disimpen "
            f"({type(error).__name__})."
        ) from error


def get_runtime_config(client: Any | None) -> dict[str, Any]:
    """Return cached settings immediately and refresh stale data in the background."""
    global _RUNTIME_CONFIG_LAST_ATTEMPT_AT, _RUNTIME_CONFIG_REFRESH_ID
    global _RUNTIME_CONFIG_REFRESHING
    now = time.monotonic()
    refresh: tuple[int, int] | None = None
    with _RUNTIME_CONFIG_LOCK:
        cached = dict(_RUNTIME_CONFIG)
        stale = now - _RUNTIME_CONFIG_LOADED_AT >= RUNTIME_CONFIG_CACHE_SECONDS
        should_refresh = (
            client is not None
            and stale
            and not _RUNTIME_CONFIG_REFRESHING
            and (
                _RUNTIME_CONFIG_LAST_ATTEMPT_AT == 0.0
                or now - _RUNTIME_CONFIG_LAST_ATTEMPT_AT
                >= RUNTIME_CONFIG_RETRY_SECONDS
            )
        )
        if should_refresh:
            _RUNTIME_CONFIG_REFRESHING = True
            _RUNTIME_CONFIG_LAST_ATTEMPT_AT = now
            _RUNTIME_CONFIG_REFRESH_ID += 1
            refresh = (_RUNTIME_CONFIG_REFRESH_ID, _RUNTIME_CONFIG_VERSION)
    if refresh is not None and client is not None:
        refresh_id, version = refresh
        future = _READ_EXECUTOR.submit(_read_runtime_config, client)
        future.add_done_callback(
            lambda completed: _finish_runtime_config_refresh(
                completed, refresh_id, version
            )
        )
        timer = threading.Timer(
            RUNTIME_CONFIG_READ_TIMEOUT_SECONDS,
            _runtime_config_budget_expired,
            args=(refresh_id,),
        )
        timer.daemon = True
        timer.start()
    emergency_enabled = bool(cached.get("guest_emergency_enabled", False))
    emergency_expires_at = str(
        cached.get("guest_emergency_expires_at", "") or ""
    )
    if emergency_enabled:
        try:
            expires_at = datetime.fromisoformat(emergency_expires_at)
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            if (
                expires_at <= now_utc
                or expires_at > now_utc + timedelta(hours=6)
            ):
                emergency_enabled = False
        except ValueError:
            emergency_enabled = False
    cached["guest_emergency_enabled"] = emergency_enabled
    cached["guest_emergency_expires_at"] = emergency_expires_at
    return cached


def _read_runtime_config(client: Any) -> dict[str, Any]:
    return (
        client.collection("settings")
        .document("runtime")
        .get(timeout=RUNTIME_CONFIG_READ_TIMEOUT_SECONDS, retry=None)
        .to_dict()
        or {}
    )


def _runtime_config_budget_expired(refresh_id: int) -> None:
    global _RUNTIME_CONFIG_REFRESHING, _RUNTIME_CONFIG_SKIPPED_TIMEOUTS
    with _RUNTIME_CONFIG_LOCK:
        if (
            refresh_id == _RUNTIME_CONFIG_REFRESH_ID
            and _RUNTIME_CONFIG_REFRESHING
        ):
            _RUNTIME_CONFIG_SKIPPED_TIMEOUTS += 1
            _RUNTIME_CONFIG_REFRESHING = False


def _finish_runtime_config_refresh(
    future: Any,
    refresh_id: int,
    version: int,
) -> None:
    global _RUNTIME_CONFIG_LOADED_AT, _RUNTIME_CONFIG_REFRESHING
    try:
        loaded = future.result()
    except Exception as error:
        if _is_local_timeout(error):
            _runtime_config_budget_expired(refresh_id)
        else:
            if _firestore_failure_kind(error) is not None:
                _trip_circuit(error, "pembacaan konfigurasi runtime")
            else:
                logger.warning(
                    "Firestore read failed context=pembacaan konfigurasi runtime "
                    "error_class=%s: %s",
                    type(error).__name__,
                    _safe_error_message(error),
                )
            _record_recent_error(
                RuntimeError(
                    f"{type(error).__name__}: {_safe_error_message(error)}"
                ),
                "pembacaan konfigurasi runtime",
            )
        with _RUNTIME_CONFIG_LOCK:
            if refresh_id == _RUNTIME_CONFIG_REFRESH_ID:
                _RUNTIME_CONFIG_REFRESHING = False
        return

    try:
        daily_question_limit = max(
            0, int(loaded.get("daily_question_limit", 0))
        )
    except (TypeError, ValueError):
        daily_question_limit = 0
    try:
        guest_daily_question_limit = max(
            1, min(10_000, int(loaded.get("guest_daily_question_limit", 10)))
        )
    except (TypeError, ValueError):
        guest_daily_question_limit = 10
    guest_emergency_expires_at = str(
        loaded.get("guest_emergency_expires_at", "") or ""
    )[:64]
    sanitized = {
        "maintenance": bool(loaded.get("maintenance", False)),
        "announcement": str(loaded.get("announcement", ""))[:1000],
        "daily_question_limit": daily_question_limit,
        "guest_daily_question_limit": guest_daily_question_limit,
        "guest_emergency_enabled": bool(
            loaded.get("guest_emergency_enabled", False)
        ),
        "guest_emergency_expires_at": guest_emergency_expires_at,
        "settings_password_hash": str(
            loaded.get("settings_password_hash", "")
        ),
    }
    with _RUNTIME_CONFIG_LOCK:
        if (
            refresh_id == _RUNTIME_CONFIG_REFRESH_ID
            and version == _RUNTIME_CONFIG_VERSION
            and _RUNTIME_CONFIG_REFRESHING
        ):
            _RUNTIME_CONFIG.update(sanitized)
            _RUNTIME_CONFIG_LOADED_AT = time.monotonic()
        if refresh_id == _RUNTIME_CONFIG_REFRESH_ID:
            _RUNTIME_CONFIG_REFRESHING = False


def update_runtime_config(client: Any | None, **updates: Any) -> None:
    global _RUNTIME_CONFIG_LOADED_AT, _RUNTIME_CONFIG_VERSION
    allowed = {
        "maintenance",
        "announcement",
        "daily_question_limit",
        "guest_daily_question_limit",
        "guest_emergency_enabled",
        "guest_emergency_expires_at",
    }
    sanitized = {key: value for key, value in updates.items() if key in allowed}
    if "announcement" in sanitized:
        sanitized["announcement"] = str(sanitized["announcement"])[:1000]
    if "daily_question_limit" in sanitized:
        sanitized["daily_question_limit"] = max(
            0, int(sanitized["daily_question_limit"])
        )
    if "guest_daily_question_limit" in sanitized:
        limit = int(sanitized["guest_daily_question_limit"])
        if isinstance(sanitized["guest_daily_question_limit"], bool) or not (
            1 <= limit <= 10_000
        ):
            raise ValueError("Batas tamu kudu 1 dugi 10.000.")
        sanitized["guest_daily_question_limit"] = limit
    if "guest_emergency_enabled" in sanitized:
        sanitized["guest_emergency_enabled"] = bool(
            sanitized["guest_emergency_enabled"]
        )
    if "guest_emergency_expires_at" in sanitized:
        expires_at = str(sanitized["guest_emergency_expires_at"] or "")
        if expires_at:
            try:
                parsed_expiry = datetime.fromisoformat(expires_at)
            except ValueError as error:
                raise ValueError("Waktu kedaluwarsa mode tamu tidak valid.") from error
            if parsed_expiry.tzinfo is None:
                parsed_expiry = parsed_expiry.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            if (
                "guest_emergency_enabled" in sanitized
                and sanitized["guest_emergency_enabled"]
                and not (
                    now_utc < parsed_expiry <= now_utc + timedelta(hours=6)
                )
            ):
                raise ValueError("Katup darurat hanya boleh aktif maksimal 6 jam.")
        sanitized["guest_emergency_expires_at"] = expires_at[:64]
    if client is None:
        raise RuntimeError("Firestore tidak tersedia untuk menyimpan kontrol admin.")
    try:
        batch = client.batch()
        batch.set(
            client.collection("settings").document("runtime"),
            sanitized,
            merge=True,
        )
        batch.commit(timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        raise RuntimeError(
            "Penyimpanan kontrol admin gagal "
            f"({type(error).__name__}): {_safe_error_message(error)}"
        ) from error
    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG.update(sanitized)
        _RUNTIME_CONFIG_LOADED_AT = time.monotonic()
        _RUNTIME_CONFIG_VERSION += 1


def save_settings_password_hash(client: Any | None, encoded_hash: str) -> None:
    """Persist an admin-entered settings password as a hash."""
    global _RUNTIME_CONFIG_LOADED_AT, _RUNTIME_CONFIG_VERSION
    if not encoded_hash.startswith("pbkdf2_sha256$"):
        raise ValueError("Format hash sandi Pengaturan tidak valid.")
    if client is None:
        raise RuntimeError("Firestore tidak tersedia untuk menyimpan sandi.")
    try:
        _write_settings_password_hash(client, encoded_hash)
    except Exception as error:
        raise RuntimeError(
            "Penyimpanan sandi Pengaturan gagal "
            f"({type(error).__name__}): {_safe_error_message(error, encoded_hash)}"
        ) from error

    with _RUNTIME_CONFIG_LOCK:
        _RUNTIME_CONFIG["settings_password_hash"] = encoded_hash
        _RUNTIME_CONFIG_LOADED_AT = time.monotonic()
        _RUNTIME_CONFIG_VERSION += 1


def _write_settings_password_hash(client: Any, encoded_hash: str) -> None:
    batch = client.batch()
    batch.set(
        client.collection("settings").document("runtime"),
        {"settings_password_hash": encoded_hash},
        merge=True,
    )
    batch.commit(timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS, retry=None)


def set_user_blocked(client: Any | None, uid: str, blocked: bool) -> None:
    if client is None:
        raise RuntimeError("Firestore tidak tersedia untuk mengubah status pengguna.")
    _write_user_blocked(client, uid, blocked)


def _write_user_blocked(client: Any, uid: str, blocked: bool) -> None:
    try:
        batch = client.batch()
        batch.set(
            client.collection("users").document(uid),
            {"blocked": blocked},
            merge=True,
        )
        batch.commit(timeout=FIRESTORE_ADMIN_TIMEOUT_SECONDS, retry=None)
    except Exception as error:
        raise RuntimeError(
            "Perubahan status pengguna gagal "
            f"({type(error).__name__}): {_safe_error_message(error, uid)}"
        ) from error


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
            "name": str(user_data.get("name", "")),
            "total_messages": int(user_data.get("total_messages", 0)),
            "total_chunks": int(user_data.get("total_chunks", 0)),
            "active_chunk_count": int(active_chunk.get("count", 0)),
            "active_chunk_number": (
                _chunk_number(recent_chunks[-1].id) if recent_chunks else 0
            ),
            "blocked": bool(user_data.get("blocked", False)),
            "day": str(user_data.get("day", "")),
            "day_count": int(user_data.get("day_count", 0)),
            "daily_question_limit": max(
                0, int(user_data.get("daily_question_limit", 0))
            ),
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


def save_user_name(client: Any | None, uid: str, name: str) -> None:
    """Create the user's profile on deliberate submission, or update its name."""
    if not _DEVICE_UID_RE.fullmatch(uid):
        raise ValueError("ID perangkat henteu valid.")
    clean_name = validate_user_name(name)
    _run_chat_mutation(
        client,
        _save_user_name,
        uid,
        clean_name,
        context="nyimpen nami pangguna",
    )


def _save_user_name(client: Any, uid: str, name: str) -> None:
    from google.cloud import firestore

    user_ref = client.collection("users").document(uid)
    now = _now()

    @firestore.transactional
    def save_profile(transaction: Any) -> None:
        user_snapshot = _transaction_snapshot(
            transaction,
            user_ref,
            timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        )
        user_fields: dict[str, Any] = {
            "name": name,
            "name_lower": name.casefold(),
        }
        if not user_snapshot.exists:
            user_fields.update(
                {
                    "first_seen": now,
                    "total_questions": 0,
                    "total_messages": 0,
                    "total_chunks": 0,
                    "blocked": False,
                }
            )
        transaction.set(user_ref, user_fields, merge=True)
        if not user_snapshot.exists:
            transaction.set(
                client.collection("stats").document("global"),
                {"total_devices": firestore.Increment(1)},
                merge=True,
            )

    save_profile(client.transaction())


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
    snapshot = user_ref.get(
        timeout=FIRESTORE_READ_TIMEOUT_SECONDS,
        retry=None,
    )
    user_data = snapshot.to_dict() or {}
    saved = _valid_saved_chats(user_data.get("saved_chats", []))
    saved = _upsert_saved_chat(saved, chat_id, name)
    user_ref.set(
        {
            "saved_chats": saved,
            "chat_ids": firestore.ArrayUnion([chat_id]),
        },
        merge=True,
        timeout=FIRESTORE_WRITE_TIMEOUT_SECONDS,
        retry=None,
    )
    return saved


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
    *,
    count_for_daily_limit: bool = True,
    answer_id: str | None = None,
    quality_metrics: Mapping[str, Any] | None = None,
    model_used: str | None = None,
) -> None:
    """Queue history and each independent question summary write."""
    if client is None:
        return
    now = _now()
    _enqueue_write(
        client,
        "question_pair",
        _write_question_pair,
        uid,
        chat_id,
        question_text,
        answer_text,
        sources,
        now,
        count_for_daily_limit,
        answer_id,
        quality_metrics,
        model_used,
    )
    _enqueue_write(
        client,
        "daily_questions",
        _write_daily_questions,
        question_text,
        _stats_uid(uid),
        now,
    )
    normalized_question = normalize_question(question_text)
    if normalized_question:
        _enqueue_write(
            client,
            "faq",
            _write_faq_question,
            normalized_question,
            question_text[:500],
            now,
        )


def enqueue_guest_question(client: Any | None) -> None:
    """Queue only the permitted guest counter update in the daily stats document."""
    if client is None:
        return
    _enqueue_write(
        client,
        "daily_questions",
        _write_daily_questions,
        "",
        "",
        _now(),
        True,
    )


def _write_question_pair(
    client: Any,
    deadline: float,
    uid: str,
    chat_id: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
    now: datetime,
    count_for_daily_limit: bool,
    answer_id: str | None = None,
    quality_metrics: Mapping[str, Any] | None = None,
    model_used: str | None = None,
) -> None:
    from google.cloud import firestore

    user_ref = client.collection("users").document(uid)
    chunks_ref = user_ref.collection("chats").document(chat_id).collection("chunks")

    @firestore.transactional
    def write_pair(transaction: Any) -> None:
        user_snapshot = _transaction_snapshot(
            transaction,
            user_ref,
            timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
        )
        if not user_snapshot.exists:
            raise ChatStoreError(
                "Profil kedah disimpen heula sateuacan nyimpen paguneman."
            )
        latest_chunks = list(
            transaction.get(
                chunks_ref.order_by(
                    "__name__", direction=firestore.Query.DESCENDING
                ).limit(1),
                timeout=_remaining_timeout(deadline, FIRESTORE_READ_TIMEOUT_SECONDS),
                retry=None,
            )
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
            "model_used": str(
                model_used or (quality_metrics or {}).get("model_used") or ""
            )[:100]
            or None,
            "sources": sources,
        }
        if isinstance(answer_id, str) and re.fullmatch(r"[a-f0-9]{32}", answer_id):
            answer["answer_id"] = answer_id
        if quality_metrics is not None:
            try:
                latency_ms = int(quality_metrics.get("latency_ms", 0))
            except (TypeError, ValueError, OverflowError):
                latency_ms = 0
            try:
                source_count = int(
                    quality_metrics.get("source_count", len(sources))
                )
            except (TypeError, ValueError, OverflowError):
                source_count = len(sources)
            answer["latency_ms"] = max(0, min(600_000, latency_ms))
            answer["source_count"] = max(0, min(100, source_count))
            top_score = quality_metrics.get("top_score")
            answer["top_score"] = (
                float(top_score)
                if isinstance(top_score, (int, float))
                and not isinstance(top_score, bool)
                and math.isfinite(float(top_score))
                else None
            )
            answer["fallback"] = bool(quality_metrics.get("fallback", False))
            raw_context = quality_metrics.get("source_context", [])
            if isinstance(raw_context, list):
                answer["source_context"] = [
                    {
                        "name": str(item.get("name", "Dokumen"))[:100],
                        "excerpt": str(item.get("excerpt", ""))[:350],
                    }
                    for item in raw_context[:3]
                    if isinstance(item, Mapping)
                ]
        next_messages, needs_new_chunk = append_message_pair(
            current_messages, question, answer
        )
        if needs_new_chunk or current_chunk is None:
            chunk_number += 1
        chunk_ref = chunks_ref.document(f"{chunk_number:06d}")
        day = now.strftime("%Y%m%d")
        user_data = user_snapshot.to_dict() or {}
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
        }
        if count_for_daily_limit:
            user_fields.update(
                {
                    "day": day,
                    "day_count": (
                        int(user_data.get("day_count", 0)) + 1
                        if user_data.get("day") == day
                        else 1
                    ),
                }
            )
        transaction.set(
            chunk_ref,
            {"messages": next_messages, "count": len(next_messages)},
        )
        transaction.set(user_ref, user_fields, merge=True)
        hashed_uid = _stats_uid(uid)
        hour_key = now.strftime("%Y%m%d%H")
        transaction.set(
            client.collection("stats").document(f"hourly_{hour_key}"),
            {
                "uids": firestore.ArrayUnion([hashed_uid]),
                "active_at": {hashed_uid: now},
            },
            merge=True,
        )

    write_pair(client.transaction())


def _write_daily_questions(
    client: Any,
    deadline: float,
    question_text: str,
    stats_uid: str,
    now: datetime,
    guest: bool = False,
) -> None:
    from google.cloud import firestore

    batch = client.batch()
    daily_fields = (
        {"guest_questions": firestore.Increment(1)}
        if guest
        else {
            "questions": firestore.Increment(1),
            "uids": firestore.ArrayUnion([stats_uid]),
        }
    )
    batch.set(
        client.collection("stats").document(f"daily_{now:%Y%m%d}"),
        daily_fields,
        merge=True,
    )
    batch.commit(
        timeout=_remaining_timeout(deadline, FIRESTORE_WRITE_TIMEOUT_SECONDS),
        retry=None,
    )


def _write_faq_question(
    client: Any,
    deadline: float,
    normalized_question: str,
    display_question: str,
    now: datetime,
) -> None:
    from google.cloud import firestore

    faq_id = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
    batch = client.batch()
    batch.set(
        client.collection("faq").document(faq_id),
        {
            "question": normalized_question,
            "display_question": display_question,
            "count": firestore.Increment(1),
            "last_seen": now,
        },
        merge=True,
    )
    batch.commit(
        timeout=_remaining_timeout(deadline, FIRESTORE_WRITE_TIMEOUT_SECONDS),
        retry=None,
    )


class _FirestoreFieldValueError(ValueError):
    def __init__(self, message: str, *, field_name: str) -> None:
        super().__init__(message)
        self.field_name = field_name


def clear_user_history(client: Any | None, uid: str) -> None:
    """Compatibility wrapper for callers that clear all chats."""
    delete_all_user_chats(client, uid)
