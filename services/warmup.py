"""Pemanasan cache in-process di background, sekali per proses server.

Tujuan: ketika user pertama kali mengirim pesan / membuka Admin, runtime config,
API key, urutan model, dan epoch sesi admin sudah ada di memori sehingga tidak
menunggu Firestore di depan layar.

Aman dipanggil di setiap run: hanya berjalan sekali. Kegagalan apa pun diabaikan
(circuit breaker Firestore tetap melindungi). Sesuaikan nama fungsi di ``_TASKS``
jika berbeda dari dokumen arsitektur.
"""
from __future__ import annotations

import importlib
import logging
import threading

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_STARTED = False

# (modul, nama fungsi, kelompok key opsional)
_TASKS = (
    ("services.firestore_store", "get_runtime_config", None),
    ("services.key_pool", "get_keys", "chat"),
    ("services.key_pool", "get_keys", "indexing"),
    ("services.key_pool", "get_keys", "admin"),
    ("services.key_pool", "get_model_order", None),
    ("services.admin_session", "get_admin_session_epoch", None),
)


def _run_task(
    module_name: str,
    function_name: str,
    key_group: str | None,
    client: object,
    secrets: dict[str, object],
) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return
    func = getattr(module, function_name, None)
    if not callable(func):
        return
    if function_name == "get_runtime_config":
        args = (client,)
    elif function_name == "get_keys":
        if key_group is None:
            return
        args = (key_group, client, secrets)
    elif function_name == "get_model_order":
        args = (client, secrets)
    else:
        args = (client,)
    try:
        func(*args)
    except Exception as exc:  # jaringan/Firestore: abaikan
        _LOG.debug("warmup %s.%s gagal: %s", module_name, function_name, exc)


def _worker(client: object, secrets: dict[str, object]) -> None:
    for module_name, function_name, key_group in _TASKS:
        _run_task(module_name, function_name, key_group, client, secrets)


def warm_start_once(
    client: object | None = None,
    secrets: dict[str, object] | None = None,
) -> None:
    global _STARTED
    if client is None:
        return
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(
        target=_worker,
        args=(client, dict(secrets or {})),
        name="jh-warmup",
        daemon=True,
    ).start()
