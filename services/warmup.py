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

# (modul, [nama fungsi alternatif], [variasi argumen yang dicoba berurutan])
_TASKS = (
    ("services.firestore_store", ["get_runtime_config"], [()]),
    ("services.key_pool", ["get_keys"], [("chat",), ()]),
    ("services.key_pool", ["get_model_order", "ordered_models"], [()]),
    ("services.admin_session", ["get_current_epoch", "get_epoch", "current_epoch"], [()]),
)


def _run_task(module_name: str, func_names: list, arg_variants: list) -> None:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return
    for name in func_names:
        func = getattr(module, name, None)
        if not callable(func):
            continue
        for args in arg_variants:
            try:
                func(*args)
                return
            except TypeError:
                continue
            except Exception as exc:  # jaringan/Firestore: abaikan
                _LOG.debug("warmup %s.%s gagal: %s", module_name, name, exc)
                return


def _worker() -> None:
    for module_name, func_names, arg_variants in _TASKS:
        _run_task(module_name, func_names, arg_variants)


def warm_start_once() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(target=_worker, name="jh-warmup", daemon=True).start()
