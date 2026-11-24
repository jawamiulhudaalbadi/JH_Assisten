"""Local Device Cache untuk JH Asisten (localStorage browser, ringan, non-blocking).

Prinsip desain
--------------
* Cache hanya untuk TAMPILAN (restore chat aktif + UID). Otorisasi (admin, blokir,
  limit harian) tetap divalidasi server -- jangan pernah percaya data dari browser.
* Satu komponen Streamlit tanpa build (HTML statis), tinggi 0, dipasang SEKALI per run
  lewat ``persist()`` di akhir script. Payload dikirim ke browser hanya jika berubah.
* Hasil baca browser ("hydrate") tiba sebagai nilai komponen dan dibaca di awal run
  berikutnya lewat ``st.session_state["jh_lc"]``; ``mount()`` memakainya.

API (3 fungsi yang dipanggil dari app.py):
    mount(uid, server_loader=None)   # awal script, setelah UID diketahui
    persist(uid, chat_id, messages)  # akhir script (WAJIB dipanggil tiap run)
    clear(uid)                       # saat "Chat baru" / hapus riwayat / keluar

Bantuan untuk device_identity.py:
    hydrated_uid(), component_ready(), remember_uid(uid)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import streamlit as st
import streamlit.components.v1 as components

SCHEMA_VERSION = 1
MAX_MESSAGES = 30  # sama dengan load_chat_history() (2 chunk terakhir)
MAX_CONTENT_CHARS = 8000  # potong jawaban super panjang
MAX_SNAPSHOT_BYTES = 120_000  # aman untuk HP jadul (kuota localStorage ~5 MB)
TTL_SECONDS = 14 * 24 * 3600  # snapshot kedaluwarsa 14 hari

_STATE_KEY = "jh_lc"
_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_KEEP_FIELDS = ("role", "content", "timestamp", "model_used", "sources", "answer_id")

_component = components.declare_component(
    "jh_local_cache",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_cache_component"),
)
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jh-lc")


# --------------------------------------------------------------------------- helpers
def _hydrated() -> dict:
    value = st.session_state.get(_STATE_KEY)
    return value if isinstance(value, dict) else {}


def component_ready() -> bool:
    """True jika browser sudah membalas (sudah hydrate) di sesi ini."""
    return bool(_hydrated().get("ready"))


def hydrated_uid() -> Optional[str]:
    """UID dari localStorage browser (None jika belum ada / belum siap)."""
    uid = _hydrated().get("uid")
    return uid if isinstance(uid, str) and _UID_RE.match(uid) else None


def remember_uid(uid: str) -> None:
    """Minta browser menyimpan UID (localStorage + cookie jh_uid) pada render berikutnya."""
    if isinstance(uid, str) and _UID_RE.match(uid):
        st.session_state["_lc_uid_store"] = uid


def _slim(messages: Any) -> list[dict]:
    out: list[dict] = []
    for msg in list(messages or [])[-MAX_MESSAGES:]:
        if not isinstance(msg, dict):
            continue
        item = {k: msg[k] for k in _KEEP_FIELDS if k in msg}
        content = item.get("content")
        if isinstance(content, str) and len(content) > MAX_CONTENT_CHARS:
            item["content"] = content[:MAX_CONTENT_CHARS]
        out.append(item)
    return out


def _snapshot_json(uid: str, chat_id: str, messages: Any) -> str:
    slim = _slim(messages)
    while True:
        payload = json.dumps(
            {"v": SCHEMA_VERSION, "uid": uid, "chat_id": chat_id, "ts": time.time(), "messages": slim},
            ensure_ascii=False,
            default=str,
            separators=(",", ":"),
        )
        if len(payload.encode("utf-8")) <= MAX_SNAPSHOT_BYTES or len(slim) <= 2:
            return payload
        slim = slim[2:]  # buang sepasang Q&A tertua sampai muat


def _is_guest() -> bool:
    return bool(st.session_state.get("guest_mode"))


# --------------------------------------------------------------------------- public API
def mount(uid: Optional[str], server_loader: Optional[Callable[[str], list]] = None) -> None:
    """Pulihkan chat aktif terakhir dari cache lokal (sekali per sesi).

    ``server_loader(chat_id) -> list[dict]`` dijalankan di thread terpisah (JANGAN
    pakai st.* di dalamnya) untuk rekonsiliasi: jika server punya pesan lebih banyak
    (mis. dari perangkat lain), pesan server dipakai pada akhir run.
    """
    ss = st.session_state
    if ss.get("_lc_restored") or not component_ready():
        return
    ss["_lc_restored"] = True
    if not uid or _is_guest():
        return
    raw = _hydrated().get("snapshot")
    if not raw:
        return
    try:
        snap = json.loads(raw) if isinstance(raw, str) else raw
        if snap.get("v") != SCHEMA_VERSION or snap.get("uid") != uid:
            return
        if time.time() - float(snap.get("ts", 0)) > TTL_SECONDS:
            return
        messages, chat_id = snap.get("messages"), snap.get("chat_id")
        if (
            not isinstance(messages, list)
            or not messages
            or not isinstance(chat_id, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", chat_id)
        ):
            return
    except Exception:
        return
    if ss.get("messages"):  # layar sudah berisi chat -> jangan timpa
        return
    ss["messages"] = messages
    ss["chat_id"] = chat_id
    ss["loaded_chat_id"] = chat_id  # lewati reload Firestore sinkron di app.py
    if not st.query_params.get("chat"):
        st.query_params["chat"] = chat_id
    ss["_lc_last_wid"] = None
    if server_loader is not None:
        ss["_lc_future"] = _EXECUTOR.submit(server_loader, chat_id)


def _reconcile() -> bool:
    ss = st.session_state
    fut: Optional[Future] = ss.get("_lc_future")
    if fut is None or not fut.done():
        return False
    ss["_lc_future"] = None
    try:
        server_messages = fut.result()
    except Exception:
        return False
    local = ss.get("messages") or []
    if isinstance(server_messages, list) and len(server_messages) > len(local):
        ss["messages"] = server_messages
        return True
    return False


def clear(uid: Optional[str] = None) -> None:
    """Hapus snapshot lokal (panggil saat 'Chat baru', hapus riwayat, atau keluar)."""
    ss = st.session_state
    ss["_lc_clear_nonce"] = int(ss.get("_lc_clear_nonce", 0)) + 1
    ss["_lc_last_wid"] = None
    ss["_lc_future"] = None


def persist(uid: Optional[str], chat_id: Optional[str], messages: Any) -> None:
    """Pasang komponen (sekali per run) dan kirim snapshot HANYA jika berubah."""
    ss = st.session_state
    snapshot, wid = None, ""
    if uid and chat_id and messages and not _is_guest():
        candidate = _snapshot_json(uid, str(chat_id), messages)
        # hash tanpa 'ts' agar tidak selalu berubah
        stable = candidate.split('"messages":', 1)[-1]
        wid = hashlib.md5(f"{chat_id}|{stable}".encode("utf-8")).hexdigest()[:16]
        if wid != ss.get("_lc_last_wid"):
            snapshot = candidate
            ss["_lc_last_wid"] = wid
    _component(
        uid=uid,
        snapshot=snapshot,
        write_id=wid,
        clear_nonce=int(ss.get("_lc_clear_nonce", 0)),
        store_uid=ss.get("_lc_uid_store"),
        ttl=TTL_SECONDS,
        schema=SCHEMA_VERSION,
        key=_STATE_KEY,
        default=None,
    )
    if _reconcile():
        st.rerun()
