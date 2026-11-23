"""Identifikasi perangkat berbasis penyimpanan lokal browser."""

from __future__ import annotations

import json
import re
import uuid

import streamlit as st
from streamlit_javascript import st_javascript

_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LOCAL_STORAGE_KEY = "jh_asisten_device_uid"
_STORAGE_ERROR = "__device_uid_storage_error__"


def _json_key() -> str:
    """Kembalikan nama kunci localStorage sebagai literal JavaScript aman."""
    return json.dumps(_LOCAL_STORAGE_KEY)


def _is_valid_uid(value: object) -> bool:
    return isinstance(value, str) and _UID_RE.fullmatch(value) is not None


def _resolve_uid(uid: str) -> str:
    st.session_state.device_uid = uid
    st.session_state._device_uid_browser_resolved = True
    return uid


_LOCAL_UID_SCRIPT = f"""
(() => {{
  try {{
    const key = {_json_key()};
    const storage = window.localStorage;
    let uid = storage.getItem(key);
    if (!/^[a-f0-9]{{16,32}}$/.test(uid || "")) {{
      if (crypto.randomUUID) {{
        uid = crypto.randomUUID().replace(/-/g, "");
      }} else {{
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        uid = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
      }}
      storage.setItem(key, uid);
    }}
    return uid;
  }} catch (error) {{
    return {json.dumps(_STORAGE_ERROR)};
  }}
}})()
"""


def get_device_uid() -> str:
    """Ambil UID dari localStorage; jangan gunakan UID yang diberikan lewat URL."""
    st.query_params.pop("uid", None)
    if st.session_state.get("_device_uid_browser_resolved"):
        session_uid = st.session_state.get("device_uid")
        if _is_valid_uid(session_uid):
            return session_uid
        st.session_state.pop("_device_uid_browser_resolved", None)

    browser_uid = st_javascript(_LOCAL_UID_SCRIPT, key="read_device_uid")
    if browser_uid is None:
        st.info("Menunggu penyimpanan perangkat dimuat dari browser...")
        st.stop()

    uid = browser_uid.strip() if isinstance(browser_uid, str) else ""
    if _is_valid_uid(uid):
        return _resolve_uid(uid)

    repaired_uid = st_javascript(_LOCAL_UID_SCRIPT, key="repair_device_uid")
    if _is_valid_uid(repaired_uid):
        return _resolve_uid(repaired_uid)

    session_uid = st.session_state.get("device_uid")
    if not _is_valid_uid(session_uid):
        session_uid = uuid.uuid4().hex
    return _resolve_uid(session_uid)
