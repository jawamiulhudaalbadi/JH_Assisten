"""Identifikasi perangkat berbasis URL dan penyimpanan lokal browser."""

from __future__ import annotations

import json
import re
import uuid

import streamlit as st
from streamlit_javascript import st_javascript

_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LOCAL_STORAGE_KEY = "jh_asisten_device_uid"


def _json_key() -> str:
    """Kembalikan nama kunci localStorage sebagai literal JavaScript aman."""
    return json.dumps(_LOCAL_STORAGE_KEY)


_LOCAL_UID_SCRIPT = f"""
(() => {{
  try {{
    const key = {_json_key()};
    let uid = localStorage.getItem(key);
    if (!/^[a-f0-9]{{16,32}}$/.test(uid || "")) {{
      if (crypto.randomUUID) {{
        uid = crypto.randomUUID().replace(/-/g, "");
      }} else {{
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        uid = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
      }}
      localStorage.setItem(key, uid);
    }}
    return uid;
  }} catch (error) {{
    return null;
  }}
}})()
"""


def get_device_uid() -> str:
    """Ambil ID dari URL atau localStorage, lalu hilangkan UID dari URL."""
    if st.session_state.get("_device_uid_browser_resolved"):
        return str(st.session_state.device_uid)

    url_uid = str(st.query_params.get("uid", "")).strip()
    st.query_params.pop("uid", None)
    if _UID_RE.fullmatch(url_uid):
        uid = url_uid
        st_javascript(
            f"localStorage.setItem({_json_key()}, {json.dumps(uid)}); true",
            key="persist_device_uid",
        )
        st.session_state.device_uid = uid
        st.session_state._device_uid_browser_resolved = True
    else:
        browser_uid = st_javascript(_LOCAL_UID_SCRIPT, key="read_device_uid")
        uid = str(browser_uid).strip() if browser_uid is not None else ""
        if _UID_RE.fullmatch(uid):
            st.session_state.device_uid = uid
            st.session_state._device_uid_browser_resolved = True
        elif "device_uid" not in st.session_state:
            st.session_state.device_uid = uuid.uuid4().hex
    return str(st.session_state.device_uid)
