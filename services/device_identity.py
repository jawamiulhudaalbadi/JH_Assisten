"""Identifikasi perangkat berbasis penyimpanan browser."""

from __future__ import annotations

import json
import re
import uuid

import streamlit as st
from streamlit_javascript import st_javascript

_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LOCAL_STORAGE_KEY = "jh_asisten_device_uid"
_COOKIE_KEY = "jh_uid"
_STORAGE_MISSING = "__device_uid_missing__"
_STORAGE_ERROR = "__device_uid_storage_error__"


def _json_literal(value: str) -> str:
    return json.dumps(value)


def _is_valid_uid(value: object) -> bool:
    return isinstance(value, str) and _UID_RE.fullmatch(value) is not None


def _resolve_uid(uid: str) -> str:
    st.session_state.device_uid = uid
    st.session_state._device_uid_browser_resolved = True
    return uid


_READ_UID_SCRIPT = f"""
(() => {{
  try {{
    const uid = window.localStorage.getItem({_json_literal(_LOCAL_STORAGE_KEY)});
    if (!/^[a-f0-9]{{16,32}}$/.test(uid || "")) {{
      return uid === null ? {_json_literal(_STORAGE_MISSING)} : {_json_literal(_STORAGE_ERROR)};
    }}
    document.cookie = {_json_literal(_COOKIE_KEY + "=")} + uid
      + "; Max-Age=31536000; Path=/; SameSite=Lax"
      + (location.protocol === "https:" ? "; Secure" : "");
    return uid;
  }} catch (error) {{
    return {_json_literal(_STORAGE_ERROR)};
  }}
}})()
"""

_CREATE_UID_SCRIPT = f"""
(() => {{
  try {{
    const key = {_json_literal(_LOCAL_STORAGE_KEY)};
    let uid = window.localStorage.getItem(key);
    if (!/^[a-f0-9]{{16,32}}$/.test(uid || "")) {{
      if (crypto.randomUUID) {{
        uid = crypto.randomUUID().replace(/-/g, "");
      }} else {{
        const bytes = new Uint8Array(16);
        crypto.getRandomValues(bytes);
        uid = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
      }}
      window.localStorage.setItem(key, uid);
    }}
    document.cookie = {_json_literal(_COOKIE_KEY + "=")} + uid
      + "; Max-Age=31536000; Path=/; SameSite=Lax"
      + (location.protocol === "https:" ? "; Secure" : "");
    return uid;
  }} catch (error) {{
    return {_json_literal(_STORAGE_ERROR)};
  }}
}})()
"""


def _temporary_uid() -> str:
    uid = st.session_state.get("_device_uid_temporary")
    if not _is_valid_uid(uid):
        uid = uuid.uuid4().hex
        st.session_state._device_uid_temporary = uid
    st.session_state.device_uid = uid
    st.session_state._device_uid_browser_resolved = False
    return uid


def get_device_uid() -> str:
    """Resolve the URL, request cookie, or browser storage UID in that order."""
    requested_uid = st.query_params.get("uid")
    st.query_params.pop("uid", None)
    if _is_valid_uid(requested_uid):
        return _resolve_uid(requested_uid)

    cookie_uid = st.context.cookies.get(_COOKIE_KEY)
    if _is_valid_uid(cookie_uid):
        return _resolve_uid(cookie_uid)

    if st.session_state.get("_device_uid_browser_resolved"):
        session_uid = st.session_state.get("device_uid")
        if _is_valid_uid(session_uid):
            return session_uid
        st.session_state.pop("_device_uid_browser_resolved", None)

    browser_uid = st_javascript(_READ_UID_SCRIPT, key="read_device_uid")
    if _is_valid_uid(browser_uid.strip() if isinstance(browser_uid, str) else None):
        return _resolve_uid(browser_uid.strip())
    if browser_uid == _STORAGE_MISSING or browser_uid == _STORAGE_ERROR:
        created_uid = st_javascript(_CREATE_UID_SCRIPT, key="create_device_uid")
        if isinstance(created_uid, str) and _is_valid_uid(created_uid.strip()):
            return _resolve_uid(created_uid.strip())

    return _temporary_uid()
