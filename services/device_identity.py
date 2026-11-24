"""Identifikasi perangkat berbasis penyimpanan browser."""

from __future__ import annotations

import json
import re
import time
import uuid

import streamlit as st
from streamlit_javascript import st_javascript

_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LOCAL_STORAGE_KEY = "jh_asisten_device_uid"
_COOKIE_KEY = "jh_uid"
_MAX_BROWSER_ATTEMPTS = 3
_MAX_BROWSER_WAIT_SECONDS = 4.0


def _json_literal(value: str) -> str:
    return json.dumps(value)


def _is_valid_uid(value: object) -> bool:
    return isinstance(value, str) and _UID_RE.fullmatch(value) is not None


def _resolve_uid(uid: str) -> str:
    st.session_state.device_uid = uid
    st.session_state._device_uid_browser_resolved = True
    st.session_state.pop("_device_uid_wait_started", None)
    st.session_state.pop("_device_uid_wait_attempts", None)
    st.session_state.pop("_device_uid_waiting", None)
    st.session_state.pop("_device_uid_wait_timed_out", None)
    return uid


_READ_UID_SCRIPT = f"""
(() => {{
  try {{
    const uid = window.localStorage.getItem({_json_literal(_LOCAL_STORAGE_KEY)});
    if (uid === null) {{
      return {{status: "empty"}};
    }}
    if (!/^[a-f0-9]{{16,32}}$/.test(uid)) {{
      return {{status: "error"}};
    }}
    document.cookie = {_json_literal(_COOKIE_KEY + "=")} + uid
      + "; Max-Age=31536000; Path=/; SameSite=Lax"
      + (location.protocol === "https:" ? "; Secure" : "");
    return {{status: "found", uid}};
  }} catch (error) {{
    return {{status: "error"}};
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
    return {{status: "found", uid}};
  }} catch (error) {{
    return {{status: "error"}};
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


def _record_browser_wait() -> None:
    now = time.monotonic()
    started = st.session_state.setdefault("_device_uid_wait_started", now)
    attempts = min(
        st.session_state.get("_device_uid_wait_attempts", 0) + 1,
        _MAX_BROWSER_ATTEMPTS,
    )
    timed_out = (
        attempts >= _MAX_BROWSER_ATTEMPTS
        or now - started >= _MAX_BROWSER_WAIT_SECONDS
    )
    st.session_state._device_uid_wait_attempts = attempts
    st.session_state._device_uid_wait_timed_out = timed_out
    st.session_state._device_uid_waiting = not timed_out


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
    browser_result = st_javascript(_READ_UID_SCRIPT, key="read_device_uid")
    if isinstance(browser_result, dict):
        status = browser_result.get("status")
        browser_uid = browser_result.get("uid")
        if status == "found" and _is_valid_uid(browser_uid):
            return _resolve_uid(browser_uid)
        if status == "empty":
            created_result = st_javascript(
                _CREATE_UID_SCRIPT,
                key="create_device_uid",
            )
            if isinstance(created_result, dict):
                created_uid = created_result.get("uid")
                if (
                    created_result.get("status") == "found"
                    and _is_valid_uid(created_uid)
                ):
                    return _resolve_uid(created_uid)

    _record_browser_wait()
    return _temporary_uid()
