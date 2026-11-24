"""Identifikasi perangkat berbasis penyimpanan browser."""

from __future__ import annotations

import json
import re
import time
import uuid

import streamlit as st

from services import local_cache

_UID_RE = re.compile(r"^[a-f0-9]{16,32}$")
_LOCAL_STORAGE_KEY = "jh_asisten_device_uid"
_COOKIE_KEY = "jh_uid"
_MAX_BROWSER_WAIT_SECONDS = 4.0


def _json_literal(value: str) -> str:
    return json.dumps(value)


def _js_expression(body: str) -> str:
    expression = body.strip()
    return expression if expression.startswith("(") else f"({expression})"


def _is_valid_uid(value: object) -> bool:
    return isinstance(value, str) and _UID_RE.fullmatch(value) is not None


def _record_diagnostic(
    *,
    source: str | None = None,
    status: str | None = None,
    raw_value: object | None = None,
    uid: str | None = None,
) -> None:
    if source is not None:
        st.session_state["_device_uid_source"] = source
    if status is not None:
        st.session_state["_device_uid_status"] = status
    if raw_value is not None or status == "belum_balas":
        st.session_state["_device_uid_raw_type"] = type(raw_value).__name__
    if uid is not None:
        st.session_state["_device_uid_prefix"] = uid[:4]


def _resolve_uid(uid: str) -> str:
    st.session_state.device_uid = uid
    st.session_state._device_uid_browser_resolved = True
    st.session_state.pop("_device_uid_wait_started", None)
    st.session_state.pop("_device_uid_wait_attempts", None)
    st.session_state.pop("_device_uid_waiting", None)
    st.session_state.pop("_device_uid_wait_timed_out", None)
    _record_diagnostic(status="found", uid=uid)
    return uid


_READ_UID_SCRIPT = _js_expression(f"""
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
""")

_CREATE_UID_SCRIPT = _js_expression(f"""
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
""")


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
    timed_out = now - started >= _MAX_BROWSER_WAIT_SECONDS
    st.session_state["_device_uid_wait_attempts"] = (
        st.session_state.get("_device_uid_wait_attempts", 0) + 1
    )
    st.session_state._device_uid_wait_timed_out = timed_out
    st.session_state._device_uid_waiting = not timed_out


def _record_browser_result(value: object) -> str:
    if not isinstance(value, dict):
        status = "belum_balas"
    else:
        status = value.get("status")
        if not isinstance(status, str) or status not in {
            "found",
            "empty",
            "error",
        }:
            status = "error"
    _record_diagnostic(status=status, raw_value=value)
    return status


def get_device_uid(*, polling: bool = False) -> str:
    """Resolve the URL, request cookie, or browser storage UID in that order."""
    if st.session_state.get("_device_uid_browser_resolved"):
        session_uid = st.session_state.get("device_uid")
        if _is_valid_uid(session_uid):
            return session_uid
        st.session_state.pop("_device_uid_browser_resolved", None)

    requested_uid = st.query_params.get("uid")
    st.query_params.pop("uid", None)
    if _is_valid_uid(requested_uid):
        _record_diagnostic(source="url")
        return _resolve_uid(requested_uid)

    cookie_uid = st.context.cookies.get(_COOKIE_KEY)
    if _is_valid_uid(cookie_uid):
        _record_diagnostic(source="cookie")
        return _resolve_uid(cookie_uid)

    browser_uid = local_cache.hydrated_uid()
    if browser_uid:
        _record_diagnostic(source="localStorage")
        return _resolve_uid(browser_uid)

    if not local_cache.component_ready():
        _record_browser_wait()
        if not st.session_state["_device_uid_wait_timed_out"]:
            if not polling:
                local_cache.persist(None, None, None)
                st.stop()
            return _temporary_uid()
    else:
        browser_uid = uuid.uuid4().hex
        local_cache.remember_uid(browser_uid)
        _record_diagnostic(source="baru")
        return _resolve_uid(browser_uid)

    _record_diagnostic(source="temporary", uid=_temporary_uid())
    return st.session_state["_device_uid_temporary"]
