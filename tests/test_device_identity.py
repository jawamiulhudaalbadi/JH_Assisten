import json
import shutil
import subprocess
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from services import device_identity


class _SessionState(dict):
    __setattr__ = dict.__setitem__


class DeviceIdentityTests(unittest.TestCase):
    def _patch_identity(self, stack, state, query_params, cookies=None):
        stack.enter_context(
            patch.object(device_identity.st, "session_state", state)
        )
        stack.enter_context(
            patch.object(device_identity.st, "query_params", query_params)
        )
        stack.enter_context(
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies=cookies or {}),
            )
        )

    def test_query_uid_and_cookie_precedence_are_unchanged(self) -> None:
        uid = "fedcba9876543210"
        state = _SessionState()
        query_params = {"uid": uid}
        with ExitStack() as stack:
            self._patch_identity(
                stack,
                state,
                query_params,
                {"jh_uid": "abcdef0123456789"},
            )
            cache = stack.enter_context(
                patch.object(device_identity, "local_cache")
            )
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["_device_uid_source"], "url")
        cache.hydrated_uid.assert_not_called()

        state.clear()
        query_params = {}
        with ExitStack() as stack:
            self._patch_identity(
                stack,
                state,
                query_params,
                {"jh_uid": "abcdef0123456789"},
            )
            cache = stack.enter_context(
                patch.object(device_identity, "local_cache")
            )
            self.assertEqual(
                device_identity.get_device_uid(),
                "abcdef0123456789",
            )

        cache.hydrated_uid.assert_not_called()

    def test_waits_for_cache_hydration_before_resolving_uid(self) -> None:
        state = _SessionState()
        with ExitStack() as stack:
            self._patch_identity(stack, state, {})
            stack.enter_context(
                patch.object(
                    device_identity.local_cache, "hydrated_uid", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "component_ready",
                    return_value=False,
                )
            )
            persist = stack.enter_context(
                patch.object(device_identity.local_cache, "persist")
            )
            stack.enter_context(
                patch.object(
                    device_identity.st, "stop", side_effect=RuntimeError("stop")
                )
            )
            stack.enter_context(
                patch.object(device_identity.time, "monotonic", return_value=10.0)
            )
            with self.assertRaisesRegex(RuntimeError, "stop"):
                device_identity.get_device_uid()

        persist.assert_called_once_with(None, None, None)
        self.assertEqual(state["_device_uid_wait_attempts"], 1)

    def test_fragment_polling_advances_timeout_without_stopping_or_repersisting(self):
        state = _SessionState(_device_uid_wait_started=10.0)
        with ExitStack() as stack:
            self._patch_identity(stack, state, {})
            stack.enter_context(
                patch.object(
                    device_identity.local_cache, "hydrated_uid", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "component_ready",
                    return_value=False,
                )
            )
            persist = stack.enter_context(
                patch.object(device_identity.local_cache, "persist")
            )
            stop = stack.enter_context(patch.object(device_identity.st, "stop"))
            stack.enter_context(
                patch.object(device_identity.time, "monotonic", return_value=11.0)
            )
            uid = device_identity.get_device_uid(polling=True)

        self.assertRegex(uid, r"^[a-f0-9]{32}$")
        self.assertTrue(state["_device_uid_waiting"])
        persist.assert_not_called()
        stop.assert_not_called()

    def test_ready_cache_uid_is_resolved(self) -> None:
        uid = "0123456789abcdef"
        state = _SessionState()
        with ExitStack() as stack:
            self._patch_identity(stack, state, {})
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "hydrated_uid",
                    return_value=uid,
                )
            )
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "component_ready",
                    return_value=True,
                )
            )
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertEqual(state["_device_uid_source"], "localStorage")

    def test_ready_cache_without_uid_creates_and_remembers_uid(self) -> None:
        state = _SessionState()
        with ExitStack() as stack:
            self._patch_identity(stack, state, {})
            stack.enter_context(
                patch.object(
                    device_identity.local_cache, "hydrated_uid", return_value=None
                )
            )
            stack.enter_context(
                patch.object(
                    device_identity.local_cache, "component_ready", return_value=True
                )
            )
            remember = stack.enter_context(
                patch.object(device_identity.local_cache, "remember_uid")
            )
            uid = device_identity.get_device_uid()

        self.assertRegex(uid, r"^[a-f0-9]{16,32}$")
        remember.assert_called_once_with(uid)
        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])

    def test_timeout_falls_back_to_temporary_uid_then_accepts_hydrated_uid(self) -> None:
        uid = "abcdef0123456789"
        state = _SessionState()
        with ExitStack() as stack:
            self._patch_identity(stack, state, {})
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "hydrated_uid",
                    side_effect=[None, uid],
                )
            )
            stack.enter_context(
                patch.object(
                    device_identity.local_cache,
                    "component_ready",
                    return_value=False,
                )
            )
            persist = stack.enter_context(
                patch.object(device_identity.local_cache, "persist")
            )
            stack.enter_context(
                patch.object(
                    device_identity.time,
                    "monotonic",
                    side_effect=[10.0, 14.1],
                )
            )
            temporary_uid = device_identity.get_device_uid()
            self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
            self.assertFalse(state["_device_uid_browser_resolved"])
            self.assertEqual(device_identity.get_device_uid(), uid)

        persist.assert_called_once_with(None, None, None)
        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertNotIn("_device_uid_wait_timed_out", state)


class DeviceIdentityScriptTests(unittest.TestCase):
    def test_uid_scripts_are_trimmed_expressions(self) -> None:
        for script in (
            device_identity._READ_UID_SCRIPT,
            device_identity._CREATE_UID_SCRIPT,
        ):
            with self.subTest(script=script):
                self.assertEqual(script, script.strip())
                self.assertTrue(script.startswith("("))

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_uid_scripts_execute_in_streamlit_javascript_wrapper(self) -> None:
        harness = r"""
const jsCode = process.argv[1];
const scenario = process.argv[2];
const entries = new Map();
if (scenario === "found") entries.set("jh_asisten_device_uid", "0123456789abcdef");
const storage = {
  getItem(key) {
    if (scenario === "error") throw new Error("storage unavailable");
    return entries.has(key) ? entries.get(key) : null;
  },
  setItem(key, value) { entries.set(key, value); }
};
globalThis.window = {localStorage: storage};
let cookie = "";
globalThis.document = {};
Object.defineProperty(document, "cookie", {
  get() { return cookie; },
  set(value) { cookie = value; }
});
globalThis.location = {protocol: "https:"};
globalThis.crypto = {
  randomUUID: () => "fedcba98-7654-3210-fedc-ba9876543210",
  getRandomValues: bytes => bytes
};
(async () => {
  const result = await eval("(async () => {return " + jsCode + "})()");
  console.log(JSON.stringify({result, cookie, stored: entries.get("jh_asisten_device_uid")}));
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
"""
        cases = (
            (device_identity._READ_UID_SCRIPT, "found", "found"),
            (device_identity._READ_UID_SCRIPT, "empty", "empty"),
            (device_identity._READ_UID_SCRIPT, "error", "error"),
            (device_identity._CREATE_UID_SCRIPT, "empty", "found"),
        )
        for script, scenario, expected_status in cases:
            with self.subTest(scenario=scenario, expected_status=expected_status):
                completed = subprocess.run(
                    [shutil.which("node"), "-e", harness, script, scenario],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                output = json.loads(completed.stdout)
                self.assertEqual(output["result"]["status"], expected_status)
                if expected_status == "found":
                    self.assertTrue(output["cookie"].startswith("jh_uid="))
                    self.assertIsNotNone(output["stored"])


if __name__ == "__main__":
    unittest.main()
