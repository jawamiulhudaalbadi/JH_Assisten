import json
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from services import device_identity


class _SessionState(dict):
    __setattr__ = dict.__setitem__


class DeviceIdentityTests(unittest.TestCase):
    def test_empty_component_round_then_reads_existing_uid_without_replacing_it(
        self,
    ) -> None:
        uid = "0123456789abcdef"
        state = _SessionState()
        query_params = {}
        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                side_effect=[
                    {"status": "empty"},
                    0,
                    {"status": "found", "uid": uid},
                ],
            ) as javascript,
        ):
            temporary_uid = device_identity.get_device_uid()
            self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
            self.assertFalse(state["_device_uid_browser_resolved"])
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertEqual(
            [call.kwargs["key"] for call in javascript.call_args_list],
            ["read_device_uid", "create_device_uid", "read_device_uid"],
        )

    def test_timeout_uses_temporary_uid_without_persisting_it(self) -> None:
        state = _SessionState(device_uid="previous-session-value")
        query_params = {"uid": "invalid"}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                return_value=0,
            ) as javascript,
            patch.object(
                device_identity.time,
                "monotonic",
                side_effect=[10.0, 11.0, 12.0, 14.1],
            ),
        ):
            temporary_uid = ""
            for _ in range(4):
                temporary_uid = device_identity.get_device_uid()

        self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
        self.assertFalse(state["_device_uid_browser_resolved"])
        self.assertNotIn("uid", query_params)
        self.assertEqual(state["_device_uid_wait_attempts"], 4)
        self.assertTrue(state["_device_uid_wait_timed_out"])
        self.assertEqual(javascript.call_count, 4)
        self.assertEqual(
            {call.kwargs["key"] for call in javascript.call_args_list},
            {"read_device_uid"},
        )
        for call in javascript.call_args_list:
            self.assertEqual(call.args[0], device_identity._READ_UID_SCRIPT)
            self.assertNotIn(temporary_uid, call.args[0])

    def test_times_out_after_four_seconds_before_three_attempts(self) -> None:
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                return_value=0,
            ) as javascript,
            patch.object(device_identity.time, "monotonic", side_effect=[10.0, 14.1]),
        ):
            device_identity.get_device_uid()
            device_identity.get_device_uid()

        self.assertTrue(state["_device_uid_wait_timed_out"])
        self.assertEqual(state["_device_uid_wait_attempts"], 2)
        self.assertEqual(javascript.call_count, 2)

    def test_empty_component_values_use_temporary_uid(self) -> None:
        for component_value in (None, ""):
            with self.subTest(component_value=component_value):
                state = _SessionState()
                with (
                    patch.object(device_identity.st, "session_state", state),
                    patch.object(device_identity.st, "query_params", {}),
                    patch.object(
                        device_identity.st,
                        "context",
                        SimpleNamespace(cookies={}),
                    ),
                    patch.object(
                        device_identity,
                        "st_javascript",
                        return_value=component_value,
                    ),
                ):
                    uid = device_identity.get_device_uid()

                self.assertRegex(uid, r"^[a-f0-9]{32}$")
                self.assertFalse(state["_device_uid_browser_resolved"])
                self.assertTrue(state["_device_uid_waiting"])

    def test_prefers_request_cookie_and_supports_uid_query_parameter(self) -> None:
        cookie_uid = "abcdef0123456789"
        query_uid = "fedcba9876543210"
        state = _SessionState()
        query_params = {}
        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={"jh_uid": cookie_uid}),
            ),
            patch.object(device_identity, "st_javascript") as javascript,
        ):
            self.assertEqual(device_identity.get_device_uid(), cookie_uid)
            query_params["uid"] = query_uid
            self.assertEqual(device_identity.get_device_uid(), query_uid)

        javascript.assert_not_called()
        self.assertNotIn("uid", query_params)

    def test_uses_browser_uid_and_marks_it_resolved(self) -> None:
        uid = "0123456789abcdef"
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                return_value={"status": "found", "uid": uid},
            ),
        ):
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])

    def test_resolves_browser_uid_on_a_later_rerun_after_error(self) -> None:
        uid = "abcdef0123456789"
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                side_effect=[{"status": "error"}, {"status": "found", "uid": uid}],
            ) as javascript,
        ):
            self.assertRegex(device_identity.get_device_uid(), r"^[a-f0-9]{32}$")
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertEqual(
            [call.kwargs["key"] for call in javascript.call_args_list],
            ["read_device_uid", "read_device_uid"],
        )

    def test_resolves_browser_uid_after_temporary_uid_timeout(self) -> None:
        uid = "0123456789abcdef"
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                side_effect=[
                    *([0] * 4),
                    {"status": "found", "uid": uid},
                ],
            ) as javascript,
            patch.object(
                device_identity.time,
                "monotonic",
                side_effect=[10.0, 11.0, 12.0, 14.1],
            ),
        ):
            for _ in range(4):
                temporary_uid = device_identity.get_device_uid()
            self.assertTrue(state["_device_uid_wait_timed_out"])
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertNotEqual(temporary_uid, uid)
        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertNotIn("_device_uid_wait_timed_out", state)
        self.assertEqual(javascript.call_count, 5)

    def test_storage_errors_do_not_create_a_persistent_uid(self) -> None:
        state = _SessionState(device_uid="abcdef0123456789")
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                return_value={"status": "error"},
            ) as javascript,
        ):
            temporary_uid = device_identity.get_device_uid()

        self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
        self.assertNotEqual(temporary_uid, "abcdef0123456789")
        self.assertFalse(state["_device_uid_browser_resolved"])
        javascript.assert_called_once()

    def test_create_result_is_not_assumed_when_component_is_empty(self) -> None:
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                side_effect=[{"status": "empty"}, 0],
            ),
        ):
            uid = device_identity.get_device_uid()

        self.assertRegex(uid, r"^[a-f0-9]{32}$")
        self.assertEqual(state["device_uid"], uid)
        self.assertFalse(state["_device_uid_browser_resolved"])

    def test_fragment_polling_uses_distinct_component_keys(self) -> None:
        state = _SessionState()
        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", {}),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(
                device_identity,
                "st_javascript",
                side_effect=[{"status": "empty"}, 0],
            ) as javascript,
        ):
            device_identity.get_device_uid(polling=True)

        self.assertEqual(
            [call.kwargs["key"] for call in javascript.call_args_list],
            ["read_device_uid_poll", "create_device_uid_poll"],
        )


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
