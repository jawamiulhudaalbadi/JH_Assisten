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
        ):
            temporary_uid = ""
            for _ in range(device_identity._MAX_BROWSER_ATTEMPTS):
                temporary_uid = device_identity.get_device_uid()

        self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
        self.assertFalse(state["_device_uid_browser_resolved"])
        self.assertNotIn("uid", query_params)
        self.assertEqual(state["_device_uid_wait_attempts"], 3)
        self.assertTrue(state["_device_uid_wait_timed_out"])
        self.assertEqual(javascript.call_count, device_identity._MAX_BROWSER_ATTEMPTS)
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
                    *([0] * device_identity._MAX_BROWSER_ATTEMPTS),
                    {"status": "found", "uid": uid},
                ],
            ) as javascript,
        ):
            for _ in range(device_identity._MAX_BROWSER_ATTEMPTS):
                temporary_uid = device_identity.get_device_uid()
            self.assertTrue(state["_device_uid_wait_timed_out"])
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertNotEqual(temporary_uid, uid)
        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertNotIn("_device_uid_wait_timed_out", state)
        self.assertEqual(javascript.call_count, device_identity._MAX_BROWSER_ATTEMPTS + 1)

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


if __name__ == "__main__":
    unittest.main()
