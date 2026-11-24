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
                side_effect=["", uid],
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
            ["read_device_uid", "read_device_uid"],
        )

    def test_waits_for_browser_storage_instead_of_creating_a_server_uid(self) -> None:
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
            patch.object(device_identity, "st_javascript", return_value=None),
        ):
            temporary_uid = device_identity.get_device_uid()

        self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
        self.assertFalse(state["_device_uid_browser_resolved"])
        self.assertNotIn("uid", query_params)

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
            patch.object(device_identity, "st_javascript", return_value=uid),
        ):
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])

    def test_repairs_corrupt_browser_uid(self) -> None:
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
                side_effect=[device_identity._STORAGE_ERROR, uid],
            ) as javascript,
        ):
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])
        self.assertEqual(
            [call.kwargs["key"] for call in javascript.call_args_list],
            ["read_device_uid", "create_device_uid"],
        )

    def test_storage_errors_keep_a_temporary_unresolved_uid(self) -> None:
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
                side_effect=[
                    device_identity._STORAGE_ERROR,
                    device_identity._STORAGE_ERROR,
                ],
            ),
        ):
            temporary_uid = device_identity.get_device_uid()

        self.assertRegex(temporary_uid, r"^[a-f0-9]{32}$")
        self.assertNotEqual(temporary_uid, "abcdef0123456789")
        self.assertFalse(state["_device_uid_browser_resolved"])

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
                side_effect=["", ""],
            ),
        ):
            uid = device_identity.get_device_uid()

        self.assertRegex(uid, r"^[a-f0-9]{32}$")
        self.assertEqual(state["device_uid"], uid)
        self.assertFalse(state["_device_uid_browser_resolved"])


if __name__ == "__main__":
    unittest.main()
