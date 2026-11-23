import unittest
from unittest.mock import patch

from services import device_identity


class _SessionState(dict):
    __setattr__ = dict.__setitem__


class DeviceIdentityTests(unittest.TestCase):
    def test_waits_for_browser_storage_instead_of_creating_a_server_uid(self) -> None:
        state = _SessionState(device_uid="previous-session-value")
        query_params = {"uid": "0123456789abcdef"}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(device_identity, "st_javascript", return_value=None),
            patch.object(device_identity.st, "info"),
            patch.object(
                device_identity.st,
                "stop",
                side_effect=RuntimeError("stopped"),
            ) as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "stopped"):
                device_identity.get_device_uid()

        self.assertEqual(state, {"device_uid": "previous-session-value"})
        self.assertNotIn("uid", query_params)
        stop.assert_called_once()

    def test_uses_browser_uid_and_marks_it_resolved(self) -> None:
        uid = "0123456789abcdef"
        state = _SessionState()
        query_params = {}

        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", query_params),
            patch.object(device_identity, "st_javascript", return_value=uid),
        ):
            self.assertEqual(device_identity.get_device_uid(), uid)

        self.assertEqual(state["device_uid"], uid)
        self.assertTrue(state["_device_uid_browser_resolved"])


if __name__ == "__main__":
    unittest.main()
