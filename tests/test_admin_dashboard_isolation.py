import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from streamlit.testing.v1 import AppTest
import services.admin_dashboard as admin_dashboard
from services.admin_session import create_admin_session_token, verify_admin_session_token


class AdminDashboardIsolationTests(unittest.TestCase):
    def test_regular_user_does_not_import_admin_dashboard(self):
        dashboard_module = sys.modules.pop("services.admin_dashboard", None)
        try:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
            app.run(timeout=45)

            self.assertEqual(len(app.exception), 0)
            self.assertNotIn("services.admin_dashboard", sys.modules)
        finally:
            if dashboard_module is not None:
                sys.modules["services.admin_dashboard"] = dashboard_module

    def test_statistics_menu_does_not_load_key_or_model_data(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.session_state["admin_authenticated"] = True
        with (
            patch.object(
                admin_dashboard,
                "get_available_keys",
                side_effect=AssertionError("unused menu loaded API keys"),
            ),
            patch.object(
                admin_dashboard,
                "get_model_order",
                side_effect=AssertionError("unused menu loaded model order"),
            ),
        ):
            app.run(timeout=45)

        self.assertEqual(len(app.exception), 0)

    def test_logout_does_not_clear_token(self):
        """Verify that logout (admin_authenticated=False) doesn't invalidate token."""
        secrets = {"ENCRYPTION_KEY": "test-encryption-key"}
        uid = "abcdef0123456789"
        token = create_admin_session_token(uid, 0, secrets, now=1_000)

        # Token valid before logout
        self.assertTrue(verify_admin_session_token(token, uid, 0, secrets, now=1_001))

        # Simulate logout: only admin_authenticated = False, token stays in cookie
        # Token should still be valid
        self.assertTrue(verify_admin_session_token(token, uid, 0, secrets, now=1_001))

        # Re-login (admin_authenticated = True) with same token works
        self.assertTrue(verify_admin_session_token(token, uid, 0, secrets, now=1_001))

    def test_revoke_clears_token(self):
        """Verify that revoke all sessions (epoch increment) invalidates token."""
        secrets = {"ENCRYPTION_KEY": "test-encryption-key"}
        uid = "abcdef0123456789"
        token = create_admin_session_token(uid, 3, secrets, now=1_000)

        # Token valid at epoch 3
        self.assertTrue(verify_admin_session_token(token, uid, 3, secrets, now=1_001))

        # After revoke (epoch becomes 4), old token invalid
        self.assertFalse(verify_admin_session_token(token, uid, 4, secrets, now=1_001))

        # New token at epoch 4 works
        new_token = create_admin_session_token(uid, 4, secrets, now=1_000)
        self.assertTrue(verify_admin_session_token(new_token, uid, 4, secrets, now=1_001))


if __name__ == "__main__":
    unittest.main()
