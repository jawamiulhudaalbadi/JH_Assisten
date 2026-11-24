import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import services.admin_dashboard as admin_dashboard


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


if __name__ == "__main__":
    unittest.main()
