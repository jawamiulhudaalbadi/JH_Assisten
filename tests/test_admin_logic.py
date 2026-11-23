import hashlib
import unittest
from datetime import datetime, timezone

from services.admin_logic import (
    process_admin_login,
    question_is_allowed,
    register_admin_button_click,
    verify_admin_code,
)
from services.firestore_store import normalize_question


class AdminLogicTests(unittest.TestCase):
    def test_normalize_question_groups_case_punctuation_and_spacing(self) -> None:
        self.assertEqual(
            normalize_question("  Bagaimana   Hukum-Nya? "),
            normalize_question("BAGAIMANA hukum nya"),
        )

    def test_blocked_user_is_rejected(self) -> None:
        allowed, message = question_is_allowed(
            {"blocked": True, "day": "20261124", "day_count": 0},
            daily_question_limit=10,
        )

        self.assertFalse(allowed)
        self.assertIn("diblokir", message)

    def test_daily_limit_uses_wib_day_boundary(self) -> None:
        now = datetime(2026, 11, 23, 23, 30, tzinfo=timezone.utc)

        allowed, message = question_is_allowed(
            {"day": "20261124", "day_count": 2},
            daily_question_limit=2,
            now=now,
        )

        self.assertFalse(allowed)
        self.assertIn("Batas pertanyaan", message)

    def test_admin_code_verifies_pbkdf2_secret(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"

        self.assertTrue(verify_admin_code("secret", encoded_hash))
        self.assertFalse(verify_admin_code("wrong", encoded_hash))

    def test_admin_login_authenticates_matching_secret(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        state: dict[str, object] = {}

        self.assertTrue(
            process_admin_login(
                "secret",
                f"pbkdf2_sha256$200000$unit-test-salt${digest}",
                state,
                now=100.0,
            )
        )
        self.assertTrue(state["admin_authenticated"])
        self.assertEqual(state["admin_failed_attempts"], 0)
        self.assertFalse(state["_admin_login_prompt_open"])
        self.assertEqual(state["_admin_button_clicks"], 0)

    def test_admin_login_locks_after_five_wrong_codes(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"
        state: dict[str, object] = {}

        for _ in range(5):
            self.assertFalse(
                process_admin_login("wrong", encoded_hash, state, now=100.0)
            )

        self.assertEqual(state["admin_locked_until"], 100.0 + 15 * 60)
        self.assertFalse(
            process_admin_login("secret", encoded_hash, state, now=101.0)
        )
        self.assertNotIn("admin_authenticated", state)
        self.assertIn("terkunci", str(state["_admin_login_notice"]))

    def test_sidebar_admin_prompt_opens_after_three_button_clicks(self) -> None:
        state: dict[str, object] = {}

        self.assertFalse(register_admin_button_click(state))
        self.assertEqual(state["_admin_button_clicks"], 1)
        self.assertFalse(register_admin_button_click(state))
        self.assertEqual(state["_admin_button_clicks"], 2)
        self.assertTrue(register_admin_button_click(state))
        self.assertTrue(state["_admin_login_prompt_open"])
        self.assertEqual(state["_admin_button_clicks"], 0)


if __name__ == "__main__":
    unittest.main()
