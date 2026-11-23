import hashlib
import unittest
from datetime import datetime, timezone

from services.admin_logic import question_is_allowed, verify_admin_code
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


if __name__ == "__main__":
    unittest.main()
