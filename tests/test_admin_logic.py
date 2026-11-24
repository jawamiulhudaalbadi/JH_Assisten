import hashlib
import unittest
from datetime import datetime, timezone

import services.admin_logic as admin_logic
from services.admin_logic import (
    hash_settings_password,
    process_admin_login,
    question_is_allowed,
    register_admin_button_click,
    verify_admin_code,
)
from services.firestore_store import normalize_question


class AdminLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        with admin_logic._PROCESS_ADMIN_LOCK:
            admin_logic._PROCESS_ADMIN_FAILED_ATTEMPTS_BY_UID.clear()
            admin_logic._PROCESS_ADMIN_LOCKED_UNTIL_BY_UID.clear()
            admin_logic._PROCESS_ADMIN_FAILED_ATTEMPTS = 0
            admin_logic._PROCESS_ADMIN_LOCKED_UNTIL = 0.0
            admin_logic._PROCESS_GUEST_FAILED_ATTEMPTS_BY_UID.clear()
            admin_logic._PROCESS_GUEST_LOCKED_UNTIL_BY_UID.clear()
            admin_logic._PROCESS_GUEST_FAILED_ATTEMPTS = 0
            admin_logic._PROCESS_GUEST_LOCKED_UNTIL = 0.0

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

    def test_per_user_daily_limit_overrides_global_limit(self) -> None:
        now = datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc)

        allowed, message = question_is_allowed(
            {
                "day": "20261124",
                "day_count": 2,
                "daily_question_limit": 2,
            },
            daily_question_limit=10,
            now=now,
        )

        self.assertFalse(allowed)
        self.assertIn("(2)", message)

    def test_admin_code_verifies_pbkdf2_secret(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"

        self.assertTrue(verify_admin_code("secret", encoded_hash))
        self.assertFalse(verify_admin_code("wrong", encoded_hash))

    def test_settings_password_is_hashed_and_verifiable(self) -> None:
        encoded_hash = hash_settings_password("settings-secret")

        self.assertNotIn("settings-secret", encoded_hash)
        self.assertTrue(verify_admin_code("settings-secret", encoded_hash))
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
                uid="login-success-uid",
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
                process_admin_login(
                    "wrong", encoded_hash, state, now=100.0, uid="lockout-uid"
                )
            )

        self.assertEqual(state["admin_locked_until"], 100.0 + 15 * 60)
        self.assertFalse(
            process_admin_login(
                "secret", encoded_hash, state, now=101.0, uid="lockout-uid"
            )
        )
        self.assertNotIn("admin_authenticated", state)
        self.assertIn("terkunci", str(state["_admin_login_notice"]))

    def test_admin_lockout_survives_session_refresh(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"
        original_session: dict[str, object] = {}

        for _ in range(3):
            self.assertFalse(
                process_admin_login(
                    "wrong", encoded_hash, original_session, now=100.0, uid="shared-uid"
                )
            )

        another_uid_session: dict[str, object] = {}
        self.assertTrue(
            process_admin_login(
                "secret",
                encoded_hash,
                another_uid_session,
                now=100.0,
                uid="another-uid",
            )
        )

        second_session: dict[str, object] = {}
        for _ in range(2):
            self.assertFalse(
                process_admin_login(
                    "wrong", encoded_hash, second_session, now=100.0, uid="shared-uid"
                )
            )

        refreshed_session: dict[str, object] = {}
        self.assertFalse(
            process_admin_login(
                "secret", encoded_hash, refreshed_session, now=101.0, uid="shared-uid"
            )
        )
        self.assertIn("terkunci", str(refreshed_session["_admin_login_notice"]))

    def test_admin_lockout_uses_global_failures_across_uids(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"

        for uid in ("first-uid", "second-uid", "third-uid", "fourth-uid"):
            self.assertFalse(
                process_admin_login(
                    "wrong", encoded_hash, {}, now=100.0, uid=uid
                )
            )
        self.assertFalse(
            process_admin_login("wrong", encoded_hash, {}, now=100.0, uid="fifth-uid")
        )

        other_uid_state: dict[str, object] = {}
        self.assertFalse(
            process_admin_login(
                "secret", encoded_hash, other_uid_state, now=101.0, uid="other-uid"
            )
        )
        self.assertIn("terkunci", str(other_uid_state["_admin_login_notice"]))

    def test_guest_login_lockout_is_separate_and_hides_submitted_code(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"secret", salt, 200_000).hex()
        encoded_hash = f"pbkdf2_sha256$200000$unit-test-salt${digest}"
        state: dict[str, object] = {}
        submitted_code = "never-log-this-guest-code"

        with self.assertNoLogs("services.admin_logic", level="DEBUG"):
            for _ in range(5):
                self.assertFalse(
                    admin_logic.process_guest_login(
                        submitted_code,
                        encoded_hash,
                        state,
                        now=100.0,
                        uid="guest-lockout-uid",
                    )
                )

        self.assertEqual(state["guest_locked_until"], 100.0 + 15 * 60)
        self.assertNotIn(submitted_code, str(state["_guest_login_notice"]))
        self.assertEqual(admin_logic._PROCESS_ADMIN_FAILED_ATTEMPTS, 0)
        self.assertFalse(
            admin_logic.process_guest_login(
                "secret",
                encoded_hash,
                state,
                now=999.0,
                uid="guest-lockout-uid",
            )
        )
        self.assertTrue(
            admin_logic.process_guest_login(
                "secret",
                encoded_hash,
                state,
                now=1000.0,
                uid="guest-lockout-uid",
            )
        )
        self.assertTrue(
            process_admin_login(
                "secret",
                encoded_hash,
                {},
                now=101.0,
                uid="admin-independent-uid",
            )
        )

    def test_guest_login_accepts_hash_without_persisting_code(self) -> None:
        salt = b"unit-test-salt"
        digest = hashlib.pbkdf2_hmac("sha256", b"guest-pass", salt, 200_000).hex()
        state: dict[str, object] = {}

        self.assertTrue(
            admin_logic.process_guest_login(
                "guest-pass",
                f"pbkdf2_sha256$200000$unit-test-salt${digest}",
                state,
                now=100.0,
                uid="guest-login-uid",
            )
        )
        self.assertNotIn("guest-pass", repr(state))

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
