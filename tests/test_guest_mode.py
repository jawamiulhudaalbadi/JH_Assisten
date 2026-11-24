import hashlib
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

from services import device_identity, firestore_store, local_cache


_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _guest_hash(code: str) -> str:
    salt = "guest-test-salt"
    digest = hashlib.pbkdf2_hmac(
        "sha256", code.encode(), salt.encode(), 200_000
    ).hex()
    return f"pbkdf2_sha256$200000${salt}${digest}"


class _RejectingFirestoreClient:
    def __init__(self):
        self.calls = []

    def collection(self, name):
        self.calls.append(("collection", name))
        raise AssertionError(f"Unexpected Firestore collection: {name}")

    def batch(self):
        self.calls.append(("batch",))
        raise AssertionError("Unexpected Firestore batch")

    def transaction(self):
        self.calls.append(("transaction",))
        raise AssertionError("Unexpected Firestore transaction")


class GuestModeAppTests(unittest.TestCase):
    @contextmanager
    def _firestore_isolation(self, app, runtime=None, guest_hash=True):
        strict_client = _RejectingFirestoreClient()

        def get_test_uid(*_args, **_kwargs):
            uid = (
                app.session_state.get("device_uid", "0123456789abcdef")
                if app.session_state.get("_device_uid_browser_resolved")
                else "fedcba9876543210"
            )
            app.session_state["device_uid"] = uid
            return uid

        config = {
            "maintenance": False,
            "announcement": "",
            "daily_question_limit": 0,
            "guest_daily_question_limit": 10,
            "guest_emergency_enabled": False,
            "guest_emergency_expires_at": "",
            "settings_password_hash": "",
        }
        config.update(runtime or {})
        if guest_hash:
            app.secrets["GUEST_CODE_HASH"] = _guest_hash("guest-pass")
        with (
            patch.object(
                firestore_store,
                "get_firestore_client",
                return_value=strict_client,
            ),
            patch.object(
                firestore_store,
                "get_runtime_config",
                return_value=config,
            ),
            patch("services.key_pool.get_available_keys", return_value=[]),
            patch(
                "services.key_pool.get_model_order",
                return_value={
                    "mode": "automatic",
                    "models": [],
                    "secrets_models": [],
                },
            ),
            patch.object(
                device_identity,
                "get_device_uid",
                side_effect=get_test_uid,
            ),
            patch("services.warmup.warm_start_once"),
            patch.object(local_cache, "_component"),
        ):
            yield strict_client

    def test_unfinalized_id_has_disabled_chat_and_no_name_form(self):
        app = AppTest.from_file(str(_APP_PATH))
        with self._firestore_isolation(app):
            app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertNotIn("Nami", [item.label for item in app.text_input])

    def test_final_id_without_name_shows_required_name_and_disables_chat(self):
        app = AppTest.from_file(str(_APP_PATH))
        app.session_state["device_uid"] = "0123456789abcdef"
        app.session_state["_device_uid_browser_resolved"] = True
        with self._firestore_isolation(app):
            app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Nami", [item.label for item in app.text_input])
        self.assertTrue(app.chat_input[0].disabled)

    def test_saving_chat_keeps_saved_name_and_profile_state(self):
        uid = "0123456789abcdef"
        chat_id = "abcdef012345"
        app = AppTest.from_file(str(_APP_PATH))
        app.query_params["chat"] = chat_id
        app.session_state["device_uid"] = uid
        app.session_state["_device_uid_browser_resolved"] = True
        app.session_state["chat_id"] = chat_id
        app.session_state["loaded_chat_id"] = chat_id
        app.session_state["_loaded_chat_uid"] = uid
        app.session_state["_history_stats_loaded"] = True
        app.session_state["messages"] = [
            {"role": "user", "content": "Patarosan"},
            {"role": "assistant", "content": "Waleran"},
        ]
        app.session_state["history_stats"] = {
            "name": "Asep",
            "total_messages": 2,
            "total_chunks": 1,
            "active_chunk_count": 2,
            "blocked": True,
            "day_count": 4,
        }

        with (
            self._firestore_isolation(app),
            patch.object(firestore_store, "get_saved_chats", return_value=[]),
            patch.object(
                firestore_store,
                "save_named_chat",
                return_value=[{"chat_id": chat_id, "name": "Paguneman"}],
            ),
        ):
            app.run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            next(
                item
                for item in app.text_input
                if item.label == "Ngaran riwayat"
            ).set_value("Paguneman")
            next(
                button
                for button in app.form_submit_button
                if button.label == "Simpan Paguneman"
            ).click().run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["history_stats"]["name"], "Asep")
        self.assertTrue(app.session_state["history_stats"]["blocked"])
        self.assertEqual(app.session_state["history_stats"]["day_count"], 4)
        self.assertFalse(app.chat_input[0].disabled)
        self.assertNotIn("Nami", [item.label for item in app.text_input])

        reloaded_app = AppTest.from_file(str(_APP_PATH))
        reloaded_app.query_params["chat"] = app.session_state["chat_id"]
        reloaded_app.session_state["device_uid"] = uid
        reloaded_app.session_state["_device_uid_browser_resolved"] = True
        with (
            self._firestore_isolation(reloaded_app),
            patch.object(
                firestore_store,
                "load_chat_history",
                return_value=(
                    [],
                    {
                        "name": "Asep",
                        "total_messages": 2,
                        "total_chunks": 1,
                    },
                ),
            ),
            patch.object(firestore_store, "get_saved_chats", return_value=[]),
        ):
            reloaded_app.run(timeout=60)

        self.assertEqual(len(reloaded_app.exception), 0)
        self.assertFalse(reloaded_app.chat_input[0].disabled)
        self.assertNotIn("Nami", [item.label for item in reloaded_app.text_input])

    def test_correct_guest_code_activates_chat_and_avoids_firestore_reads(self):
        app = AppTest.from_file(str(_APP_PATH))
        app.session_state["_guest_code_form_open"] = True
        with self._firestore_isolation(app) as strict_client:
            app.run(timeout=60)
            self.assertEqual(len(app.exception), 0)

            app.text_input(key="guest_unrecognized_id_guest_code").set_value(
                "guest-pass"
            )
            submit = next(
                button
                for button in app.form_submit_button
                if button.label == "Buka mode tamu"
            )
            submit.click().run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["guest_mode"])
        self.assertFalse(app.chat_input[0].disabled)
        self.assertTrue(
            any(
                "MODE TAMU - riwayat teu disimpen" in item.value
                for item in app.markdown
            )
        )
        self.assertIn("Tamu", [item.value for item in app.success])
        self.assertEqual(strict_client.calls, [])

    def test_active_emergency_button_enters_guest_without_code(self):
        expiry = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
        app = AppTest.from_file(str(_APP_PATH))
        with self._firestore_isolation(
            app,
            {
                "guest_emergency_enabled": True,
                "guest_emergency_expires_at": expiry,
            },
            guest_hash=False,
        ) as strict_client:
            app.run(timeout=60)
            self.assertEqual(len(app.exception), 0)
            next(
                button for button in app.button if button.label == "Mode tamu"
            ).click().run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.session_state["guest_mode"])
        self.assertTrue(app.session_state["guest_mode_via_emergency"])
        self.assertFalse(app.chat_input[0].disabled)
        self.assertTrue(
            any(
                "Katup darurat mode tamu keur dibuka" in item.value
                for item in app.error
            )
        )
        self.assertEqual(strict_client.calls, [])

    def test_expired_emergency_access_does_not_render_bypass(self):
        expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        app = AppTest.from_file(str(_APP_PATH))
        with self._firestore_isolation(
            app,
            {
                "guest_emergency_enabled": False,
                "guest_emergency_expires_at": expired,
            },
            guest_hash=False,
        ):
            app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Mode tamu", [item.label for item in app.button])

    def test_guest_session_question_limit_is_kept_in_session_state(self):
        app = AppTest.from_file(str(_APP_PATH))
        app.session_state["guest_mode"] = True
        app.session_state["guest_questions_day"] = datetime.now(
            ZoneInfo("Asia/Jakarta")
        ).strftime("%Y%m%d")
        app.session_state["guest_questions_count"] = 10
        with self._firestore_isolation(app) as strict_client:
            app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertTrue(
            any("wates 10 patarosan" in item.value for item in app.warning)
        )
        self.assertEqual(strict_client.calls, [])


if __name__ == "__main__":
    unittest.main()
