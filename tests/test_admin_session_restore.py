import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import services.admin_dashboard as admin_dashboard
import services.admin_session as admin_session
import services.device_identity as device_identity
import services.firestore_store as firestore_store
from services.admin_session import clear_admin_session_epoch_cache, create_admin_session_token
from services import local_cache


_APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
_UID = "0123456789abcdef"
_SECRETS = {"ENCRYPTION_KEY": "test-encryption-key"}
_HISTORY_STATS = {
    "name": "",
    "total_messages": 0,
    "total_chunks": 0,
    "active_chunk_count": 0,
    "active_chunk_number": 0,
    "blocked": False,
    "day": "",
    "day_count": 0,
    "daily_question_limit": 0,
}


class _Snapshot:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class _Document:
    def __init__(self, client):
        self.client = client

    def get(self, *, timeout, retry):
        assert timeout > 0
        assert retry is None
        self.client.reads += 1
        if self.client.unavailable:
            raise OSError("unavailable")
        return _Snapshot({"epoch": self.client.epoch})


class _Collection:
    def __init__(self, client):
        self.client = client

    def document(self, _name):
        return _Document(self.client)


class _Client:
    def __init__(self, epoch=4, unavailable=False):
        self.epoch = epoch
        self.unavailable = unavailable
        self.reads = 0

    def collection(self, _name):
        return _Collection(self)


class AdminSessionRestoreTests(unittest.TestCase):
    def setUp(self):
        clear_admin_session_epoch_cache()

    def _run_app(self, client):
        app = AppTest.from_file(str(_APP_PATH))
        app.secrets["ENCRYPTION_KEY"] = _SECRETS["ENCRYPTION_KEY"]
        app.session_state["device_uid"] = _UID
        app.session_state["_device_uid_browser_resolved"] = True
        app.session_state["admin_authenticated"] = True
        app.session_state["admin_device_verified"] = True
        app.session_state["_admin_session_token"] = create_admin_session_token(
            _UID, client.epoch, _SECRETS
        )
        app.session_state["_admin_auth_method"] = "token"

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    firestore_store,
                    "get_firestore_client",
                    return_value=client,
                )
            )
            stack.enter_context(
                patch.object(
                    firestore_store,
                    "get_runtime_config",
                    return_value={
                        "maintenance": False,
                        "announcement": "",
                        "daily_question_limit": 0,
                        "guest_daily_question_limit": 10,
                        "guest_emergency_enabled": False,
                        "guest_emergency_expires_at": "",
                        "settings_password_hash": "",
                    },
                )
            )
            stack.enter_context(
                patch.object(device_identity, "get_device_uid", return_value=_UID)
            )
            stack.enter_context(patch("services.key_pool.get_available_keys", return_value=[]))
            stack.enter_context(
                patch(
                    "services.key_pool.get_model_order",
                    return_value={
                        "mode": "automatic",
                        "models": [],
                        "secrets_models": [],
                    },
                )
            )
            stack.enter_context(patch.object(local_cache, "_component"))
            stack.enter_context(patch("services.warmup.warm_start_once"))
            stack.enter_context(
                patch.object(
                    firestore_store,
                    "load_chat_history",
                    return_value=([], dict(_HISTORY_STATS)),
                )
            )
            stack.enter_context(
                patch.object(firestore_store, "get_active_chat_id", return_value=None)
            )
            render_dashboard = stack.enter_context(
                patch.object(admin_dashboard, "render_dashboard")
            )
            epoch_reader = stack.enter_context(
                patch(
                    "services.admin_session.get_admin_session_epoch",
                    wraps=admin_session.get_admin_session_epoch,
                )
            )
            app.run(timeout=60)

        return app, render_dashboard, epoch_reader

    def test_cold_cache_reads_epoch_synchronously_and_restores_dashboard(self):
        client = _Client()

        app, render_dashboard, epoch_reader = self._run_app(client)

        self.assertEqual(len(app.exception), 0)
        epoch_reader.assert_called_once_with(client, background_refresh=False)
        self.assertEqual(client.reads, 1)
        self.assertTrue(app.session_state["admin_authenticated"])
        self.assertTrue(app.session_state["admin_device_verified"])
        render_dashboard.assert_called_once()

    def test_cold_cache_firestore_failure_preserves_state_and_blocks_dashboard(self):
        client = _Client(unavailable=True)

        app, render_dashboard, epoch_reader = self._run_app(client)

        self.assertEqual(len(app.exception), 0)
        epoch_reader.assert_called_once_with(client, background_refresh=False)
        self.assertEqual(client.reads, 1)
        self.assertTrue(app.session_state["admin_authenticated"])
        self.assertTrue(app.session_state["admin_device_verified"])
        render_dashboard.assert_not_called()

    def test_warm_cache_uses_background_refresh_and_restores_dashboard(self):
        client = _Client()
        self.assertEqual(admin_session.get_admin_session_epoch(client), 4)

        app, render_dashboard, epoch_reader = self._run_app(client)

        self.assertEqual(len(app.exception), 0)
        epoch_reader.assert_called_once_with(client, background_refresh=True)
        self.assertTrue(app.session_state["admin_authenticated"])
        self.assertTrue(app.session_state["admin_device_verified"])
        render_dashboard.assert_called_once()


if __name__ == "__main__":
    unittest.main()
