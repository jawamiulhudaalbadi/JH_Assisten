import unittest
from unittest.mock import patch

from cryptography.fernet import Fernet

from services.ai_service import (
    ChatKeyQuotaExceeded,
    _AllChatModelsFailedError,
    answer_question,
)
from services import key_pool
from services.firestore_store import reset_firestore_circuit


class _Snapshot:
    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


class _Document:
    def __init__(self, client, path):
        self.client = client
        self.path = path

    def get(self, *, timeout, retry):
        assert 0 < timeout <= 2
        assert retry is None
        if self.client.error:
            raise TimeoutError("Firestore unavailable")
        return _Snapshot(self.client.data.get(self.path, {}))

    def set(self, data, *, timeout):
        assert 0 < timeout <= 2
        if self.client.error:
            raise TimeoutError("Firestore unavailable")
        self.client.data[self.path] = data


class _Collection:
    def __init__(self, client, name):
        self.client = client
        self.name = name

    def document(self, name):
        return _Document(self.client, f"{self.name}/{name}")


class _Client:
    def __init__(self):
        self.data = {}
        self.error = False

    def collection(self, name):
        return _Collection(self, name)


class KeyPoolTests(unittest.TestCase):
    def setUp(self):
        reset_firestore_circuit()
        key_pool.invalidate_cache()
        key_pool._SEED_DISABLED.update(
            {group: False for group in key_pool.KEY_GROUPS}
        )
        self.client = _Client()
        self.secrets = {
            "ENCRYPTION_KEY": Fernet.generate_key().decode("ascii"),
            "GOOGLE_API_KEY": "chat-secret-key",
        }

    def tearDown(self):
        reset_firestore_circuit()
        key_pool.invalidate_cache()

    def test_seed_groups_are_isolated_and_empty_groups_stay_empty(self):
        chat = key_pool.get_keys("chat", None, self.secrets)
        indexing = key_pool.get_keys("indexing", None, self.secrets)
        admin = key_pool.get_keys("admin", None, self.secrets)

        self.assertEqual([item.key for item in chat], ["chat-secret-key"])
        self.assertEqual(indexing, ())
        self.assertEqual(admin, ())

    def test_keys_are_encrypted_and_decrypted_from_firestore(self):
        added = key_pool.add_key(
            "admin",
            self.client,
            self.secrets,
            "admin-secret-123456789",
            "Admin utama",
        )
        stored = self.client.data["config/keys_admin"]["keys"][0]

        self.assertNotIn("admin-secret-123456789", str(stored))
        self.assertEqual(added.sid, key_pool.key_fingerprint("admin-secret-123456789"))
        self.assertNotIn("admin-secret-123456789", repr(added))

        key_pool.invalidate_cache("admin")
        loaded = key_pool.get_keys("admin", self.client, self.secrets)
        self.assertEqual(loaded[0].key, "admin-secret-123456789")
        self.assertEqual(key_pool.mask_key(loaded[0].key), "admi…6789")

    def test_daily_model_quota_cools_only_that_model_and_uses_pacific_reset(self):
        key_pool.get_keys("chat", None, self.secrets)
        key_pool.record_quota(
            "chat",
            self.client,
            self.secrets,
            "chat-secret-key",
            RuntimeError("429 requests per day exceeded"),
            model="gemini-test",
            daily=True,
        )
        record = key_pool.get_keys("chat", self.client, self.secrets)[0]

        self.assertNotIn("gemini-test", key_pool.models_for_key(record, ["gemini-test"]))
        self.assertEqual(key_pool.models_for_key(record, ["gemini-other"]), ["gemini-other"])
        self.assertEqual(record.last_error, "daily_quota")
        self.assertIsNotNone(record.model_cooldowns["gemini-test"])

    def test_firestore_failure_uses_chat_secret_seed(self):
        self.client.error = True

        keys = key_pool.get_keys("chat", self.client, self.secrets, refresh=True)

        self.assertEqual([item.key for item in keys], ["chat-secret-key"])
        self.assertEqual(key_pool.get_keys("indexing", self.client, self.secrets), ())

    def test_removed_seed_does_not_reappear_during_firestore_outage(self):
        seed = key_pool.get_keys("chat", self.client, self.secrets)[0]
        key_pool.remove_key("chat", self.client, self.secrets, seed.sid)
        self.client.error = True

        self.assertEqual(
            key_pool.get_keys("chat", self.client, self.secrets, refresh=True),
            (),
        )

    def test_session_override_never_calls_database(self):
        session_state = {}
        key_pool.set_session_chat_override(session_state, "session-only-key")

        self.assertEqual(
            session_state["chat_api_key_override"],
            "session-only-key",
        )
        self.assertEqual(self.client.data, {})
        key_pool.clear_session_chat_override(session_state)
        self.assertNotIn("chat_api_key_override", session_state)

    def test_admin_model_order_precedes_secret_order(self):
        self.assertEqual(
            key_pool.ordered_models(
                ["model-a", "model-b"],
                {
                    "mode": "manual",
                    "models": ["model-b"],
                    "secrets_models": ["model-a"],
                },
            ),
            ["model-b"],
        )
        self.assertEqual(
            key_pool.ordered_models(
                ["model-a", "model-b"],
                {
                    "mode": "automatic",
                    "models": [],
                    "secrets_models": ["model-b", "model-a"],
                },
            ),
            ["model-b", "model-a"],
        )

    def test_chat_rotates_key_after_all_models_on_current_key_are_limited(self):
        used_keys = []

        def generate(_messages, api_key, **_kwargs):
            used_keys.append(api_key)
            if api_key == "first-key":
                raise _AllChatModelsFailedError(is_quota=True)
            return "jawaban", "gemini-test"

        with patch("services.ai_service._generate_with_failover", side_effect=generate):
            answer, _sources = answer_question(
                None,
                "Patarosan",
                api_key="first-key",
                api_key_candidates=["second-key"],
                model_names=["gemini-test"],
                media_parts=[
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": "image/jpeg",
                        "data": "dGVzdA==",
                    }
                ],
            )

        self.assertEqual(answer, "jawaban")
        self.assertEqual(used_keys, ["first-key", "second-key"])

    def test_session_override_does_not_rotate_after_quota(self):
        with patch(
            "services.ai_service._generate_with_failover",
            side_effect=_AllChatModelsFailedError(is_quota=True),
        ) as generate:
            with self.assertRaises(ChatKeyQuotaExceeded):
                answer_question(
                    None,
                    "Patarosan",
                    api_key="session-only-key",
                    api_key_candidates=["configured-key"],
                    model_names=["gemini-test"],
                    media_parts=[
                        {
                            "type": "file",
                            "source_type": "base64",
                            "mime_type": "image/jpeg",
                            "data": "dGVzdA==",
                        }
                    ],
                    session_key_override=True,
                )

        self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
