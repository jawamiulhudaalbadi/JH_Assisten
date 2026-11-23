import unittest
from unittest.mock import Mock, patch
import threading
import time

from services import firestore_store
from services.firestore_store import (
    SAVED_CHAT_LIMIT,
    SavedChatLimitError,
    _delete_user_chat,
    _delete_all_user_chats,
    _empty_history,
    _enqueue_write,
    _save_named_chat,
    _transaction_snapshot,
    _upsert_saved_chat,
    append_message_pair,
    firestore_status,
    firestore_write_health,
    enqueue_question_answer,
    load_chat_history,
    reset_firestore_circuit,
    save_settings_password_hash,
)


class _UnavailableError(Exception):
    code = "UNAVAILABLE"


class _FakeSnapshot:
    def __init__(self, path: str, data: dict | None = None, exists: bool = True):
        self.id = path.rsplit("/", 1)[-1]
        self.path = path
        self._data = data or {}
        self.exists = exists

    def to_dict(self):
        return self._data


class _FakeReference:
    def __init__(self, client, path: str, is_query: bool = False):
        self.client = client
        self.path = path
        self.is_query = is_query

    def collection(self, name: str):
        return _FakeReference(self.client, f"{self.path}/{name}")

    def document(self, name: str):
        return _FakeReference(self.client, f"{self.path}/{name}")

    def order_by(self, _field: str, *, direction: str):
        assert direction
        return _FakeReference(self.client, self.path, is_query=True)

    def limit(self, _count: int):
        return self


class _FakeTransaction:
    def __init__(self, client):
        self.client = client
        self.writes = []

    def get(self, reference, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        if reference.is_query:
            return iter([])
        return iter([_FakeSnapshot(reference.path, exists=False)])

    def set(self, reference, data, merge: bool = False):
        self.writes.append((reference.path, data, merge))

    def commit(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        if (
            self.client.fail_pair_once
            and any("/chunks/" in path for path, _data, _merge in self.writes)
        ):
            self.client.fail_pair_once = False
            raise _UnavailableError("transient failure")
        self.client.persisted.extend(self.writes)


class _FakeBatch:
    def __init__(self, client):
        self.client = client
        self.writes = []

    def set(self, reference, data, merge: bool = False):
        self.writes.append((reference.path, data, merge))

    def commit(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        if self.client.fail_daily and any(
            "/daily_" in path for path, _data, _merge in self.writes
        ):
            raise ValueError("permanent failure")
        self.client.persisted.extend(self.writes)


class _FakeClient:
    def __init__(self):
        self.persisted = []
        self.fail_pair_once = False
        self.fail_daily = False

    def collection(self, name: str):
        return _FakeReference(self, name)

    def transaction(self):
        return _FakeTransaction(self)

    def batch(self):
        return _FakeBatch(self)


class FirestoreStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_firestore_circuit()

    def tearDown(self) -> None:
        reset_firestore_circuit()

    def _wait_until(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def test_transaction_snapshot_reads_snapshot_from_generator(self) -> None:
        snapshot = object()

        class Transaction:
            def get(
                self,
                document_ref: object,
                *,
                timeout: float,
                retry: object,
            ):
                assert document_ref is not None
                assert timeout > 0
                assert retry is None
                return iter([snapshot])

        self.assertIs(_transaction_snapshot(Transaction(), object()), snapshot)

    def test_pair_fits_exactly_in_chunk(self) -> None:
        messages = [{"role": "user"} for _ in range(18)]

        updated, created_new_chunk = append_message_pair(
            messages,
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        )

        self.assertFalse(created_new_chunk)
        self.assertEqual(len(updated), 20)
        self.assertEqual(
            [item["content"] for item in updated[-2:]],
            ["question", "answer"],
        )

    def test_pair_starts_new_chunk_when_current_chunk_is_full(self) -> None:
        messages = [{"role": "user"} for _ in range(20)]

        updated, created_new_chunk = append_message_pair(
            messages,
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        )

        self.assertTrue(created_new_chunk)
        self.assertEqual(
            updated,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_saved_chat_limit_blocks_sixth_and_allows_renaming(self) -> None:
        saved = [
            {"chat_id": f"chat-{index}", "name": f"Chat {index}"}
            for index in range(SAVED_CHAT_LIMIT)
        ]

        with self.assertRaises(SavedChatLimitError):
            _upsert_saved_chat(saved, "new-chat", "New chat")

        renamed = _upsert_saved_chat(saved, "chat-0", "Renamed")
        self.assertEqual(len(renamed), SAVED_CHAT_LIMIT)
        self.assertEqual(renamed[0], {"chat_id": "chat-0", "name": "Renamed"})

    def test_save_named_chat_persists_name_with_transaction(self) -> None:
        snapshot = Mock()
        snapshot.to_dict.return_value = {
            "saved_chats": [{"chat_id": "old-chat", "name": "Old"}]
        }
        transaction = Mock()
        transaction.get.return_value = iter([snapshot])
        client = Mock()
        client.collection.return_value.document.return_value = object()
        client.transaction.return_value = transaction
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=lambda callback: lambda transaction: callback(transaction),
        ):
            saved = _save_named_chat(client, "uid", "new-chat", "New")

        self.assertEqual(
            saved,
            [
                {"chat_id": "new-chat", "name": "New"},
                {"chat_id": "old-chat", "name": "Old"},
            ],
        )
        persisted = transaction.set.call_args.args[1]
        self.assertEqual(persisted["saved_chats"], saved)
        self.assertIn("chat_ids", persisted)

    def test_delete_single_chat_clears_its_metadata_and_updates_counts(self) -> None:
        chunk = Mock()
        chunk.to_dict.return_value = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            "count": 2,
        }
        chunk.reference = "chunk-ref"
        chunks = Mock()
        chunks.stream.return_value = [chunk]
        chat_ref = Mock()
        chat_ref.collection.return_value = chunks
        chats = Mock()
        chats.document.return_value = chat_ref
        user_ref = Mock()
        user_ref.get.return_value.to_dict.return_value = {
            "chat_ids": ["active-chat", "other-chat"],
            "saved_chats": [
                {"chat_id": "active-chat", "name": "Saved"},
                {"chat_id": "other-chat", "name": "Other"},
            ],
            "active_chat_id": "active-chat",
            "total_messages": 6,
            "total_questions": 3,
            "total_chunks": 4,
        }
        user_ref.collection.return_value = chats
        batch = Mock()
        client = Mock()
        client.collection.return_value.document.return_value = user_ref
        client.batch.return_value = batch

        updated = _delete_user_chat(client, "uid", "active-chat")

        self.assertEqual(updated["total_messages"], 4)
        self.assertEqual(updated["total_questions"], 2)
        self.assertEqual(updated["total_chunks"], 3)
        persisted = user_ref.set.call_args.args[0]
        self.assertEqual(persisted["chat_ids"], ["other-chat"])
        self.assertEqual(
            persisted["saved_chats"],
            [{"chat_id": "other-chat", "name": "Other"}],
        )
        self.assertIsNone(persisted["active_chat_id"])
        self.assertEqual(batch.delete.call_args_list[0].args[0], "chunk-ref")
        self.assertEqual(batch.delete.call_args_list[1].args[0], chat_ref)

    def test_delete_all_chats_removes_saved_and_legacy_chat_ids(self) -> None:
        user_ref = Mock()
        user_ref.get.return_value.to_dict.return_value = {
            "chat_ids": ["chat-1"],
            "active_chat_id": "chat-2",
            "saved_chats": [
                {"chat_id": "chat-1", "name": "First"},
                {"chat_id": "chat-3", "name": "Third"},
            ],
        }
        chats = Mock()
        chats.stream.return_value = [Mock(id="chat-4")]
        user_ref.collection.return_value = chats
        chat_ids = ("chat-1", "chat-2", "chat-3", "chat-4")
        chat_refs = {chat_id: Mock() for chat_id in chat_ids}
        for chat_ref in chat_refs.values():
            chat_ref.collection.return_value.stream.return_value = []
        chats.document.side_effect = chat_refs.__getitem__
        batch = Mock()
        client = Mock()
        client.collection.return_value.document.return_value = user_ref
        client.batch.return_value = batch

        _delete_all_user_chats(client, "uid")

        self.assertEqual(chats.document.call_count, 4)
        self.assertEqual(batch.delete.call_count, 4)
        update = user_ref.set.call_args.args[0]
        self.assertEqual(update["chat_ids"], [])
        self.assertEqual(update["saved_chats"], [])
        self.assertIsNone(update["active_chat_id"])
        self.assertEqual(update["total_messages"], 0)
        self.assertEqual(update["total_questions"], 0)

    def test_settings_password_hash_is_written_to_runtime_config(self) -> None:
        batch = Mock()
        runtime_ref = object()
        client = Mock()
        client.collection.return_value.document.return_value = runtime_ref
        client.batch.return_value = batch
        encoded_hash = "pbkdf2_sha256$200000$salt$digest"

        with patch.dict("services.firestore_store._RUNTIME_CONFIG", clear=True):
            save_settings_password_hash(client, encoded_hash)
            self.assertEqual(
                firestore_store._RUNTIME_CONFIG["settings_password_hash"],
                encoded_hash,
            )

        batch.set.assert_called_once_with(
            runtime_ref,
            {"settings_password_hash": encoded_hash},
            merge=True,
        )
        batch.commit.assert_called_once()

    def test_firestore_failure_returns_empty_history_and_opens_circuit(self) -> None:
        class Document:
            def get(self, *, timeout: float, retry: object):
                self.assert_timeout(timeout, retry)
                raise OSError("Firestore unavailable")

            @staticmethod
            def assert_timeout(timeout: float, retry: object) -> None:
                assert 0 < timeout <= 5
                assert retry is None

        class Collection:
            def document(self, _document_id: str):
                return Document()

        class Client:
            def collection(self, _name: str):
                return Collection()

        with patch("services.firestore_store.logger.error") as log_error:
            self.assertEqual(load_chat_history(Client(), "uid", "chat"), _empty_history())
            self.assertEqual(load_chat_history(Client(), "uid", "chat"), _empty_history())

        self.assertEqual(log_error.call_count, 1)
        self.assertFalse(firestore_status()["connected"])

    def test_slow_firestore_does_not_hold_history_render(self) -> None:
        class Document:
            def get(self, *, timeout: float, retry: object):
                self.timeout = timeout
                self.retry = retry
                time.sleep(0.3)
                return self

            @staticmethod
            def to_dict():
                return {}

        class Collection:
            def document(self, _document_id: str):
                return Document()

        class Client:
            def collection(self, _name: str):
                return Collection()

        with patch("services.firestore_store.HISTORY_LOAD_TIMEOUT_SECONDS", 0.05):
            started = time.monotonic()
            history = load_chat_history(Client(), "uid", "chat")
            elapsed = time.monotonic() - started

        self.assertEqual(history, _empty_history())
        self.assertLess(elapsed, 0.2)
        self.assertFalse(firestore_status()["connected"])

    def test_question_write_retries_once_and_persists_all_domains(self) -> None:
        client = _FakeClient()
        client.fail_pair_once = True
        with patch(
            "services.firestore_store.FIRESTORE_WRITE_RETRY_DELAYS",
            (0.0, 0.0, 0.0),
        ):
            enqueue_question_answer(
                client,
                "private-user-id",
                "chat",
                "Question",
                "Answer",
                [],
            )
            self.assertTrue(
                self._wait_until(
                    lambda: firestore_write_health()["writes_ok"] >= 3
                    and firestore_write_health()["writes_queued"] == 0
                )
            )

        paths = [path for path, _data, _merge in client.persisted]
        self.assertTrue(any("/chunks/" in path for path in paths))
        self.assertTrue(any(path.startswith("stats/daily_") for path in paths))
        self.assertTrue(any(path.startswith("faq/") for path in paths))
        health = firestore_write_health()
        self.assertEqual(health["writes_retried"], 1)
        self.assertEqual(health["writes_failed"], 1)

    def test_open_breaker_keeps_write_queued_until_available(self) -> None:
        client = _FakeClient()
        persisted = []
        completed = threading.Event()

        def write(_client, _deadline):
            persisted.append("written")
            completed.set()

        with firestore_store._CIRCUIT_LOCK:
            firestore_store._CIRCUIT_OPEN_UNTIL = time.monotonic() + 0.15
        _enqueue_write(client, "test_write", write)

        self.assertTrue(self._wait_until(lambda: firestore_write_health()["writes_queued"] == 1))
        self.assertFalse(completed.is_set())
        self.assertTrue(completed.wait(2.0))
        self.assertEqual(persisted, ["written"])

    def test_write_queue_drops_oldest_item_at_its_bound(self) -> None:
        client = _FakeClient()
        noop = lambda _client, _deadline: None
        with patch("services.firestore_store.FIRESTORE_WRITE_QUEUE_LIMIT", 3):
            with firestore_store._CIRCUIT_LOCK:
                firestore_store._CIRCUIT_OPEN_UNTIL = time.monotonic() + 30
            for index in range(4):
                _enqueue_write(client, f"write-{index}", noop)

            self.assertEqual(firestore_write_health()["writes_queued"], 3)
            self.assertEqual(firestore_write_health()["writes_dropped"], 1)
            self.assertEqual(
                [job.kind for job in firestore_store._WRITE_QUEUE],
                ["write-1", "write-2", "write-3"],
            )

    def test_daily_stats_failure_does_not_cancel_chunk_write(self) -> None:
        client = _FakeClient()
        client.fail_daily = True
        with patch(
            "services.firestore_store.FIRESTORE_WRITE_RETRY_DELAYS",
            (0.0, 0.0, 0.0),
        ):
            enqueue_question_answer(
                client,
                "private-user-id",
                "chat",
                "Question",
                "Answer",
                [],
            )
            self.assertTrue(
                self._wait_until(
                    lambda: firestore_write_health()["writes_ok"] >= 2
                    and firestore_write_health()["writes_failed"] >= 1
                    and firestore_write_health()["writes_queued"] == 0
                )
            )

        paths = [path for path, _data, _merge in client.persisted]
        self.assertTrue(any("/chunks/" in path for path in paths))
        self.assertTrue(any(path.startswith("faq/") for path in paths))
        self.assertFalse(any("/daily_" in path for path in paths))


if __name__ == "__main__":
    unittest.main()
