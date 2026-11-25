import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch
import threading
import time

from services import device_identity, firestore_store
from services.firestore_store import (
    SAVED_CHAT_LIMIT,
    SavedChatLimitError,
    _delete_user_chat,
    _delete_all_user_chats,
    _empty_history,
    _enqueue_write,
    _now,
    _save_named_chat,
    _transaction_snapshot,
    _upsert_saved_chat,
    _write_question_pair,
    _write_daily_questions,
    append_message_pair,
    delete_users_without_name_and_recount,
    firestore_status,
    firestore_write_health,
    enqueue_question_answer,
    enqueue_guest_question,
    get_users_without_name,
    get_admin_users,
    get_runtime_config,
    load_chat_history,
    _save_user_name,
    submit_answer_feedback,
    reset_firestore_circuit,
    save_settings_password_hash,
    update_runtime_config,
    get_admin_statistics,
)


class _SessionState(dict):
    __setattr__ = dict.__setitem__


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

    def get(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        self.client.read_paths.append(self.path)
        return _FakeSnapshot(
            self.path,
            data=self.client.document_data.get(self.path, {}),
            exists=self.path in self.client.created_documents,
        )

    def stream(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        self.client.read_paths.append(self.path)
        return iter(self.client.history_chunks.get(self.path, []))

    def create(self, data, *, timeout: float, retry: object):
        from google.api_core.exceptions import AlreadyExists

        assert timeout > 0
        assert retry is None
        if self.path in self.client.created_documents:
            raise AlreadyExists("already exists")
        self.client.created_documents.add(self.path)
        self.client.persisted.append((self.path, data, False))


class _FakeTransaction:
    def __init__(self, client):
        self.client = client
        self.writes = []
        self.active = False
        self.completed = False
        self.owner_thread_id = None

    def begin(self):
        if self.active or self.completed:
            raise ValueError("Transaction not in progress, cannot be used in API requests.")
        self.active = True
        self.owner_thread_id = threading.get_ident()

    def _ensure_active(self):
        if (
            not self.active
            or self.completed
            or self.owner_thread_id != threading.get_ident()
        ):
            raise ValueError("Transaction not in progress, cannot be used in API requests.")

    def get(self, reference, *, timeout: float, retry: object):
        self._ensure_active()
        assert timeout > 0
        assert retry is None
        if reference.is_query:
            return iter([])
        return iter(
            [
                _FakeSnapshot(
                    reference.path,
                    exists=reference.path in self.client.created_documents,
                )
            ]
        )

    def set(self, reference, data, merge: bool = False):
        self._ensure_active()
        self.writes.append((reference.path, data, merge))

    def commit(self, *, timeout: float, retry: object):
        self._ensure_active()
        assert timeout > 0
        assert retry is None
        self.active = False
        self.completed = True
        if self.client.fail_pair_once and any(
            "/chunks/" in path for path, _data, _merge in self.writes
        ):
            self.client.fail_pair_once = False
            raise _UnavailableError("transient failure")
        self.client.persisted.extend(self.writes)
        for path, _data, _merge in self.writes:
            if path.startswith("users/") and path.count("/") == 1:
                self.client.created_documents.add(path)
            if path.startswith("feedback/"):
                self.client.created_documents.add(path)

    def abort(self):
        self.active = False
        self.completed = True


class _FakeBatch:
    def __init__(self, client):
        self.client = client
        self.writes = []
        self.timeout = None

    def set(self, reference, data, merge: bool = False):
        self.writes.append((reference.path, data, merge))

    def commit(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        self.timeout = timeout
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
        self.batches = []
        self.transactions = []
        self.created_documents = set()
        self.read_paths = []
        self.document_data = {}
        self.history_chunks = {}

    def collection(self, name: str):
        return _FakeReference(self, name)

    def transaction(self):
        transaction = _FakeTransaction(self)
        self.transactions.append(transaction)
        return transaction

    def batch(self):
        batch = _FakeBatch(self)
        self.batches.append(batch)
        return batch


class _CleanupSnapshot:
    def __init__(
        self,
        doc_id: str,
        data: dict | None = None,
        *,
        path: str | None = None,
    ):
        self.id = doc_id
        self._data = data or {}
        self.reference = SimpleNamespace(path=path or doc_id)

    def to_dict(self):
        return self._data


class _CleanupReference:
    def __init__(self, client, path: str):
        self.client = client
        self.path = path

    def collection(self, name: str):
        return _CleanupReference(self.client, f"{self.path}/{name}")

    def document(self, name: str):
        return _CleanupReference(self.client, f"{self.path}/{name}")

    def stream(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        parts = self.path.split("/")
        if self.path == "users":
            return iter(
                _CleanupSnapshot(uid, data)
                for uid, data in self.client.users.items()
            )
        if len(parts) == 3 and parts[0] == "users" and parts[2] == "chats":
            return iter(
                _CleanupSnapshot(chat_id)
                for chat_id in self.client.chats.get(parts[1], [])
            )
        if len(parts) == 5 and parts[0] == "users" and parts[2] == "chats":
            return iter(
                _CleanupSnapshot(chunk_id)
                for chunk_id in self.client.chunks.get((parts[1], parts[3]), [])
            )
        raise AssertionError(f"Unexpected stream path: {self.path}")


class _CleanupBatch:
    def __init__(self, client):
        self.client = client
        self.deletions = []
        self.sets = []

    def delete(self, reference):
        self.deletions.append(reference.path)

    def set(self, reference, data, merge: bool = False):
        self.sets.append((reference.path, data, merge))

    def commit(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        for path in self.deletions:
            parts = path.split("/")
            if len(parts) == 2 and parts[0] == "users":
                self.client.users.pop(parts[1], None)
            elif len(parts) == 4 and parts[0] == "users":
                self.client.chats[parts[1]].remove(parts[3])
        for path, data, _merge in self.sets:
            if path == "stats/global":
                self.client.global_stats.update(data)


class _CleanupClient:
    def __init__(self):
        self.users = {
            "no_name": {"total_questions": 0},  # tidak punya nama
            "has_name": {"total_questions": 0, "name": "Budi"},  # punya nama
            "has_questions": {"total_questions": 3, "name": "Siti"},  # punya nama & pertanyaan
        }
        self.chats = {"no_name": [], "has_name": ["chat"], "has_questions": []}
        self.chunks = {("has_name", "chat"): ["000001"]}
        self.global_stats = {"total_devices": 26}

    def collection(self, name: str):
        return _CleanupReference(self, name)

    def collection_group(self, name: str):
        assert name == "chunks"
        return _CleanupChunkGroup(self)

    def batch(self):
        return _CleanupBatch(self)


class _CleanupChunkGroup:
    def __init__(self, client):
        self.client = client

    def stream(self, *, timeout: float, retry: object):
        assert timeout > 0
        assert retry is None
        return iter(
            _CleanupSnapshot(
                chunk_id,
                path=f"users/{uid}/chats/{chat_id}/chunks/{chunk_id}",
            )
            for (uid, chat_id), chunk_ids in self.client.chunks.items()
            for chunk_id in chunk_ids
        )


def _fake_transactional(callback):
    def run(transaction):
        transaction.begin()
        try:
            result = callback(transaction)
            transaction.commit(timeout=10.0, retry=None)
            return result
        except Exception:
            transaction.abort()
            raise

    return run


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

    def test_save_named_chat_persists_name_without_transaction(self) -> None:
        snapshot = Mock()
        snapshot.to_dict.return_value = {
            "saved_chats": [{"chat_id": "old-chat", "name": "Old"}]
        }
        client = Mock()
        user_ref = client.collection.return_value.document.return_value
        user_ref.get.return_value = snapshot
        saved = _save_named_chat(client, "uid", "new-chat", "New")

        self.assertEqual(
            saved,
            [
                {"chat_id": "new-chat", "name": "New"},
                {"chat_id": "old-chat", "name": "Old"},
            ],
        )
        persisted = user_ref.set.call_args.args[0]
        self.assertEqual(persisted["saved_chats"], saved)
        self.assertIn("chat_ids", persisted)
        client.transaction.assert_not_called()

    def test_fake_transaction_rejects_use_outside_callback_and_after_commit(self) -> None:
        transaction = _FakeTransaction(_FakeClient())
        with self.assertRaisesRegex(ValueError, "Transaction not in progress"):
            transaction.get(
                _FakeReference(transaction.client, "users/uid"),
                timeout=1,
                retry=None,
            )

        transaction.begin()
        transaction.commit(timeout=1, retry=None)
        with self.assertRaisesRegex(ValueError, "Transaction not in progress"):
            transaction.set(_FakeReference(transaction.client, "users/uid"), {})

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
                raise _UnavailableError("Firestore unavailable")

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
        self.assertTrue(firestore_status()["connected"])

    def test_slow_runtime_config_read_does_not_delay_render_or_trip_breaker(self) -> None:
        class Document:
            def get(self, *, timeout: float, retry: object):
                assert timeout == 5.0
                assert retry is None
                time.sleep(0.5)
                return self

            @staticmethod
            def to_dict():
                return {"announcement": "loaded in background"}

        class Collection:
            def document(self, _document_id: str):
                return Document()

        class Client:
            def collection(self, _name: str):
                return Collection()

        started = time.monotonic()
        config = get_runtime_config(Client())
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.1)
        self.assertEqual(config["announcement"], "")
        self.assertTrue(
            self._wait_until(
                lambda: get_runtime_config(Client())["announcement"]
                == "loaded in background"
            )
        )
        self.assertTrue(firestore_status()["connected"])

    def test_runtime_config_real_rpc_failure_opens_breaker(self) -> None:
        class Document:
            def get(self, *, timeout: float, retry: object):
                assert timeout == 5.0
                raise _UnavailableError("service unavailable")

        class Collection:
            def document(self, _document_id: str):
                return Document()

        class Client:
            def collection(self, _name: str):
                return Collection()

        self.assertEqual(get_runtime_config(Client())["maintenance"], False)
        self.assertTrue(
            self._wait_until(lambda: firestore_status()["circuit_open"]),
            (
                firestore_status(),
                firestore_store._RUNTIME_CONFIG_REFRESH_ID,
                firestore_store._RUNTIME_CONFIG_REFRESHING,
                firestore_write_health()["runtime_config_skipped_timeouts"],
            ),
        )

    def test_runtime_config_local_budget_timeout_is_counted_not_breaking(self) -> None:
        class Document:
            def get(self, *, timeout: float, retry: object):
                time.sleep(0.2)
                return self

            @staticmethod
            def to_dict():
                return {"announcement": "too late"}

        class Collection:
            def document(self, _document_id: str):
                return Document()

        class Client:
            def collection(self, _name: str):
                return Collection()

        with patch(
            "services.firestore_store.RUNTIME_CONFIG_READ_TIMEOUT_SECONDS",
            0.05,
        ):
            get_runtime_config(Client())
            self.assertTrue(
                self._wait_until(
                    lambda: firestore_write_health()[
                        "runtime_config_skipped_timeouts"
                    ]
                    == 1
                )
            )

        self.assertTrue(firestore_status()["connected"])

    def test_admin_runtime_write_bypasses_open_chat_breaker(self) -> None:
        client = _FakeClient()
        with firestore_store._CIRCUIT_LOCK:
            firestore_store._CIRCUIT_OPEN_UNTIL = time.monotonic() + 30

        update_runtime_config(client, maintenance=True)

        self.assertFalse(firestore_status()["connected"])
        self.assertEqual(firestore_store._RUNTIME_CONFIG["maintenance"], True)
        self.assertEqual(client.persisted[-1][1], {"maintenance": True})
        self.assertEqual(client.batches[-1].timeout, 10.0)

    def test_guest_daily_write_contains_only_guest_counter(self) -> None:
        client = _FakeClient()
        _write_daily_questions(
            client,
            time.monotonic() + 10,
            "",
            "",
            _now(),
            guest=True,
        )

        self.assertEqual(len(client.persisted), 1)
        path, fields, merge = client.persisted[0]
        self.assertRegex(path, r"^stats/daily_\d{8}$")
        self.assertEqual(set(fields), {"guest_questions"})
        self.assertEqual(fields["guest_questions"].value, 1)
        self.assertTrue(merge)
        self.assertEqual(client.read_paths, [])

    def test_guest_enqueue_uses_only_existing_daily_write_kind(self) -> None:
        client = object()
        with patch.object(firestore_store, "_enqueue_write") as enqueue:
            enqueue_guest_question(client)

        enqueue.assert_called_once()
        self.assertEqual(enqueue.call_args.args[:2], (client, "daily_questions"))
        self.assertIs(enqueue.call_args.args[2], _write_daily_questions)
        self.assertEqual(enqueue.call_args.args[3:5], ("", ""))
        self.assertTrue(enqueue.call_args.args[-1])

    def test_expired_guest_emergency_is_ignored(self) -> None:
        expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        with patch.dict(
            firestore_store._RUNTIME_CONFIG,
            {
                "guest_emergency_enabled": True,
                "guest_emergency_expires_at": expired,
            },
        ):
            runtime = firestore_store.get_runtime_config(None)

        self.assertFalse(runtime["guest_emergency_enabled"])

    def test_guest_emergency_cannot_be_saved_past_six_hours(self) -> None:
        client = _FakeClient()
        too_late = (
            datetime.now(timezone.utc) + timedelta(hours=6, seconds=1)
        ).isoformat()

        with self.assertRaisesRegex(ValueError, "maksimal 6 jam"):
            update_runtime_config(
                client,
                guest_emergency_enabled=True,
                guest_emergency_expires_at=too_late,
            )

        self.assertEqual(client.persisted, [])

    def test_question_write_requires_user_profile_to_exist(self) -> None:
        client = _FakeClient()
        uid = "0123456789abcdef"
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=_fake_transactional,
        ):
            with self.assertRaises(firestore_store.ChatStoreError):
                _write_question_pair(
                    client,
                    time.monotonic() + 10,
                    uid,
                    "chat",
                    "Question",
                    "Answer",
                    [],
                    _now(),
                    True,
                )

        self.assertEqual(client.persisted, [])
        self.assertNotIn(f"users/{uid}", client.created_documents)

    def test_saving_new_profile_creates_user_and_counts_device_once(self) -> None:
        client = _FakeClient()
        uid = "0123456789abcdef"
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=_fake_transactional,
        ):
            _save_user_name(client, uid, "Éta Asep")
            _save_user_name(client, uid, "Asep")

        user_writes = [
            data
            for path, data, _merge in client.persisted
            if path == f"users/{uid}"
        ]
        global_writes = [
            data for path, data, _merge in client.persisted if path == "stats/global"
        ]
        self.assertEqual(len(user_writes), 2)
        self.assertEqual(user_writes[0]["name"], "Éta Asep")
        self.assertEqual(user_writes[0]["name_lower"], "éta asep")
        self.assertEqual(user_writes[0]["total_questions"], 0)
        self.assertEqual(len(global_writes), 1)
        self.assertEqual(global_writes[0]["total_devices"].value, 1)

    def test_admin_user_rows_mark_duplicate_names_case_insensitively(self) -> None:
        snapshots = [
            SimpleNamespace(
                id=uid,
                to_dict=lambda name=name: {
                    "name": name,
                    "name_lower": name.casefold(),
                },
            )
            for uid, name in (
                ("0123456789abcdef", "Asep"),
                ("abcdef0123456789", "ASEP"),
                ("fedcba9876543210", ""),
            )
        ]
        users_query = Mock()
        users_query.limit.return_value = users_query
        users_query.stream.return_value = iter(snapshots)
        client = Mock()
        client.collection.return_value = users_query

        users = get_admin_users(client)

        self.assertTrue(users[0]["duplicate_name"])
        self.assertTrue(users[1]["duplicate_name"])
        self.assertFalse(users[2]["duplicate_name"])

    def test_cookie_uid_restores_history_without_page_load_writes(self) -> None:
        uid = "abcdef0123456789"
        chat_id = "saved-chat"
        client = _FakeClient()
        client.document_data[f"users/{uid}"] = {
            "name": "Asep",
            "total_messages": 2,
        }
        chunk_path = f"users/{uid}/chats/{chat_id}/chunks"
        client.history_chunks[chunk_path] = [
            _FakeSnapshot(
                f"{chunk_path}/000001",
                {
                    "messages": [
                        {"role": "user", "content": "remembered question"},
                        {"role": "assistant", "content": "remembered answer"},
                    ],
                    "count": 2,
                },
            )
        ]
        state = _SessionState()
        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", {}),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={"jh_uid": uid}),
            ),
        ):
            resolved_uid = device_identity.get_device_uid()
            history, loaded_stats = load_chat_history(
                client,
                resolved_uid,
                chat_id,
            )

        self.assertEqual(resolved_uid, uid)
        self.assertEqual(
            [message["content"] for message in history],
            ["remembered question", "remembered answer"],
        )
        self.assertEqual(loaded_stats["name"], "Asep")
        self.assertIn(f"users/{uid}", client.read_paths)
        self.assertEqual(client.persisted, [])
        self.assertNotIn(f"users/{uid}", client.created_documents)
        self.assertFalse(
            any("stats/global" in path for path in client.persisted)
        )
        self.assertFalse(
            any(path.startswith("stats/hourly_") for path in client.persisted)
        )

    def test_temporary_uid_never_reaches_device_firestore_operations(self) -> None:
        state = _SessionState()
        state["_device_uid_wait_started"] = 10.0
        client = _FakeClient()
        with (
            patch.object(device_identity.st, "session_state", state),
            patch.object(device_identity.st, "query_params", {}),
            patch.object(
                device_identity.st,
                "context",
                SimpleNamespace(cookies={}),
            ),
            patch.object(device_identity.local_cache, "hydrated_uid", return_value=None),
            patch.object(
                device_identity.local_cache,
                "component_ready",
                return_value=False,
            ),
            patch.object(
                device_identity.time,
                "monotonic",
                return_value=14.1,
            ),
        ):
            uid = device_identity.get_device_uid()
            device_client = firestore_store.get_device_firestore_client(client)

            self.assertFalse(state["_device_uid_browser_resolved"])
            self.assertIsNone(device_client)
            self.assertEqual(load_chat_history(device_client, uid, "chat")[0], [])
            self.assertIsNone(firestore_store.get_active_chat_id(device_client, uid))
            with self.assertRaises(firestore_store.ChatStoreError):
                firestore_store.get_saved_chats(device_client, uid)
            with self.assertRaises(firestore_store.ChatStoreError):
                firestore_store.save_named_chat(device_client, uid, "chat", "saved")
            with self.assertRaises(firestore_store.ChatStoreError):
                firestore_store.delete_user_chat(device_client, uid, "chat")
            with self.assertRaises(firestore_store.ChatStoreError):
                firestore_store.delete_all_user_chats(device_client, uid)
            with patch.object(firestore_store, "_enqueue_write") as enqueue:
                enqueue_question_answer(
                    device_client,
                    uid,
                    "chat",
                    "question",
                    "answer",
                    [],
                )
                enqueue.assert_not_called()

        self.assertEqual(client.read_paths, [])
        self.assertEqual(client.persisted, [])
        self.assertEqual(client.created_documents, set())

    def test_existing_user_question_does_not_increment_total_devices(self) -> None:
        client = _FakeClient()
        uid = "0123456789abcdef"
        client.created_documents.add(f"users/{uid}")
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=_fake_transactional,
        ):
            _write_question_pair(
                client,
                time.monotonic() + 10,
                uid,
                "chat",
                "Question",
                "Answer",
                [],
                _now(),
                True,
            )

        paths = [path for path, _data, _merge in client.persisted]
        self.assertNotIn("stats/global", paths)
        self.assertTrue(any(path.startswith("stats/hourly_") for path in paths))

    def test_quality_metrics_share_the_existing_question_chunk_write(self) -> None:
        client = _FakeClient()
        uid = "0123456789abcdef"
        client.created_documents.add(f"users/{uid}")
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=_fake_transactional,
        ):
            _write_question_pair(
                client,
                time.monotonic() + 10,
                uid,
                "chat",
                "Question",
                "Answer",
                [{"name": "Kitab"}],
                _now(),
                True,
                "a" * 32,
                {
                    "latency_ms": 123,
                    "source_count": 1,
                    "top_score": 0.75,
                    "fallback": True,
                    "model_used": "gemini-test",
                    "source_context": [
                        {"name": "Kitab", "excerpt": "petikan"}
                    ],
                },
                "gemini-test",
            )

        chunk_data = next(
            data for path, data, _merge in client.persisted if "/chunks/" in path
        )
        answer = chunk_data["messages"][-1]
        self.assertEqual(answer["answer_id"], "a" * 32)
        self.assertEqual(answer["latency_ms"], 123)
        self.assertEqual(answer["source_count"], 1)
        self.assertEqual(answer["top_score"], 0.75)
        self.assertTrue(answer["fallback"])
        self.assertEqual(answer["model_used"], "gemini-test")
        self.assertEqual(answer["source_context"][0]["excerpt"], "petikan")

    def test_feedback_is_idempotent_and_updates_summary_once(self) -> None:
        client = _FakeClient()
        args = {
            "uid": "0123456789abcdef",
            "name": "Nami",
            "chat_id": "chat",
            "answer_id": "b" * 32,
            "question": "Question",
            "answer": "Answer",
            "rating": "cocok",
            "reason": None,
            "quality_metrics": {
                "latency_ms": 200,
                "source_count": 1,
                "fallback": False,
                "model_used": "gemini-test",
            },
            "sources": [{"name": "Kitab", "excerpt": "petikan"}],
        }
        with patch(
            "google.cloud.firestore.transactional",
            side_effect=_fake_transactional,
        ):
            self.assertTrue(submit_answer_feedback(client, **args))
            self.assertFalse(submit_answer_feedback(client, **args))

        feedback_writes = [
            data for path, data, _merge in client.persisted
            if path == f"feedback/{args['answer_id']}"
        ]
        summary_writes = [
            data for path, data, _merge in client.persisted
            if path == "stats/quality"
        ]
        self.assertEqual(len(feedback_writes), 1)
        self.assertEqual(len(summary_writes), 1)
        self.assertEqual(feedback_writes[0]["question"], "Question")
        self.assertEqual(feedback_writes[0]["sources"][0]["excerpt"], "petikan")

    def test_users_without_name_cleanup_preserves_history_and_recounts_devices(self) -> None:
        client = _CleanupClient()

        candidates = get_users_without_name(client)
        deleted_count, remaining_count = delete_users_without_name_and_recount(
            client, candidates
        )

        self.assertEqual(candidates, ["no_name"])
        self.assertEqual((deleted_count, remaining_count), (1, 2))
        self.assertNotIn("no_name", client.users)
        self.assertIn("has_name", client.users)
        self.assertIn("has_questions", client.users)
        self.assertEqual(client.global_stats["total_devices"], 2)

    def test_question_write_retries_once_and_persists_all_domains(self) -> None:
        client = _FakeClient()
        client.created_documents.add("users/private-user-id")
        client.fail_pair_once = True
        with (
            patch(
                "services.firestore_store.FIRESTORE_WRITE_RETRY_DELAYS",
                (0.0, 0.0, 0.0),
            ),
            patch("google.cloud.firestore.transactional", side_effect=_fake_transactional),
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
        self.assertEqual(len(client.transactions), 2)
        self.assertIsNot(client.transactions[0], client.transactions[1])

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
        client.created_documents.add("users/private-user-id")
        client.fail_daily = True
        with (
            patch(
                "services.firestore_store.FIRESTORE_WRITE_RETRY_DELAYS",
                (0.0, 0.0, 0.0),
            ),
            patch("google.cloud.firestore.transactional", side_effect=_fake_transactional),
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


class AdminStatisticsGuestTests(unittest.TestCase):
    """Tests for get_admin_statistics including guest_questions field."""

    @staticmethod
    def _make_snapshot(data: dict) -> Any:
        class _Snapshot:
            def __init__(self, d):
                self._data = d

            def to_dict(self):
                return self._data

        return _Snapshot(data)

    def _make_client(self, daily_data: list[dict], global_devices: int = 5):
        class _Client:
            def __init__(self, daily_data, global_devices):
                self.daily_data = {f"daily_{i['day']}": i for i in daily_data}
                self.global_devices = global_devices

            def collection(self, name: str):
                class _Collection:
                    def __init__(self, outer, coll_name):
                        self.outer = outer
                        self.coll_name = coll_name

                    def document(self, doc_id: str):
                        class _Document:
                            def __init__(self, outer, coll_name, doc_id):
                                self.outer = outer
                                self.coll_name = coll_name
                                self.doc_id = doc_id

                            def get(self, *, timeout, retry):
                                assert timeout > 0
                                assert retry is None
                                if self.coll_name == "stats" and self.doc_id == "global":
                                    return AdminStatisticsGuestTests._make_snapshot({"total_devices": self.outer.global_devices})
                                if self.coll_name == "stats" and self.doc_id.startswith("daily_"):
                                    return AdminStatisticsGuestTests._make_snapshot(self.outer.daily_data.get(self.doc_id, {}))
                                if self.coll_name == "stats" and self.doc_id.startswith("hourly_"):
                                    return AdminStatisticsGuestTests._make_snapshot({"active_at": {}})
                                return AdminStatisticsGuestTests._make_snapshot({})

                        return _Document(self.outer, self.coll_name, doc_id)

                return _Collection(self, name)

        return _Client(daily_data, global_devices)

    @patch("services.firestore_store._now")
    def test_get_admin_statistics_includes_guest_questions(self, mock_now):
        """Verify that guest_questions field is read from daily stats."""
        # Mock _now to return a fixed date (2024-01-15 WIB)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        _WIB = ZoneInfo("Asia/Jakarta")
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=_WIB)

        daily_data = [
            {"day": "20240115", "questions": 10, "uids": ["uid1", "uid2"], "guest_questions": 3},
            {"day": "20240114", "questions": 5, "uids": ["uid1"], "guest_questions": 0},
            {"day": "20240113", "questions": 8, "uids": ["uid2", "uid3"], "guest_questions": 2},
        ]
        client = self._make_client(daily_data, global_devices=3)

        stats = get_admin_statistics(client)

        self.assertEqual(stats["total_devices"], 3)
        daily = stats["daily_questions"]
        # Should have 14 days, but only 3 have data
        self.assertGreaterEqual(len(daily), 3)

        # Check guest_questions field exists and has correct values for the 3 days
        day_map = {row["day"]: row for row in daily}
        self.assertEqual(day_map["20240115"]["questions"], 10)
        self.assertEqual(day_map["20240115"]["guest_questions"], 3)
        self.assertEqual(day_map["20240114"]["questions"], 5)
        self.assertEqual(day_map["20240114"]["guest_questions"], 0)
        self.assertEqual(day_map["20240113"]["questions"], 8)
        self.assertEqual(day_map["20240113"]["guest_questions"], 2)

    @patch("services.firestore_store._now")
    def test_guest_questions_not_counted_as_registered_users(self, mock_now):
        """Verify that guest questions don't affect total_devices or unique counts."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        _WIB = ZoneInfo("Asia/Jakarta")
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=_WIB)

        # Many guest questions but few registered users
        daily_data = [
            {"day": "20240115", "questions": 2, "uids": ["uid1"], "guest_questions": 50},
            {"day": "20240114", "questions": 1, "uids": ["uid1"], "guest_questions": 30},
        ]
        client = self._make_client(daily_data, global_devices=1)

        stats = get_admin_statistics(client)

        # total_devices only counts registered users (global_devices)
        self.assertEqual(stats["total_devices"], 1)
        # unique_today/unique_7_days only count registered UIDs
        self.assertEqual(stats["unique_today"], 1)
        self.assertEqual(stats["unique_7_days"], 1)
        # But guest_questions are tracked separately
        day_map = {row["day"]: row for row in stats["daily_questions"]}
        self.assertEqual(day_map["20240115"]["guest_questions"], 50)
        self.assertEqual(day_map["20240114"]["guest_questions"], 30)

    @patch("services.firestore_store._now")
    def test_get_admin_statistics_missing_guest_questions_defaults_to_zero(self, mock_now):
        """Verify backward compatibility: missing guest_questions defaults to 0."""
        from datetime import datetime
        from zoneinfo import ZoneInfo
        _WIB = ZoneInfo("Asia/Jakarta")
        mock_now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=_WIB)

        # Old daily docs without guest_questions field
        daily_data = [
            {"day": "20240115", "questions": 10, "uids": ["uid1"]},  # no guest_questions
            {"day": "20240114", "questions": 5, "uids": [], "guest_questions": 0},  # explicit 0
        ]
        client = self._make_client(daily_data, global_devices=1)

        stats = get_admin_statistics(client)

        day_map = {row["day"]: row for row in stats["daily_questions"]}
        self.assertEqual(day_map["20240115"]["guest_questions"], 0)
        self.assertEqual(day_map["20240114"]["guest_questions"], 0)


if __name__ == "__main__":
    unittest.main()
