import json
import time
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from services import local_cache


class LocalCacheTests(unittest.TestCase):
    uid = "0123456789abcdef"

    def setUp(self) -> None:
        self.state = {}
        self.session_state = patch.object(
            local_cache.st,
            "session_state",
            self.state,
        )
        self.session_state.start()
        self.addCleanup(self.session_state.stop)
        self.query_params = {}
        self.query_patch = patch.object(
            local_cache.st,
            "query_params",
            self.query_params,
        )
        self.query_patch.start()
        self.addCleanup(self.query_patch.stop)

    def _snapshot(self, **updates):
        snapshot = {
            "v": local_cache.SCHEMA_VERSION,
            "uid": self.uid,
            "chat_id": "chat-1",
            "ts": time.time(),
            "messages": [{"role": "user", "content": "cached"}],
        }
        snapshot.update(updates)
        return json.dumps(snapshot)

    def test_snapshot_keeps_only_last_thirty_messages(self) -> None:
        messages = [
            {"role": "user", "content": str(index)}
            for index in range(35)
        ]

        payload = json.loads(
            local_cache._snapshot_json(self.uid, "chat-1", messages)
        )

        self.assertEqual(len(payload["messages"]), 30)
        self.assertEqual(payload["messages"][0]["content"], "5")
        self.assertEqual(payload["messages"][-1]["content"], "34")

    def test_snapshot_truncates_content_and_fits_size_limit(self) -> None:
        raw = local_cache._snapshot_json(
            self.uid,
            "chat-1",
            [
                {"role": "user", "content": "x" * 20_000}
                for _ in range(30)
            ],
        )
        payload = json.loads(raw)

        self.assertLessEqual(
            len(raw.encode("utf-8")),
            local_cache.MAX_SNAPSHOT_BYTES,
        )
        self.assertEqual(len(payload["messages"][0]["content"]), 8_000)
        self.assertTrue(
            all(
                len(message["content"]) <= local_cache.MAX_CONTENT_CHARS
                for message in payload["messages"]
            )
        )
        self.assertLess(len(payload["messages"]), 30)

    def test_snapshot_serializes_datetime_timestamp(self) -> None:
        timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)

        payload = json.loads(
            local_cache._snapshot_json(
                self.uid,
                "chat-1",
                [{"role": "assistant", "content": "hello", "timestamp": timestamp}],
            )
        )

        self.assertEqual(payload["messages"][0]["timestamp"], str(timestamp))

    def test_mount_does_not_replace_existing_messages(self) -> None:
        current_messages = [{"role": "user", "content": "current"}]
        self.state.update(
            {
                "jh_lc": {"ready": True, "snapshot": self._snapshot()},
                "messages": current_messages,
            }
        )

        local_cache.mount(self.uid)

        self.assertIs(self.state["messages"], current_messages)
        self.assertNotIn("chat_id", self.state)

    def test_mount_updates_chat_query_only_when_no_chat_was_requested(self) -> None:
        self.state["jh_lc"] = {
            "ready": True,
            "snapshot": self._snapshot(chat_id="restored-chat"),
        }

        local_cache.mount(self.uid)

        self.assertEqual(self.state["chat_id"], "restored-chat")
        self.assertEqual(self.query_params["chat"], "restored-chat")

        self.state.clear()
        self.query_params["chat"] = "X"
        self.state["jh_lc"] = {
            "ready": True,
            "snapshot": self._snapshot(chat_id="restored-chat"),
        }

        local_cache.mount(self.uid)

        self.assertEqual(self.state["chat_id"], "restored-chat")
        self.assertEqual(self.query_params["chat"], "X")

    def test_mount_rejects_uid_schema_and_expired_snapshots(self) -> None:
        expired = self._snapshot(ts=time.time() - local_cache.TTL_SECONDS - 1)
        invalid_snapshots = (
            self._snapshot(uid="fedcba9876543210"),
            self._snapshot(v=local_cache.SCHEMA_VERSION + 1),
            expired,
            self._snapshot(chat_id="../other-user"),
        )

        for snapshot in invalid_snapshots:
            with self.subTest(snapshot=snapshot):
                self.state.clear()
                self.state["jh_lc"] = {"ready": True, "snapshot": snapshot}

                local_cache.mount(self.uid)

                self.assertNotIn("messages", self.state)
                self.assertNotIn("chat_id", self.state)

    def test_mount_ignores_guest_cache(self) -> None:
        self.state.update(
            {
                "jh_lc": {"ready": True, "snapshot": self._snapshot()},
                "guest_mode": True,
            }
        )

        local_cache.mount(self.uid)

        self.assertNotIn("messages", self.state)
        self.assertNotIn("chat_id", self.state)

    def test_persist_sends_snapshot_only_when_content_changes(self) -> None:
        messages = [{"role": "user", "content": "same"}]
        with patch.object(local_cache, "_component") as component:
            local_cache.persist(self.uid, "chat-1", messages)
            local_cache.persist(self.uid, "chat-1", messages)
            messages[0]["content"] = "changed"
            local_cache.persist(self.uid, "chat-1", messages)

        self.assertEqual(component.call_count, 3)
        self.assertIsNotNone(component.call_args_list[0].kwargs["snapshot"])
        self.assertIsNone(component.call_args_list[1].kwargs["snapshot"])
        self.assertIsNotNone(component.call_args_list[2].kwargs["snapshot"])


if __name__ == "__main__":
    unittest.main()
