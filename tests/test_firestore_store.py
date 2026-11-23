import unittest

from services.firestore_store import _transaction_snapshot, append_message_pair


class FirestoreStoreTests(unittest.TestCase):
    def test_transaction_snapshot_reads_snapshot_from_generator(self) -> None:
        snapshot = object()

        class Transaction:
            def get(self, document_ref: object):
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


if __name__ == "__main__":
    unittest.main()
