import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from services.firestore_store import (
    get_admin_user_page,
    get_answer_quality_page,
    save_answer_reviews,
)


class _Snapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _DocumentReference:
    def __init__(self, collection, doc_id):
        self.collection_name = collection
        self.doc_id = doc_id


class _Query:
    def __init__(self, client, collection):
        self.client = client
        self.collection_name = collection
        self.filters = []
        self.fields = None
        self.orders = []
        self.maximum = None
        self.cursor = None

    def document(self, doc_id):
        return _DocumentReference(self.collection_name, doc_id)

    def select(self, fields):
        self.fields = list(fields)
        return self

    def where(self, field, operator, value):
        self.filters.append((field, operator, value))
        return self

    def order_by(self, field, direction="ASCENDING"):
        self.orders.append((field, direction))
        return self

    def limit(self, maximum):
        self.maximum = maximum
        return self

    def start_after(self, cursor):
        self.cursor = cursor
        return self

    def stream(self, *, timeout, retry):
        assert timeout > 0
        assert retry is None
        self.client.queries.append(self)
        rows = [
            _Snapshot(doc_id, data)
            for doc_id, data in self.client.data.get(self.collection_name, {}).items()
        ]
        for field, operator, expected in self.filters:
            if operator == "in":
                rows = [
                    row for row in rows
                    if row.to_dict().get(field) in expected
                ]
            elif operator == "==":
                rows = [
                    row for row in rows
                    if row.to_dict().get(field) == expected
                ]
            elif operator == ">=":
                rows = [
                    row for row in rows
                    if row.to_dict().get(field, "") >= expected
                ]
            elif operator == "<=":
                rows = [
                    row for row in rows
                    if row.to_dict().get(field, "") <= expected
                ]
        for field, direction in reversed(self.orders):
            rows.sort(
                key=lambda row: (
                    row.id if field == "__name__" else row.to_dict().get(field)
                ),
                reverse=direction == "DESCENDING",
            )
        if self.cursor:
            cursor_values = self.cursor
            if isinstance(cursor_values, dict):
                cursor_id = cursor_values["__name__"].doc_id
                if self.collection_name == "users":
                    if "name_lower" in cursor_values:
                        cursor_key = (
                            cursor_values["name_lower"],
                            cursor_id,
                        )
                        rows = [
                            row
                            for row in rows
                            if (
                                row.to_dict().get("name_lower", ""),
                                row.id,
                            )
                            > cursor_key
                        ]
                    else:
                        rows = [row for row in rows if row.id > cursor_id]
                elif "name_lower" in cursor_values:
                    cursor_key = (cursor_values["name_lower"], cursor_id)
                    rows = [
                        row
                        for row in rows
                        if (
                            row.to_dict().get("name_lower", ""),
                            row.id,
                        )
                        > cursor_key
                    ]
                else:
                    rows = [row for row in rows if row.id < cursor_id]
        if self.maximum is not None:
            rows = rows[: self.maximum]
        if self.fields is not None:
            rows = [
                _Snapshot(
                    row.id,
                    {
                        field: row.to_dict()[field]
                        for field in self.fields
                        if field in row.to_dict()
                    },
                )
                for row in rows
            ]
        return rows


class _Batch:
    def __init__(self, client):
        self.client = client
        self.writes = []

    def set(self, reference, data, merge=False):
        self.writes.append((reference.collection_name, reference.doc_id, data, merge))

    def commit(self, *, timeout, retry):
        assert timeout > 0
        assert retry is None
        self.client.writes.extend(self.writes)


class _Client:
    def __init__(self, data=None):
        self.data = data or {}
        self.queries = []
        self.writes = []

    def collection(self, name):
        return _Query(self, name)

    def batch(self):
        return _Batch(self)


class AdminFirestoreTests(unittest.TestCase):
    def test_quality_filter_is_applied_in_firestore_query(self):
        client = _Client(
            {
                "feedback": {
                    "a" * 32: {
                        "answer_id": "a" * 32,
                        "rating": "Cocog",
                        "source_count": 1,
                        "created_at": 3,
                    },
                    "b" * 32: {
                        "answer_id": "b" * 32,
                        "rating": "Teu cocog",
                        "source_count": 0,
                        "created_at": 2,
                    },
                }
            }
        )

        page = get_answer_quality_page(client, filter_type="Tanpa sumber")

        self.assertEqual([item["answer_id"] for item in page["items"]], ["b" * 32])
        self.assertIn(("source_count", "==", 0), client.queries[0].filters)
        self.assertEqual(client.queries[0].fields, [
            "answer_id", "uid", "name", "name_lower", "chat_id", "created_at",
            "rating", "reason", "question", "answer", "source_count", "fallback",
            "model_used", "sources", "latency_ms",
        ])

    def test_quality_model_and_name_filters_are_query_filters(self):
        client = _Client(
            {
                "feedback": {
                    "a" * 32: {
                        "answer_id": "a" * 32,
                        "name_lower": "asep",
                        "model_used": "gemini-flash",
                        "created_at": 3,
                    },
                    "b" * 32: {
                        "answer_id": "b" * 32,
                        "name_lower": "dewi",
                        "model_used": "gemini-pro",
                        "created_at": 2,
                    },
                }
            }
        )

        model_page = get_answer_quality_page(
            client,
            filter_type="Per model",
            filter_value="gemini-flash",
        )
        self.assertEqual(
            [item["answer_id"] for item in model_page["items"]], ["a" * 32]
        )
        self.assertIn(
            ("model_used", "==", "gemini-flash"),
            client.queries[-1].filters,
        )

        name_page = get_answer_quality_page(
            client,
            filter_type="Per nama",
            filter_value="as",
        )
        self.assertEqual(
            [item["answer_id"] for item in name_page["items"]], ["a" * 32]
        )
        self.assertIn(("name_lower", ">=", "as"), client.queries[-1].filters)

    def test_user_page_marks_duplicate_names_and_activity_flags(self):
        client = _Client(
            {
                "users": {
                    "abcdef0123456789": {
                        "name": "Asep",
                        "name_lower": "asep",
                        "total_questions": 3,
                    },
                    "bcdef0123456789a": {
                        "name": "Asep",
                        "name_lower": "asep",
                        "total_questions": 0,
                    },
                    "cdef0123456789ab": {
                        "name": "Dewi",
                        "name_lower": "dewi",
                        "total_questions": 22,
                        "day_count": 20,
                        "day": datetime.now(ZoneInfo("Asia/Jakarta")).strftime(
                            "%Y%m%d"
                        ),
                    },
                }
            }
        )

        page = get_admin_user_page(client, page_size=2)

        self.assertEqual(len(page["items"]), 2)
        self.assertTrue(all(item["duplicate_name"] for item in page["items"]))
        self.assertTrue(page["items"][1]["no_questions"])
        self.assertIsNotNone(page["next_cursor"])
        next_page = get_admin_user_page(
            client,
            page_size=2,
            start_after=page["next_cursor"],
        )
        self.assertTrue(next_page["items"][0]["question_spike"])
        self.assertEqual(len(next_page["items"][0]["short_id"]), 6)

    def test_review_labels_are_written_and_invalid_labels_rejected(self):
        client = _Client()
        answer_id = "d" * 32

        save_answer_reviews(
            client,
            [{"answer_id": answer_id, "label": "Kaluar kontéks"}],
        )

        self.assertEqual(client.writes[0][0:2], ("reviews", answer_id))
        self.assertEqual(client.writes[0][2]["label"], "Kaluar kontéks")
        with self.assertRaises(ValueError):
            save_answer_reviews(
                client,
                [{"answer_id": answer_id, "label": "invalid"}],
            )


if __name__ == "__main__":
    unittest.main()
