import unittest
from unittest.mock import Mock, patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from services.ai_service import (
    EMBEDDING_MAX_BATCH_SIZE,
    EMBEDDING_MAX_REQUESTS_PER_MINUTE,
    EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS,
    EMBEDDING_MAX_QUOTA_RETRIES,
    EMBEDDING_TARGET_BATCH_TOKENS,
    _EmbeddingRateLimiter,
    _QuotaAwareEmbeddings,
    _get_embeddings,
    _is_chat_model_fallback_error,
    _make_embedding_batches,
    _to_langchain_documents,
    answer_question,
    is_embedding_daily_quota_error,
    is_embedding_quota_error,
    update_search_index,
)
from services.gdrive_service import DriveDocument


class _DeterministicEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(sum(map(ord, text)) % 997)] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 997)]


class UpdateSearchIndexTests(unittest.TestCase):
    def test_embedding_client_does_not_retry_api_requests(self) -> None:
        with (
            patch("services.ai_service.GoogleGenerativeAIEmbeddings") as make_embeddings,
            patch("services.ai_service.genai.Client") as make_client,
        ):
            _get_embeddings("test-api-key")

        make_client.assert_called_once()
        retry_options = make_client.call_args.kwargs["http_options"].retry_options
        self.assertEqual(retry_options.attempts, 1)
        self.assertIs(make_embeddings.return_value.client, make_client.return_value)

    def test_replacing_file_chunks_keeps_the_new_chunks(self) -> None:
        embeddings = _DeterministicEmbeddings()
        index = FAISS.from_documents(
            [
                Document(
                    page_content="old contents",
                    metadata={"file_id": "file-1"},
                )
            ],
            embeddings,
        )
        replacement = DriveDocument(
            file_id="file-1",
            name="document.txt",
            text="replacement contents",
            modified_time="2026-11-12T00:00:00Z",
        )

        with patch("services.ai_service._get_embeddings", return_value=embeddings):
            updated_index = update_search_index(
                index,
                [replacement],
                {"file-1"},
                "unused-api-key",
            )

        self.assertIsNotNone(updated_index)
        assert updated_index is not None
        documents = [
            updated_index.docstore.search(document_id)
            for document_id in updated_index.index_to_docstore_id.values()
        ]
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "replacement contents")
        self.assertEqual(documents[0].metadata["file_id"], "file-1")

    def test_one_file_chunks_are_sent_in_batches_of_at_most_twenty(self) -> None:
        client = Mock()
        client.embed_documents.side_effect = lambda texts, batch_size: [
            [1.0, 2.0] for _ in texts
        ]
        embeddings = _QuotaAwareEmbeddings(client)

        with patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"):
            vectors = embeddings.embed_documents(["x" * 900] * 100)

        self.assertEqual(len(vectors), 100)
        self.assertEqual(client.embed_documents.call_count, 5)
        self.assertEqual(
            [len(call.args[0]) for call in client.embed_documents.call_args_list],
            [20] * 5,
        )
        self.assertTrue(
            all(
                call.kwargs["batch_size"] == 20
                for call in client.embed_documents.call_args_list
            )
        )

    def test_embedding_batches_respect_count_and_token_limits(self) -> None:
        batches = _make_embedding_batches(["x" * 1500] * 41)

        self.assertEqual(EMBEDDING_MAX_BATCH_SIZE, 20)
        self.assertEqual(EMBEDDING_TARGET_BATCH_TOKENS, 22_000)
        self.assertEqual([len(batch) for batch, _ in batches], [20, 20, 1])
        self.assertTrue(
            all(tokens <= EMBEDDING_TARGET_BATCH_TOKENS for _, tokens in batches)
        )

    def test_embedding_limiter_allows_thirty_requests_then_waits_for_window(self) -> None:
        limiter = _EmbeddingRateLimiter()
        now = [100.0]
        waits: list[float] = []

        def fake_sleep(seconds: float) -> None:
            waits.append(seconds)
            now[0] += seconds

        with (
            patch("services.ai_service.time.monotonic", side_effect=lambda: now[0]),
            patch("services.ai_service.time.sleep", side_effect=fake_sleep),
        ):
            for _ in range(EMBEDDING_MAX_REQUESTS_PER_MINUTE + 1):
                limiter.acquire(1)

        self.assertEqual(waits, [60.0])

    def test_embedding_limiter_waits_only_when_token_window_is_full(self) -> None:
        limiter = _EmbeddingRateLimiter()
        now = [100.0]
        waits: list[float] = []

        def fake_sleep(seconds: float) -> None:
            waits.append(seconds)
            now[0] += seconds

        with (
            patch("services.ai_service.time.monotonic", side_effect=lambda: now[0]),
            patch("services.ai_service.time.sleep", side_effect=fake_sleep),
        ):
            limiter.acquire(EMBEDDING_TARGET_BATCH_TOKENS)
            limiter.acquire(1)

        self.assertEqual(waits, [60.0])

    def test_long_document_chunks_use_requested_size_and_overlap(self) -> None:
        chunks = _to_langchain_documents(
            [
                DriveDocument(
                    file_id="file-1",
                    name="long.txt",
                    text="x" * 4000,
                    modified_time="2026-11-12T00:00:00Z",
                )
            ]
        )

        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(chunk.page_content) <= 1500 for chunk in chunks))
        self.assertTrue(
            all(
                previous.page_content[-200:] == following.page_content[:200]
                for previous, following in zip(chunks, chunks[1:])
            )
        )

    def test_non_quota_embedding_error_is_not_retried(self) -> None:
        client = Mock()
        error = RuntimeError("400 INVALID_ARGUMENT: malformed input")
        client.embed_documents.side_effect = error
        embeddings = _QuotaAwareEmbeddings(client)

        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            self.assertRaises(RuntimeError),
        ):
            embeddings.embed_documents(["short text"])

        self.assertEqual(client.embed_documents.call_count, 1)
        self.assertFalse(is_embedding_quota_error(error))

    def test_quota_retry_retries_only_the_failed_batch(self) -> None:
        client = Mock()
        client.embed_documents.side_effect = [
            [[1.0, 2.0] for _ in range(20)],
            RuntimeError("429 RESOURCE_EXHAUSTED: retry in 30s"),
            [[3.0, 4.0] for _ in range(5)],
        ]
        on_wait = Mock()
        embeddings = _QuotaAwareEmbeddings(client, on_wait=on_wait)

        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            patch("services.ai_service.time.sleep") as sleep,
        ):
            vectors = embeddings.embed_documents(["x" * 1500] * 25)

        self.assertEqual(len(vectors), 25)
        self.assertEqual(client.embed_documents.call_count, 3)
        self.assertEqual(len(client.embed_documents.call_args_list[0].args[0]), 20)
        self.assertEqual(len(client.embed_documents.call_args_list[1].args[0]), 5)
        self.assertEqual(
            client.embed_documents.call_args_list[1].args[0],
            client.embed_documents.call_args_list[2].args[0],
        )
        on_wait.assert_called_once_with(30.0)
        sleep.assert_called_once_with(30.0)

    def test_daily_quota_error_stops_without_retrying(self) -> None:
        client = Mock()
        error = RuntimeError(
            "429 RESOURCE_EXHAUSTED: quota exceeded for requests per day"
        )
        client.embed_documents.side_effect = error
        embeddings = _QuotaAwareEmbeddings(client)

        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            patch("services.ai_service.time.sleep") as sleep,
            self.assertRaises(RuntimeError),
        ):
            embeddings.embed_documents(["short text"])

        self.assertEqual(client.embed_documents.call_count, 1)
        sleep.assert_not_called()
        self.assertTrue(is_embedding_quota_error(error))
        self.assertTrue(is_embedding_daily_quota_error(error))

    def test_long_quota_wait_does_not_retry_automatically(self) -> None:
        client = Mock()
        client.embed_documents.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED: retry in 1h"
        )
        embeddings = _QuotaAwareEmbeddings(client)

        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            patch("services.ai_service.time.sleep") as sleep,
            self.assertRaises(RuntimeError),
        ):
            embeddings.embed_documents(["short text"])

        self.assertEqual(client.embed_documents.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS, 300.0)

    def test_quota_retry_attempts_are_bounded(self) -> None:
        client = Mock()
        client.embed_documents.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED: quota exceeded"
        )
        on_wait = Mock()
        embeddings = _QuotaAwareEmbeddings(client, on_wait=on_wait)

        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            patch("services.ai_service.time.sleep") as sleep,
            self.assertRaises(RuntimeError),
        ):
            embeddings.embed_documents(["short text"])

        self.assertEqual(
            client.embed_documents.call_count,
            EMBEDDING_MAX_QUOTA_RETRIES + 1,
        )
        self.assertEqual(
            [call.args for call in sleep.call_args_list],
            [(60.0,), (60.0,)],
        )
        self.assertEqual(on_wait.call_count, EMBEDDING_MAX_QUOTA_RETRIES)

    def test_chat_503_high_demand_uses_next_model(self) -> None:
        index = FAISS.from_documents(
            [
                Document(
                    page_content="isi uji untuk pertanyaan",
                    metadata={"source": "uji", "file_id": "file-1"},
                )
            ],
            _DeterministicEmbeddings(),
        )
        prompt = Mock()
        pipeline = Mock()
        prompt.__or__ = Mock(return_value=pipeline)
        pipeline.invoke.side_effect = [
            RuntimeError(
                "503 UNAVAILABLE: This model is currently experiencing high demand."
            ),
            Mock(content="Jawaban dari model cadangan"),
        ]

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["model-ramai", "model-cadangan"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI") as make_model,
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(index, "Apa isi uji?", api_key="test-key")

        self.assertEqual(pipeline.invoke.call_count, 2)
        self.assertEqual(make_model.call_count, 2)
        self.assertTrue(answer.startswith("Jawaban dari model cadangan"))
        self.assertTrue(
            _is_chat_model_fallback_error(
                RuntimeError("503 UNAVAILABLE: high demand")
            )
        )
