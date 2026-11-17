from collections.abc import Callable
import unittest
from unittest.mock import Mock, patch

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from services.ai_service import (
    CHAT_MODEL_REQUEST_TIMEOUT_SECONDS,
    CHAT_QUERY_EMBEDDING_TIMEOUT_MS,
    EMBEDDING_MAX_BATCH_SIZE,
    EMBEDDING_MAX_BATCH_CHARACTERS,
    EMBEDDING_MAX_REQUESTS_PER_MINUTE,
    EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS,
    EMBEDDING_REQUEST_TIMEOUT_MS,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MAX_QUOTA_RETRIES,
    EMBEDDING_TARGET_BATCH_TOKENS,
    _EmbeddingRateLimiter,
    _QuotaAwareEmbeddings,
    _get_chat_query_embeddings,
    _get_embeddings,
    _is_chat_model_fallback_error,
    _make_embedding_batches,
    _to_langchain_documents,
    answer_question,
    is_embedding_daily_quota_error,
    is_embedding_quota_error,
    transcribe_audio,
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
        http_options = make_client.call_args.kwargs["http_options"]
        retry_options = http_options.retry_options
        self.assertEqual(retry_options.attempts, EMBEDDING_MAX_RETRIES + 1)
        self.assertEqual(http_options.timeout, EMBEDDING_REQUEST_TIMEOUT_MS)
        self.assertIs(make_embeddings.return_value.client, make_client.return_value)

    def test_chat_query_embedding_has_short_timeout_and_no_rate_limiter_wait(self) -> None:
        embedding_client = Mock()
        embedding_client.embed_query.return_value = [1.0, 2.0]
        with (
            patch(
                "services.ai_service.GoogleGenerativeAIEmbeddings",
                return_value=embedding_client,
            ),
            patch("services.ai_service.genai.Client") as make_client,
        ):
            embeddings = _get_chat_query_embeddings("test-api-key")

        http_options = make_client.call_args.kwargs["http_options"]
        self.assertEqual(http_options.timeout, CHAT_QUERY_EMBEDDING_TIMEOUT_MS)
        self.assertEqual(
            http_options.retry_options.attempts,
            EMBEDDING_MAX_RETRIES + 1,
        )
        with patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire") as acquire:
            self.assertEqual(embeddings.embed_query("query"), [1.0, 2.0])
        acquire.assert_not_called()

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

    def test_partial_file_checkpoint_resumes_without_reembedding_saved_chunks(self) -> None:
        chunks = [
            Document(
                page_content=f"chunk {number}",
                metadata={
                    "file_id": "file-1",
                    "modified_time": "2026-11-12T00:00:00Z",
                    "chunk_index": number,
                },
            )
            for number in range(21)
        ]
        embeddings = Mock(spec=Embeddings)
        embeddings.embed_documents.side_effect = lambda texts: [
            [float(len(text)), 1.0] for text in texts
        ]
        source = DriveDocument(
            file_id="file-1",
            name="document.txt",
            text="unused",
            modified_time="2026-11-12T00:00:00Z",
        )
        checkpointed_indices: list[FAISS] = []

        def fail_after_first_checkpoint(index: FAISS) -> None:
            checkpointed_indices.append(index)
            raise RuntimeError("simulated interruption")

        with (
            patch("services.ai_service._to_langchain_documents", return_value=chunks),
            patch("services.ai_service._get_embeddings", return_value=embeddings),
            self.assertRaisesRegex(RuntimeError, "simulated interruption"),
        ):
            update_search_index(
                None,
                [source],
                {"file-1"},
                "unused-api-key",
                on_checkpoint=fail_after_first_checkpoint,
            )

        self.assertEqual(len(checkpointed_indices), 1)
        self.assertEqual(len(checkpointed_indices[0].index_to_docstore_id), 20)
        with (
            patch("services.ai_service._to_langchain_documents", return_value=chunks),
            patch("services.ai_service._get_embeddings", return_value=embeddings),
        ):
            resumed_index = update_search_index(
                checkpointed_indices[0],
                [source],
                {"file-1"},
                "unused-api-key",
            )

        self.assertIsNotNone(resumed_index)
        assert resumed_index is not None
        self.assertEqual(len(resumed_index.index_to_docstore_id), 21)
        self.assertEqual(
            [len(call.args[0]) for call in embeddings.embed_documents.call_args_list],
            [20, 1],
        )

    def test_update_search_index_reports_embedding_batch_progress(self) -> None:
        chunks = [
            Document(
                page_content=f"chunk {number}",
                metadata={
                    "file_id": "file-1",
                    "modified_time": "2026-11-12T00:00:00Z",
                    "chunk_index": number,
                },
            )
            for number in range(21)
        ]
        client = Mock()

        def embed_documents(texts: list[str], batch_size: int) -> list[list[float]]:
            self.assertEqual(batch_size, len(texts))
            return [[1.0, 2.0] for _ in texts]

        client.embed_documents.side_effect = embed_documents
        progress: list[tuple[int, int, bool]] = []
        source = DriveDocument(
            file_id="file-1",
            name="document.txt",
            text="unused",
            modified_time="2026-11-12T00:00:00Z",
        )

        def create_embeddings(
            _api_key: str,
            _on_wait: Callable[[float], None] | None,
            on_progress: Callable[[int, int], None] | None,
        ) -> _QuotaAwareEmbeddings:
            self.assertEqual(_api_key, "unused-api-key")
            self.assertIsNone(_on_wait)
            return _QuotaAwareEmbeddings(client, on_progress=on_progress)

        with (
            patch("services.ai_service._to_langchain_documents", return_value=chunks),
            patch(
                "services.ai_service._get_embeddings",
                side_effect=create_embeddings,
            ),
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
        ):
            update_search_index(
                None,
                [source],
                {"file-1"},
                "unused-api-key",
                on_progress=lambda done, total: progress.append(
                    (done, total)
                ),
            )

        self.assertEqual(
            progress,
            [
                (0, 21),
                (20, 21),
                (20, 21),
                (21, 21),
            ],
        )

    def test_one_file_chunks_are_sent_in_batches_of_at_most_twenty(self) -> None:
        client = Mock()
        client.embed_documents.side_effect = lambda texts, batch_size: [
            [1.0, 2.0] for _ in texts
        ]
        embeddings = _QuotaAwareEmbeddings(client)

        with patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"):
            vectors = embeddings.embed_documents(["x" * 900] * 100)

        self.assertEqual(len(vectors), 100)
        self.assertEqual(client.embed_documents.call_count, 7)
        self.assertEqual(
            [len(call.args[0]) for call in client.embed_documents.call_args_list],
            [16] * 6 + [4],
        )
        self.assertTrue(
            all(
                call.kwargs["batch_size"] == len(call.args[0])
                for call in client.embed_documents.call_args_list
            )
        )
        self.assertTrue(
            all(
                sum(map(len, call.args[0])) <= EMBEDDING_MAX_BATCH_CHARACTERS
                for call in client.embed_documents.call_args_list
            )
        )

    def test_embedding_batches_respect_count_and_token_limits(self) -> None:
        batches = _make_embedding_batches(["x" * 1500] * 41)

        self.assertEqual(EMBEDDING_MAX_BATCH_SIZE, 20)
        self.assertEqual(EMBEDDING_TARGET_BATCH_TOKENS, 22_000)
        self.assertEqual([len(batch) for batch, _ in batches], [10, 10, 10, 10, 1])
        self.assertTrue(
            all(
                tokens <= EMBEDDING_TARGET_BATCH_TOKENS
                and len(batch) <= EMBEDDING_MAX_BATCH_SIZE
                and sum(map(len, batch)) <= EMBEDDING_MAX_BATCH_CHARACTERS
                for batch, tokens in batches
            )
        )

    def test_embedding_batches_split_at_character_limit(self) -> None:
        batches = _make_embedding_batches(["x" * 9000, "y" * 6000, "z"])

        self.assertEqual([len(batch) for batch, _ in batches], [2, 1])
        self.assertEqual(
            [sum(map(len, batch)) for batch, _ in batches],
            [15_000, 1],
        )
        with self.assertRaises(ValueError):
            _make_embedding_batches(["x" * (EMBEDDING_MAX_BATCH_CHARACTERS + 1)])

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

    def test_query_embedding_forwards_rate_limit_wait_callback(self) -> None:
        client = Mock()
        client.embed_query.return_value = [1.0, 2.0]
        embeddings = _QuotaAwareEmbeddings(client)
        on_wait = Mock()

        with patch(
            "services.ai_service._EMBEDDING_RATE_LIMITER.acquire"
        ) as acquire:
            query_vector = embeddings.embed_query_with_wait("query", on_wait)

        self.assertEqual(query_vector, [1.0, 2.0])
        acquire.assert_called_once_with(10, on_wait)
        client.embed_query.assert_called_once_with("query")

    def test_query_embedding_timeout_falls_back_to_local_keyword_search(self) -> None:
        index = FAISS.from_documents(
            [
                Document(
                    page_content="isi uji untuk pertanyaan",
                    metadata={"source": "uji", "file_id": "file-1"},
                )
            ],
            _DeterministicEmbeddings(),
        )
        failing_embedding_client = Mock()
        failing_embedding_client.embed_query.side_effect = TimeoutError(
            "request timed out"
        )
        index.embedding_function = _QuotaAwareEmbeddings(failing_embedding_client)
        prompt = Mock()
        pipeline = Mock()
        prompt.__or__ = Mock(return_value=pipeline)
        pipeline.invoke.return_value = Mock(content="Jawaban dari pencarian lokal.")
        progress_messages: list[str] = []

        query_embeddings = _QuotaAwareEmbeddings(
            failing_embedding_client,
            limit_query_rate=False,
        )
        with (
            patch("services.ai_service._EMBEDDING_RATE_LIMITER.acquire"),
            patch(
                "services.ai_service._get_chat_query_embeddings",
                return_value=query_embeddings,
            ),
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["gemini-flash"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI"),
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            for error, expected_prefix in (
                (
                    TimeoutError("request timed out"),
                    "Catatan: layanan embedding Gemini",
                ),
                (
                    RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded"),
                    "Catatan: kuota embedding Gemini habis",
                ),
            ):
                with self.subTest(error=error):
                    failing_embedding_client.embed_query.side_effect = error
                    answer, _ = answer_question(
                        index,
                        "Apa isi uji?",
                        api_key="test-key",
                        on_progress=progress_messages.append,
                    )
                    self.assertTrue(answer.startswith(expected_prefix))

        self.assertIn("Mencari di indeks lokal...", progress_messages)
        self.assertIn("Menyusun jawaban dengan Gemini...", progress_messages)

    def test_busy_sync_uses_local_keyword_search_without_query_embedding(self) -> None:
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
        pipeline.invoke.return_value = Mock(content="Jawaban dari indeks lokal.")
        progress_messages: list[str] = []

        with (
            patch("services.ai_service._get_chat_query_embeddings") as query_embeddings,
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["gemini-flash"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI"),
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(
                index,
                "Apa isi uji?",
                api_key="test-key",
                on_progress=progress_messages.append,
                force_local_search=True,
            )

        query_embeddings.assert_not_called()
        self.assertTrue(
            answer.startswith(
                "Catatan: sinkronisasi indeks sedang berlangsung, jadi dokumen dicari"
            )
        )
        self.assertLess(
            progress_messages.index("Mencari di indeks lokal..."),
            progress_messages.index("Menyusun jawaban dengan Gemini..."),
        )

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
            [[1.0, 2.0] for _ in range(10)],
            RuntimeError("429 RESOURCE_EXHAUSTED: retry in 30s"),
            [[2.0, 3.0] for _ in range(10)],
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
        self.assertEqual(client.embed_documents.call_count, 4)
        self.assertEqual(len(client.embed_documents.call_args_list[0].args[0]), 10)
        self.assertEqual(len(client.embed_documents.call_args_list[1].args[0]), 10)
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
        progress_messages: list[str] = []

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["model-ramai", "model-cadangan"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ) as make_prompt,
            patch("services.ai_service.ChatGoogleGenerativeAI") as make_model,
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(
                index,
                "Apa isi uji?",
                api_key="test-key",
                on_progress=progress_messages.append,
            )

        system_prompt = make_prompt.call_args.args[0][0][1]
        self.assertIn("Selalu pertahankan struktur baris asli", system_prompt)
        self.assertIn("dua spasi di akhir setiap baris", system_prompt)
        self.assertIn("judul atau kata kunci singkat", system_prompt)
        self.assertEqual(pipeline.invoke.call_count, 2)
        self.assertEqual(make_model.call_count, 2)
        self.assertTrue(answer.startswith("Jawaban dari model cadangan"))
        self.assertIn(
            "Menghubungi Gemini untuk embedding pertanyaan...",
            progress_messages,
        )
        self.assertIn("Memeriksa model Gemini yang tersedia...", progress_messages)
        self.assertIn("Menunggu jawaban dari model-ramai...", progress_messages)
        self.assertTrue(
            _is_chat_model_fallback_error(
                RuntimeError("503 UNAVAILABLE: high demand")
            )
        )

    def test_chat_request_timeout_uses_next_model(self) -> None:
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
            TimeoutError("request timed out"),
            Mock(content="Jawaban dari model cadangan"),
        ]

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["model-timeout", "model-cadangan"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI") as make_model,
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(index, "Apa isi uji?", api_key="test-key")

        self.assertTrue(answer.startswith("Jawaban dari model cadangan"))
        self.assertEqual(pipeline.invoke.call_count, 2)
        self.assertEqual(make_model.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["request_timeout"] == CHAT_MODEL_REQUEST_TIMEOUT_SECONDS
                for call in make_model.call_args_list
            )
        )

    def test_answer_question_streams_response_tokens(self) -> None:
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
        pipeline.stream.return_value = iter(
            [Mock(content="Jawaban "), Mock(content="bertahap.")]
        )
        tokens: list[str | None] = []

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["gemini-flash"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI"),
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(
                index,
                "Apa isi uji?",
                api_key="test-key",
                on_token=tokens.append,
            )

        self.assertTrue(answer.startswith("Jawaban bertahap."))
        self.assertEqual(tokens, ["Jawaban ", "bertahap."])
        pipeline.stream.assert_called_once()

    def test_streamed_partial_answer_is_cleared_before_model_fallback(self) -> None:
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

        def partial_answer():
            yield Mock(content="Jawaban parsial")
            raise RuntimeError("503 UNAVAILABLE: temporarily unavailable")

        pipeline.stream.side_effect = [
            partial_answer(),
            iter([Mock(content="Jawaban lengkap")]),
        ]
        tokens: list[str | None] = []

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["model-1", "model-2"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI"),
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, _ = answer_question(
                index,
                "Apa isi uji?",
                api_key="test-key",
                on_token=tokens.append,
            )

        self.assertTrue(answer.startswith("Jawaban lengkap"))
        self.assertEqual(
            tokens,
            ["Jawaban parsial", None, "Jawaban lengkap"],
        )

    def test_chat_rate_limit_is_reported_as_quota_and_tries_fallback(self) -> None:
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
        pipeline.invoke.side_effect = RuntimeError(
            "429 RESOURCE_EXHAUSTED: quota exceeded"
        )
        progress_messages: list[str] = []

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["model-1", "model-2"],
            ),
            patch(
                "services.ai_service.ChatPromptTemplate.from_messages",
                return_value=prompt,
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI"),
            patch("services.ai_service.load_index_manifest", return_value=None),
            self.assertRaisesRegex(ValueError, "limit/kuota"),
        ):
            answer_question(
                index,
                "Apa isi uji?",
                api_key="test-key",
                on_progress=progress_messages.append,
            )

        self.assertEqual(pipeline.invoke.call_count, 2)
        self.assertTrue(
            any("membatasi permintaan" in message for message in progress_messages)
        )

    def test_transcribe_audio_sends_audio_to_a_gemini_model(self) -> None:
        model = Mock()
        model.invoke.return_value = Mock(content="Carikan informasi tentang zakat.")

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["gemini-audio-model", "gemma-text-model"],
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI", return_value=model),
        ):
            transcript = transcribe_audio(
                b"audio bytes",
                "audio/wav",
                api_key="test-key",
            )

        self.assertEqual(transcript, "Carikan informasi tentang zakat.")
        make_model_call = model.invoke.call_args.args[0]
        audio_part = make_model_call[0].content[1]
        self.assertEqual(audio_part["mime_type"], "audio/wav")
        self.assertEqual(audio_part["source_type"], "base64")

    def test_answer_question_sends_image_even_without_a_local_index(self) -> None:
        model = Mock()
        model.invoke.return_value = Mock(content="Gambar menampilkan sebuah masjid.")
        media_parts = [
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": "image/jpeg",
                "data": "aW1hZ2UtZGF0YQ==",
            }
        ]

        with (
            patch(
                "services.ai_service.get_available_chat_models",
                return_value=["gemini-vision-model"],
            ),
            patch("services.ai_service.ChatGoogleGenerativeAI", return_value=model),
            patch("services.ai_service.load_index_manifest", return_value=None),
        ):
            answer, sources = answer_question(
                None,
                "Jelaskan gambar ini.",
                api_key="test-key",
                media_parts=media_parts,
            )

        self.assertEqual(answer, "Gambar menampilkan sebuah masjid.")
        self.assertEqual(sources, [])
        human_message = model.invoke.call_args.args[0][1]
        self.assertEqual(human_message.content[1], media_parts[0])
