"""Build a semantic search index and answer questions with Gemini."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import faiss
import numpy as np
from collections import Counter
from collections.abc import Sequence
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Callable

from google import genai
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.gdrive_service import DriveDocument

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "faiss_index"
MANIFEST_FILENAME = "drive_manifest.json"
EMBEDDING_TOKENS_PER_MINUTE = 27_000
EMBEDDING_WINDOW_SECONDS = 60.0
EMBEDDING_TARGET_BATCH_TOKENS = 22_000
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 30
EMBEDDING_MAX_BATCH_SIZE = 20
EMBEDDING_MAX_BATCH_CHARACTERS = 15_000
EMBEDDING_MAX_QUOTA_RETRIES = 2
EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS = EMBEDDING_WINDOW_SECONDS
EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS = 300.0
EMBEDDING_REQUEST_TIMEOUT_MS = 30_000
CHAT_QUERY_EMBEDDING_TIMEOUT_MS = 5_000
CHAT_MODEL_REQUEST_TIMEOUT_SECONDS = 180
CHAT_MAX_OUTPUT_TOKENS = 8192
CHAT_MAX_CONTINUATIONS = 5
CONTINUE_PROMPT = (
    "Lanjutkan jawaban di atas tepat dari titik terakhir yang terpotong. "
    "Jangan mengulang bagian yang sudah ditulis dan jangan membuat pembuka baru."
)
EMBEDDING_MAX_RETRIES = 0
MODEL_LIST_CACHE_SECONDS = 300.0
FALLBACK_CHAT_MODELS = (
    "gemini-omni-1.2-flash",
    "gemini-omni-1.1-flash",
    "gemini-3.8-flash",
    "gemma-4-26b-it",
)
PREFERRED_CHAT_MODEL_ORDER = {
    model: position for position, model in enumerate(FALLBACK_CHAT_MODELS)
}
_CHAT_MODEL_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_CHAT_MODEL_CACHE_LOCK = Lock()


class _EmbeddingRateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: deque[tuple[float, int]] = deque()

    def acquire(
        self,
        estimated_tokens: int,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        if estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
            raise ValueError(
                "Մեկ embedding հարցման գնահատված token-ների քանակը գերազանցում է "
                f"{EMBEDDING_TARGET_BATCH_TOKENS}-ի անվտանգ սահմանը։"
            )
        estimated_tokens = max(estimated_tokens, 1)
        while True:
            with self._lock:
                now = time.monotonic()
                while (
                    self._requests
                    and now - self._requests[0][0] >= EMBEDDING_WINDOW_SECONDS
                ):
                    self._requests.popleft()

                tokens_in_window = sum(tokens for _, tokens in self._requests)
                requests_wait = (
                    max(
                        0.0,
                        self._requests[0][0] + EMBEDDING_WINDOW_SECONDS - now,
                    )
                    if len(self._requests) >= EMBEDDING_MAX_REQUESTS_PER_MINUTE
                    else 0.0
                )
                tokens_wait = 0.0
                if tokens_in_window + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
                    remaining_tokens = tokens_in_window
                    for request_time, request_tokens in self._requests:
                        remaining_tokens -= request_tokens
                        if (
                            remaining_tokens + estimated_tokens
                            <= EMBEDDING_TARGET_BATCH_TOKENS
                        ):
                            tokens_wait = max(
                                0.0,
                                request_time + EMBEDDING_WINDOW_SECONDS - now,
                            )
                            break

                if requests_wait == 0 and tokens_wait == 0:
                    self._requests.append((now, estimated_tokens))
                    return

                wait_seconds = max(0.1, requests_wait, tokens_wait)

            if on_wait:
                on_wait(wait_seconds)
            time.sleep(wait_seconds)


_EMBEDDING_RATE_LIMITER = _EmbeddingRateLimiter()


class _QuotaAwareEmbeddings(Embeddings):
    def __init__(
        self,
        client: GoogleGenerativeAIEmbeddings,
        on_wait: Callable[[float], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        limit_query_rate: bool = True,
    ) -> None:
        self._client = client
        self._on_wait = on_wait
        self._on_progress = on_progress
        self._limit_query_rate = limit_query_rate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        completed = 0
        for batch, estimated_tokens in _make_embedding_batches(texts):
            retries = 0
            while True:
                if self._on_progress:
                    self._on_progress(completed, len(texts))
                _EMBEDDING_RATE_LIMITER.acquire(estimated_tokens, self._on_wait)
                try:
                    batch_vectors = self._client.embed_documents(
                        batch,
                        batch_size=len(batch),
                    )
                except Exception as error:
                    if (
                        not is_embedding_quota_error(error)
                        or is_embedding_daily_quota_error(error)
                        or retries >= EMBEDDING_MAX_QUOTA_RETRIES
                    ):
                        raise
                    retry_delay = (
                        get_quota_retry_delay_seconds(error)
                        or EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS
                    )
                    if retry_delay > EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS:
                        raise
                    if self._on_wait:
                        self._on_wait(retry_delay)
                    time.sleep(retry_delay)
                    retries += 1
                    continue
                vectors.extend(batch_vectors)
                completed += len(batch)
                if self._on_progress:
                    self._on_progress(completed, len(texts))
                break
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_query_with_wait(text, self._on_wait)

    def embed_query_with_wait(
        self,
        text: str,
        on_wait: Callable[[float], None] | None = None,
    ) -> list[float]:
        if self._limit_query_rate:
            _EMBEDDING_RATE_LIMITER.acquire(
                (len(text.encode("utf-8")) + 3) // 4 + 8,
                self._on_wait if on_wait is None else on_wait,
            )
        return self._client.embed_query(text)


def is_embedding_quota_error(error: Exception) -> bool:
    """Return whether an exception indicates a Gemini quota or rate limit."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited:
            break
        visited.add(id(current))
        message = str(current).casefold()
        status = str(
            getattr(current, "status", None)
            or getattr(current, "status_code", None)
            or getattr(current, "code", None)
            or "",
        ).casefold()
        if (
            "429" in status
            or "resource_exhausted" in status
            or "429" in message
            or "resource_exhausted" in message
            or "quota exceeded" in message
            or "quota_exceeded" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_transient_embedding_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = str(
            getattr(current, "status", None)
            or getattr(current, "status_code", None)
            or getattr(current, "code", None)
            or ""
        )
        class_name = type(current).__name__.casefold()
        message = str(current).casefold()
        if (
            status in {"408", "425", "429", "500", "502", "503", "504"}
            or "timeout" in class_name
            or "connection" in class_name
            or "network" in class_name
            or any(
                marker in message
                for marker in (
                    "timed out",
                    "timeout",
                    "connection error",
                    "connection reset",
                    "temporarily unavailable",
                    "service unavailable",
                    "resource_exhausted",
                    "quota exceeded",
                    "too many requests",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                )
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_embedding_daily_quota_error(error: Exception) -> bool:
    """Return whether a Gemini quota error is explicitly limited per day."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited:
            break
        visited.add(id(current))
        message = str(current).casefold()
        compact_message = re.sub(r"[^a-z0-9]", "", message)
        if is_embedding_quota_error(current) and (
            "per day" in message
            or "per-day" in message
            or "daily" in message
            or "perday" in compact_message
            or "requestsperday" in compact_message
            or "tokensperday" in compact_message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def get_quota_retry_delay_seconds(error: Exception) -> float | None:
    """Extract retry duration from Gemini's RetryInfo or human-readable message."""
    current: BaseException | None = error
    while current is not None:
        message = str(current)
        retry_delay = re.search(
            r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
            message,
            re.IGNORECASE,
        )
        if retry_delay:
            return float(retry_delay.group(1))

        human_delay = re.search(
            r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
            message,
            re.IGNORECASE,
        )
        if human_delay and any(human_delay.groups()):
            hours, minutes, seconds = human_delay.groups()
            return (
                int(hours or 0) * 3600
                + int(minutes or 0) * 60
                + float(seconds or 0)
            )
        current = current.__cause__
    return None


def _make_embedding_batches(texts: list[str]) -> list[tuple[list[str], int]]:
    batches: list[tuple[list[str], int]] = []
    current_batch: list[str] = []
    current_tokens = 0
    current_characters = 0

    for text in texts:
        estimated_tokens = max(1, len(text) // 3)
        character_count = len(text)
        if current_batch and (
            len(current_batch) >= EMBEDDING_MAX_BATCH_SIZE
            or current_characters + character_count > EMBEDDING_MAX_BATCH_CHARACTERS
            or current_tokens + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS
        ):
            batches.append((current_batch, current_tokens))
            current_batch = []
            current_tokens = 0
            current_characters = 0
        if character_count > EMBEDDING_MAX_BATCH_CHARACTERS:
            raise ValueError(
                "Մեկ embedding chunk-ի նիշերի քանակը գերազանցում է "
                f"{EMBEDDING_MAX_BATCH_CHARACTERS}-ի սահմանը։"
            )
        if estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
            raise ValueError(
                "Մեկ embedding chunk-ի գնահատված token-ների քանակը գերազանցում է "
                f"{EMBEDDING_TARGET_BATCH_TOKENS}-ի անվտանգ սահմանը։"
            )

        current_batch.append(text)
        current_tokens += estimated_tokens
        current_characters += character_count

    if current_batch:
        batches.append((current_batch, current_tokens))
    return batches


def _check_api_key(api_key: str | None = None) -> None:
    if not (api_key or os.getenv("GOOGLE_API_KEY")):
        raise ValueError("GOOGLE_API_KEY belum diatur.")


def _get_embeddings(
    api_key: str | None = None,
    on_wait: Callable[[float], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Embeddings:
    client = _create_embedding_client(api_key, EMBEDDING_REQUEST_TIMEOUT_MS)
    return _QuotaAwareEmbeddings(
        client,
        on_wait=on_wait,
        on_progress=on_progress,
    )


def _create_embedding_client(
    api_key: str | None,
    timeout_ms: int,
) -> GoogleGenerativeAIEmbeddings:
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    client = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=resolved_api_key,
    )
    client.client = genai.Client(
        api_key=resolved_api_key,
        http_options=genai.types.HttpOptions(
            timeout=timeout_ms,
            retry_options=genai.types.HttpRetryOptions(
                attempts=EMBEDDING_MAX_RETRIES + 1,
            ),
        ),
    )
    return client


def _get_chat_query_embeddings(api_key: str | None = None) -> _QuotaAwareEmbeddings:
    client = _create_embedding_client(api_key, CHAT_QUERY_EMBEDDING_TIMEOUT_MS)
    return _QuotaAwareEmbeddings(client, limit_query_rate=False)


def _to_langchain_documents(
    source_documents: Sequence[DriveDocument],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks: list[Document] = []
    for source in source_documents:
        if not source.text.strip():
            continue
        base_metadata = {
            "source": source.name,
            "file_id": source.file_id,
            "modified_time": source.modified_time,
            "web_view_link": source.web_view_link,
            "webViewLink": source.web_view_link,
            "download_url": source.download_url,
        }
        sections = source.page_sections or (("", source.text),)
        for page_label, section_text in sections:
            if not section_text.strip():
                continue
            metadata = dict(base_metadata)
            if page_label:
                metadata["page_label"] = page_label
            else:
                metadata["section_type"] = "section"
            chunks.extend(
                splitter.split_documents(
                    [Document(page_content=section_text, metadata=metadata)]
                )
            )

    next_chunk_index: Counter[str] = Counter()
    for chunk in chunks:
        file_id = str(chunk.metadata.get("file_id", ""))
        chunk_index = next_chunk_index[file_id]
        chunk.metadata["chunk_index"] = chunk_index
        if not chunk.metadata.get("page_label"):
            chunk.metadata["section_index"] = chunk_index + 1
        next_chunk_index[file_id] += 1
    return chunks


def create_search_index(
    source_documents: Sequence[DriveDocument],
    api_key: str | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> FAISS:
    """Create a multilingual semantic index from Drive documents."""
    if not source_documents:
        raise ValueError(
            "Tidak ada teks DOCX atau PDF yang dapat diindeks di folder Google Drive tersebut."
        )
    _check_api_key(api_key)

    documents = _to_langchain_documents(source_documents)
    return FAISS.from_documents(documents, _get_embeddings(api_key, on_wait))


def load_search_index(
    api_key: str,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> FAISS | None:
    """Load a previously persisted FAISS index, if one is present."""
    if not (index_directory / "index.faiss").is_file():
        return None
    return FAISS.load_local(
        str(index_directory),
        _get_embeddings(api_key),
        allow_dangerous_deserialization=True,
    )


def update_search_index(
    index: FAISS | None,
    source_documents: Sequence[DriveDocument],
    replaced_file_ids: set[str],
    api_key: str,
    on_wait: Callable[[float], None] | None = None,
    on_checkpoint: Callable[[FAISS], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FAISS | None:
    """Replace changed file chunks and checkpoint each successfully embedded batch."""
    chunks = _to_langchain_documents(source_documents)
    if not chunks and index is None:
        return None

    chunks_by_file: dict[str, dict[int, Document]] = {}
    for chunk in chunks:
        file_id = str(chunk.metadata.get("file_id", ""))
        chunk_index = int(chunk.metadata.get("chunk_index", 0))
        chunks_by_file.setdefault(file_id, {})[chunk_index] = chunk

    existing_chunks: set[tuple[str, str, int]] = set()
    if index is not None:
        stale_ids: list[str] = []
        for document_id in index.index_to_docstore_id.values():
            stored_document = index.docstore.search(document_id)
            if not isinstance(stored_document, Document):
                continue
            file_id = str(stored_document.metadata.get("file_id", ""))
            if file_id not in replaced_file_ids:
                continue
            desired_chunks = chunks_by_file.get(file_id)
            chunk_index = int(stored_document.metadata.get("chunk_index", -1))
            desired_chunk = desired_chunks.get(chunk_index) if desired_chunks else None
            if (
                desired_chunk is not None
                and stored_document.metadata.get("modified_time")
                == desired_chunk.metadata.get("modified_time")
                and stored_document.page_content == desired_chunk.page_content
            ):
                existing_chunks.add(
                    (
                        file_id,
                        str(stored_document.metadata.get("modified_time", "")),
                        chunk_index,
                    )
                )
            else:
                stale_ids.append(document_id)
        if stale_ids:
            index.delete(ids=stale_ids)

    pending_chunks = [
        chunk
        for chunk in chunks
        if (
            str(chunk.metadata.get("file_id", "")),
            str(chunk.metadata.get("modified_time", "")),
            int(chunk.metadata.get("chunk_index", 0)),
        )
        not in existing_chunks
    ]
    if pending_chunks:
        total_pending = len(pending_chunks)
        completed = 0
        batch_offset = 0
        batch_total = 0

        def report_embedding_progress(
            batch_done: int,
            current_batch_total: int,
        ) -> None:
            if current_batch_total != batch_total:
                raise RuntimeError("Embedding progress does not match its batch.")
            if on_progress:
                on_progress(
                    batch_offset + batch_done,
                    total_pending,
                )

        embeddings = _get_embeddings(
            api_key,
            on_wait,
            report_embedding_progress,
        )
        for batch_info in _make_embedding_batches(
            [chunk.page_content for chunk in pending_chunks]
        ):
            batch = batch_info[0]
            batch_chunks = pending_chunks[: len(batch)]
            del pending_chunks[: len(batch)]
            batch_offset = completed
            batch_total = len(batch_chunks)
            if index is None:
                index = FAISS.from_documents(batch_chunks, embeddings)
            else:
                index.add_documents(batch_chunks, embedding=embeddings)
            completed += len(batch_chunks)
            if on_checkpoint:
                on_checkpoint(index)
    return index


def get_index_file_manifest(index: FAISS | None) -> dict[str, str]:
    """Recover file IDs and modification times from a legacy local index."""
    if index is None:
        return {}

    manifest: dict[str, str] = {}
    for document_id in index.index_to_docstore_id.values():
        stored_document = index.docstore.search(document_id)
        if isinstance(stored_document, Document):
            file_id = stored_document.metadata.get("file_id")
            if file_id:
                manifest[str(file_id)] = str(
                    stored_document.metadata.get("modified_time", "")
                )
    return manifest


def load_index_manifest(
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> dict[str, object] | None:
    """Load the local Drive inventory associated with the FAISS index."""
    manifest_path = index_directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return None
    return manifest


def save_index_manifest(
    folder_id: str,
    files: dict[str, str],
    has_index: bool,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
    sources: dict[str, dict[str, str]] | None = None,
) -> None:
    """Persist Drive modification times next to the local FAISS index."""
    index_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = index_directory / MANIFEST_FILENAME
    temporary_path = manifest_path.with_suffix(".tmp")
    previous_manifest = load_index_manifest(index_directory) or {}
    saved_sources = sources if sources is not None else previous_manifest.get("sources", {})
    temporary_path.write_text(
        json.dumps(
            {
                "folder_id": folder_id,
                "files": files,
                "has_index": has_index,
                "sources": saved_sources,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _extract_stream_chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    return ""


def _keyword_search(index: FAISS, question: str, k: int = 5) -> list[Document]:
    """Rank indexed chunks locally when Gemini query embeddings are unavailable."""
    query_terms = {
        term
        for term in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(term) > 2
    }
    if not query_terms:
        return []

    indexed_documents: list[tuple[Document, Counter[str]]] = []
    document_frequencies: Counter[str] = Counter()
    for document_id in index.index_to_docstore_id.values():
        document = index.docstore.search(document_id)
        if not isinstance(document, Document):
            continue
        counts = Counter(
            term
            for term in re.findall(
                r"\w+", document.page_content.casefold(), flags=re.UNICODE
            )
            if len(term) > 2
        )
        indexed_documents.append((document, counts))
        document_frequencies.update(query_terms & counts.keys())

    document_count = len(indexed_documents)
    ranked_documents = []
    for position, (document, counts) in enumerate(indexed_documents):
        score = sum(
            (1 + math.log(counts[term]))
            * (1 + math.log((document_count + 1) / (document_frequencies[term] + 1)))
            for term in query_terms
            if counts[term]
        )
        if score:
            ranked_documents.append((score, position, document))

    ranked_documents.sort(key=lambda result: (-result[0], result[1]))
    return [document for _, _, document in ranked_documents[:k]]


_FULL_CONTEXT_RE = re.compile(
    r"\b(?:lengkap|sampai\s+selesai|seluruh(?:nya)?|utuh|semua(?:\s+bait)?|full|bacakan|tuliskan|tampilkan|bait|nadhom|nadzom|syair)\b",
    re.IGNORECASE,
)
FULL_CONTEXT_CHUNK_COUNT = 25


def is_full_context_request(question: str) -> bool:
    return bool(_FULL_CONTEXT_RE.search(question))


def _full_context_window(
    index: FAISS,
    matches: list[Document],
    window_size: int = FULL_CONTEXT_CHUNK_COUNT,
    anchor_vector_position: int | None = None,
) -> list[Document]:
    """Expand the strongest hit with sequential chunks from the same source."""
    if not matches:
        return []
    file_id = str(matches[0].metadata.get("file_id", ""))
    if not file_id:
        return matches

    source_chunks: list[tuple[int, int, Document]] = []
    fallback_index = 0
    for vector_position, document_id in sorted(index.index_to_docstore_id.items()):
        document = index.docstore.search(document_id)
        if not isinstance(document, Document):
            continue
        if str(document.metadata.get("file_id", "")) != file_id:
            continue
        chunk_index = document.metadata.get("chunk_index")
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index = fallback_index
        source_chunks.append((chunk_index, vector_position, document))
        fallback_index += 1

    if not source_chunks:
        return matches
    source_chunks.sort(key=lambda item: (item[0], item[1]))
    if anchor_vector_position is not None:
        best_position = next(
            (
                position
                for position, (_, vector_position, _) in enumerate(source_chunks)
                if vector_position == anchor_vector_position
            ),
            None,
        )
    else:
        best_position = next(
            (
                position
                for position, (_, _, document) in enumerate(source_chunks)
                if document is matches[0]
            ),
            None,
        )
    if best_position is None:
        best_chunk_index = matches[0].metadata.get("chunk_index")
        try:
            best_position = next(
                position
                for position, (chunk_index, _, _) in enumerate(source_chunks)
                if chunk_index == int(best_chunk_index)
            )
        except (StopIteration, TypeError, ValueError):
            matching_positions = [
                position
                for position, (_, _, document) in enumerate(source_chunks)
                if document.page_content == matches[0].page_content
            ]
            best_position = matching_positions[-1] if matching_positions else 0

    count = min(window_size, len(source_chunks))
    start = max(0, best_position - 2)
    start = min(start, len(source_chunks) - count)
    return [document for _, _, document in source_chunks[start : start + count]]


def _similarity_search_with_position(
    index: FAISS,
    query_vector: list[float],
    k: int,
) -> tuple[list[Document], int | None]:
    """Search FAISS while retaining the exact vector row for the best hit."""
    query_matrix = np.asarray([query_vector], dtype=np.float32)
    if index._normalize_L2:
        faiss.normalize_L2(query_matrix)
    _, vector_ids = index.index.search(query_matrix, k)
    matches: list[Document] = []
    best_position: int | None = None
    for vector_id in vector_ids[0]:
        vector_position = int(vector_id)
        if vector_position < 0:
            continue
        document_id = index.index_to_docstore_id[vector_position]
        document = index.docstore.search(document_id)
        if isinstance(document, Document):
            matches.append(document)
            if best_position is None:
                best_position = vector_position
    return matches, best_position


_SOURCE_REQUEST_RE = re.compile(
    r"\b(?:link|tautan|download|unduh|unduhkan|kirim(?:kan)?|minta\s+(?:buku|file))\b",
    re.IGNORECASE,
)
_SOURCE_REQUEST_WORDS = {
    "link", "tautan", "download", "unduh", "unduhkan", "minta", "tolong",
    "kirim", "kirimkan", "buku", "file", "pdf", "doc", "docx", "dong",
    "ya", "nya", "untuk", "dari", "yang", "saya", "aku",
}


def is_source_link_request(question: str) -> bool:
    return bool(_SOURCE_REQUEST_RE.search(question))


def _normalize_source_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def find_source_link_matches(
    question: str,
    index: FAISS | None = None,
    limit: int = 3,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> list[dict[str, str]]:
    """Find up to three Drive files whose names resemble the requested title."""
    manifest = load_index_manifest(index_directory) or {}
    manifest_sources = manifest.get("sources", {})
    candidates: dict[str, dict[str, str]] = {}
    if isinstance(manifest_sources, dict):
        for file_id, source in manifest_sources.items():
            if isinstance(source, dict):
                candidates[str(file_id)] = {
                    key: str(value)
                    for key, value in source.items()
                    if value is not None
                }

    if index is not None:
        for document_id in index.index_to_docstore_id.values():
            document = index.docstore.search(document_id)
            if not isinstance(document, Document):
                continue
            file_id = str(document.metadata.get("file_id", ""))
            if not file_id:
                continue
            source = candidates.setdefault(file_id, {"file_id": file_id})
            if not source.get("name"):
                source["name"] = str(document.metadata.get("source", file_id))
            for key in ("web_view_link", "download_url"):
                if not source.get(key):
                    value = document.metadata.get(key, "")
                    if key == "web_view_link":
                        value = value or document.metadata.get("webViewLink", "")
                    source[key] = str(value)

    query_terms = [
        term
        for term in _normalize_source_query(question).split()
        if term not in _SOURCE_REQUEST_WORDS
    ]
    if not query_terms:
        return []
    query = " ".join(query_terms)
    query_term_set = set(query_terms)
    ranked: list[tuple[float, dict[str, str]]] = []
    for file_id, source in candidates.items():
        name = source.get("name", file_id)
        normalized_name = _normalize_source_query(re.sub(r"\.[a-z0-9]{1,6}$", "", name))
        name_terms = set(normalized_name.split())
        overlap = len(query_term_set & name_terms) / len(query_term_set)
        partial = max(
            (
                SequenceMatcher(None, query_term, name_term).ratio()
                for query_term in query_term_set
                for name_term in name_terms
            ),
            default=0.0,
        )
        score = max(
            SequenceMatcher(None, query, normalized_name).ratio(),
            overlap * 0.95,
            partial * 0.78,
        )
        if query in normalized_name or normalized_name in query:
            score = max(score, 0.9)
        if score < 0.4:
            continue
        source.setdefault("file_id", file_id)
        source["web_view_link"] = source.get("web_view_link") or (
            f"https://drive.google.com/file/d/{file_id}/view"
        )
        source["download_url"] = source.get("download_url") or (
            f"https://drive.google.com/uc?export=download&id={file_id}"
        )
        ranked.append((score, source))

    ranked.sort(
        key=lambda result: (-result[0], result[1].get("name", "").casefold())
    )
    return [source for _, source in ranked[: min(limit, 3)]]


def get_available_chat_models(api_key: str | None = None) -> list[str]:
    """Discover and briefly cache models that support generateContent."""
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CHAT_MODEL_CACHE_LOCK:
        cached = _CHAT_MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < MODEL_LIST_CACHE_SECONDS:
            return list(cached[1])

    discovered: set[str] = set()
    client = None
    try:
        client = genai.Client(api_key=resolved_api_key)
        for model in client.models.list(config={"query_base": True}):
            actions = getattr(model, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            model_name = str(getattr(model, "name", "")).rsplit("/", 1)[-1]
            if model_name:
                discovered.add(model_name)
    except Exception:
        discovered.clear()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    candidates = discovered or set(FALLBACK_CHAT_MODELS)
    model_names = tuple(
        sorted(
            candidates,
            key=lambda name: (
                PREFERRED_CHAT_MODEL_ORDER.get(name, len(PREFERRED_CHAT_MODEL_ORDER)),
                0 if "flash" in name.casefold() else 1,
                name.casefold(),
            ),
        )
    )
    with _CHAT_MODEL_CACHE_LOCK:
        _CHAT_MODEL_CACHE[cache_key] = (time.monotonic(), model_names)
    return list(model_names)


def _is_chat_model_fallback_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = (
            getattr(current, "status_code", None)
            or getattr(current, "status", None)
            or getattr(current, "code", None)
        )
        if status in {
            404,
            408,
            425,
            429,
            500,
            502,
            503,
            504,
            "404",
            "408",
            "425",
            "429",
            "500",
            "502",
            "503",
            "504",
        }:
            return True
        class_name = type(current).__name__.casefold()
        message = str(current).casefold()
        if any(
            marker in message
            for marker in (
                "resource_exhausted",
                "too many requests",
                "model_not_found",
                "not_found",
                "not found",
                "no longer available",
                "model is unavailable",
                "model unavailable",
                "model not supported",
                "currently experiencing high demand",
                "temporarily unavailable",
                "service unavailable",
                "timed out",
                "timeout",
                "connection error",
                "connection reset",
                "network error",
                "408",
                "500",
                "502",
                "503",
                "504",
            )
        ) or any(marker in class_name for marker in ("timeout", "connection", "network")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_chat_quota_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = str(
            getattr(current, "status_code", None)
            or getattr(current, "status", None)
            or getattr(current, "code", None)
            or ""
        ).casefold()
        message = str(current).casefold()
        if (
            "429" in status
            or "resource_exhausted" in status
            or "429" in message
            or "resource_exhausted" in message
            or "quota exceeded" in message
            or "quota_exceeded" in message
            or "too many requests" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _format_reference_values(values: list[str], unit: str) -> str:
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        return ""
    if all(value.isdigit() for value in unique_values):
        numbers = sorted({int(value) for value in unique_values})
        ranges: list[str] = []
        start = previous = numbers[0]
        for number in numbers[1:]:
            if number == previous + 1:
                previous = number
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = number
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return f"{unit} " + ", ".join(ranges)
    return ", ".join(f"{unit} {value}" for value in unique_values)


def _format_source_references(matches: list[Document]) -> str:
    references: dict[str, dict[str, object]] = {}
    for document in matches:
        metadata = document.metadata
        file_id = str(metadata.get("file_id", ""))
        name = str(metadata.get("source", "Dokumen"))
        key = file_id or name
        reference = references.setdefault(
            key,
            {"name": name, "pages": [], "sections": []},
        )
        page_label = metadata.get("page_label")
        if page_label is not None:
            reference["pages"].append(str(page_label))
            continue
        section_index = metadata.get("section_index", metadata.get("chunk_index"))
        if section_index is not None:
            try:
                reference["sections"].append(str(int(section_index) + (0 if "section_index" in metadata else 1)))
            except (TypeError, ValueError):
                pass

    formatted = []
    for number, reference in enumerate(references.values(), start=1):
        page_reference = _format_reference_values(reference["pages"], "Halaman")
        section_reference = _format_reference_values(reference["sections"], "Bagian")
        location = page_reference or section_reference
        if location:
            formatted.append(f"{number}. {reference['name']} ({location})")
        else:
            formatted.append(f"{number}. {reference['name']}")
    return "Sumber Rujukan: " + "; ".join(formatted) if formatted else ""


def _is_max_tokens_finish(reason: object) -> bool:
    text = str(reason or "").upper()
    return "MAX_TOKENS" in text or text == "LENGTH"


def _create_chat_model(model_name: str, api_key: str | None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        retries=0,
        request_timeout=CHAT_MODEL_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
    )


def _generate_complete_answer(
    model: ChatGoogleGenerativeAI,
    messages: Sequence[HumanMessage | SystemMessage],
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    """Generate an answer and keep continuing while Gemini stops at the token limit."""
    full_answer = ""
    conversation = list(messages)
    for _ in range(CHAT_MAX_CONTINUATIONS + 1):
        finish_reason = None
        parts: list[str] = []
        if on_token:
            for chunk in model.stream(conversation):
                text = _extract_stream_chunk_text(chunk.content)
                if text:
                    parts.append(text)
                    on_token(text)
                metadata = getattr(chunk, "response_metadata", None) or {}
                if metadata.get("finish_reason"):
                    finish_reason = metadata["finish_reason"]
        else:
            response = model.invoke(conversation)
            parts.append(_extract_stream_chunk_text(response.content))
            metadata = getattr(response, "response_metadata", None) or {}
            finish_reason = metadata.get("finish_reason")

        piece = "".join(parts)
        full_answer += piece
        if not piece.strip() or not _is_max_tokens_finish(finish_reason):
            break
        conversation = list(messages) + [
            AIMessage(content=full_answer),
            HumanMessage(content=CONTINUE_PROMPT),
        ]
    return full_answer.strip()


def _invoke_chat_model_messages(
    messages: Sequence[HumanMessage | SystemMessage],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    candidates = list(model_names or get_available_chat_models(api_key))
    last_model_error: Exception | None = None
    for model_name in candidates:
        try:
            model = _create_chat_model(model_name, api_key)
            answer = _generate_complete_answer(model, messages, on_token)
            if not answer:
                raise ValueError(f"Model {model_name} tidak menghasilkan teks jawaban.")
            return answer
        except Exception as error:
            if not _is_chat_model_fallback_error(error):
                raise
            if on_token:
                on_token(None)
            last_model_error = error

    if last_model_error is not None:
        raise last_model_error
    raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    api_key: str | None = None,
) -> str:
    """Transcribe a recorded voice message using an available Gemini model."""
    if not audio_bytes:
        raise ValueError("Rekaman suara kosong.")
    if not mime_type.startswith("audio/"):
        raise ValueError("Format rekaman suara tidak didukung.")

    model_names = [
        model_name
        for model_name in get_available_chat_models(api_key)
        if "gemini" in model_name.casefold()
    ]
    if not model_names:
        raise ValueError("Tidak ada model Gemini yang tersedia untuk memproses suara.")

    audio_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transkripsikan rekaman suara secara akurat. Ikuti bahasa "
                    "yang digunakan pembicara dan keluarkan hanya transkripsinya."
                ),
            },
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            },
        ]
    )
    return _invoke_chat_model_messages([audio_message], api_key, model_names)


def answer_question(
    index: FAISS | None,
    question: str,
    api_key: str | None = None,
    media_parts: Sequence[dict[str, str]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_token: Callable[[str | None], None] | None = None,
    force_local_search: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """Answer from retrieved passages and any supplied image attachments."""
    if not question.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    _check_api_key(api_key)

    media_parts = media_parts or ()
    full_context_mode = is_full_context_request(question)
    retrieval_count = FULL_CONTEXT_CHUNK_COUNT if full_context_mode else 8
    keyword_fallback_reason: str | None = None
    anchor_vector_position: int | None = None
    matches: list[Document] = []
    if index is not None:
        if force_local_search:
            if on_progress:
                on_progress("Mencari di indeks lokal...")
            matches = _keyword_search(index, question, k=retrieval_count)
            keyword_fallback_reason = "sinkronisasi indeks sedang berlangsung"
        else:
            try:
                if on_progress:
                    on_progress("Menghubungi Gemini untuk embedding pertanyaan...")

                embedding_function = index.embedding_function
                if isinstance(embedding_function, _QuotaAwareEmbeddings):
                    query_vector = _get_chat_query_embeddings(api_key).embed_query(
                        question,
                    )
                elif isinstance(embedding_function, Embeddings):
                    query_vector = embedding_function.embed_query(question)
                else:
                    query_vector = index._embed_query(question)

                if on_progress:
                    on_progress(f"Mencari {retrieval_count} bagian yang relevan...")
                if full_context_mode:
                    matches, anchor_vector_position = _similarity_search_with_position(
                        index,
                        query_vector,
                        retrieval_count,
                    )
                else:
                    matches = index.similarity_search_by_vector(
                        query_vector,
                        k=retrieval_count,
                    )
            except Exception as error:
                if not _is_transient_embedding_error(error):
                    raise
                if on_progress:
                    on_progress("Mencari di indeks lokal...")
                matches = _keyword_search(index, question, k=retrieval_count)
                keyword_fallback_reason = (
                    "kuota embedding Gemini habis"
                    if is_embedding_quota_error(error)
                    else "layanan embedding Gemini sedang lambat/tidak tersedia"
                )

    if full_context_mode and index is not None:
        matches = _full_context_window(
            index,
            matches,
            anchor_vector_position=anchor_vector_position,
        )

    if not matches and not media_parts:
        if keyword_fallback_reason:
            return (
                f"Pencarian lokal tidak menemukan kata kunci yang cocok "
                f"({keyword_fallback_reason}). Coba gunakan istilah yang lebih spesifik.",
                [],
            )
        return "Saya belum menemukan informasi yang relevan di dokumen.", []

    context = "\n\n".join(
        f"[{document.metadata.get('source', 'Dokumen')}]\n{document.page_content}"
        for document in matches
    )
    completeness_instruction = (
        "Pengguna meminta isi lengkap. Pertahankan urutan konteks, gunakan semua "
        "bagian yang tersedia, dan jangan meringkas bait atau bagian yang "
        "diminta utuh. Jika konteks tidak memuat seluruh dokumen, jelaskan batasnya."
        if full_context_mode
        else "Sesuaikan panjang jawaban dengan pertanyaan. Pertanyaan sederhana dijawab singkat; jika isi yang diminta panjang (misalnya teks, bait, daftar, atau penjelasan rinci), tulis sampai tuntas dan jangan berhenti di tengah atau memotong jawaban."
    )
    source_line_instruction = (
        "Selalu pertahankan struktur baris asli dari dokumen rujukan. Jika teks "
        "rujukan tersusun dalam baris-baris pendek berurutan (seperti lirik lagu, "
        "nadhom, syair, pantun, doa, atau daftar), tampilkan kembali persis per "
        "baris sesuai aslinya dengan menambahkan dua spasi di akhir setiap baris "
        "(hard line break), meskipun pengguna hanya mengetik judul atau kata "
        "kunci singkat tanpa menyebut kata lagu/syair."
        if matches
        else ""
    )
    system_instruction = (
        "Anda adalah asisten pribadi yang menjawab dalam Bahasa Indonesia "
        "atau Bahasa Sunda, mengikuti bahasa pertanyaan pengguna. "
        + (
            "Gunakan konteks dokumen dan gambar terlampir jika relevan. Bedakan "
            "informasi dari dokumen dan pengamatan gambar; jika konteks dokumen "
            "tidak cukup, jelaskan batasnya. "
            if matches
            else "Tidak ada bagian dokumen yang relevan. Jika ada gambar terlampir, "
            "jawab berdasarkan pengamatan gambar dan jangan mengarang isi dokumen. "
        )
        + completeness_instruction
        + (" " + source_line_instruction if source_line_instruction else "")
    )
    answer = ""
    last_model_error: Exception | None = None
    if on_progress:
        on_progress("Menyusun jawaban dengan Gemini...")
    if media_parts:
        if on_progress:
            on_progress("Menyiapkan Gemini untuk memeriksa lampiran...")
        system_message = (
            f"{system_instruction}\n\nKonteks:\n{context or '(tidak tersedia)'}"
        )
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                *media_parts,
            ]
        )
        answer = _invoke_chat_model_messages(
            [SystemMessage(content=system_message), human_message],
            api_key,
            on_token=on_token,
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system_instruction}\n\nKonteks:\n{{context}}"),
                ("human", "{question}"),
            ]
        )
        if on_progress:
            on_progress("Memeriksa model Gemini yang tersedia...")
        for model_name in get_available_chat_models(api_key):
            if on_progress:
                on_progress(f"Menunggu jawaban dari {model_name}...")
            try:
                model = _create_chat_model(model_name, api_key)
                prompt_messages = prompt.format_messages(
                    context=context or "(tidak tersedia)",
                    question=question,
                )
                answer = _generate_complete_answer(model, prompt_messages, on_token)
                if not answer:
                    raise ValueError(
                        f"Model {model_name} tidak menghasilkan teks jawaban."
                    )
                break
            except Exception as error:
                if not _is_chat_model_fallback_error(error):
                    raise
                if on_progress and _is_chat_quota_error(error):
                    on_progress(
                        f"Gemini membatasi permintaan ke {model_name}; mencoba model cadangan..."
                    )
                if on_token:
                    on_token(None)
                last_model_error = error

        if not answer:
            if last_model_error is not None:
                if _is_chat_quota_error(last_model_error):
                    raise ValueError(
                        "Gemini sedang membatasi permintaan (limit/kuota). "
                        "Tunggu sebentar lalu coba lagi."
                    ) from last_model_error
                raise last_model_error
            raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")
    if keyword_fallback_reason:
        answer = (
            f"Catatan: {keyword_fallback_reason}, jadi dokumen dicari berdasarkan "
            f"kecocokan kata pada indeks lokal.\n\n{answer}"
        )
    source_references = _format_source_references(matches)
    if source_references:
        answer = f"{answer.rstrip()}\n\n{source_references}"

    sources_by_id: dict[str, dict[str, str]] = {}
    manifest_sources = (load_index_manifest() or {}).get("sources", {})
    if not isinstance(manifest_sources, dict):
        manifest_sources = {}
    for document in matches:
        metadata = document.metadata
        file_id = str(metadata.get("file_id", ""))
        name = str(metadata.get("source", "Dokumen"))
        source_key = file_id or name
        if source_key in sources_by_id:
            continue
        manifest_source = manifest_sources.get(file_id, {})
        if not isinstance(manifest_source, dict):
            manifest_source = {}
        sources_by_id[source_key] = {
            "file_id": file_id,
            "name": name,
            "web_view_link": str(
                metadata.get("web_view_link")
                or metadata.get("webViewLink")
                or manifest_source.get("web_view_link")
                or manifest_source.get("webViewLink")
                or f"https://drive.google.com/file/d/{file_id}/view"
            ),
            "download_url": str(
                metadata.get("download_url")
                or manifest_source.get("download_url")
                or f"https://drive.google.com/uc?export=download&id={file_id}"
            ),
        }
    sources = list(sources_by_id.values())
    return answer, sources"""Build a semantic search index and answer questions with Gemini."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import time
import unicodedata
import faiss
import numpy as np
from collections import Counter
from collections.abc import Sequence
from collections import deque
from difflib import SequenceMatcher
from pathlib import Path
from threading import Lock
from typing import Callable

from google import genai
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from services.gdrive_service import DriveDocument

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "faiss_index"
MANIFEST_FILENAME = "drive_manifest.json"
EMBEDDING_TOKENS_PER_MINUTE = 27_000
EMBEDDING_WINDOW_SECONDS = 60.0
EMBEDDING_TARGET_BATCH_TOKENS = 22_000
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 30
EMBEDDING_MAX_BATCH_SIZE = 20
EMBEDDING_MAX_BATCH_CHARACTERS = 15_000
EMBEDDING_MAX_QUOTA_RETRIES = 2
EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS = EMBEDDING_WINDOW_SECONDS
EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS = 300.0
EMBEDDING_REQUEST_TIMEOUT_MS = 30_000
CHAT_QUERY_EMBEDDING_TIMEOUT_MS = 5_000
CHAT_MODEL_REQUEST_TIMEOUT_SECONDS = 180
CHAT_MAX_OUTPUT_TOKENS = 8192
CHAT_MAX_CONTINUATIONS = 5
CONTINUE_PROMPT = (
    "Lanjutkan jawaban di atas tepat dari titik terakhir yang terpotong. "
    "Jangan mengulang bagian yang sudah ditulis dan jangan membuat pembuka baru."
)
EMBEDDING_MAX_RETRIES = 0
MODEL_LIST_CACHE_SECONDS = 300.0
FALLBACK_CHAT_MODELS = (
    "gemini-omni-1.2-flash",
    "gemini-omni-1.1-flash",
    "gemini-3.8-flash",
    "gemma-4-26b-it",
)
PREFERRED_CHAT_MODEL_ORDER = {
    model: position for position, model in enumerate(FALLBACK_CHAT_MODELS)
}
_CHAT_MODEL_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_CHAT_MODEL_CACHE_LOCK = Lock()


class _EmbeddingRateLimiter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: deque[tuple[float, int]] = deque()

    def acquire(
        self,
        estimated_tokens: int,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        if estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
            raise ValueError(
                "Մեկ embedding հարցման գնահատված token-ների քանակը գերազանցում է "
                f"{EMBEDDING_TARGET_BATCH_TOKENS}-ի անվտանգ սահմանը։"
            )
        estimated_tokens = max(estimated_tokens, 1)
        while True:
            with self._lock:
                now = time.monotonic()
                while (
                    self._requests
                    and now - self._requests[0][0] >= EMBEDDING_WINDOW_SECONDS
                ):
                    self._requests.popleft()

                tokens_in_window = sum(tokens for _, tokens in self._requests)
                requests_wait = (
                    max(
                        0.0,
                        self._requests[0][0] + EMBEDDING_WINDOW_SECONDS - now,
                    )
                    if len(self._requests) >= EMBEDDING_MAX_REQUESTS_PER_MINUTE
                    else 0.0
                )
                tokens_wait = 0.0
                if tokens_in_window + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
                    remaining_tokens = tokens_in_window
                    for request_time, request_tokens in self._requests:
                        remaining_tokens -= request_tokens
                        if (
                            remaining_tokens + estimated_tokens
                            <= EMBEDDING_TARGET_BATCH_TOKENS
                        ):
                            tokens_wait = max(
                                0.0,
                                request_time + EMBEDDING_WINDOW_SECONDS - now,
                            )
                            break

                if requests_wait == 0 and tokens_wait == 0:
                    self._requests.append((now, estimated_tokens))
                    return

                wait_seconds = max(0.1, requests_wait, tokens_wait)

            if on_wait:
                on_wait(wait_seconds)
            time.sleep(wait_seconds)


_EMBEDDING_RATE_LIMITER = _EmbeddingRateLimiter()


class _QuotaAwareEmbeddings(Embeddings):
    def __init__(
        self,
        client: GoogleGenerativeAIEmbeddings,
        on_wait: Callable[[float], None] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        limit_query_rate: bool = True,
    ) -> None:
        self._client = client
        self._on_wait = on_wait
        self._on_progress = on_progress
        self._limit_query_rate = limit_query_rate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        completed = 0
        for batch, estimated_tokens in _make_embedding_batches(texts):
            retries = 0
            while True:
                if self._on_progress:
                    self._on_progress(completed, len(texts))
                _EMBEDDING_RATE_LIMITER.acquire(estimated_tokens, self._on_wait)
                try:
                    batch_vectors = self._client.embed_documents(
                        batch,
                        batch_size=len(batch),
                    )
                except Exception as error:
                    if (
                        not is_embedding_quota_error(error)
                        or is_embedding_daily_quota_error(error)
                        or retries >= EMBEDDING_MAX_QUOTA_RETRIES
                    ):
                        raise
                    retry_delay = (
                        get_quota_retry_delay_seconds(error)
                        or EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS
                    )
                    if retry_delay > EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS:
                        raise
                    if self._on_wait:
                        self._on_wait(retry_delay)
                    time.sleep(retry_delay)
                    retries += 1
                    continue
                vectors.extend(batch_vectors)
                completed += len(batch)
                if self._on_progress:
                    self._on_progress(completed, len(texts))
                break
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_query_with_wait(text, self._on_wait)

    def embed_query_with_wait(
        self,
        text: str,
        on_wait: Callable[[float], None] | None = None,
    ) -> list[float]:
        if self._limit_query_rate:
            _EMBEDDING_RATE_LIMITER.acquire(
                (len(text.encode("utf-8")) + 3) // 4 + 8,
                self._on_wait if on_wait is None else on_wait,
            )
        return self._client.embed_query(text)


def is_embedding_quota_error(error: Exception) -> bool:
    """Return whether an exception indicates a Gemini quota or rate limit."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited:
            break
        visited.add(id(current))
        message = str(current).casefold()
        status = str(
            getattr(current, "status", None)
            or getattr(current, "status_code", None)
            or getattr(current, "code", None)
            or "",
        ).casefold()
        if (
            "429" in status
            or "resource_exhausted" in status
            or "429" in message
            or "resource_exhausted" in message
            or "quota exceeded" in message
            or "quota_exceeded" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_transient_embedding_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = str(
            getattr(current, "status", None)
            or getattr(current, "status_code", None)
            or getattr(current, "code", None)
            or ""
        )
        class_name = type(current).__name__.casefold()
        message = str(current).casefold()
        if (
            status in {"408", "425", "429", "500", "502", "503", "504"}
            or "timeout" in class_name
            or "connection" in class_name
            or "network" in class_name
            or any(
                marker in message
                for marker in (
                    "timed out",
                    "timeout",
                    "connection error",
                    "connection reset",
                    "temporarily unavailable",
                    "service unavailable",
                    "resource_exhausted",
                    "quota exceeded",
                    "too many requests",
                    "429",
                    "500",
                    "502",
                    "503",
                    "504",
                )
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def is_embedding_daily_quota_error(error: Exception) -> bool:
    """Return whether a Gemini quota error is explicitly limited per day."""
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None:
        if id(current) in visited:
            break
        visited.add(id(current))
        message = str(current).casefold()
        compact_message = re.sub(r"[^a-z0-9]", "", message)
        if is_embedding_quota_error(current) and (
            "per day" in message
            or "per-day" in message
            or "daily" in message
            or "perday" in compact_message
            or "requestsperday" in compact_message
            or "tokensperday" in compact_message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def get_quota_retry_delay_seconds(error: Exception) -> float | None:
    """Extract retry duration from Gemini's RetryInfo or human-readable message."""
    current: BaseException | None = error
    while current is not None:
        message = str(current)
        retry_delay = re.search(
            r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s",
            message,
            re.IGNORECASE,
        )
        if retry_delay:
            return float(retry_delay.group(1))

        human_delay = re.search(
            r"retry in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?",
            message,
            re.IGNORECASE,
        )
        if human_delay and any(human_delay.groups()):
            hours, minutes, seconds = human_delay.groups()
            return (
                int(hours or 0) * 3600
                + int(minutes or 0) * 60
                + float(seconds or 0)
            )
        current = current.__cause__
    return None


def _make_embedding_batches(texts: list[str]) -> list[tuple[list[str], int]]:
    batches: list[tuple[list[str], int]] = []
    current_batch: list[str] = []
    current_tokens = 0
    current_characters = 0

    for text in texts:
        estimated_tokens = max(1, len(text) // 3)
        character_count = len(text)
        if current_batch and (
            len(current_batch) >= EMBEDDING_MAX_BATCH_SIZE
            or current_characters + character_count > EMBEDDING_MAX_BATCH_CHARACTERS
            or current_tokens + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS
        ):
            batches.append((current_batch, current_tokens))
            current_batch = []
            current_tokens = 0
            current_characters = 0
        if character_count > EMBEDDING_MAX_BATCH_CHARACTERS:
            raise ValueError(
                "Մեկ embedding chunk-ի նիշերի քանակը գերազանցում է "
                f"{EMBEDDING_MAX_BATCH_CHARACTERS}-ի սահմանը։"
            )
        if estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
            raise ValueError(
                "Մեկ embedding chunk-ի գնահատված token-ների քանակը գերազանցում է "
                f"{EMBEDDING_TARGET_BATCH_TOKENS}-ի անվտանգ սահմանը։"
            )

        current_batch.append(text)
        current_tokens += estimated_tokens
        current_characters += character_count

    if current_batch:
        batches.append((current_batch, current_tokens))
    return batches


def _check_api_key(api_key: str | None = None) -> None:
    if not (api_key or os.getenv("GOOGLE_API_KEY")):
        raise ValueError("GOOGLE_API_KEY belum diatur.")


def _get_embeddings(
    api_key: str | None = None,
    on_wait: Callable[[float], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Embeddings:
    client = _create_embedding_client(api_key, EMBEDDING_REQUEST_TIMEOUT_MS)
    return _QuotaAwareEmbeddings(
        client,
        on_wait=on_wait,
        on_progress=on_progress,
    )


def _create_embedding_client(
    api_key: str | None,
    timeout_ms: int,
) -> GoogleGenerativeAIEmbeddings:
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    client = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=resolved_api_key,
    )
    client.client = genai.Client(
        api_key=resolved_api_key,
        http_options=genai.types.HttpOptions(
            timeout=timeout_ms,
            retry_options=genai.types.HttpRetryOptions(
                attempts=EMBEDDING_MAX_RETRIES + 1,
            ),
        ),
    )
    return client


def _get_chat_query_embeddings(api_key: str | None = None) -> _QuotaAwareEmbeddings:
    client = _create_embedding_client(api_key, CHAT_QUERY_EMBEDDING_TIMEOUT_MS)
    return _QuotaAwareEmbeddings(client, limit_query_rate=False)


def _to_langchain_documents(
    source_documents: Sequence[DriveDocument],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    chunks: list[Document] = []
    for source in source_documents:
        if not source.text.strip():
            continue
        base_metadata = {
            "source": source.name,
            "file_id": source.file_id,
            "modified_time": source.modified_time,
            "web_view_link": source.web_view_link,
            "webViewLink": source.web_view_link,
            "download_url": source.download_url,
        }
        sections = source.page_sections or (("", source.text),)
        for page_label, section_text in sections:
            if not section_text.strip():
                continue
            metadata = dict(base_metadata)
            if page_label:
                metadata["page_label"] = page_label
            else:
                metadata["section_type"] = "section"
            chunks.extend(
                splitter.split_documents(
                    [Document(page_content=section_text, metadata=metadata)]
                )
            )

    next_chunk_index: Counter[str] = Counter()
    for chunk in chunks:
        file_id = str(chunk.metadata.get("file_id", ""))
        chunk_index = next_chunk_index[file_id]
        chunk.metadata["chunk_index"] = chunk_index
        if not chunk.metadata.get("page_label"):
            chunk.metadata["section_index"] = chunk_index + 1
        next_chunk_index[file_id] += 1
    return chunks


def create_search_index(
    source_documents: Sequence[DriveDocument],
    api_key: str | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> FAISS:
    """Create a multilingual semantic index from Drive documents."""
    if not source_documents:
        raise ValueError(
            "Tidak ada teks DOCX atau PDF yang dapat diindeks di folder Google Drive tersebut."
        )
    _check_api_key(api_key)

    documents = _to_langchain_documents(source_documents)
    return FAISS.from_documents(documents, _get_embeddings(api_key, on_wait))


def load_search_index(
    api_key: str,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> FAISS | None:
    """Load a previously persisted FAISS index, if one is present."""
    if not (index_directory / "index.faiss").is_file():
        return None
    return FAISS.load_local(
        str(index_directory),
        _get_embeddings(api_key),
        allow_dangerous_deserialization=True,
    )


def update_search_index(
    index: FAISS | None,
    source_documents: Sequence[DriveDocument],
    replaced_file_ids: set[str],
    api_key: str,
    on_wait: Callable[[float], None] | None = None,
    on_checkpoint: Callable[[FAISS], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> FAISS | None:
    """Replace changed file chunks and checkpoint each successfully embedded batch."""
    chunks = _to_langchain_documents(source_documents)
    if not chunks and index is None:
        return None

    chunks_by_file: dict[str, dict[int, Document]] = {}
    for chunk in chunks:
        file_id = str(chunk.metadata.get("file_id", ""))
        chunk_index = int(chunk.metadata.get("chunk_index", 0))
        chunks_by_file.setdefault(file_id, {})[chunk_index] = chunk

    existing_chunks: set[tuple[str, str, int]] = set()
    if index is not None:
        stale_ids: list[str] = []
        for document_id in index.index_to_docstore_id.values():
            stored_document = index.docstore.search(document_id)
            if not isinstance(stored_document, Document):
                continue
            file_id = str(stored_document.metadata.get("file_id", ""))
            if file_id not in replaced_file_ids:
                continue
            desired_chunks = chunks_by_file.get(file_id)
            chunk_index = int(stored_document.metadata.get("chunk_index", -1))
            desired_chunk = desired_chunks.get(chunk_index) if desired_chunks else None
            if (
                desired_chunk is not None
                and stored_document.metadata.get("modified_time")
                == desired_chunk.metadata.get("modified_time")
                and stored_document.page_content == desired_chunk.page_content
            ):
                existing_chunks.add(
                    (
                        file_id,
                        str(stored_document.metadata.get("modified_time", "")),
                        chunk_index,
                    )
                )
            else:
                stale_ids.append(document_id)
        if stale_ids:
            index.delete(ids=stale_ids)

    pending_chunks = [
        chunk
        for chunk in chunks
        if (
            str(chunk.metadata.get("file_id", "")),
            str(chunk.metadata.get("modified_time", "")),
            int(chunk.metadata.get("chunk_index", 0)),
        )
        not in existing_chunks
    ]
    if pending_chunks:
        total_pending = len(pending_chunks)
        completed = 0
        batch_offset = 0
        batch_total = 0

        def report_embedding_progress(
            batch_done: int,
            current_batch_total: int,
        ) -> None:
            if current_batch_total != batch_total:
                raise RuntimeError("Embedding progress does not match its batch.")
            if on_progress:
                on_progress(
                    batch_offset + batch_done,
                    total_pending,
                )

        embeddings = _get_embeddings(
            api_key,
            on_wait,
            report_embedding_progress,
        )
        for batch_info in _make_embedding_batches(
            [chunk.page_content for chunk in pending_chunks]
        ):
            batch = batch_info[0]
            batch_chunks = pending_chunks[: len(batch)]
            del pending_chunks[: len(batch)]
            batch_offset = completed
            batch_total = len(batch_chunks)
            if index is None:
                index = FAISS.from_documents(batch_chunks, embeddings)
            else:
                index.add_documents(batch_chunks, embedding=embeddings)
            completed += len(batch_chunks)
            if on_checkpoint:
                on_checkpoint(index)
    return index


def get_index_file_manifest(index: FAISS | None) -> dict[str, str]:
    """Recover file IDs and modification times from a legacy local index."""
    if index is None:
        return {}

    manifest: dict[str, str] = {}
    for document_id in index.index_to_docstore_id.values():
        stored_document = index.docstore.search(document_id)
        if isinstance(stored_document, Document):
            file_id = stored_document.metadata.get("file_id")
            if file_id:
                manifest[str(file_id)] = str(
                    stored_document.metadata.get("modified_time", "")
                )
    return manifest


def load_index_manifest(
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> dict[str, object] | None:
    """Load the local Drive inventory associated with the FAISS index."""
    manifest_path = index_directory / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        return None
    return manifest


def save_index_manifest(
    folder_id: str,
    files: dict[str, str],
    has_index: bool,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
    sources: dict[str, dict[str, str]] | None = None,
) -> None:
    """Persist Drive modification times next to the local FAISS index."""
    index_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = index_directory / MANIFEST_FILENAME
    temporary_path = manifest_path.with_suffix(".tmp")
    previous_manifest = load_index_manifest(index_directory) or {}
    saved_sources = sources if sources is not None else previous_manifest.get("sources", {})
    temporary_path.write_text(
        json.dumps(
            {
                "folder_id": folder_id,
                "files": files,
                "has_index": has_index,
                "sources": saved_sources,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def _extract_response_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "\n".join(text_parts).strip()
    return str(content).strip()


def _extract_stream_chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        return "".join(text_parts)
    return ""


def _keyword_search(index: FAISS, question: str, k: int = 5) -> list[Document]:
    """Rank indexed chunks locally when Gemini query embeddings are unavailable."""
    query_terms = {
        term
        for term in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(term) > 2
    }
    if not query_terms:
        return []

    indexed_documents: list[tuple[Document, Counter[str]]] = []
    document_frequencies: Counter[str] = Counter()
    for document_id in index.index_to_docstore_id.values():
        document = index.docstore.search(document_id)
        if not isinstance(document, Document):
            continue
        counts = Counter(
            term
            for term in re.findall(
                r"\w+", document.page_content.casefold(), flags=re.UNICODE
            )
            if len(term) > 2
        )
        indexed_documents.append((document, counts))
        document_frequencies.update(query_terms & counts.keys())

    document_count = len(indexed_documents)
    ranked_documents = []
    for position, (document, counts) in enumerate(indexed_documents):
        score = sum(
            (1 + math.log(counts[term]))
            * (1 + math.log((document_count + 1) / (document_frequencies[term] + 1)))
            for term in query_terms
            if counts[term]
        )
        if score:
            ranked_documents.append((score, position, document))

    ranked_documents.sort(key=lambda result: (-result[0], result[1]))
    return [document for _, _, document in ranked_documents[:k]]


_FULL_CONTEXT_RE = re.compile(
    r"\b(?:lengkap|sampai\s+selesai|seluruh(?:nya)?|utuh|semua(?:\s+bait)?|full|bacakan|tuliskan|tampilkan|bait|nadhom|nadzom|syair)\b",
    re.IGNORECASE,
)
FULL_CONTEXT_CHUNK_COUNT = 25


def is_full_context_request(question: str) -> bool:
    return bool(_FULL_CONTEXT_RE.search(question))


def _full_context_window(
    index: FAISS,
    matches: list[Document],
    window_size: int = FULL_CONTEXT_CHUNK_COUNT,
    anchor_vector_position: int | None = None,
) -> list[Document]:
    """Expand the strongest hit with sequential chunks from the same source."""
    if not matches:
        return []
    file_id = str(matches[0].metadata.get("file_id", ""))
    if not file_id:
        return matches

    source_chunks: list[tuple[int, int, Document]] = []
    fallback_index = 0
    for vector_position, document_id in sorted(index.index_to_docstore_id.items()):
        document = index.docstore.search(document_id)
        if not isinstance(document, Document):
            continue
        if str(document.metadata.get("file_id", "")) != file_id:
            continue
        chunk_index = document.metadata.get("chunk_index")
        try:
            chunk_index = int(chunk_index)
        except (TypeError, ValueError):
            chunk_index = fallback_index
        source_chunks.append((chunk_index, vector_position, document))
        fallback_index += 1

    if not source_chunks:
        return matches
    source_chunks.sort(key=lambda item: (item[0], item[1]))
    if anchor_vector_position is not None:
        best_position = next(
            (
                position
                for position, (_, vector_position, _) in enumerate(source_chunks)
                if vector_position == anchor_vector_position
            ),
            None,
        )
    else:
        best_position = next(
            (
                position
                for position, (_, _, document) in enumerate(source_chunks)
                if document is matches[0]
            ),
            None,
        )
    if best_position is None:
        best_chunk_index = matches[0].metadata.get("chunk_index")
        try:
            best_position = next(
                position
                for position, (chunk_index, _, _) in enumerate(source_chunks)
                if chunk_index == int(best_chunk_index)
            )
        except (StopIteration, TypeError, ValueError):
            matching_positions = [
                position
                for position, (_, _, document) in enumerate(source_chunks)
                if document.page_content == matches[0].page_content
            ]
            best_position = matching_positions[-1] if matching_positions else 0

    count = min(window_size, len(source_chunks))
    start = max(0, best_position - 2)
    start = min(start, len(source_chunks) - count)
    return [document for _, _, document in source_chunks[start : start + count]]


def _similarity_search_with_position(
    index: FAISS,
    query_vector: list[float],
    k: int,
) -> tuple[list[Document], int | None]:
    """Search FAISS while retaining the exact vector row for the best hit."""
    query_matrix = np.asarray([query_vector], dtype=np.float32)
    if index._normalize_L2:
        faiss.normalize_L2(query_matrix)
    _, vector_ids = index.index.search(query_matrix, k)
    matches: list[Document] = []
    best_position: int | None = None
    for vector_id in vector_ids[0]:
        vector_position = int(vector_id)
        if vector_position < 0:
            continue
        document_id = index.index_to_docstore_id[vector_position]
        document = index.docstore.search(document_id)
        if isinstance(document, Document):
            matches.append(document)
            if best_position is None:
                best_position = vector_position
    return matches, best_position


_SOURCE_REQUEST_RE = re.compile(
    r"\b(?:link|tautan|download|unduh|unduhkan|kirim(?:kan)?|minta\s+(?:buku|file))\b",
    re.IGNORECASE,
)
_SOURCE_REQUEST_WORDS = {
    "link", "tautan", "download", "unduh", "unduhkan", "minta", "tolong",
    "kirim", "kirimkan", "buku", "file", "pdf", "doc", "docx", "dong",
    "ya", "nya", "untuk", "dari", "yang", "saya", "aku",
}


def is_source_link_request(question: str) -> bool:
    return bool(_SOURCE_REQUEST_RE.search(question))


def _normalize_source_query(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    normalized = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[\w]+", normalized, flags=re.UNICODE))


def find_source_link_matches(
    question: str,
    index: FAISS | None = None,
    limit: int = 3,
    index_directory: Path = DEFAULT_INDEX_DIRECTORY,
) -> list[dict[str, str]]:
    """Find up to three Drive files whose names resemble the requested title."""
    manifest = load_index_manifest(index_directory) or {}
    manifest_sources = manifest.get("sources", {})
    candidates: dict[str, dict[str, str]] = {}
    if isinstance(manifest_sources, dict):
        for file_id, source in manifest_sources.items():
            if isinstance(source, dict):
                candidates[str(file_id)] = {
                    key: str(value)
                    for key, value in source.items()
                    if value is not None
                }

    if index is not None:
        for document_id in index.index_to_docstore_id.values():
            document = index.docstore.search(document_id)
            if not isinstance(document, Document):
                continue
            file_id = str(document.metadata.get("file_id", ""))
            if not file_id:
                continue
            source = candidates.setdefault(file_id, {"file_id": file_id})
            if not source.get("name"):
                source["name"] = str(document.metadata.get("source", file_id))
            for key in ("web_view_link", "download_url"):
                if not source.get(key):
                    value = document.metadata.get(key, "")
                    if key == "web_view_link":
                        value = value or document.metadata.get("webViewLink", "")
                    source[key] = str(value)

    query_terms = [
        term
        for term in _normalize_source_query(question).split()
        if term not in _SOURCE_REQUEST_WORDS
    ]
    if not query_terms:
        return []
    query = " ".join(query_terms)
    query_term_set = set(query_terms)
    ranked: list[tuple[float, dict[str, str]]] = []
    for file_id, source in candidates.items():
        name = source.get("name", file_id)
        normalized_name = _normalize_source_query(re.sub(r"\.[a-z0-9]{1,6}$", "", name))
        name_terms = set(normalized_name.split())
        overlap = len(query_term_set & name_terms) / len(query_term_set)
        partial = max(
            (
                SequenceMatcher(None, query_term, name_term).ratio()
                for query_term in query_term_set
                for name_term in name_terms
            ),
            default=0.0,
        )
        score = max(
            SequenceMatcher(None, query, normalized_name).ratio(),
            overlap * 0.95,
            partial * 0.78,
        )
        if query in normalized_name or normalized_name in query:
            score = max(score, 0.9)
        if score < 0.4:
            continue
        source.setdefault("file_id", file_id)
        source["web_view_link"] = source.get("web_view_link") or (
            f"https://drive.google.com/file/d/{file_id}/view"
        )
        source["download_url"] = source.get("download_url") or (
            f"https://drive.google.com/uc?export=download&id={file_id}"
        )
        ranked.append((score, source))

    ranked.sort(
        key=lambda result: (-result[0], result[1].get("name", "").casefold())
    )
    return [source for _, source in ranked[: min(limit, 3)]]


def get_available_chat_models(api_key: str | None = None) -> list[str]:
    """Discover and briefly cache models that support generateContent."""
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CHAT_MODEL_CACHE_LOCK:
        cached = _CHAT_MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < MODEL_LIST_CACHE_SECONDS:
            return list(cached[1])

    discovered: set[str] = set()
    client = None
    try:
        client = genai.Client(api_key=resolved_api_key)
        for model in client.models.list(config={"query_base": True}):
            actions = getattr(model, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            model_name = str(getattr(model, "name", "")).rsplit("/", 1)[-1]
            if model_name:
                discovered.add(model_name)
    except Exception:
        discovered.clear()
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    candidates = discovered or set(FALLBACK_CHAT_MODELS)
    model_names = tuple(
        sorted(
            candidates,
            key=lambda name: (
                PREFERRED_CHAT_MODEL_ORDER.get(name, len(PREFERRED_CHAT_MODEL_ORDER)),
                0 if "flash" in name.casefold() else 1,
                name.casefold(),
            ),
        )
    )
    with _CHAT_MODEL_CACHE_LOCK:
        _CHAT_MODEL_CACHE[cache_key] = (time.monotonic(), model_names)
    return list(model_names)


def _is_chat_model_fallback_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = (
            getattr(current, "status_code", None)
            or getattr(current, "status", None)
            or getattr(current, "code", None)
        )
        if status in {
            404,
            408,
            425,
            429,
            500,
            502,
            503,
            504,
            "404",
            "408",
            "425",
            "429",
            "500",
            "502",
            "503",
            "504",
        }:
            return True
        class_name = type(current).__name__.casefold()
        message = str(current).casefold()
        if any(
            marker in message
            for marker in (
                "resource_exhausted",
                "too many requests",
                "model_not_found",
                "not_found",
                "not found",
                "no longer available",
                "model is unavailable",
                "model unavailable",
                "model not supported",
                "currently experiencing high demand",
                "temporarily unavailable",
                "service unavailable",
                "timed out",
                "timeout",
                "connection error",
                "connection reset",
                "network error",
                "408",
                "500",
                "502",
                "503",
                "504",
            )
        ) or any(marker in class_name for marker in ("timeout", "connection", "network")):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_chat_quota_error(error: Exception) -> bool:
    current: BaseException | None = error
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        status = str(
            getattr(current, "status_code", None)
            or getattr(current, "status", None)
            or getattr(current, "code", None)
            or ""
        ).casefold()
        message = str(current).casefold()
        if (
            "429" in status
            or "resource_exhausted" in status
            or "429" in message
            or "resource_exhausted" in message
            or "quota exceeded" in message
            or "quota_exceeded" in message
            or "too many requests" in message
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _format_reference_values(values: list[str], unit: str) -> str:
    unique_values = list(dict.fromkeys(values))
    if not unique_values:
        return ""
    if all(value.isdigit() for value in unique_values):
        numbers = sorted({int(value) for value in unique_values})
        ranges: list[str] = []
        start = previous = numbers[0]
        for number in numbers[1:]:
            if number == previous + 1:
                previous = number
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = number
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return f"{unit} " + ", ".join(ranges)
    return ", ".join(f"{unit} {value}" for value in unique_values)


def _format_source_references(matches: list[Document]) -> str:
    references: dict[str, dict[str, object]] = {}
    for document in matches:
        metadata = document.metadata
        file_id = str(metadata.get("file_id", ""))
        name = str(metadata.get("source", "Dokumen"))
        key = file_id or name
        reference = references.setdefault(
            key,
            {"name": name, "pages": [], "sections": []},
        )
        page_label = metadata.get("page_label")
        if page_label is not None:
            reference["pages"].append(str(page_label))
            continue
        section_index = metadata.get("section_index", metadata.get("chunk_index"))
        if section_index is not None:
            try:
                reference["sections"].append(str(int(section_index) + (0 if "section_index" in metadata else 1)))
            except (TypeError, ValueError):
                pass

    formatted = []
    for number, reference in enumerate(references.values(), start=1):
        page_reference = _format_reference_values(reference["pages"], "Halaman")
        section_reference = _format_reference_values(reference["sections"], "Bagian")
        location = page_reference or section_reference
        if location:
            formatted.append(f"{number}. {reference['name']} ({location})")
        else:
            formatted.append(f"{number}. {reference['name']}")
    return "Sumber Rujukan: " + "; ".join(formatted) if formatted else ""


def _is_max_tokens_finish(reason: object) -> bool:
    text = str(reason or "").upper()
    return "MAX_TOKENS" in text or text == "LENGTH"


def _create_chat_model(model_name: str, api_key: str | None) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        retries=0,
        request_timeout=CHAT_MODEL_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=CHAT_MAX_OUTPUT_TOKENS,
    )


def _generate_complete_answer(
    model: ChatGoogleGenerativeAI,
    messages: Sequence[HumanMessage | SystemMessage],
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    """Generate an answer and keep continuing while Gemini stops at the token limit."""
    full_answer = ""
    conversation = list(messages)
    for _ in range(CHAT_MAX_CONTINUATIONS + 1):
        finish_reason = None
        parts: list[str] = []
        if on_token:
            for chunk in model.stream(conversation):
                text = _extract_stream_chunk_text(chunk.content)
                if text:
                    parts.append(text)
                    on_token(text)
                metadata = getattr(chunk, "response_metadata", None) or {}
                if metadata.get("finish_reason"):
                    finish_reason = metadata["finish_reason"]
        else:
            response = model.invoke(conversation)
            parts.append(_extract_stream_chunk_text(response.content))
            metadata = getattr(response, "response_metadata", None) or {}
            finish_reason = metadata.get("finish_reason")

        piece = "".join(parts)
        full_answer += piece
        if not piece.strip() or not _is_max_tokens_finish(finish_reason):
            break
        conversation = list(messages) + [
            AIMessage(content=full_answer),
            HumanMessage(content=CONTINUE_PROMPT),
        ]
    return full_answer.strip()


def _invoke_chat_model_messages(
    messages: Sequence[HumanMessage | SystemMessage],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    candidates = list(model_names or get_available_chat_models(api_key))
    last_model_error: Exception | None = None
    for model_name in candidates:
        try:
            model = _create_chat_model(model_name, api_key)
            answer = _generate_complete_answer(model, messages, on_token)
            if not answer:
                raise ValueError(f"Model {model_name} tidak menghasilkan teks jawaban.")
            return answer
        except Exception as error:
            if not _is_chat_model_fallback_error(error):
                raise
            if on_token:
                on_token(None)
            last_model_error = error

    if last_model_error is not None:
        raise last_model_error
    raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    api_key: str | None = None,
) -> str:
    """Transcribe a recorded voice message using an available Gemini model."""
    if not audio_bytes:
        raise ValueError("Rekaman suara kosong.")
    if not mime_type.startswith("audio/"):
        raise ValueError("Format rekaman suara tidak didukung.")

    model_names = [
        model_name
        for model_name in get_available_chat_models(api_key)
        if "gemini" in model_name.casefold()
    ]
    if not model_names:
        raise ValueError("Tidak ada model Gemini yang tersedia untuk memproses suara.")

    audio_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Transkripsikan rekaman suara secara akurat. Ikuti bahasa "
                    "yang digunakan pembicara dan keluarkan hanya transkripsinya."
                ),
            },
            {
                "type": "file",
                "source_type": "base64",
                "mime_type": mime_type,
                "data": base64.b64encode(audio_bytes).decode("ascii"),
            },
        ]
    )
    return _invoke_chat_model_messages([audio_message], api_key, model_names)


def answer_question(
    index: FAISS | None,
    question: str,
    api_key: str | None = None,
    media_parts: Sequence[dict[str, str]] | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_token: Callable[[str | None], None] | None = None,
    force_local_search: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """Answer from retrieved passages and any supplied image attachments."""
    if not question.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    _check_api_key(api_key)

    media_parts = media_parts or ()
    full_context_mode = is_full_context_request(question)
    retrieval_count = FULL_CONTEXT_CHUNK_COUNT if full_context_mode else 8
    keyword_fallback_reason: str | None = None
    anchor_vector_position: int | None = None
    matches: list[Document] = []
    if index is not None:
        if force_local_search:
            if on_progress:
                on_progress("Mencari di indeks lokal...")
            matches = _keyword_search(index, question, k=retrieval_count)
            keyword_fallback_reason = "sinkronisasi indeks sedang berlangsung"
        else:
            try:
                if on_progress:
                    on_progress("Menghubungi Gemini untuk embedding pertanyaan...")

                embedding_function = index.embedding_function
                if isinstance(embedding_function, _QuotaAwareEmbeddings):
                    query_vector = _get_chat_query_embeddings(api_key).embed_query(
                        question,
                    )
                elif isinstance(embedding_function, Embeddings):
                    query_vector = embedding_function.embed_query(question)
                else:
                    query_vector = index._embed_query(question)

                if on_progress:
                    on_progress(f"Mencari {retrieval_count} bagian yang relevan...")
                if full_context_mode:
                    matches, anchor_vector_position = _similarity_search_with_position(
                        index,
                        query_vector,
                        retrieval_count,
                    )
                else:
                    matches = index.similarity_search_by_vector(
                        query_vector,
                        k=retrieval_count,
                    )
            except Exception as error:
                if not _is_transient_embedding_error(error):
                    raise
                if on_progress:
                    on_progress("Mencari di indeks lokal...")
                matches = _keyword_search(index, question, k=retrieval_count)
                keyword_fallback_reason = (
                    "kuota embedding Gemini habis"
                    if is_embedding_quota_error(error)
                    else "layanan embedding Gemini sedang lambat/tidak tersedia"
                )

    if full_context_mode and index is not None:
        matches = _full_context_window(
            index,
            matches,
            anchor_vector_position=anchor_vector_position,
        )

    if not matches and not media_parts:
        if keyword_fallback_reason:
            return (
                f"Pencarian lokal tidak menemukan kata kunci yang cocok "
                f"({keyword_fallback_reason}). Coba gunakan istilah yang lebih spesifik.",
                [],
            )
        return "Saya belum menemukan informasi yang relevan di dokumen.", []

    context = "\n\n".join(
        f"[{document.metadata.get('source', 'Dokumen')}]\n{document.page_content}"
        for document in matches
    )
    completeness_instruction = (
        "Pengguna meminta isi lengkap. Pertahankan urutan konteks, gunakan semua "
        "bagian yang tersedia, dan jangan meringkas bait atau bagian yang "
        "diminta utuh. Jika konteks tidak memuat seluruh dokumen, jelaskan batasnya."
        if full_context_mode
        else "Sesuaikan panjang jawaban dengan pertanyaan. Pertanyaan sederhana dijawab singkat; jika isi yang diminta panjang (misalnya teks, bait, daftar, atau penjelasan rinci), tulis sampai tuntas dan jangan berhenti di tengah atau memotong jawaban."
    )
    source_line_instruction = (
        "Selalu pertahankan struktur baris asli dari dokumen rujukan. Jika teks "
        "rujukan tersusun dalam baris-baris pendek berurutan (seperti lirik lagu, "
        "nadhom, syair, pantun, doa, atau daftar), tampilkan kembali persis per "
        "baris sesuai aslinya dengan menambahkan dua spasi di akhir setiap baris "
        "(hard line break), meskipun pengguna hanya mengetik judul atau kata "
        "kunci singkat tanpa menyebut kata lagu/syair."
        if matches
        else ""
    )
    system_instruction = (
        "Anda adalah asisten pribadi yang menjawab dalam Bahasa Indonesia "
        "atau Bahasa Sunda, mengikuti bahasa pertanyaan pengguna. "
        + (
            "Gunakan konteks dokumen dan gambar terlampir jika relevan. Bedakan "
            "informasi dari dokumen dan pengamatan gambar; jika konteks dokumen "
            "tidak cukup, jelaskan batasnya. "
            if matches
            else "Tidak ada bagian dokumen yang relevan. Jika ada gambar terlampir, "
            "jawab berdasarkan pengamatan gambar dan jangan mengarang isi dokumen. "
        )
        + completeness_instruction
        + (" " + source_line_instruction if source_line_instruction else "")
    )
    answer = ""
    last_model_error: Exception | None = None
    if on_progress:
        on_progress("Menyusun jawaban dengan Gemini...")
    if media_parts:
        if on_progress:
            on_progress("Menyiapkan Gemini untuk memeriksa lampiran...")
        system_message = (
            f"{system_instruction}\n\nKonteks:\n{context or '(tidak tersedia)'}"
        )
        human_message = HumanMessage(
            content=[
                {"type": "text", "text": question},
                *media_parts,
            ]
        )
        answer = _invoke_chat_model_messages(
            [SystemMessage(content=system_message), human_message],
            api_key,
            on_token=on_token,
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system_instruction}\n\nKonteks:\n{{context}}"),
                ("human", "{question}"),
            ]
        )
        if on_progress:
            on_progress("Memeriksa model Gemini yang tersedia...")
        for model_name in get_available_chat_models(api_key):
            if on_progress:
                on_progress(f"Menunggu jawaban dari {model_name}...")
            try:
                model = _create_chat_model(model_name, api_key)
                prompt_messages = prompt.format_messages(
                    context=context or "(tidak tersedia)",
                    question=question,
                )
                answer = _generate_complete_answer(model, prompt_messages, on_token)
                if not answer:
                    raise ValueError(
                        f"Model {model_name} tidak menghasilkan teks jawaban."
                    )
                break
            except Exception as error:
                if not _is_chat_model_fallback_error(error):
                    raise
                if on_progress and _is_chat_quota_error(error):
                    on_progress(
                        f"Gemini membatasi permintaan ke {model_name}; mencoba model cadangan..."
                    )
                if on_token:
                    on_token(None)
                last_model_error = error

        if not answer:
            if last_model_error is not None:
                if _is_chat_quota_error(last_model_error):
                    raise ValueError(
                        "Gemini sedang membatasi permintaan (limit/kuota). "
                        "Tunggu sebentar lalu coba lagi."
                    ) from last_model_error
                raise last_model_error
            raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")
    if keyword_fallback_reason:
        answer = (
            f"Catatan: {keyword_fallback_reason}, jadi dokumen dicari berdasarkan "
            f"kecocokan kata pada indeks lokal.\n\n{answer}"
        )
    source_references = _format_source_references(matches)
    if source_references:
        answer = f"{answer.rstrip()}\n\n{source_references}"

    sources_by_id: dict[str, dict[str, str]] = {}
    manifest_sources = (load_index_manifest() or {}).get("sources", {})
    if not isinstance(manifest_sources, dict):
        manifest_sources = {}
    for document in matches:
        metadata = document.metadata
        file_id = str(metadata.get("file_id", ""))
        name = str(metadata.get("source", "Dokumen"))
        source_key = file_id or name
        if source_key in sources_by_id:
            continue
        manifest_source = manifest_sources.get(file_id, {})
        if not isinstance(manifest_source, dict):
            manifest_source = {}
        sources_by_id[source_key] = {
            "file_id": file_id,
            "name": name,
            "web_view_link": str(
                metadata.get("web_view_link")
                or metadata.get("webViewLink")
                or manifest_source.get("web_view_link")
                or manifest_source.get("webViewLink")
                or f"https://drive.google.com/file/d/{file_id}/view"
            ),
            "download_url": str(
                metadata.get("download_url")
                or manifest_source.get("download_url")
                or f"https://drive.google.com/uc?export=download&id={file_id}"
            ),
        }
    sources = list(sources_by_id.values())
    return answer, sources