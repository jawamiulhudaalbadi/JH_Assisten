"""Build a semantic search index and answer questions with Gemini."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import queue
import re
import threading
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
try:  # model_router.py boleh diletakkan di folder yang sama dengan file ini
    from services.model_router import router
except ImportError:
    try:
        from .model_router import router
    except ImportError:
        from model_router import router

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "faiss_index"
MANIFEST_FILENAME = "drive_manifest.json"


def _env_number(name: str, default, cast=int):
    """Baca angka positif dari environment variable; pakai default jika kosong/salah."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# --- Kalibrasi kuota gemini-embedding-001 (tier gratis) ---------------------
EMBEDDING_FREE_TIER_RPM = _env_number("GEMINI_EMBED_RPM", 100)
EMBEDDING_FREE_TIER_TPM = _env_number("GEMINI_EMBED_TPM", 30_000)
EMBEDDING_SAFETY_FACTOR = 0.85
EMBEDDING_WINDOW_SECONDS = 60.0
EMBEDDING_TOKENS_PER_MINUTE = int(EMBEDDING_FREE_TIER_TPM * EMBEDDING_SAFETY_FACTOR)
EMBEDDING_TEXTS_PER_MINUTE = int(EMBEDDING_FREE_TIER_RPM * EMBEDDING_SAFETY_FACTOR)
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 30
EMBEDDING_MAX_BATCH_SIZE = 100
EMBEDDING_TARGET_BATCH_TOKENS = min(18_000, EMBEDDING_TOKENS_PER_MINUTE // 2)
EMBEDDING_MAX_BATCH_CHARACTERS = EMBEDDING_TARGET_BATCH_TOKENS * 3
EMBEDDING_MAX_TEXT_CHARACTERS = 6_000
EMBEDDING_MAX_QUOTA_RETRIES = 3
EMBEDDING_UNKNOWN_QUOTA_RETRIES = 1
EMBEDDING_MIN_QUOTA_WAIT_SECONDS = 30.0
EMBEDDING_MAX_QUOTA_WAIT_SECONDS = 70.0
EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS = EMBEDDING_WINDOW_SECONDS
EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS = 300.0
# --- Chunking ----------------------------------------------------------------
RAG_CHUNK_SIZE = _env_number("RAG_CHUNK_SIZE", 3_500)
RAG_CHUNK_OVERLAP = min(_env_number("RAG_CHUNK_OVERLAP", 350), RAG_CHUNK_SIZE // 4)
RAG_CLEAN_OCR = os.getenv("RAG_CLEAN_OCR", "").strip().lower() in {"1", "true", "yes"}
EMBEDDING_REQUEST_TIMEOUT_MS = 30_000
CHAT_QUERY_EMBEDDING_TIMEOUT_MS = 5_000
CHAT_MODEL_REQUEST_TIMEOUT_SECONDS = 180
CHAT_MAX_OUTPUT_TOKENS = 8192
CHAT_MAX_CONTINUATIONS = 5
CHAT_SAME_MODEL_RESUME_RETRIES = 2
CONTINUE_PROMPT = (
    "Lanjutkan jawaban di atas tepat dari titik terakhir yang terpotong. "
    "Jangan mengulang bagian yang sudah ditulis dan jangan membuat pembuka baru. "
    "Pertahankan format baris dan ejaan yang sama persis."
)
EMBEDDING_MAX_RETRIES = 0
MODEL_LIST_CACHE_SECONDS = 1800.0
_DEFAULT_CHAT_MODELS = (
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
)
FALLBACK_CHAT_MODELS = tuple(
    name.strip()
    for name in os.getenv("GEMINI_CHAT_MODELS", "").split(",")
    if name.strip()
) or _DEFAULT_CHAT_MODELS
PREFERRED_CHAT_MODEL_ORDER = {
    model: position for position, model in enumerate(FALLBACK_CHAT_MODELS)
}
EXCLUDED_CHAT_MODEL_MARKERS = (
    "image",
    "deep-research",
    "antigravity",
    "omni",
    "nano-banana",
    "tts",
    "embedding",
    "live",
    "aqa",
    "robotics",
    "computer-use",
    "veo",
    "imagen",
    "lyria",
    "learnlm",
)
# --- Balapan model (hedged request) -----------------------------------------
CHAT_HEDGE_SECONDS = _env_number("CHAT_HEDGE_SECONDS", 7.0, float)
CHAT_MAX_PARALLEL = _env_number("CHAT_MAX_PARALLEL", 2)
CHAT_MAX_ATTEMPTS = _env_number("CHAT_MAX_ATTEMPTS", 5)
CHAT_FIRST_TOKEN_TIMEOUT_SECONDS = _env_number("CHAT_FIRST_TOKEN_TIMEOUT", 18.0, float)
CHAT_CHUNK_TIMEOUT_SECONDS = _env_number("CHAT_CHUNK_TIMEOUT", 45.0, float)
CHAT_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()
_CHAT_MODEL_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_CHAT_MODEL_CACHE_LOCK = Lock()
_QUERY_EMBEDDING_CACHE: dict[str, _QuotaAwareEmbeddings] = {}
_QUERY_EMBEDDING_CACHE_LOCK = Lock()
EMBEDDING_DIMENSIONS = 768


def reduce_embedding_vector(vector: Sequence[float]) -> list[float]:
    """Potong vektor Gemini ke EMBEDDING_DIMENSIONS lalu normalisasi L2."""
    if len(vector) <= EMBEDDING_DIMENSIONS:
        return list(vector)
    reduced = np.asarray(vector[:EMBEDDING_DIMENSIONS], dtype=np.float32)
    norm = float(np.linalg.norm(reduced))
    if norm > 0:
        reduced = reduced / norm
    return reduced.tolist()


class _EmbeddingRateLimiter:
    """Jendela geser 60 detik: token (TPM), teks (RPM per teks), dan panggilan HTTP."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._events: deque[tuple[float, int, int]] = deque()
        self._throttle_factor = 1.0

    @staticmethod
    def _wait_for(
        events: deque[tuple[float, int, int]],
        value_of: Callable[[tuple[float, int, int]], int],
        limit: int,
        need: int,
        now: float,
    ) -> float:
        used = sum(value_of(event) for event in events)
        if used + need <= limit:
            return 0.0
        remaining = used
        for event in events:
            remaining -= value_of(event)
            if remaining + need <= limit:
                return max(0.0, event[0] + EMBEDDING_WINDOW_SECONDS - now)
        return EMBEDDING_WINDOW_SECONDS

    def penalize(self) -> None:
        with self._lock:
            self._throttle_factor = max(0.4, self._throttle_factor * 0.7)

    def reward(self) -> None:
        with self._lock:
            self._throttle_factor = min(1.0, self._throttle_factor + 0.05)

    def acquire(
        self,
        estimated_tokens: int,
        text_count: int = 1,
        on_wait: Callable[[float], None] | None = None,
    ) -> None:
        estimated_tokens = max(int(estimated_tokens), 1)
        text_count = max(int(text_count), 1)
        if estimated_tokens > EMBEDDING_TOKENS_PER_MINUTE:
            raise ValueError(
                "Estimasi token satu permintaan embedding melebihi anggaran aman "
                f"{EMBEDDING_TOKENS_PER_MINUTE} token/menit."
            )
        if text_count > EMBEDDING_TEXTS_PER_MINUTE:
            raise ValueError(
                "Jumlah teks satu permintaan embedding melebihi anggaran aman "
                f"{EMBEDDING_TEXTS_PER_MINUTE} teks/menit."
            )
        while True:
            with self._lock:
                now = time.monotonic()
                while (
                    self._events
                    and now - self._events[0][0] >= EMBEDDING_WINDOW_SECONDS
                ):
                    self._events.popleft()

                factor = self._throttle_factor
                token_limit = max(int(EMBEDDING_TOKENS_PER_MINUTE * factor), estimated_tokens)
                text_limit = max(int(EMBEDDING_TEXTS_PER_MINUTE * factor), text_count)
                wait_seconds = max(
                    self._wait_for(self._events, lambda e: e[1], token_limit, estimated_tokens, now),
                    self._wait_for(self._events, lambda e: e[2], text_limit, text_count, now),
                    self._wait_for(
                        self._events, lambda e: 1, EMBEDDING_MAX_REQUESTS_PER_MINUTE, 1, now
                    ),
                )
                if wait_seconds <= 0:
                    self._events.append((now, estimated_tokens, text_count))
                    return
                wait_seconds = max(0.1, wait_seconds)

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
            minute_retries = 0
            unknown_retries = 0
            while True:
                if self._on_progress:
                    self._on_progress(completed, len(texts))
                _EMBEDDING_RATE_LIMITER.acquire(
                    estimated_tokens, len(batch), self._on_wait
                )
                try:
                    batch_vectors = self._client.embed_documents(
                        batch,
                        batch_size=len(batch),
                    )
                except Exception as error:
                    kind = classify_embedding_quota_error(error)
                    if kind is None or kind == "daily":
                        raise
                    _EMBEDDING_RATE_LIMITER.penalize()
                    if kind == "minute":
                        if minute_retries >= EMBEDDING_MAX_QUOTA_RETRIES:
                            raise
                        minute_retries += 1
                    else:
                        if unknown_retries >= EMBEDDING_UNKNOWN_QUOTA_RETRIES:
                            raise
                        unknown_retries += 1
                    retry_delay = _quota_wait_seconds(error)
                    if retry_delay is None:
                        raise
                    if self._on_wait:
                        self._on_wait(retry_delay)
                    time.sleep(retry_delay)
                    continue
                _EMBEDDING_RATE_LIMITER.reward()
                vectors.extend(reduce_embedding_vector(vector) for vector in batch_vectors)
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
                1,
                self._on_wait if on_wait is None else on_wait,
            )
        return reduce_embedding_vector(self._client.embed_query(text))


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


def _quota_message_markers(error: Exception) -> tuple[bool, bool]:
    current: BaseException | None = error
    visited: set[int] = set()
    daily = minute = False
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        compact = re.sub(r"[^a-z0-9]", "", str(current).casefold())
        if "perday" in compact or "daily" in compact or "requestsperday" in compact:
            daily = True
        if "perminute" in compact or "permin" in compact:
            minute = True
        current = current.__cause__ or current.__context__
    return daily, minute


def classify_embedding_quota_error(error: Exception) -> str | None:
    if not is_embedding_quota_error(error):
        return None
    daily, minute = _quota_message_markers(error)
    if daily:
        return "daily"
    return "minute" if minute else "unknown"


def _quota_wait_seconds(error: Exception) -> float | None:
    parsed = get_quota_retry_delay_seconds(error)
    if parsed is None:
        return EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS
    if parsed > EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS:
        return None
    return min(
        max(parsed + 2.0, EMBEDDING_MIN_QUOTA_WAIT_SECONDS),
        EMBEDDING_MAX_QUOTA_WAIT_SECONDS,
    )


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
        character_count = len(text)
        if character_count > EMBEDDING_MAX_TEXT_CHARACTERS:
            raise ValueError(
                f"Satu chunk berisi {character_count} karakter; batas aman "
                f"{EMBEDDING_MAX_TEXT_CHARACTERS} (model memotong teks di 2.048 token). "
                "Kecilkan RAG_CHUNK_SIZE."
            )
        estimated_tokens = max(1, character_count // 3)
        if current_batch and (
            len(current_batch) >= EMBEDDING_MAX_BATCH_SIZE
            or current_characters + character_count > EMBEDDING_MAX_BATCH_CHARACTERS
            or current_tokens + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS
        ):
            batches.append((current_batch, current_tokens))
            current_batch = []
            current_tokens = 0
            current_characters = 0

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
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    with _QUERY_EMBEDDING_CACHE_LOCK:
        cached = _QUERY_EMBEDDING_CACHE.get(cache_key)
        if cached is not None:
            return cached
        client = _create_embedding_client(resolved_api_key, CHAT_QUERY_EMBEDDING_TIMEOUT_MS)
        wrapper = _QuotaAwareEmbeddings(client, limit_query_rate=False)
        _QUERY_EMBEDDING_CACHE[cache_key] = wrapper
        return wrapper


try:  # pembersih OCR opsional (aktif jika RAG_CLEAN_OCR=1)
    from services.text_cleaner import clean_ocr_text as _clean_ocr
except ImportError:  # pragma: no cover
    def _clean_ocr(text: str) -> str:
        return text


def _to_langchain_documents(
    source_documents: Sequence[DriveDocument],
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
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
            if RAG_CLEAN_OCR:
                section_text = _clean_ocr(section_text)
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


def _chunk_fingerprint(chunk: Document) -> tuple[str, str]:
    return (
        str(chunk.metadata.get("page_label", "")),
        hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest(),
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
    """Ganti chunk file yang berubah; chunk dengan isi identik dipakai ulang (tanpa embedding)."""
    chunks = _to_langchain_documents(source_documents)
    if not chunks and index is None:
        return None

    chunks_by_file: dict[str, list[Document]] = {}
    for chunk in chunks:
        file_id = str(chunk.metadata.get("file_id", ""))
        chunks_by_file.setdefault(file_id, []).append(chunk)

    reused_chunk_ids: set[int] = set()
    if index is not None and replaced_file_ids:
        stored_by_file: dict[str, dict[tuple[str, str], list[str]]] = {}
        for document_id in list(index.index_to_docstore_id.values()):
            stored_document = index.docstore.search(document_id)
            if not isinstance(stored_document, Document):
                continue
            file_id = str(stored_document.metadata.get("file_id", ""))
            if file_id not in replaced_file_ids:
                continue
            stored_by_file.setdefault(file_id, {}).setdefault(
                _chunk_fingerprint(stored_document), []
            ).append(document_id)

        stale_ids: list[str] = []
        for file_id, buckets in stored_by_file.items():
            for chunk in chunks_by_file.get(file_id, []):
                candidates = buckets.get(_chunk_fingerprint(chunk))
                if not candidates:
                    continue
                stored_document = index.docstore.search(candidates.pop(0))
                if isinstance(stored_document, Document):
                    stored_document.metadata.update(chunk.metadata)
                    reused_chunk_ids.add(id(chunk))
            for leftover_ids in buckets.values():
                stale_ids.extend(leftover_ids)
        if stale_ids:
            index.delete(ids=stale_ids)

    pending_chunks = [chunk for chunk in chunks if id(chunk) not in reused_chunk_ids]
    if not pending_chunks:
        return index

    total_pending = len(pending_chunks)
    completed = 0
    batch_offset = 0
    batch_total = 0

    def report_embedding_progress(batch_done: int, current_batch_total: int) -> None:
        if current_batch_total != batch_total:
            raise RuntimeError("Embedding progress does not match its batch.")
        if on_progress:
            on_progress(batch_offset + batch_done, total_pending)

    embeddings = _get_embeddings(api_key, on_wait, report_embedding_progress)
    original_embedding_function = index.embedding_function if index is not None else None
    if index is not None:
        index.embedding_function = embeddings
    try:
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
                original_embedding_function = index.embedding_function
            else:
                index.add_documents(batch_chunks)
            completed += len(batch_chunks)
            if on_checkpoint:
                on_checkpoint(index)
    finally:
        if index is not None and original_embedding_function is not None:
            index.embedding_function = original_embedding_function
    return index


def get_index_file_manifest(index: FAISS | None) -> dict[str, str]:
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


_QUERY_STOPWORDS = {
    "apa", "apakah", "bagaimana", "mengapa", "kenapa", "siapa", "kapan", "dimana",
    "tolong", "coba", "mohon", "bisa", "minta", "tampilkan", "tuliskan", "bacakan",
    "jelaskan", "uraikan", "terangkan", "bahas", "lengkap", "semua", "utuh", "full",
    "tentang", "mengenai", "dari", "yang", "dan", "atau", "untuk", "pada", "dalam",
    "dengan", "secara", "detail", "rinci", "lirik", "nadzom", "nadhom", "syair",
    "bait", "dangding", "pupuh", "lagu", "sholawat", "shalawat", "qasidah", "qosidah",
    "sama", "sampai", "selesai", "seluruhnya", "seluruh", "itu", "ini", "tersebut",
}


def _keyword_search(index: FAISS, question: str, k: int = 5) -> list[Document]:
    """Rank indexed chunks locally when Gemini embeddings or chat models are unavailable."""
    raw_terms = [
        term
        for term in re.findall(r"\w+", question.casefold(), flags=re.UNICODE)
        if len(term) > 2
    ]
    meaningful_terms = {term for term in raw_terms if term not in _QUERY_STOPWORDS}
    query_terms = meaningful_terms or set(raw_terms)
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
        source_name = str(document.metadata.get("source", "")).casefold()
        title_hits = sum(1 for term in query_terms if term in source_name)
        if title_hits:
            score += title_hits * 2.5
        if score:
            ranked_documents.append((score, position, document))

    ranked_documents.sort(key=lambda result: (-result[0], result[1]))
    return [document for _, _, document in ranked_documents[:k]]


_FULL_CONTEXT_RE = re.compile(
    r"\b(?:lengkap|sampai\s+selesai|seluruh(?:nya)?|utuh|semua(?:\s+bait)?|full|"
    r"bacakan|tuliskan|tampilkan|lirik|bait|nadhom|nadzom|syair|dangding|pupuh|"
    r"sholawat|shalawat|qasidah|qosidah|lagu)\b",
    re.IGNORECASE,
)
_DISCUSSION_RE = re.compile(
    r"\b(?:jelas(?:kan|in)?|urai(?:kan)?|terang(?:kan)?|bahas(?:kan)?|diskusi(?:kan)?|"
    r"persoalan|masalah|hukum(?:nya)?|makna|arti(?:nya)?|maksud(?:nya)?|tafsir|kandungan|"
    r"hikmah|syarat|rukun|dalil|alasan|penyebab|sebab|dampak|solusi|cara|langkah|"
    r"bagaimana|mengapa|kenapa|perbedaan|banding(?:kan)?|analisis|kesimpulan|"
    r"detail|rinci|apa\s+(?:itu|saja|maksud|hukum|pendapat))\b",
    re.IGNORECASE,
)
_JUDUL_MARKER_RE = re.compile(r"(?:^|\n)\s*JUDUL\s*:", re.IGNORECASE)

FULL_CONTEXT_CHUNK_COUNT = 14
FULL_CONTEXT_RETRIEVAL_COUNT = 10
DISCUSSION_RETRIEVAL_COUNT = 9
DISCUSSION_MAX_CHUNKS = 14


STRICT_READING_RULES = (
    "ATURAN MEMBACA SUMBER (WAJIB UNTUK SEMUA TOPIK; jika pengguna meminta menyalin "
    "teks, salin persis tanpa mengubahnya). "
    "1) Jawab HANYA dari konteks. Jangan menambah fakta, angka, hukum, atau kesimpulan "
    "yang tidak tertulis. Jika tidak ada di konteks, katakan tidak ditemukan. "
    "2) Konteks disusun berurutan sesuai dokumen. Baca dari awal sampai akhir tanpa "
    "melompati bagian. Teks asli beserta terjemahannya adalah satu kesatuan. "
    "3) Jika sumber membagi sesuatu ke dalam kategori, jenis, bab, atau poin, setiap "
    "bagian punya definisi, contoh, dan ketentuannya sendiri. Jangan memindahkan "
    "ketentuan satu bagian ke bagian lain dan jangan menggabungkan dua bagian yang berbeda. "
    "4) Pisahkan sifat/status suatu hal dari ketentuan penggunaannya atau keterangan "
    "tambahannya. Keterangan tambahan tidak mengubah statusnya. "
    "5) Sebelum menjawab, periksa tiap butir jawaban: bagian mana dari konteks yang "
    "mendukungnya, dan apakah tidak tertukar dengan bagian lain. "
    "6) Jika konteks saling bertentangan atau tidak lengkap, sebutkan apa adanya. "
    "CONTOH POLA PENALARAN: "
    "(a) Fikih: air musyammas (dipanaskan matahari dalam wadah logam selain emas dan "
    "perak) berstatus suci menyucikan, hukum penggunaan di badan makruh. Ia TIDAK masuk "
    "suci tidak menyucikan. Air musta'mal (bekas bersuci wajib) berstatus suci tidak "
    "menyucikan, itu kategori lain. Status zat (suci menyucikan, suci tidak menyucikan, "
    "najis) harus dipisah dari hukum penggunaan (mubah, makruh, haram). "
    "(b) Umum: jika dokumen menyebut produk A berstatus aktif tetapi dibatasi, jawab "
    "keduanya: A aktif, dengan pembatasannya. Jangan menulis A nonaktif. "
    "Jika redaksi konteks berbeda dari contoh, ikuti konteks. "
    "Untuk pertanyaan klasifikasi, gunakan urutan: kategori, status, ketentuan, rujukan."
)


def _order_matches_for_reading(matches: list[Document]) -> list[Document]:
    file_order: dict[str, int] = {}
    for document in matches:
        file_order.setdefault(str(document.metadata.get("file_id", "")), len(file_order))

    def key(document: Document) -> tuple[int, int]:
        try:
            chunk_index = int(document.metadata.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = 10**9
        return (file_order[str(document.metadata.get("file_id", ""))], chunk_index)

    return sorted(matches, key=key)


def is_full_context_request(question: str) -> bool:
    return bool(_FULL_CONTEXT_RE.search(question))


def is_discussion_request(question: str) -> bool:
    return bool(_DISCUSSION_RE.search(question))


def _ordered_file_chunks(
    index: FAISS,
    file_id: str,
) -> list[tuple[int, int, Document]]:
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
    source_chunks.sort(key=lambda item: (item[0], item[1]))
    return source_chunks


def _expand_discussion_neighbors(
    index: FAISS,
    matches: list[Document],
    max_total: int = DISCUSSION_MAX_CHUNKS,
) -> list[Document]:
    """Tambahkan chunk tetangga (±1) dari hit teratas agar pembahasan kitab tidak terpotong."""
    if not matches:
        return []
    expanded: list[Document] = list(matches)
    seen_keys: set[tuple[str, int | None, str]] = {
        (
            str(doc.metadata.get("file_id", "")),
            doc.metadata.get("chunk_index"),
            doc.page_content[:80],
        )
        for doc in expanded
    }
    cache_by_file: dict[str, list[tuple[int, int, Document]]] = {}
    for top_doc in matches[:2]:
        file_id = str(top_doc.metadata.get("file_id", ""))
        if not file_id:
            continue
        source_chunks = cache_by_file.setdefault(
            file_id, _ordered_file_chunks(index, file_id)
        )
        pos = next(
            (i for i, (_, _, d) in enumerate(source_chunks) if d is top_doc),
            None,
        )
        if pos is None:
            top_idx = top_doc.metadata.get("chunk_index")
            pos = next(
                (i for i, (ci, _, _) in enumerate(source_chunks) if ci == top_idx),
                None,
            )
        if pos is None:
            continue
        for offset in (1, -1):
            neighbor_pos = pos + offset
            if 0 <= neighbor_pos < len(source_chunks):
                neighbor = source_chunks[neighbor_pos][2]
                key = (
                    file_id,
                    neighbor.metadata.get("chunk_index"),
                    neighbor.page_content[:80],
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    expanded.append(neighbor)
                    if len(expanded) >= max_total:
                        return expanded
    return expanded[:max_total]


def _full_context_window(
    index: FAISS,
    matches: list[Document],
    window_size: int = FULL_CONTEXT_CHUNK_COUNT,
    anchor_vector_position: int | None = None,
) -> list[Document]:
    """Otomatis menyesuaikan jumlah chunk: sedikit untuk nadzom pendek, banyak untuk nadzom panjang."""
    if not matches:
        return []
    file_id = str(matches[0].metadata.get("file_id", ""))
    if not file_id:
        return matches

    source_chunks = _ordered_file_chunks(index, file_id)
    if not source_chunks:
        return matches

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

    total = len(source_chunks)
    cap = max(1, min(window_size, total))

    judul_positions = [
        i
        for i, (_, _, doc) in enumerate(source_chunks)
        if _JUDUL_MARKER_RE.search(doc.page_content)
    ]
    if len(judul_positions) <= 1 and total <= cap:
        return [document for _, _, document in source_chunks]

    if len(judul_positions) > 1:
        section_start = max(
            (p for p in judul_positions if p <= best_position),
            default=0,
        )
        section_end = min(
            (p for p in judul_positions if p > best_position),
            default=total,
        )
        if section_end - section_start <= cap:
            return [document for _, _, document in source_chunks[section_start:section_end]]
        low = max(section_start, best_position - 2)
        high = min(section_end, low + cap)
        return [document for _, _, document in source_chunks[low:high]]

    position_by_object = {
        id(document): position
        for position, (_, _, document) in enumerate(source_chunks)
    }
    hit_positions = [best_position]
    for other in matches[1:]:
        if str(other.metadata.get("file_id", "")) != file_id:
            continue
        position = position_by_object.get(id(other))
        if position is not None:
            hit_positions.append(position)

    low = max(0, min(hit_positions) - 1)
    high = min(total, max(hit_positions) + 2)
    if high - low > cap:
        start = min(max(low, best_position - 3), total - cap)
        low, high = start, start + cap
    return [document for _, _, document in source_chunks[low:high]]


def _similarity_search_with_position(
    index: FAISS,
    query_vector: list[float],
    k: int,
) -> tuple[list[Document], int | None]:
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


def _model_sort_key(name: str) -> tuple:
    lowered = name.casefold()
    if name in PREFERRED_CHAT_MODEL_ORDER:
        return (0, PREFERRED_CHAT_MODEL_ORDER[name], 0, 0, lowered)
    version_match = re.search(r"gemini-(\d+)(?:\.(\d+))?", lowered)
    major = int(version_match.group(1)) if version_match else 0
    minor = int(version_match.group(2)) if version_match and version_match.group(2) else 0
    is_flash = 0 if "flash" in lowered else 1
    is_lite = 1 if "lite" in lowered else 0
    is_preview = 1 if ("preview" in lowered or "exp" in lowered) else 0
    return (1, is_flash, is_lite, -major, -minor, is_preview, lowered)


def get_available_chat_models(api_key: str | None = None) -> list[str]:
    """Discover and cache Gemini models that support generateContent."""
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
    cache_key = hashlib.sha256(resolved_api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    with _CHAT_MODEL_CACHE_LOCK:
        cached = _CHAT_MODEL_CACHE.get(cache_key)
        if cached and now - cached[0] < MODEL_LIST_CACHE_SECONDS:
            return router.rank(cached[1])

    discovered: set[str] = set()
    client = None
    try:
        client = genai.Client(api_key=resolved_api_key)
        for model in client.models.list(config={"query_base": True}):
            actions = getattr(model, "supported_actions", None) or []
            if "generateContent" not in actions:
                continue
            model_name = str(getattr(model, "name", "")).rsplit("/", 1)[-1]
            if (
                model_name
                and "gemini" in model_name.casefold()
                and not _is_excluded_chat_model(model_name)
            ):
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
    model_names = tuple(sorted(candidates, key=_model_sort_key))
    with _CHAT_MODEL_CACHE_LOCK:
        _CHAT_MODEL_CACHE[cache_key] = (time.monotonic(), model_names)
    return router.rank(model_names)


def _is_excluded_chat_model(model_name: str) -> bool:
    lowered = model_name.casefold()
    return any(marker in lowered for marker in EXCLUDED_CHAT_MODEL_MARKERS)


def _mark_chat_model_failed(model_name: str, error: Exception | str = "") -> None:
    router.record_failure(model_name, error)


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
            404, 408, 425, 429, 500, 502, 503, 504,
            "404", "408", "425", "429", "500", "502", "503", "504",
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
                "thinking_level",
                "thinking level",
                "thinking_budget",
                "temporarily unavailable",
                "service unavailable",
                "timed out",
                "timeout",
                "connection error",
                "connection reset",
                "network error",
                "overloaded",
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


def _is_chat_quota_error(error: BaseException) -> bool:
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
                reference["sections"].append(
                    str(int(section_index) + (0 if "section_index" in metadata else 1))
                )
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


CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.0") or 0.0)


def _chat_temperature(model_name: str) -> float:
    return 1.0 if "gemini-3" in model_name.casefold() else CHAT_TEMPERATURE


def _create_chat_model(
    model_name: str,
    api_key: str | None,
    max_output_tokens: int = CHAT_MAX_OUTPUT_TOKENS,
    thinking: bool = True,
) -> ChatGoogleGenerativeAI:
    kwargs = dict(
        model=model_name,
        temperature=_chat_temperature(model_name),
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        retries=0,
        request_timeout=CHAT_MODEL_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=max_output_tokens,
    )
    if thinking and router.thinking_supported(model_name):
        lowered = model_name.casefold()
        if "gemini-2.5-flash" in lowered:
            try:
                return ChatGoogleGenerativeAI(**kwargs, thinking_budget=0)
            except Exception:
                pass
        elif "gemini-3" in lowered and CHAT_THINKING_LEVEL:
            try:
                return ChatGoogleGenerativeAI(**kwargs, thinking_level=CHAT_THINKING_LEVEL)
            except Exception:
                pass
    return ChatGoogleGenerativeAI(**kwargs)


# ==========================================================================
# Pembuatan jawaban: balapan model + lanjutkan di model yang sama.
# ==========================================================================
class _Attempt:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.cancel = threading.Event()
        self.started = time.monotonic()
        self.finished = False
        self.failed = False


class _WinnerInterrupted(Exception):
    """Model pemenang terputus setelah sebagian jawaban sempat mengalir."""

    def __init__(self, model_name: str, partial_text: str, error: BaseException) -> None:
        super().__init__(str(error))
        self.model_name = model_name
        self.partial_text = partial_text
        self.error = error


class _AllChatModelsFailedError(RuntimeError):
    """Seluruh model Gemini gagal atau kehabisan kuota."""

    def __init__(self, is_quota: bool, cause: BaseException | None = None) -> None:
        message = (
            "Gemini sedang membatasi permintaan (limit/kuota). "
            "Tunggu sebentar lalu coba lagi."
            if is_quota
            else "Semua model Gemini sedang sibuk atau mengalami gangguan."
        )
        super().__init__(message)
        self.is_quota = is_quota
        self.cause = cause


def _messages_size(messages: Sequence[object]) -> int:
    total = 0
    for message in messages:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, Sequence):
            for part in content:
                if isinstance(part, str):
                    total += len(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
                else:
                    total += 2_000
    return total


def _hedge_delay(model_name: str, prompt_chars: int) -> float:
    expected = router.expected_ttft(model_name)
    if expected is None:
        base = CHAT_HEDGE_SECONDS
    else:
        base = min(max(expected * 2.0, 5.0), 10.0)
    return base + prompt_chars / 35_000


def _stream_with_continuation(
    model: ChatGoogleGenerativeAI,
    messages: Sequence[object],
    cancel: threading.Event,
):
    conversation = list(messages)
    full_answer = ""
    for _ in range(CHAT_MAX_CONTINUATIONS + 1):
        finish_reason = None
        parts: list[str] = []
        for chunk in model.stream(conversation):
            if cancel.is_set():
                return
            text = _extract_stream_chunk_text(chunk.content)
            if text:
                parts.append(text)
                yield text
            metadata = getattr(chunk, "response_metadata", None) or {}
            if metadata.get("finish_reason"):
                finish_reason = metadata["finish_reason"]
        piece = "".join(parts)
        full_answer += piece
        if not piece.strip() or not _is_max_tokens_finish(finish_reason):
            return
        conversation = list(messages) + [
            AIMessage(content=full_answer),
            HumanMessage(content=CONTINUE_PROMPT),
        ]


def _attempt_worker(
    attempt: _Attempt,
    events: queue.Queue,
    messages: Sequence[object],
    api_key: str | None,
) -> None:
    def put(kind: str, payload: object = None) -> None:
        events.put((attempt, kind, payload))

    use_thinking = True
    try:
        while True:
            model = _create_chat_model(attempt.model_name, api_key, thinking=use_thinking)
            emitted = False
            try:
                for text in _stream_with_continuation(model, messages, attempt.cancel):
                    emitted = True
                    put("chunk", text)
                put("done")
                return
            except Exception as error:
                if use_thinking and not emitted and "thinking" in str(error).casefold():
                    router.mark_thinking_unsupported(attempt.model_name)
                    use_thinking = False
                    continue
                raise
    except Exception as error:
        put("error", error)


def _hedged_generate(
    messages: Sequence[object],
    api_key: str | None,
    candidates: Sequence[str],
    on_token: Callable[[str | None], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    if not candidates:
        raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")

    prompt_chars = _messages_size(messages)
    attempt_limit = max(CHAT_FIRST_TOKEN_TIMEOUT_SECONDS, 12.0 + prompt_chars / 5_000)
    events: queue.Queue = queue.Queue()
    attempts: list[_Attempt] = []
    next_candidate = 0
    last_error: BaseException | None = None
    winner: _Attempt | None = None
    parts: list[str] = []

    def launch() -> bool:
        nonlocal next_candidate
        if next_candidate >= min(len(candidates), CHAT_MAX_ATTEMPTS):
            return False
        name = candidates[next_candidate]
        next_candidate += 1
        attempt = _Attempt(name)
        if on_progress:
            if attempts:
                on_progress(f"Menunggu {attempts[-1].model_name}; mencoba {name} bersamaan...")
            else:
                on_progress(f"Menunggu jawaban dari {name}...")
        attempts.append(attempt)
        threading.Thread(
            target=_attempt_worker,
            args=(attempt, events, messages, api_key),
            daemon=True,
        ).start()
        return True

    launch()
    last_launch = time.monotonic()
    try:
        while True:
            if winner is None:
                now = time.monotonic()
                for item in attempts:
                    if not item.finished and now - item.started > attempt_limit:
                        item.finished = item.failed = True
                        item.cancel.set()
                        last_error = TimeoutError(
                            f"timed out: {item.model_name} tidak merespons dalam "
                            f"{attempt_limit:.0f} detik"
                        )
                        router.record_failure(item.model_name, last_error)
                active = [item for item in attempts if not item.finished]
                if not active:
                    if not launch():
                        raise last_error or ValueError(
                            "Semua model Gemini tidak menghasilkan jawaban."
                        )
                    last_launch = now
                elif (
                    len(active) < CHAT_MAX_PARALLEL
                    and now - last_launch >= _hedge_delay(active[-1].model_name, prompt_chars)
                ):
                    if launch():
                        last_launch = now
                wait = 0.25
            else:
                wait = CHAT_CHUNK_TIMEOUT_SECONDS

            try:
                attempt, kind, payload = events.get(timeout=wait)
            except queue.Empty:
                if winner is not None:
                    raise _WinnerInterrupted(
                        winner.model_name,
                        "".join(parts),
                        TimeoutError("timed out: jawaban berhenti mengalir"),
                    )
                continue

            if attempt.failed or (winner is not None and attempt is not winner):
                continue

            if kind == "chunk":
                if winner is None:
                    winner = attempt
                    now = time.monotonic()
                    winner_position = attempts.index(winner)
                    for position, other in enumerate(attempts):
                        if other is winner:
                            continue
                        if position < winner_position and not other.finished:
                            router.record_slow(other.model_name, (now - other.started) * 2.5)
                        other.cancel.set()
                    router.record_success(winner.model_name, now - winner.started)
                    if on_progress:
                        on_progress(f"Dijawab oleh {winner.model_name}")
                parts.append(payload)
                if on_token:
                    on_token(payload)
            elif kind == "done":
                if attempt is winner:
                    break
                attempt.finished = attempt.failed = True
                last_error = ValueError(
                    f"Model {attempt.model_name} tidak menghasilkan teks jawaban."
                )
            elif kind == "error":
                if attempt is winner:
                    raise _WinnerInterrupted(attempt.model_name, "".join(parts), payload)
                attempt.finished = attempt.failed = True
                if not _is_chat_model_fallback_error(payload):
                    raise payload
                router.record_failure(attempt.model_name, payload)
                last_error = payload
                if on_progress:
                    if _is_chat_quota_error(payload):
                        on_progress(f"Gemini membatasi {attempt.model_name}; beralih ke model lain...")
                    else:
                        on_progress(f"{attempt.model_name} bermasalah; beralih ke model lain...")
                if launch():
                    last_launch = time.monotonic()
    finally:
        for item in attempts:
            item.cancel.set()

    return "".join(parts).lstrip("\r\n").rstrip(), winner.model_name


def _continue_partial_answer(
    messages: Sequence[object],
    partial_text: str,
    model_name: str,
    api_key: str | None,
    candidates: Sequence[str],
    on_token: Callable[[str | None], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Lanjutkan jawaban yang terputus di tengah TANPA menghapus atau ganti model dulu."""
    accumulated = partial_text
    ordered_models = [model_name] * CHAT_SAME_MODEL_RESUME_RETRIES + [
        name for name in candidates if name != model_name
    ]
    last_error: BaseException | None = None
    for index, target_model in enumerate(ordered_models):
        if index > 0 and target_model == model_name:
            time.sleep(1.5)
        elif target_model != model_name and on_progress:
            on_progress(f"{model_name} terputus permanen; melanjutkan dengan {target_model}...")

        continuation_messages = list(messages) + [
            AIMessage(content=accumulated),
            HumanMessage(content=CONTINUE_PROMPT),
        ]
        cancel = threading.Event()
        try:
            model = _create_chat_model(target_model, api_key, thinking=True)
            for text in _stream_with_continuation(model, continuation_messages, cancel):
                accumulated += text
                if on_token:
                    on_token(text)
            return accumulated.lstrip("\r\n").rstrip(), target_model
        except Exception as error:
            last_error = error
            if not _is_chat_model_fallback_error(error):
                raise
            if target_model != model_name:
                router.record_failure(target_model, error)

    if accumulated.strip():
        return accumulated.lstrip("\r\n").rstrip(), model_name
    raise _AllChatModelsFailedError(
        is_quota=last_error is not None and _is_chat_quota_error(last_error),
        cause=last_error,
    ) from last_error


def _generate_with_failover(
    messages: Sequence[object],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    candidates = list(model_names or get_available_chat_models(api_key))
    if not candidates:
        raise _AllChatModelsFailedError(is_quota=False)

    last_error: BaseException | None = None
    for _ in range(2):
        try:
            return _hedged_generate(messages, api_key, candidates, on_token, on_progress)
        except _WinnerInterrupted as interrupted:
            if not _is_chat_model_fallback_error(interrupted.error):
                raise interrupted.error
            if interrupted.partial_text.strip():
                if on_progress:
                    on_progress(
                        f"Koneksi {interrupted.model_name} terputus sebentar; melanjutkan jawaban..."
                    )
                return _continue_partial_answer(
                    messages,
                    interrupted.partial_text,
                    interrupted.model_name,
                    api_key,
                    candidates,
                    on_token=on_token,
                    on_progress=on_progress,
                )
            router.record_failure(interrupted.model_name, interrupted.error)
            last_error = interrupted.error
            candidates = [name for name in candidates if name != interrupted.model_name]
            if not candidates:
                break
        except Exception as error:
            if _is_chat_model_fallback_error(error) or isinstance(error, TimeoutError):
                last_error = error
                break
            raise
    raise _AllChatModelsFailedError(
        is_quota=last_error is not None and _is_chat_quota_error(last_error),
        cause=last_error,
    ) from last_error


def _invoke_chat_model_messages(
    messages: Sequence[HumanMessage | SystemMessage],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    try:
        answer, _ = _generate_with_failover(messages, api_key, model_names, on_token)
        return answer
    except _AllChatModelsFailedError as error:
        raise ValueError(str(error)) from error


def start_model_warmup(api_key: str | None = None) -> None:
    def probe(model_name: str) -> float:
        model = _create_chat_model(model_name, api_key, max_output_tokens=16)
        started = time.monotonic()
        for _ in model.stream([HumanMessage(content="Balas: ok")]):
            return time.monotonic() - started
        raise ValueError("tidak ada respons")

    def run() -> None:
        try:
            router.warm_up(get_available_chat_models(api_key), probe)
        except Exception:
            pass

    threading.Thread(target=run, daemon=True).start()


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str,
    api_key: str | None = None,
) -> str:
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


# --------------------------------------------------------------------------
# Tata letak jawaban = tata letak sumber (tanpa kotak putih kode Markdown).
# --------------------------------------------------------------------------
_LEADING_WS_RE = re.compile(r"^[ \t]+")
_MD_LIST_LINE_RE = re.compile(r"^[ \t]{0,3}(?:[-*+]|\d+\.)\s+")
_MD_BLOCK_PREFIX_RE = re.compile(r"^(?:#{1,6}\s+|>\s*|```|~~~|[-*_]{3,}\s*|\|)")
_FENCE_LINE_RE = re.compile(r"^\s*(?:```|~~~)[a-zA-Z0-9_-]*\s*$")


def _line_key(line: str) -> str:
    cleaned = line.strip()
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = re.sub(r"^\*\*(.+)\*\*$", r"\1", cleaned)
    return " ".join(cleaned.split()).casefold()


def _spaces_to_nbsp(whitespace: str) -> str:
    return whitespace.expandtabs(4).replace(" ", "\u00a0")


def _strip_accidental_markdown(line: str, source_lines: dict[str, str]) -> str:
    key = _line_key(line)
    source_line = source_lines.get(key)
    if source_line is None:
        return line
    stripped = line.lstrip()
    if stripped.startswith("#") and not source_line.lstrip().startswith("#"):
        stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    if (
        stripped.startswith("**")
        and stripped.endswith("**")
        and not source_line.lstrip().startswith("**")
    ):
        stripped = stripped[2:-2]
    return stripped


class _StreamingMarkdownLineSanitizer:
    """Ubah spasi awal baris menjadi NBSP dan jaga line-break selama streaming."""

    def __init__(self, target: Callable[[str | None], None]) -> None:
        self._target = target
        self._line_buf = ""
        self._header_flushed = False
        self._is_first_line = True

    def __call__(self, token: str | None) -> None:
        if token is None:
            self._line_buf = ""
            self._header_flushed = False
            self._is_first_line = True
            self._target(None)
            return

        out: list[str] = []
        for ch in token:
            if ch == "\n":
                if not self._header_flushed:
                    line = self._flush_prefix_with_guard(self._line_buf)
                    self._line_buf = ""
                else:
                    line = ""
                self._header_flushed = False
                self._is_first_line = False
                if line and not line.endswith("  "):
                    out.append(line + "  \n")
                else:
                    out.append(line + "\n")
                continue

            if not self._header_flushed:
                self._line_buf += ch
                if len(self._line_buf) >= 12 or (
                    self._line_buf.strip()
                    and not self._line_buf.lstrip().startswith(("`", "~", "-", "*", "+"))
                    and not re.match(r"^\s*\d+\.?$", self._line_buf)
                ):
                    out.append(self._flush_prefix_with_guard(self._line_buf))
                    self._line_buf = ""
                    self._header_flushed = True
                    self._is_first_line = False
            else:
                out.append(ch)

        if out:
            self._target("".join(out))

    def flush(self) -> None:
        if self._line_buf:
            self._target(self._flush_prefix_with_guard(self._line_buf))
            self._line_buf = ""
            self._header_flushed = True
            self._is_first_line = False

    def _flush_prefix_with_guard(self, prefix: str) -> str:
        formatted = self._flush_line_prefix(prefix)
        if self._is_first_line and formatted.startswith("\u00a0"):
            return "\u200b" + formatted
        return formatted

    @staticmethod
    def _flush_line_prefix(prefix: str) -> str:
        if _FENCE_LINE_RE.match(prefix):
            return ""
        if _MD_LIST_LINE_RE.match(prefix):
            return prefix
        match = _LEADING_WS_RE.match(prefix)
        if not match:
            return prefix
        return _spaces_to_nbsp(match.group(0)) + prefix[match.end():]


def restore_source_indentation(answer: str, matches: list[Document]) -> str:
    """Samakan indentasi & baris jawaban dengan sumber tanpa memicu blok kode putih."""
    indents: dict[str, str] = {}
    source_lines: dict[str, str] = {}
    source_has_fences = False
    for document in matches:
        if "```" in document.page_content or "~~~" in document.page_content:
            source_has_fences = True
        for line in document.page_content.splitlines():
            if not line.strip():
                continue
            key = _line_key(line)
            found = _LEADING_WS_RE.match(line)
            indents.setdefault(key, found.group(0).expandtabs(4) if found else "")
            source_lines.setdefault(key, line)

    lines = answer.splitlines()
    if not source_has_fences:
        lines = [line for line in lines if not _FENCE_LINE_RE.match(line)]

    normalized: list[str] = []
    for line in lines:
        if not line.strip():
            normalized.append("")
            continue
        key = _line_key(line)
        if key in indents:
            body = _strip_accidental_markdown(line, source_lines).lstrip()
            normalized.append(_spaces_to_nbsp(indents[key]) + body)
            continue
        if _MD_LIST_LINE_RE.match(line):
            normalized.append(line)
            continue
        found = _LEADING_WS_RE.match(line)
        if found:
            normalized.append(_spaces_to_nbsp(found.group(0)) + line[found.end():])
        else:
            normalized.append(line)

    result: list[str] = []
    total = len(normalized)
    for i, line in enumerate(normalized):
        if (
            line.strip()
            and i + 1 < total
            and normalized[i + 1].strip()
            and not _MD_BLOCK_PREFIX_RE.match(line.lstrip())
            and not _MD_LIST_LINE_RE.match(line)
            and not _MD_BLOCK_PREFIX_RE.match(normalized[i + 1].lstrip())
            and not _MD_LIST_LINE_RE.match(normalized[i + 1])
            and not line.endswith("  ")
        ):
            result.append(line + "  ")
        else:
            result.append(line)

    output = "\n".join(result)
    # Lindungi spasi awal baris pertama dari pemotongan .strip() di pemanggil (app.py).
    if output.startswith("\u00a0"):
        output = "\u200b" + output
    return output


def _stitch_sequential_chunks(matches: list[Document]) -> str:
    """Gabungkan potongan chunk berurutan dan hilangkan overlap baris di sambungan."""
    if not matches:
        return ""
    merged_lines = matches[0].page_content.splitlines()
    for document in matches[1:]:
        next_lines = document.page_content.splitlines()
        overlap = 0
        max_check = min(len(merged_lines), len(next_lines), 20)
        for size in range(max_check, 0, -1):
            if merged_lines[-size:] == next_lines[:size]:
                overlap = size
                break
        if overlap:
            merged_lines.extend(next_lines[overlap:])
        else:
            if merged_lines and merged_lines[-1].strip():
                merged_lines.append("")
            merged_lines.extend(next_lines)
    return "\n".join(merged_lines).rstrip()


def _build_local_fallback_answer(
    matches: list[Document],
    full_context_mode: bool,
    reason_note: str,
) -> str:
    """Susun jawaban langsung dari isi dokumen lokal ketika AI tidak dapat dipakai."""
    if full_context_mode and len({str(d.metadata.get("file_id", "")) for d in matches}) == 1:
        body = _stitch_sequential_chunks(matches)
    else:
        sections: list[str] = []
        for idx, doc in enumerate(matches[:5], start=1):
            source = str(doc.metadata.get("source", "Dokumen"))
            page = doc.metadata.get("page_label")
            section_idx = doc.metadata.get("section_index", doc.metadata.get("chunk_index"))
            loc = (
                f"Halaman {page}"
                if page is not None
                else (
                    f"Bagian {int(section_idx) + (0 if 'section_index' in doc.metadata else 1)}"
                    if section_idx is not None
                    else ""
                )
            )
            header = f"**[{idx}. {source}" + (f" — {loc}" if loc else "") + "]**"
            sections.append(f"{header}\n{doc.page_content.strip()}")
        body = "\n\n".join(sections)

    body = restore_source_indentation(body, matches)
    return f"Catatan: {reason_note}\n\n{body}".rstrip()


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
    discussion_mode = is_discussion_request(question)
    if full_context_mode:
        retrieval_count = FULL_CONTEXT_RETRIEVAL_COUNT
    elif discussion_mode:
        retrieval_count = DISCUSSION_RETRIEVAL_COUNT
    else:
        retrieval_count = 6

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
    elif discussion_mode and index is not None and matches:
        matches = _expand_discussion_neighbors(index, matches)
    if not full_context_mode and matches:
        matches = _order_matches_for_reading(matches)

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

    if full_context_mode and discussion_mode:
        completeness_instruction = (
            "Pengguna meminta teks sekaligus penjelasan/pembahasannya. Pertama, tampilkan teks "
            "rujukan yang diminta secara utuh (tanpa terpotong dan sesuai susunan baris aslinya). "
            "Setelah itu, berikan penjelasan/pembahasan yang mendalam dan terstruktur sedetail "
            "mungkin mencakup maksud, poin-poin penting, dalil/dasar yang ada di dokumen, serta "
            "hal penting lain yang perlu diketahui pengguna."
        )
    elif full_context_mode:
        completeness_instruction = (
            "Pengguna meminta isi lengkap. Konteks berisi beberapa bagian berurutan dari "
            "satu dokumen dan mungkin memuat nadhom/teks lain di sekitarnya. Tampilkan HANYA "
            "teks yang diminta, mulai dari judulnya sampai bait terakhirnya, tanpa terpotong, "
            "tanpa ringkasan, dan tanpa komentar tambahan. Salin persis seperti aslinya, "
            "termasuk urutan, ejaan, dan susunan barisnya."
        )
    elif discussion_mode:
        completeness_instruction = (
            "Pengguna menanyakan persoalan atau berdiskusi. Berikan jawaban yang mendalam, "
            "terstruktur, dan sedetail mungkin berdasarkan dokumen rujukan agar semua hal "
            "penting yang perlu diketahui tersampaikan dengan jelas: (1) jawaban/inti persoalan, "
            "(2) rincian poin penting, syarat, sebab-akibat, atau tahapan dari sumber, "
            "(3) kutipan, dalil, atau contoh yang tercantum di dokumen jika ada, dan "
            "(4) batasan jika ada bagian yang tidak dibahas di dokumen. Jangan berhenti di tengah."
        )
    else:
        completeness_instruction = (
            "Sesuaikan panjang jawaban dengan pertanyaan. Pertanyaan sederhana dijawab jelas "
            "dan tepat; jika pertanyaan menyangkut penjelasan, daftar, bait, atau persoalan "
            "yang luas, jawab secara rinci sampai tuntas dan jangan memotong hal penting."
        )

    source_line_instruction = (
        "Setiap kali menampilkan teks lirik lagu, nadhom, syair, pantun, doa, atau kutipan "
        "berbaris pendek dari dokumen, selalu pertahankan struktur baris dan indentasi aslinya "
        "persis per baris. JANGAN bungkus dengan blok kode (```), dan JANGAN tambahkan format "
        "tebal (**) atau heading (#) yang tidak ada di teks sumber."
        if matches
        else ""
    )
    system_instruction = (
        "Anda adalah asisten pribadi yang secara default menjawab dalam Bahasa Sunda "
        "yang sopan dan halus (basa lemes). Gunakan Bahasa Indonesia HANYA jika pengguna "
        "secara jelas meminta jawaban dalam Bahasa Indonesia atau bahasa lain. "
        "Istilah Arab/kitab, nama kitab, dan kutipan teks sumber tetap ditulis apa adanya "
        "tanpa diterjemahkan ulang. "
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
        + (" " + STRICT_READING_RULES if matches else "")
    )

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
        chat_messages: list = [SystemMessage(content=system_message), human_message]
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{system_instruction}\n\nKonteks:\n{{context}}"),
                ("human", "{question}"),
            ]
        )
        chat_messages = prompt.format_messages(
            context=context or "(tidak tersedia)",
            question=question,
        )

    stream_sanitizer = _StreamingMarkdownLineSanitizer(on_token) if on_token else None
    local_only_mode = False
    try:
        answer, _answering_model = _generate_with_failover(
            chat_messages,
            api_key,
            on_token=stream_sanitizer,
            on_progress=on_progress,
        )
        if stream_sanitizer is not None:
            stream_sanitizer.flush()
    except _AllChatModelsFailedError as chat_error:
        if index is not None and not matches:
            matches = _keyword_search(index, question, k=retrieval_count)
            if full_context_mode and matches:
                matches = _full_context_window(index, matches)
        if not matches:
            reason = (
                "limit/kuota semua model Gemini sedang habis"
                if chat_error.is_quota
                else "semua model Gemini sedang mengalami error/gangguan"
            )
            return (
                f"Maaf, {reason}, dan pencarian lokal tidak menemukan teks dokumen yang cocok.",
                [],
            )
        local_only_mode = True
        reason_note = (
            "Limit/kuota semua model Gemini sedang habis, jadi hasil ini diambil langsung "
            "dengan pencarian lokal dari dokumen (tanpa pemrosesan AI)."
            if chat_error.is_quota
            else "Semua model Gemini sedang mengalami error/gangguan, jadi hasil ini diambil "
            "langsung dengan pencarian lokal dari dokumen (tanpa pemrosesan AI)."
        )
        if on_progress:
            on_progress("Semua model Gemini tidak tersedia; menampilkan hasil pencarian lokal...")
        answer = _build_local_fallback_answer(matches, full_context_mode, reason_note)
        if on_token:
            on_token(None)
            on_token(answer)

    if matches and not local_only_mode:
        answer = restore_source_indentation(answer, matches)
    if keyword_fallback_reason and not local_only_mode:
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