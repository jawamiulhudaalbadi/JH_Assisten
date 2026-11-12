"""Build a semantic search index and answer questions with Gemini."""

from __future__ import annotations

import json
import hashlib
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
EMBEDDING_MAX_QUOTA_RETRIES = 2
EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS = EMBEDDING_WINDOW_SECONDS
EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS = 300.0
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
    ) -> None:
        self._client = client
        self._on_wait = on_wait

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch, estimated_tokens in _make_embedding_batches(texts):
            retries = 0
            while True:
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
                break
        return vectors

    def embed_query(self, text: str) -> list[float]:
        _EMBEDDING_RATE_LIMITER.acquire(
            (len(text.encode("utf-8")) + 3) // 4 + 8,
            self._on_wait,
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

    for text in texts:
        estimated_tokens = max(1, len(text) // 3)
        if current_batch and (
            len(current_batch) >= EMBEDDING_MAX_BATCH_SIZE
            or current_tokens + estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS
        ):
            batches.append((current_batch, current_tokens))
            current_batch = []
            current_tokens = 0
        if estimated_tokens > EMBEDDING_TARGET_BATCH_TOKENS:
            raise ValueError(
                "Մեկ embedding chunk-ի գնահատված token-ների քանակը գերազանցում է "
                f"{EMBEDDING_TARGET_BATCH_TOKENS}-ի անվտանգ սահմանը։"
            )

        current_batch.append(text)
        current_tokens += estimated_tokens

    if current_batch:
        batches.append((current_batch, current_tokens))
    return batches


def _check_api_key(api_key: str | None = None) -> None:
    if not (api_key or os.getenv("GOOGLE_API_KEY")):
        raise ValueError("GOOGLE_API_KEY belum diatur.")


def _get_embeddings(
    api_key: str | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> Embeddings:
    _check_api_key(api_key)
    resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
    client = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=resolved_api_key,
    )
    client.client = genai.Client(
        api_key=resolved_api_key,
        http_options=genai.types.HttpOptions(
            retry_options=genai.types.HttpRetryOptions(attempts=1),
        ),
    )
    return _QuotaAwareEmbeddings(client, on_wait=on_wait)


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
) -> FAISS | None:
    """Replace changed file chunks and append new chunks to an existing index."""
    chunks = _to_langchain_documents(source_documents)
    if index is None:
        if not chunks:
            return None
        return FAISS.from_documents(chunks, _get_embeddings(api_key, on_wait))

    if replaced_file_ids:
        stale_ids = []
        for document_id in index.index_to_docstore_id.values():
            stored_document = index.docstore.search(document_id)
            if (
                isinstance(stored_document, Document)
                and stored_document.metadata.get("file_id") in replaced_file_ids
            ):
                stale_ids.append(document_id)
        if stale_ids:
            index.delete(ids=stale_ids)

    if chunks:
        index.add_documents(chunks, embedding=_get_embeddings(api_key, on_wait))
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
    r"\b(?:lengkap|sampai\s+selesai|seluruh(?:nya)?|utuh|semua(?:\s+bait)?|full)\b",
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
    question: str,
    k: int,
) -> tuple[list[Document], int | None]:
    """Search FAISS while retaining the exact vector row for the best hit."""
    query_vector = np.asarray([index._embed_query(question)], dtype=np.float32)
    if index._normalize_L2:
        faiss.normalize_L2(query_vector)
    _, vector_ids = index.index.search(query_vector, k)
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
        if status in {404, 429, 503, "404", "429", "503"}:
            return True
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
                "503",
            )
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


def answer_question(
    index: FAISS,
    question: str,
    api_key: str | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Retrieve relevant passages and answer in Indonesian or Sundanese."""
    if not question.strip():
        raise ValueError("Pertanyaan tidak boleh kosong.")
    _check_api_key(api_key)

    full_context_mode = is_full_context_request(question)
    retrieval_count = FULL_CONTEXT_CHUNK_COUNT if full_context_mode else 5
    used_keyword_fallback = False
    anchor_vector_position: int | None = None
    try:
        if full_context_mode:
            matches, anchor_vector_position = _similarity_search_with_position(
                index,
                question,
                retrieval_count,
            )
        else:
            matches = index.similarity_search(question, k=retrieval_count)
    except Exception as error:
        if not is_embedding_quota_error(error):
            raise
        matches = _keyword_search(index, question, k=retrieval_count)
        used_keyword_fallback = True

    if full_context_mode:
        matches = _full_context_window(
            index,
            matches,
            anchor_vector_position=anchor_vector_position,
        )

    if not matches:
        if used_keyword_fallback:
            return (
                "Kuota embedding Gemini sedang habis dan kata kunci pertanyaan "
                "tidak ditemukan pada indeks lokal. Coba gunakan istilah yang "
                "lebih spesifik atau tunggu kuota pulih.",
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
        else "Jawab ringkas dan fokus pada bagian yang paling relevan."
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Anda adalah asisten pribadi yang menjawab dalam Bahasa Indonesia "
                "atau Bahasa Sunda, mengikuti bahasa pertanyaan pengguna. "
                "Gunakan hanya informasi pada konteks. Jika konteks tidak cukup, "
                "katakan dengan jujur bahwa jawabannya tidak ditemukan. "
                f"{completeness_instruction}\n\nKonteks:\n{{context}}",
            ),
            ("human", "{question}"),
        ]
    )
    answer = ""
    last_model_error: Exception | None = None
    for model_name in get_available_chat_models(api_key):
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0.2,
                google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                retries=0,
            )
            response = (prompt | model).invoke(
                {"context": context, "question": question}
            )
            answer = _extract_response_text(response.content)
            if not answer:
                raise ValueError(f"Model {model_name} tidak menghasilkan teks jawaban.")
            break
        except Exception as error:
            if not _is_chat_model_fallback_error(error):
                raise
            last_model_error = error

    if not answer:
        if last_model_error is not None:
            raise last_model_error
        raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")
    if used_keyword_fallback:
        answer = (
            "Catatan: kuota embedding Gemini habis, jadi dokumen dicari "
            "berdasarkan kecocokan kata pada indeks lokal.\n\n" + answer
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
