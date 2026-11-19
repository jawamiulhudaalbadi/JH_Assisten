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
# Kuota berlaku per PROYEK Google Cloud (bukan per API key). Nilai di bawah bisa
# diubah lewat environment variable bila batas di AI Studio Anda berbeda
# (cek: AI Studio > Rate limits). Semua dihitung dengan margin keamanan.
EMBEDDING_FREE_TIER_RPM = _env_number("GEMINI_EMBED_RPM", 100)
EMBEDDING_FREE_TIER_TPM = _env_number("GEMINI_EMBED_TPM", 30_000)
EMBEDDING_SAFETY_FACTOR = 0.85
EMBEDDING_WINDOW_SECONDS = 60.0

# Anggaran per jendela 60 detik. Teks dalam satu batch DIHITUNG SATU PER SATU
# untuk RPM (laporan komunitas: batchEmbedContents berisi N teks = N request).
EMBEDDING_TOKENS_PER_MINUTE = int(EMBEDDING_FREE_TIER_TPM * EMBEDDING_SAFETY_FACTOR)
EMBEDDING_TEXTS_PER_MINUTE = int(EMBEDDING_FREE_TIER_RPM * EMBEDDING_SAFETY_FACTOR)
EMBEDDING_MAX_REQUESTS_PER_MINUTE = 30  # panggilan HTTP, hanya agar tidak "menembak" API
EMBEDDING_MAX_BATCH_SIZE = 100  # batas teks per panggilan API

# Batas per panggilan API ~20.000 token; dibatasi juga separuh anggaran TPM agar
# dua batch bisa muat dalam satu jendela menit dan jendela terisi merata.
EMBEDDING_TARGET_BATCH_TOKENS = min(18_000, EMBEDDING_TOKENS_PER_MINUTE // 2)
EMBEDDING_MAX_BATCH_CHARACTERS = EMBEDDING_TARGET_BATCH_TOKENS * 3  # estimasi 1 token = 3 karakter
EMBEDDING_MAX_TEXT_CHARACTERS = 6_000  # 1 teks maksimal 2.048 token; lebih dari itu dipotong diam-diam
EMBEDDING_MAX_QUOTA_RETRIES = 3  # retry untuk 429 per-menit
EMBEDDING_UNKNOWN_QUOTA_RETRIES = 1  # retry untuk 429 yang tidak jelas menit/harian
EMBEDDING_MIN_QUOTA_WAIT_SECONDS = 30.0
EMBEDDING_MAX_QUOTA_WAIT_SECONDS = 70.0
EMBEDDING_DEFAULT_QUOTA_RETRY_DELAY_SECONDS = EMBEDDING_WINDOW_SECONDS
EMBEDDING_MAX_AUTOMATIC_QUOTA_WAIT_SECONDS = 300.0

# --- Chunking ----------------------------------------------------------------
# Satu halaman PDF dipecah sendiri-sendiri (page_sections), jadi halaman <= 3.500
# karakter selalu menjadi tepat 1 chunk. Dimensi vektor tidak terpengaruh.
RAG_CHUNK_SIZE = _env_number("RAG_CHUNK_SIZE", 3_500)
RAG_CHUNK_OVERLAP = min(_env_number("RAG_CHUNK_OVERLAP", 350), RAG_CHUNK_SIZE // 4)
RAG_CLEAN_OCR = os.getenv("RAG_CLEAN_OCR", "").strip().lower() in {"1", "true", "yes"}
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

# Urutan model chat. Yang paling kiri dicoba lebih dulu. Bisa diganti tanpa
# mengubah kode lewat environment variable, contoh:
# GEMINI_CHAT_MODELS="gemini-2.0-flash,gemini-1.5-flash"
# Prioritas KUALITAS: model terbaru/terkuat dulu. Model "lite" hanya jaring
# pengaman terakhir bila semua model di atasnya sibuk atau gagal.
_DEFAULT_CHAT_MODELS = (
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash-8b",
)

FALLBACK_CHAT_MODELS = tuple(
    name.strip()
    for name in os.getenv("GEMINI_CHAT_MODELS", "").split(",")
    if name.strip()
) or _DEFAULT_CHAT_MODELS

PREFERRED_CHAT_MODEL_ORDER = {
    model: position for position, model in enumerate(FALLBACK_CHAT_MODELS)
}

# Model yang bukan untuk chat teks (gambar, riset, embedding, dsb.) dibuang dari
# daftar otomatis agar tidak membuang waktu dengan error 404/429.
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
)

# --- Balapan model (hedged request) -----------------------------------------
# Model teratas dimulai lebih dulu. Bila belum ada token pertama setelah
# CHAT_HEDGE_SECONDS, model berikutnya dijalankan BERSAMAAN (yang pertama tidak
# dibuang). Pemenangnya adalah yang lebih dulu mengeluarkan token; sisanya
# dibatalkan. Model yang error langsung diganti tanpa menunggu timeout.
CHAT_HEDGE_SECONDS = _env_number("CHAT_HEDGE_SECONDS", 4.0, float)
CHAT_MAX_PARALLEL = _env_number("CHAT_MAX_PARALLEL", 3)
CHAT_MAX_ATTEMPTS = _env_number("CHAT_MAX_ATTEMPTS", 6)

# Batas mutlak menunggu token pertama untuk satu model (ditambah menurut ukuran prompt).
CHAT_FIRST_TOKEN_TIMEOUT_SECONDS = _env_number("CHAT_FIRST_TOKEN_TIMEOUT", 15.0, float)
# Jeda maksimal antar potongan jawaban setelah jawaban mulai mengalir.
CHAT_CHUNK_TIMEOUT_SECONDS = _env_number("CHAT_CHUNK_TIMEOUT", 60.0, float)

# Tingkat "berpikir" Gemini: makin rendah makin cepat jawaban pertama muncul.
# Nilai: minimal | low | medium | high. Kosongkan untuk memakai bawaan Google.
CHAT_THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()

_CHAT_MODEL_CACHE: dict[str, tuple[float, tuple[str, ...]]] = {}
_CHAT_MODEL_CACHE_LOCK = Lock()
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
        # (waktu, token, jumlah teks)
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
        """Dipanggil saat 429: turunkan laju sementara."""
        with self._lock:
            self._throttle_factor = max(0.4, self._throttle_factor * 0.7)

    def reward(self) -> None:
        """Dipanggil saat sukses: pulihkan laju bertahap."""
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
                    # Bukan 429, atau limit HARIAN: berhenti segera (jangan buang request).
                    if kind is None or kind == "daily":
                        raise
                    _EMBEDDING_RATE_LIMITER.penalize()
                    if kind == "minute":
                        if minute_retries >= EMBEDDING_MAX_QUOTA_RETRIES:
                            raise
                        minute_retries += 1
                    else:  # 429 tanpa penanda jelas: coba sekali lagi, lalu berhenti aman
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
    """Kembalikan (ada_penanda_harian, ada_penanda_menit) dari seluruh rantai error."""
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
    """Bedakan jenis 429: 'daily' (RPD/TPD), 'minute' (RPM/TPM), 'unknown', atau None."""
    if not is_embedding_quota_error(error):
        return None
    daily, minute = _quota_message_markers(error)
    if daily:  # jika pesan memuat keduanya, harian yang menentukan
        return "daily"
    return "minute" if minute else "unknown"


def _quota_wait_seconds(error: Exception) -> float | None:
    """Jeda sebelum retry 429 non-harian: 30-70 detik; None bila terlalu lama."""
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
    """Kelompokkan teks: maks 100 teks/batch, dibatasi estimasi token & karakter."""
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
    client = _create_embedding_client(api_key, CHAT_QUERY_EMBEDDING_TIMEOUT_MS)
    return _QuotaAwareEmbeddings(client, limit_query_rate=False)


try:  # pembersih OCR opsional (aktif jika RAG_CLEAN_OCR=1)
    from services.text_cleaner import clean_ocr_text as _clean_ocr
except ImportError:  # pragma: no cover
    try:
        from services.text_cleaner import clean_ocr_text as _clean_ocr
    except ImportError:
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
    """Identitas isi chunk: (label halaman, SHA-256 teks)."""
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
                    # Vektor lama tetap valid; cukup segarkan metadata (waktu ubah, nomor chunk, tautan).
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
    # FAISS.add_documents memakai index.embedding_function (kwarg `embedding=` diabaikan),
    # jadi pasang sementara agar API key, limiter, dan callback progres yang benar terpakai.
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
# Batas atas jumlah chunk berurutan yang dikirim pada permintaan "lengkap".
# 12 chunk (~40 rb karakter) cukup untuk nadzom 100 bait. Jendela sebenarnya
# ADAPTIF: nadzom pendek hanya mengirim 4-6 chunk, jadi tetap cepat.
FULL_CONTEXT_CHUNK_COUNT = 12
FULL_CONTEXT_RETRIEVAL_COUNT = 10
FULL_CONTEXT_MIN_CHUNKS = 4


def is_full_context_request(question: str) -> bool:
    return bool(_FULL_CONTEXT_RE.search(question))


def _full_context_window(
    index: FAISS,
    matches: list[Document],
    window_size: int = FULL_CONTEXT_CHUNK_COUNT,
    anchor_vector_position: int | None = None,
) -> list[Document]:
    """Perluas hasil terbaik menjadi chunk berurutan dari dokumen yang sama.
    Jendela mengikuti sebaran hasil pencarian: hasil yang berdekatan (nadzom
    pendek) menghasilkan jendela kecil; hasil yang tersebar (nadzom panjang)
    menghasilkan jendela besar, maksimal window_size.
    """
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
    total = len(source_chunks)
    cap = max(1, min(window_size, total))
    low = max(0, min(hit_positions) - 1)
    high = min(total, max(hit_positions) + 3)
    minimum = min(FULL_CONTEXT_MIN_CHUNKS, total)
    if high - low < minimum:
        high = min(total, low + minimum)
        low = max(0, high - minimum)
    if high - low > cap:
        start = min(max(low, best_position - 3), total - cap)
        low, high = start, start + cap
    return [document for _, _, document in source_chunks[low:high]]


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
            if model_name and not _is_excluded_chat_model(model_name):
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
    return router.rank(model_names)


def _is_excluded_chat_model(model_name: str) -> bool:
    lowered = model_name.casefold()
    return any(marker in lowered for marker in EXCLUDED_CHAT_MODEL_MARKERS)


def _mark_chat_model_failed(model_name: str, error: Exception | str = "") -> None:
    """Catat kegagalan ke router (cooldown bertingkat; 404 dilewati lama)."""
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
                "408", "500", "502", "503", "504",
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


def _create_chat_model(
    model_name: str,
    api_key: str | None,
    max_output_tokens: int = CHAT_MAX_OUTPUT_TOKENS,
    thinking: bool = True,
) -> ChatGoogleGenerativeAI:
    kwargs = dict(
        model=model_name,
        temperature=0.2,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        retries=0,
        request_timeout=CHAT_MODEL_REQUEST_TIMEOUT_SECONDS,
        max_output_tokens=max_output_tokens,
    )
    if thinking and CHAT_THINKING_LEVEL and router.thinking_supported(model_name):
        try:
            return ChatGoogleGenerativeAI(**kwargs, thinking_level=CHAT_THINKING_LEVEL)
        except Exception:  # versi library tanpa parameter ini
            pass
    return ChatGoogleGenerativeAI(**kwargs)


# ==========================================================================
# Pembuatan jawaban: balapan model (hedged) + lanjutan otomatis.
# ==========================================================================
class _Attempt:
    """Satu percobaan memanggil satu model (berjalan di thread sendiri)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.cancel = threading.Event()
        self.started = time.monotonic()
        self.finished = False
        self.failed = False


class _WinnerFailed(Exception):
    """Model pemenang gagal setelah jawaban mulai mengalir."""

    def __init__(self, model_name: str, error: BaseException) -> None:
        super().__init__(str(error))
        self.model_name = model_name
        self.error = error


def _messages_size(messages: Sequence[object]) -> int:
    """Perkiraan jumlah karakter prompt (teks saja; lampiran dihitung kecil)."""
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
    """Berapa lama menunggu token pertama sebelum model cadangan ikut dijalankan."""
    expected = router.expected_ttft(model_name)
    if expected is None:
        base = CHAT_HEDGE_SECONDS
    else:
        base = min(max(expected * 1.8, 3.0), 8.0)
    return base + prompt_chars / 40_000


def _stream_with_continuation(
    model: ChatGoogleGenerativeAI,
    messages: Sequence[object],
    cancel: threading.Event,
):
    """Stream teks jawaban; otomatis lanjut bila terpotong batas token."""
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
                # Parameter "berpikir" ditolak model ini: ulangi TANPA parameter itu.
                if use_thinking and not emitted and "thinking" in str(error).casefold():
                    router.mark_thinking_unsupported(attempt.model_name)
                    use_thinking = False
                    continue
                raise
    except Exception as error:  # diteruskan ke thread utama
        put("error", error)


def _hedged_generate(
    messages: Sequence[object],
    api_key: str | None,
    candidates: Sequence[str],
    on_token: Callable[[str | None], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Jalankan model teratas; bila lambat, model cadangan ikut BERSAMAAN.
    Pemenang = model pertama yang mengeluarkan token. Mengembalikan
    (jawaban, nama_model). Dijalankan di thread pemanggil, sehingga callback
    on_token/on_progress aman dipakai dari Streamlit.
    """
    if not candidates:
        raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")
    prompt_chars = _messages_size(messages)
    attempt_limit = max(CHAT_FIRST_TOKEN_TIMEOUT_SECONDS, 10.0 + prompt_chars / 5_000)
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
                    raise _WinnerFailed(
                        winner.model_name,
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
                            # Model yang lebih diutamakan tetapi kalah balapan: catat lambat.
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
                    raise _WinnerFailed(attempt.model_name, payload)
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
    return "".join(parts).strip(), winner.model_name


def _generate_with_failover(
    messages: Sequence[object],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    """Balapan model + cadangan bila pemenang gagal di tengah jawaban."""
    candidates = list(model_names or get_available_chat_models(api_key))
    if not candidates:
        raise ValueError("Tidak ada model Gemini aktif yang mendukung generateContent.")
    last_error: BaseException | None = None
    for _ in range(2):
        try:
            return _hedged_generate(messages, api_key, candidates, on_token, on_progress)
        except _WinnerFailed as failure:
            if not _is_chat_model_fallback_error(failure.error):
                raise failure.error
            router.record_failure(failure.model_name, failure.error)
            last_error = failure.error
            candidates = [name for name in candidates if name != failure.model_name]
            if on_token:
                on_token(None)  # hapus jawaban setengah jadi sebelum mengulang
            if on_progress:
                on_progress(f"{failure.model_name} terputus; mengulang dengan model lain...")
            if not candidates:
                break
        except Exception as error:
            if _is_chat_model_fallback_error(error) or isinstance(error, TimeoutError):
                last_error = error
                break
            raise
    raise _friendly_model_error(last_error) from last_error


def _invoke_chat_model_messages(
    messages: Sequence[HumanMessage | SystemMessage],
    api_key: str | None,
    model_names: Sequence[str] | None = None,
    on_token: Callable[[str | None], None] | None = None,
) -> str:
    answer, _ = _generate_with_failover(messages, api_key, model_names, on_token)
    return answer


def start_model_warmup(api_key: str | None = None) -> None:
    """Ukur kecepatan model teratas di latar belakang (panggil sekali saat start)."""
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


# --------------------------------------------------------------------------
# Tata letak jawaban = tata letak sumber.
# --------------------------------------------------------------------------
_LEADING_WS_RE = re.compile(r"^[ \t]+")


def _line_key(line: str) -> str:
    return " ".join(line.split()).casefold()


def restore_source_indentation(answer: str, matches: list[Document]) -> str:
    """Samakan indentasi baris jawaban dengan baris yang sama di dokumen sumber."""
    indents: dict[str, str] = {}
    for document in matches:
        for line in document.page_content.splitlines():
            if not line.strip():
                continue
            found = _LEADING_WS_RE.match(line)
            indents.setdefault(
                _line_key(line), found.group(0).expandtabs(4) if found else ""
            )
    if not any(indents.values()):
        return answer
    restored = []
    for line in answer.splitlines():
        key = _line_key(line)
        if key in indents:
            line = indents[key] + line.lstrip()
        restored.append(line)
    return "\n".join(restored)


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
    retrieval_count = FULL_CONTEXT_RETRIEVAL_COUNT if full_context_mode else 8
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
        "Pengguna meminta isi lengkap. Konteks berisi beberapa bagian berurutan dari "
        "satu dokumen dan mungkin memuat nadhom/teks lain di sekitarnya. Tampilkan HANYA "
        "teks yang diminta, mulai dari judulnya sampai bait terakhirnya, tanpa terpotong, "
        "tanpa ringkasan, dan tanpa komentar tambahan. Salin persis seperti aslinya, "
        "termasuk urutan, ejaan, dan susunan barisnya."
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

    answer, _answering_model = _generate_with_failover(
        chat_messages,
        api_key,
        on_token=on_token,
        on_progress=on_progress,
    )

    if matches:
        answer = restore_source_indentation(answer, matches)
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