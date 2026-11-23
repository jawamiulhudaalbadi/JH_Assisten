"""Penyimpanan riwayat dan statistik pengguna di Firestore."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)
MAX_MESSAGES_PER_CHUNK = 20
HISTORY_WARNING_MESSAGES = 200
HISTORY_WARNING_CHUNKS = 10
_WRITE_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="firestore-write")


def credentials_json_from_secrets(secrets: Mapping[str, Any]) -> str | None:
    """Validasi kredensial service account dari Streamlit Secrets."""
    raw = secrets.get("FIREBASE_SERVICE_ACCOUNT")
    if isinstance(raw, Mapping):
        return json.dumps(dict(raw), sort_keys=True)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("FIREBASE_SERVICE_ACCOUNT bukan JSON yang valid.")
            return None
        if isinstance(parsed, dict):
            return json.dumps(parsed, sort_keys=True)
    return None


@st.cache_resource(show_spinner=False)
def get_firestore_client(credentials_json: str | None) -> Any | None:
    """Buat klien database default satu kali; impor Firestore dilakukan malas."""
    if not credentials_json:
        return None
    try:
        from google.cloud import firestore
        from google.oauth2 import service_account

        service_account_info = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )
        return firestore.Client(
            project=service_account_info["project_id"],
            credentials=credentials,
        )
    except Exception:
        logger.exception("Klien Firestore tidak dapat dibuat.")
        return None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _chunk_number(document_id: str) -> int:
    try:
        return int(document_id)
    except ValueError:
        return 0


def _stats_uid(uid: str) -> str:
    """Ringkas UID agar dokumen statistik tetap kecil dan anonim."""
    return hashlib.sha256(uid.encode("utf-8")).hexdigest()[:12]


def _transaction_snapshot(transaction: Any, document_ref: Any) -> Any:
    """Ambil snapshot tunggal dari hasil iterator transaction.get()."""
    snapshot = next(iter(transaction.get(document_ref)), None)
    if snapshot is None:
        raise RuntimeError("Pembacaan dokumen Firestore tidak mengembalikan snapshot.")
    return snapshot


def append_message_pair(
    current_messages: list[dict[str, Any]],
    question: dict[str, Any],
    answer: dict[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Tambahkan satu pasang pesan tanpa membuat chunk melebihi batas."""
    if len(current_messages) + 2 > MAX_MESSAGES_PER_CHUNK:
        return [question, answer], True
    return [*current_messages, question, answer], False


def load_chat_history(
    client: Any | None, uid: str, chat_id: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Muat maksimal dua chunk terakhir dan ringkasan hitungan pengguna."""
    if client is None:
        return [], {"total_messages": 0, "total_chunks": 0, "active_chunk_count": 0}
    try:
        user_ref = client.collection("users").document(uid)
        user_data = user_ref.get().to_dict() or {}
        chunks = (
            user_ref.collection("chats")
            .document(chat_id)
            .collection("chunks")
            .order_by("__name__", direction="DESCENDING")
            .limit(2)
            .stream()
        )
        recent_chunks = list(chunks)
        recent_chunks.sort(key=lambda item: _chunk_number(item.id))
        messages: list[dict[str, Any]] = []
        for chunk in recent_chunks:
            chunk_data = chunk.to_dict() or {}
            chunk_messages = chunk_data.get("messages", [])
            if isinstance(chunk_messages, list):
                messages.extend(
                    message for message in chunk_messages if isinstance(message, dict)
                )
        active_chunk_count = (
            int((recent_chunks[-1].to_dict() or {}).get("count", 0))
            if recent_chunks
            else 0
        )
        return messages[-30:], {
            "total_messages": int(user_data.get("total_messages", 0)),
            "total_chunks": int(user_data.get("total_chunks", 0)),
            "active_chunk_count": active_chunk_count,
        }
    except Exception:
        logger.exception("Riwayat Firestore gagal dimuat untuk perangkat %s.", uid[:6])
        return [], {"total_messages": 0, "total_chunks": 0, "active_chunk_count": 0}


def get_active_chat_id(client: Any | None, uid: str) -> str | None:
    """Baca obrolan terakhir dari dokumen perangkat."""
    if client is None:
        return None
    try:
        user_data = client.collection("users").document(uid).get().to_dict() or {}
        chat_id = user_data.get("active_chat_id")
        return str(chat_id) if chat_id else None
    except Exception:
        logger.exception("Chat aktif gagal dimuat untuk perangkat %s.", uid[:6])
        return None


def enqueue_question_answer(
    client: Any | None,
    uid: str,
    chat_id: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
) -> None:
    """Jadwalkan satu penulisan pasangan pesan di thread latar."""
    if client is None:
        return
    _WRITE_EXECUTOR.submit(
        _persist_question_answer,
        client,
        uid,
        chat_id,
        question_text,
        answer_text,
        sources,
    )


def _persist_question_answer(
    client: Any,
    uid: str,
    chat_id: str,
    question_text: str,
    answer_text: str,
    sources: list[dict[str, Any]],
) -> None:
    try:
        from google.cloud import firestore

        user_ref = client.collection("users").document(uid)
        chunks_ref = (
            user_ref.collection("chats").document(chat_id).collection("chunks")
        )
        now = _now()
        stats_uid = _stats_uid(uid)
        question = {
            "role": "user",
            "content": question_text,
            "timestamp": now,
            "model_used": None,
            "sources": [],
        }
        answer = {
            "role": "assistant",
            "content": answer_text,
            "timestamp": now,
            "model_used": None,
            "sources": sources,
        }

        @firestore.transactional
        def save_pair(transaction: Any) -> None:
            user_snapshot = _transaction_snapshot(transaction, user_ref)
            query = chunks_ref.order_by(
                "__name__", direction=firestore.Query.DESCENDING
            ).limit(1)
            latest_chunks = list(transaction.get(query))
            current_chunk = latest_chunks[0] if latest_chunks else None
            chunk_number = (
                _chunk_number(current_chunk.id) if current_chunk is not None else 0
            )
            current_messages = (
                (current_chunk.to_dict() or {}).get("messages", [])
                if current_chunk is not None
                else []
            )
            if not isinstance(current_messages, list):
                current_messages = []
            next_messages, needs_new_chunk = append_message_pair(
                current_messages, question, answer
            )
            if needs_new_chunk or current_chunk is None:
                chunk_number += 1
            chunk_ref = chunks_ref.document(f"{chunk_number:06d}")
            day = now.strftime("%Y%m%d")
            user_data = user_snapshot.to_dict() or {}
            user_fields: dict[str, Any] = {
                "last_active": now,
                "active_chat_id": chat_id,
                "active_chunk_number": chunk_number,
                "active_chunk_count": len(next_messages),
                "total_questions": firestore.Increment(1),
                "total_messages": firestore.Increment(2),
                "total_chunks": firestore.Increment(
                    1 if needs_new_chunk or current_chunk is None else 0
                ),
                "day": day,
                "day_count": (
                    int(user_data.get("day_count", 0)) + 1
                    if user_data.get("day") == day
                    else 1
                ),
            }
            if not user_snapshot.exists:
                user_fields.update({"first_seen": now, "blocked": False})

            transaction.set(
                chunk_ref,
                {"messages": next_messages, "count": len(next_messages)},
            )
            transaction.set(user_ref, user_fields, merge=True)
            transaction.set(
                client.collection("stats").document(f"daily_{day}"),
                {
                    "questions": firestore.Increment(1),
                    "uids": firestore.ArrayUnion([stats_uid]),
                },
                merge=True,
            )
            transaction.set(
                client.collection("stats").document(now.strftime("hourly_%Y%m%d%H")),
                {"uids": firestore.ArrayUnion([stats_uid])},
                merge=True,
            )
            if not user_snapshot.exists:
                transaction.set(
                    client.collection("stats").document("global"),
                    {"total_devices": firestore.Increment(1)},
                    merge=True,
                )

        save_pair(client.transaction())
    except Exception:
        logger.exception("Riwayat gagal disimpan untuk perangkat %s.", uid[:6])


def mark_active(client: Any | None, uid: str) -> None:
    """Catat aktivitas paling banyak sekali per jam per sesi."""
    if client is None:
        return
    now = _now()
    previous = st.session_state.get("last_active_written_at")
    if isinstance(previous, datetime) and (now - previous).total_seconds() < 3600:
        return
    st.session_state.last_active_written_at = now
    _WRITE_EXECUTOR.submit(_mark_active, client, uid, now)


def _mark_active(client: Any, uid: str, now: datetime) -> None:
    try:
        from google.cloud import firestore

        user_ref = client.collection("users").document(uid)

        @firestore.transactional
        def update_user(transaction: Any) -> None:
            user_snapshot = _transaction_snapshot(transaction, user_ref)
            fields: dict[str, Any] = {
                "last_active": now,
            }
            if not user_snapshot.exists:
                fields.update(
                    {
                        "first_seen": now,
                        "total_questions": 0,
                        "total_messages": 0,
                        "total_chunks": 0,
                        "blocked": False,
                        "day": now.strftime("%Y%m%d"),
                        "day_count": 0,
                    }
                )
                transaction.set(
                    client.collection("stats").document("global"),
                    {"total_devices": firestore.Increment(1)},
                    merge=True,
                )
            transaction.set(user_ref, fields, merge=True)

        update_user(client.transaction())
    except Exception:
        logger.exception("Aktivitas gagal dicatat untuk perangkat %s.", uid[:6])


def clear_user_history(client: Any | None, uid: str) -> None:
    """Jadwalkan penghapusan semua chunk tanpa menghapus statistik FAQ."""
    if client is None:
        return
    _WRITE_EXECUTOR.submit(_clear_user_history, client, uid)


def _clear_user_history(client: Any, uid: str) -> None:
    try:
        user_ref = client.collection("users").document(uid)
        chats = user_ref.collection("chats").stream()
        batch = client.batch()
        pending = 0
        for chat in chats:
            for chunk in chat.reference.collection("chunks").stream():
                batch.delete(chunk.reference)
                pending += 1
                if pending == 450:
                    batch.commit()
                    batch = client.batch()
                    pending = 0
        batch.set(
            user_ref,
            {
                "total_questions": 0,
                "total_messages": 0,
                "total_chunks": 0,
            },
            merge=True,
        )
        batch.commit()
    except Exception:
        logger.exception("Riwayat gagal dihapus untuk perangkat %s.", uid[:6])
