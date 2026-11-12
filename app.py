"""Streamlit UI for the personal AI assistant."""

from __future__ import annotations

import ast
import base64
import html
import json
from datetime import datetime, timezone
from math import ceil
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import streamlit as st
from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError
from googleapiclient.errors import HttpError
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai._common import GoogleGenerativeAIError
from oauthlib.oauth2.rfc6749.errors import OAuth2Error
from requests.exceptions import RequestException

from services.ai_service import (
    answer_question,
    find_source_link_matches,
    is_full_context_request,
    is_source_link_request,
)
from services.gdrive_service import (
    create_oauth_authorization,
    load_cached_credentials,
    save_oauth_callback,
)
from services.sync_manager import DriveSyncManager


st.set_page_config(
    page_title="JH Asisten AI",
    page_icon="🕌",
    layout="centered",
)

PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.json"
_WIB_TIMEZONE = ZoneInfo("Asia/Jakarta")
_SUNDA_WEEKDAYS = (
    "Senén",
    "Salasa",
    "Rebo",
    "Kemis",
    "Jumaah",
    "Saptu",
    "Minggu",
)
_SUNDA_MONTHS = (
    "Januari",
    "Pébruari",
    "Maret",
    "April",
    "Méi",
    "Juni",
    "Juli",
    "Agustus",
    "Séptémber",
    "Oktober",
    "Nopémber",
    "Désémber",
)
if PRIVATE_CONFIG_PATH.is_file():
    PRIVATE_CONFIG = json.loads(PRIVATE_CONFIG_PATH.read_text(encoding="utf-8"))
else:
    PRIVATE_CONFIG = {}

DEFAULT_API_KEY = str(PRIVATE_CONFIG.get("google_api_key", "")).strip()
DEFAULT_FOLDER_ID = str(PRIVATE_CONFIG.get("google_drive_folder_id", "")).strip()
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def _save_private_config(config: dict[str, object]) -> None:
    PRIVATE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    PRIVATE_CONFIG_PATH.chmod(0o600)


def _clean_legacy_assistant_message(content: str) -> str:
    if "signature" not in content or not content.lstrip().startswith("["):
        return content
    try:
        parts = ast.literal_eval(content)
    except (SyntaxError, ValueError):
        return "Balasan sebelumnya berisi metadata internal. Silakan kirim ulang pertanyaan."

    text_parts = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "\n".join(text_parts) if text_parts else (
        "Balasan sebelumnya berisi metadata internal. Silakan kirim ulang pertanyaan."
    )


def _render_message_content(content: str) -> None:
    markdown_lines: list[str] = []

    def flush_markdown() -> None:
        if markdown_lines:
            st.markdown("\n".join(markdown_lines))
            markdown_lines.clear()

    for line in content.splitlines():
        if _ARABIC_TEXT_RE.search(line):
            flush_markdown()
            st.markdown(
                '<div class="arabic-chat-line" dir="rtl" lang="ar">'
                f"{html.escape(line)}"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            markdown_lines.append(line)
    flush_markdown()


def _reconstruct_source_text(chunks: list[Document]) -> str:
    text = ""
    for document in chunks:
        chunk_text = document.page_content
        if not text:
            text = chunk_text
            continue
        max_overlap = min(300, len(text), len(chunk_text))
        overlap = next(
            (
                size
                for size in range(max_overlap, 0, -1)
                if text.endswith(chunk_text[:size])
            ),
            0,
        )
        text += chunk_text[overlap:] if overlap else "\n\n" + chunk_text
    return text


def _render_copy_button(content: str, key: str) -> None:
        encoded_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
        st.html(
                f"""
                <div class="copy-answer-control">
                    <button type="button" style="padding: 6px 10px; cursor: pointer;">
                        Salin Jawaban
                    </button>
                    <span role="status" style="margin-left: 8px;"></span>
                </div>
                <script>
                    (() => {{
                        const root = document.currentScript.parentElement;
                        const encoded = "{encoded_content}";
                        const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
                        const answer = new TextDecoder("utf-8").decode(bytes);
                        const button = root.querySelector("button");
                        const status = root.querySelector("[role=status]");
                        button.addEventListener("click", async () => {{
                            try {{
                                await navigator.clipboard.writeText(answer);
                            }} catch (error) {{
                                const textarea = document.createElement("textarea");
                                textarea.value = answer;
                                document.body.appendChild(textarea);
                                textarea.select();
                                document.execCommand("copy");
                                textarea.remove();
                            }}
                            status.textContent = "Tersalin";
                        }});
                    }})();
                </script>
                """,
                unsafe_allow_javascript=True,
        )


def _render_source_links(sources: list[dict[str, str]], key: str) -> None:
    if not sources:
        return
    st.caption("Sumber:")
    for source_number, source in enumerate(sources):
        st.caption(source.get("name", "Dokumen"))
        open_link = source.get("web_view_link", "")
        download_link = source.get("download_url", "")
        columns = st.columns(2)
        if open_link:
            columns[0].link_button(
                "Buka di Drive",
                open_link,
                key=f"{key}_open_{source_number}",
                use_container_width=True,
            )
        if download_link:
            columns[1].link_button(
                "Download",
                download_link,
                key=f"{key}_download_{source_number}",
                use_container_width=True,
            )


@st.cache_resource
def _get_drive_sync_manager() -> DriveSyncManager:
    return DriveSyncManager()


def _format_rpd_reset_time(reset_at: datetime) -> str:
    wib_time = reset_at.astimezone(_WIB_TIMEZONE)
    return (
        f"{_SUNDA_WEEKDAYS[wib_time.weekday()]}, {wib_time.day} "
        f"{_SUNDA_MONTHS[wib_time.month - 1]} {wib_time.year} tabuh "
        f"{wib_time.hour:02}:{wib_time.minute:02} WIB"
    )


@st.fragment(run_every="1s")
def _render_rpd_limit_notice() -> None:
    manager = _get_drive_sync_manager()
    now = datetime.now(timezone.utc)
    status = manager.get_rpd_status(now)
    if status is None:
        if st.session_state.pop("_rpd_notice_was_active", False):
            st.rerun(scope="app")
        return

    st.session_state["_rpd_notice_was_active"] = True
    reset_at = datetime.fromisoformat(status["rpd_reset_at"])
    remaining_seconds = max(
        0,
        ceil((reset_at.astimezone(timezone.utc) - now).total_seconds()),
    )
    remaining_hours, remainder = divmod(remaining_seconds, 3600)
    remaining_minutes, remaining_seconds = divmod(remainder, 60)
    st.warning(
        "Punten, kuota poéan Gemini API parantos béak. Prosés sinkronisasi "
        "dieureunkeun samentawis supados aman.\n\n"
        f"Kuota bakal dianyarkeun deui dina: {_format_rpd_reset_time(reset_at)}.\n\n"
        "Sésa waktos ngantosan: "
        f"{remaining_hours:02} jam {remaining_minutes:02} menit "
        f"{remaining_seconds:02} detik."
    )


@st.fragment(run_every="3s")
def _render_drive_sync_status(folder_id: str, api_key: str) -> None:
    manager = _get_drive_sync_manager()
    snapshot = manager.snapshot()
    rpd_status = manager.get_rpd_status()
    result = snapshot["result"]
    file_count = int(snapshot["file_count"])

    if result:
        file_count = int(result["file_count"])
        st.session_state.document_count = file_count
        generation = int(snapshot["generation"])
        if generation != st.session_state.get("drive_sync_generation_seen", 0):
            st.session_state.drive_sync_generation_seen = generation
            st.session_state.index_manifest_files = result["files"]
            st.session_state.index_has_index = bool(result["has_index"])
            if result["index_updated"]:
                index = manager.load_local_index(folder_id, api_key)
                st.session_state.search_index = index
                st.session_state.index_folder_id = folder_id
                st.session_state.index_api_key = api_key
                st.rerun(scope="app")

    st.caption(f"File Drive terpantau: {file_count}")
    if snapshot["phase"] == "quota_error" and rpd_status is None:
        st.warning(str(snapshot["error"]))
    elif snapshot["error"] and rpd_status is None:
        st.warning(f"Sinkronisasi Drive gagal: {snapshot['error']}")
    elif snapshot["phase"] == "quota_error":
        st.caption("Sinkronisasi dieureunkeun samentawis kusabab wates kuota.")
    elif snapshot["phase"] == "downloading":
        total = int(snapshot["download_total"])
        downloaded = int(snapshot["downloaded_count"])
        fraction = downloaded / total if total else 0.0
        st.progress(fraction, text=str(snapshot["status"]))
    elif snapshot["phase"] in {"embedding", "saving"}:
        st.caption(str(snapshot["status"]))
    else:
        st.caption(str(snapshot["status"]))

    files = snapshot["files"]
    status_labels = {
        "ready": "Tersedia",
        "downloaded": "Diunduh, menunggu indeks",
        "queued": "Belum dimuat",
        "downloading": "Sedang diunduh",
        "embedding": "Sedang diindeks",
        "failed": "Gagal",
    }
    counts = {
        state: sum(file_status["state"] == state for file_status in files)
        for state in status_labels
    }
    in_progress_count = (
        counts["downloading"] + counts["downloaded"] + counts["embedding"]
    )
    st.caption(
        f"Tersedia {counts['ready']} · Sedang dimuat {in_progress_count} · "
        f"Belum dimuat {counts['queued']} · Gagal {counts['failed']}"
    )
    if snapshot["running"]:
        sync_label = "Lanjutkan muat" if snapshot["paused"] else "Jeda muat"
    else:
        sync_label = "Lanjutkan muat" if snapshot["paused"] else "Sinkronkan Sekarang"
    if st.button(
        sync_label,
        key="drive_sync_pause_toggle",
        use_container_width=True,
        disabled=rpd_status is not None,
    ):
        if snapshot["running"]:
            manager.resume() if snapshot["paused"] else manager.pause()
        else:
            manager.start_sync(folder_id, api_key)
        st.rerun(scope="fragment")

    with st.expander(f"Daftar sumber ({len(files)})"):
        state_order = {
            "downloading": 0,
            "downloaded": 1,
            "embedding": 2,
            "queued": 3,
            "failed": 4,
            "ready": 5,
        }
        for file_status in sorted(files, key=lambda item: state_order.get(item["state"], 5)):
            file_id = file_status["id"]
            state = file_status["state"]
            columns = st.columns([4, 1])
            with columns[0]:
                st.write(f"{file_status['name']} · {status_labels.get(state, state)}")
                if file_status.get("error"):
                    st.caption(file_status["error"])
            if state == "failed":
                with columns[1]:
                    if st.button(
                        "Coba ulang",
                        key=f"retry_source_{file_id}",
                        disabled=rpd_status is not None,
                    ):
                        if manager.retry_file(file_id):
                            manager.start_sync(folder_id, api_key)
                        st.rerun(scope="fragment")

    ready_files = [file_status for file_status in files if file_status["state"] == "ready"]
    with st.expander(f"Pratinjau sumber tersedia ({len(ready_files)})"):
        if not ready_files:
            st.info("Belum ada sumber yang siap dipratinjau.")
        else:
            source_names = {
                file_status["id"]: file_status["name"]
                for file_status in ready_files
            }
            selected_file_id = st.selectbox(
                "Pilih file",
                options=list(source_names),
                format_func=source_names.get,
                key="drive_source_preview_selection",
            )
            index: FAISS | None = st.session_state.get("search_index")
            indexed_chunks = []
            if index is not None:
                for document_id in index.index_to_docstore_id.values():
                    document = index.docstore.search(document_id)
                    if (
                        isinstance(document, Document)
                        and document.metadata.get("file_id") == selected_file_id
                    ):
                        indexed_chunks.append(document)

            if indexed_chunks:
                st.success(f"Sudah masuk indeks pencarian · {len(indexed_chunks)} potongan")
                extracted_text = manager.load_source_text(selected_file_id)
                if extracted_text is None:
                    extracted_text = _reconstruct_source_text(indexed_chunks)
                    st.caption(
                        "Teks lengkap asli belum tersimpan untuk sumber lama; "
                        "isi berikut dirangkai dari seluruh chunk indeks."
                    )
                preview_tab, read_tab = st.tabs(["Cuplikan", "Baca Dokumen"])
                with preview_tab:
                    st.text_area(
                        "Cuplikan teks hasil ekstraksi",
                        value=extracted_text[:1800],
                        height=220,
                        disabled=True,
                        key=f"drive_source_preview_text_{selected_file_id}",
                    )
                with read_tab:
                    section_size = 6000
                    section_count = max(1, (len(extracted_text) + section_size - 1) // section_size)
                    section_number = st.selectbox(
                        "Bagian dokumen",
                        options=range(section_count),
                        format_func=lambda number: f"Bagian {number + 1} dari {section_count}",
                        key=f"drive_source_section_{selected_file_id}",
                    )
                    start = section_number * section_size
                    st.text_area(
                        "Isi teks hasil ekstraksi",
                        value=extracted_text[start : start + section_size],
                        height=500,
                        disabled=True,
                        key=f"drive_source_full_text_{selected_file_id}_{section_number}",
                    )
                    st.caption(f"Total teks: {len(extracted_text):,} karakter")
            else:
                st.warning(
                    "File berstatus tersedia, tetapi chunk-nya belum ditemukan "
                    "di indeks pencarian lokal."
                )


st.session_state["google_api_key"] = DEFAULT_API_KEY
st.session_state["google_drive_folder_id"] = DEFAULT_FOLDER_ID

if "app_title" not in st.session_state:
    st.session_state.app_title = str(
        PRIVATE_CONFIG.get("app_title", "✨ Asisten AI Pribadi")
    )
st.markdown(
    """
    <style>
    .block-container { max-width: 860px; padding-top: 2rem; }
    [data-testid="stChatMessage"] { border-radius: 0.8rem; }
    .arabic-chat-line {
        direction: rtl;
        text-align: right;
        font-family: "Noto Naskh Arabic", "Amiri", "Traditional Arabic", serif;
        font-size: 1.25rem;
        line-height: 2;
        margin: 0.35rem 0;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        h1 { font-size: 1.8rem !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Sumber pengetahuan")
    folder_id = st.session_state.google_drive_folder_id
    if not folder_id:
        st.warning("Atur GOOGLE_DRIVE_FOLDER_ID untuk menghubungkan folder Drive.")
    st.caption(
        "File Drive terpantau: "
        f"{st.session_state.get('document_count', 0)}"
    )

    st.divider()
    st.subheader("Menu Pengaturan Instan")
    settings_password = st.text_input(
        "Password pengaturan",
        type="password",
        key="settings_password",
    )
    if settings_password == "411779":
        if "app_title_input" not in st.session_state:
            st.session_state.app_title_input = st.session_state.app_title

        edited_title = st.text_input(
            "Judul Utama Aplikasi",
            key="app_title_input",
        )
        if edited_title != st.session_state.app_title:
            st.session_state.app_title = edited_title
            PRIVATE_CONFIG["app_title"] = edited_title
            _save_private_config(PRIVATE_CONFIG)

st.title(st.session_state.app_title)
st.caption("✨ JH Asisten AI - Memori Otomatis Terhubung dengan Google Drive")

api_key = st.session_state.google_api_key
if not api_key:
    st.info("Atur GOOGLE_API_KEY sebelum menggunakan fitur pencarian AI.")
if not folder_id:
    st.info("Atur GOOGLE_DRIVE_FOLDER_ID untuk menghubungkan memori Google Drive.")

drive_authorized = False
if api_key and folder_id:
    try:
        manager = _get_drive_sync_manager()
        if (
            st.session_state.get("index_folder_id") != folder_id
            or st.session_state.get("index_api_key") != api_key
        ):
            st.session_state.search_index = manager.load_local_index(folder_id, api_key)
            st.session_state.index_folder_id = folder_id
            st.session_state.index_api_key = api_key
    except (GoogleAPIError, GoogleAuthError, HttpError, OSError, ValueError) as error:
        st.warning(f"Indeks lokal belum dapat dimuat: {error}")

    try:
        drive_authorized = load_cached_credentials() is not None
    except (GoogleAuthError, OSError, ValueError) as error:
        st.warning(f"Kredensial Google Drive tidak dapat dimuat: {error}")

    if not drive_authorized:
        st.warning(
            "Izin Google Drive diperlukan satu kali. Setelah menyetujui akses, "
            "salin seluruh URL callback dari address bar localhost dan tempel di sini."
        )
        try:
            if "drive_oauth_flow" not in st.session_state:
                flow, authorization_url, oauth_state = create_oauth_authorization()
                st.session_state.drive_oauth_flow = flow
                st.session_state.drive_oauth_url = authorization_url
                st.session_state.drive_oauth_state = oauth_state

            st.link_button(
                "Buka izin Google Drive",
                st.session_state.drive_oauth_url,
            )
            st.caption("URL callback berisi kode sekali pakai. Jangan bagikan kepada siapa pun.")
            with st.form("drive_oauth_callback_form", clear_on_submit=True):
                callback_url = st.text_input(
                    "URL callback localhost",
                    type="password",
                    placeholder="Tempel seluruh URL dari address bar setelah memberi izin",
                )
                submit_oauth = st.form_submit_button("Simpan izin Google Drive")

            if submit_oauth:
                try:
                    save_oauth_callback(
                        st.session_state.drive_oauth_flow,
                        callback_url,
                        st.session_state.drive_oauth_state,
                    )
                    st.session_state.pop("drive_oauth_flow", None)
                    st.session_state.pop("drive_oauth_url", None)
                    st.session_state.pop("drive_oauth_state", None)
                    st.rerun()
                except (
                    GoogleAuthError,
                    OAuth2Error,
                    OSError,
                    RequestException,
                    ValueError,
                ) as error:
                    st.error(f"Izin Google Drive gagal disimpan: {error}")
        except (GoogleAuthError, OSError, ValueError) as error:
            st.error(f"Tidak dapat memulai izin Google Drive: {error}")

if api_key and folder_id and drive_authorized:
    try:
        with st.sidebar:
            _render_rpd_limit_notice()
            _render_drive_sync_status(folder_id, api_key)
    except (GoogleAPIError, GoogleAuthError, HttpError, OSError, ValueError) as error:
        st.warning(f"Sinkronisasi Drive berjalan di latar belakang; indeks lokal tetap digunakan bila tersedia. {error}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message["role"] == "assistant":
        message["content"] = _clean_legacy_assistant_message(message["content"])
    with st.chat_message(message["role"]):
        _render_message_content(message["content"])
        if message["role"] == "assistant":
            _render_source_links(
                message.get("sources", []),
                f"history_{len(st.session_state.messages)}_{id(message)}",
            )
            _render_copy_button(
                message["content"],
                f"history_copy_{len(st.session_state.messages)}_{id(message)}",
            )

question = st.chat_input("Tanyakan sesuatu dari dokumen Anda...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        _render_message_content(question)

    with st.chat_message("assistant"):
        index: FAISS | None = st.session_state.get("search_index")
        response_sources: list[dict[str, str]] = []
        if is_source_link_request(question):
            response_sources = find_source_link_matches(question, index)
            if response_sources:
                response_text = "Berikut tautan sumber yang paling cocok:"
            else:
                response_text = "Belum ditemukan nama sumber yang cocok di daftar Drive."
            _render_message_content(response_text)
            _render_source_links(response_sources, "requested_source")
        elif index is None:
            response_text = (
                "Indeks awal sedang disiapkan di latar belakang. "
                "Coba kirim pertanyaan lagi setelah status memori siap."
            )
            _render_message_content(response_text)
        else:
            try:
                spinner_text = (
                    "Mencari dan menyusun hingga 25 bagian berurutan..."
                    if is_full_context_request(question)
                    else "Mencari jawaban..."
                )
                with st.spinner(spinner_text):
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                    )
                _render_message_content(response_text)
                _render_source_links(response_sources, "answer_source")
            except (GoogleAPIError, GoogleGenerativeAIError, ValueError) as error:
                response_text = f"Pencarian gagal: {error}"
                st.error(response_text)
        _render_copy_button(response_text, "current_answer_copy")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": response_sources,
        }
    )
