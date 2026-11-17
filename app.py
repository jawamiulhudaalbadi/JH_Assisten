"""Streamlit UI for the personal AI assistant."""

from __future__ import annotations

import ast
import base64
import html
import json
import os
from datetime import datetime, timezone
from importlib import import_module
from inspect import signature
from io import BytesIO
from math import ceil
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    tomllib = import_module("tomli")
from zoneinfo import ZoneInfo

import streamlit as st
from PIL import Image
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
    is_source_link_request,
    transcribe_audio,
)
from services.gdrive_service import (
    create_oauth_authorization,
    load_cached_credentials,
    save_oauth_callback,
)
from services.sync_manager import DriveSyncManager


APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app_icon.png"
st.set_page_config(
    page_title="JH Asisten AI",
    page_icon=APP_ICON_PATH,
    layout="centered",
)

PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.json"
CUSTOM_LOGO_PATH = PRIVATE_CONFIG_PATH.parent / ".app_custom_logo.png"
MAX_LOGO_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_LOGO_IMAGE_PIXELS = 16_000_000
MAX_CHAT_IMAGE_COUNT = 4
MAX_CHAT_IMAGE_TOTAL_BYTES = 10 * 1024 * 1024
APP_THEMES: dict[str, dict[str, str]] = {
    "emerald": {
        "label": "Zamrud & Gading",
        "background": "#F3F6F1",
        "surface": "#FFFFFF",
        "sidebar": "#EAF1EB",
        "text": "#20342C",
        "primary": "#1F6A54",
        "primary_text": "#FFFFFF",
        "accent": "#B58B4B",
        "chat": "#E8F0E9",
        "border": "#D8E2D9",
    },
    "midnight": {
        "label": "Galeri Tengah Malam",
        "background": "#10141F",
        "surface": "#171E2D",
        "sidebar": "#131A28",
        "text": "#E8EAF1",
        "primary": "#78D6B0",
        "primary_text": "#10251F",
        "accent": "#C9A96E",
        "chat": "#1D2638",
        "border": "#30394C",
    },
    "terracotta": {
        "label": "Atelier Terracotta",
        "background": "#FAF4EE",
        "surface": "#FFFDF9",
        "sidebar": "#F5ECE3",
        "text": "#3D2B26",
        "primary": "#9D4F3B",
        "primary_text": "#FFFFFF",
        "accent": "#C78D52",
        "chat": "#F3E7DC",
        "border": "#E6D6C8",
    },
    "ocean": {
        "label": "Biru Samudra",
        "background": "#F0F6F8",
        "surface": "#FCFEFF",
        "sidebar": "#E9F1F4",
        "text": "#203641",
        "primary": "#26718B",
        "primary_text": "#FFFFFF",
        "accent": "#6BA9A6",
        "chat": "#E3EFF2",
        "border": "#D3E1E5",
    },
    "lavender": {
        "label": "Lavender Artistik",
        "background": "#F6F2F8",
        "surface": "#FFFCFF",
        "sidebar": "#F0EAF5",
        "text": "#382D42",
        "primary": "#77568B",
        "primary_text": "#FFFFFF",
        "accent": "#C27C9A",
        "chat": "#EFE7F3",
        "border": "#E4D7E9",
    },
}
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

try:
  for _k, _v in st.secrets.items():
    if _k not in PRIVATE_CONFIG and isinstance(_v, (str, int, float, bool)):
      PRIVATE_CONFIG[_k] = _v
except Exception:
  pass


def _load_google_api_key() -> str:
  if PRIVATE_CONFIG.get("google_api_key_override") is True:
    return str(PRIVATE_CONFIG.get("google_api_key", "")).strip()

  environment_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
  if environment_api_key:
    return environment_api_key

  secrets_path = PRIVATE_CONFIG_PATH.parent / "secrets.toml"
  if secrets_path.is_file():
    secrets_config = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
    secrets_api_key = str(secrets_config.get("google_api_key", "")).strip()
    if secrets_api_key:
      return secrets_api_key

  return str(PRIVATE_CONFIG.get("google_api_key", "")).strip()


DEFAULT_API_KEY = _load_google_api_key()
DEFAULT_FOLDER_ID = (
    os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    or str(PRIVATE_CONFIG.get("google_drive_folder_id", "")).strip()
)
APP_THEME_ID = str(PRIVATE_CONFIG.get("app_theme", "emerald"))
if APP_THEME_ID not in APP_THEMES:
  APP_THEME_ID = "emerald"
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def _save_private_config(config: dict[str, object]) -> None:
    PRIVATE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    PRIVATE_CONFIG_PATH.chmod(0o600)


def _prepare_custom_logo(image_bytes: bytes) -> bytes:
    if not image_bytes:
        raise ValueError("File logo kosong.")
    if len(image_bytes) > MAX_LOGO_UPLOAD_BYTES:
        raise ValueError("Ukuran logo maksimal 5 MB.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Logo harus berformat PNG, JPG, atau WEBP.")
            if image.width * image.height > MAX_LOGO_IMAGE_PIXELS:
                raise ValueError("Resolusi logo terlalu besar.")

            image.thumbnail((512, 512))
            output = BytesIO()
            image.convert("RGBA" if "A" in image.getbands() else "RGB").save(
                output,
                format="PNG",
                optimize=True,
            )
    except (Image.DecompressionBombError, OSError) as error:
        raise ValueError("File logo tidak dapat dibaca.") from error

    return output.getvalue()


def _save_custom_logo(image_bytes: bytes) -> None:
    temporary_path = CUSTOM_LOGO_PATH.with_name(".app_custom_logo.png.tmp")
    temporary_path.write_bytes(image_bytes)
    temporary_path.chmod(0o600)
    temporary_path.replace(CUSTOM_LOGO_PATH)


def _prepare_chat_image(image_bytes: bytes) -> tuple[bytes, str]:
    if not image_bytes:
        raise ValueError("File gambar kosong.")
    if len(image_bytes) > MAX_LOGO_UPLOAD_BYTES:
        raise ValueError("Ukuran setiap gambar maksimal 5 MB.")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("Gambar harus berformat PNG, JPG, atau WEBP.")
            if image.width * image.height > MAX_LOGO_IMAGE_PIXELS:
                raise ValueError("Resolusi gambar terlalu besar.")

            image.thumbnail((1024, 1024))
            output = BytesIO()
            if "A" in image.getbands():
                image.convert("RGBA").save(output, format="PNG", optimize=True)
                mime_type = "image/png"
            else:
                image.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=85,
                    optimize=True,
                )
                mime_type = "image/jpeg"
    except (Image.DecompressionBombError, OSError) as error:
        raise ValueError("File gambar tidak dapat dibaca.") from error

    return output.getvalue(), mime_type


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
            markdown_lines.append(f"{line}  " if line else line)
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


def _format_sync_eta(seconds: float) -> str:
    remaining_seconds = ceil(max(seconds, 0))
    if remaining_seconds < 60:
        return f"{remaining_seconds} detik"
    minutes = ceil(remaining_seconds / 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return (
            f"{hours} jam {remaining_minutes} menit"
            if remaining_minutes
            else f"{hours} jam"
        )
    return f"{minutes} menit"


@st.fragment(run_every="1s")
def _render_rpd_limit_notice(api_key: str) -> None:
    manager = _get_drive_sync_manager()
    now = datetime.now(timezone.utc)
    status = manager.get_rpd_status(now, api_key=api_key)
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
    if st.session_state.pop("_quota_resume_start_failed", False):
        st.warning(
            "API key tersimpan, tetapi sinkronisasi otomatis belum dapat dimulai. "
            "Tekan Sinkronkan Sekarang untuk melanjutkan indeks."
        )
    snapshot = manager.snapshot()
    rpd_status = manager.get_rpd_status(api_key=api_key)
    result = snapshot["result"]
    file_count = int(snapshot["file_count"])
    embedding_total = int(snapshot.get("embedding_total", 0))

    if result:
        file_count = int(result["file_count"])
        st.session_state.document_count = file_count
        generation = int(snapshot["generation"])
        if generation != st.session_state.get("drive_sync_generation_seen", 0):
            st.session_state.index_manifest_files = result["files"]
            st.session_state.index_has_index = bool(result["has_index"])
            if result["index_updated"]:
                try:
                    index = manager.load_local_index(folder_id, api_key)
                except EOFError:
                    st.warning(
                        "Indeks FAISS belum terbaca karena berkasnya belum lengkap. "
                        "Aplikasi akan mencoba lagi."
                    )
                    return
                st.session_state.search_index = index
                st.session_state.index_folder_id = folder_id
                st.session_state.index_api_key = api_key
                st.session_state.drive_sync_generation_seen = generation
                st.rerun(scope="app")
            st.session_state.drive_sync_generation_seen = generation

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
    elif snapshot["phase"] == "embedding" and embedding_total:
        total = embedding_total
        done = int(snapshot.get("embedding_done_count", 0))
        fraction = done / total
        st.progress(
            fraction,
            text=f"{snapshot['status']} ({fraction:.0%})",
        )
        eta = snapshot.get("embedding_eta_seconds")
        if str(snapshot["status"]).startswith("Jeda aman embedding"):
            st.caption(str(snapshot["status"]))
        elif eta is None:
            st.caption(
                "Estimasi waktu batch ini dihitung setelah batch embedding pertama selesai."
            )
        elif done < total:
            st.caption(f"Estimasi waktu batch ini: {_format_sync_eta(float(eta))}")
        else:
            st.caption("Embedding batch selesai; hasil sedang disimpan.")
    elif (
        snapshot["phase"] == "embedding"
        and "embedding_total" not in snapshot
    ):
        st.caption(str(snapshot["status"]))
        st.info(
            "Sinkronisasi ini memakai status versi lama. Setelah selesai, "
            "mulai ulang aplikasi untuk menampilkan progres embedding."
        )
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
        failed_files = [
            file_status for file_status in files if file_status["state"] == "failed"
        ]
        if failed_files:
            st.markdown(f"**File gagal ({len(failed_files)})**")
            for file_status in failed_files:
                file_id = file_status["id"]
                columns = st.columns([4, 1])
                with columns[0]:
                    st.write(file_status["name"])
                    st.caption(
                        file_status.get("error")
                        or "Teks kosong atau rusak, silakan konversi ke Google Docs"
                    )
                with columns[1]:
                    if st.button(
                        "Coba ulang",
                        key=f"retry_source_{file_id}",
                        disabled=rpd_status is not None,
                    ):
                        if manager.retry_file(file_id):
                            manager.start_sync(folder_id, api_key)
                        st.rerun(scope="fragment")
            st.divider()
        else:
            st.caption("Tidak ada file yang gagal.")

        state_order = {
            "downloading": 0,
            "downloaded": 1,
            "embedding": 2,
            "queued": 3,
            "failed": 4,
            "ready": 5,
        }
        for file_status in sorted(files, key=lambda item: state_order.get(item["state"], 5)):
            if file_status["state"] == "failed":
                continue
            file_id = file_status["id"]
            state = file_status["state"]
            columns = st.columns([4, 1])
            with columns[0]:
                st.write(f"{file_status['name']} · {status_labels.get(state, state)}")
                if file_status.get("error"):
                    st.caption(file_status["error"])

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
theme = APP_THEMES[APP_THEME_ID]
st.markdown(
    f"""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {{
        background-color: {theme["background"]};
        color: {theme["text"]};
    }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {theme["sidebar"]};
        border-right: 1px solid {theme["border"]};
    }}
    .block-container {{ max-width: 860px; padding-top: 2rem; }}
    [data-testid="stChatMessage"] {{
        background-color: {theme["chat"]};
        border: 1px solid {theme["border"]};
        border-radius: 0.9rem;
    }}
    [data-testid="stChatInput"] {{
        background-color: {theme["surface"]};
        border-color: {theme["border"]};
        border-radius: 0.9rem;
    }}
    [data-testid="stExpander"] {{ border-color: {theme["border"]}; }}
    [data-testid="stFileUploader"] {{
        background-color: {theme["surface"]};
        border: 1px dashed {theme["accent"]};
        border-radius: 0.8rem;
    }}
    a {{ color: {theme["primary"]} !important; }}
    div.stButton > button {{
        background-color: {theme["surface"]};
        border: 1px solid {theme["accent"]};
        border-radius: 0.7rem;
        color: {theme["text"]};
        transition: transform 120ms ease, box-shadow 120ms ease;
    }}
    div.stButton > button:hover {{
        border-color: {theme["primary"]};
        box-shadow: 0 3px 12px {theme["border"]};
        transform: translateY(-1px);
    }}
    div.stButton > button[kind="primary"] {{
        background-color: {theme["primary"]};
        color: {theme["primary_text"]};
    }}
    input, textarea {{
        background-color: {theme["surface"]} !important;
        color: {theme["text"]} !important;
        border-color: {theme["border"]} !important;
    }}
    .arabic-chat-line {{
        direction: rtl;
        text-align: right;
        font-family: "Noto Naskh Arabic", "Amiri", "Traditional Arabic", serif;
        font-size: 1.25rem;
        line-height: 2;
        margin: 0.35rem 0;
    }}
    .app-brand-title {{
        font-size: 88px;
        line-height: 1;
        margin: 0;
    }}
    @media (max-width: 850px) {{
        .app-brand-title {{ font-size: 48px; }}
        .block-container [data-testid="stImage"] img {{ width: 48px !important; }}
    }}
    @media (max-width: 640px) {{
        .block-container {{ padding: 1rem 0.8rem 2rem; }}
        h1 {{ font-size: 1.8rem !important; }}
        .app-brand-title {{ font-size: 32px !important; }}
        .block-container [data-testid="stImage"] img {{ width: 32px !important; }}
    }}
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

        with st.form("google_api_key_form", clear_on_submit=True):
            edited_api_key = st.text_input(
                "API Key Google Gemini",
                type="password",
                placeholder="Tempel API key baru",
            )
            save_api_key = st.form_submit_button("Simpan API key")

        if save_api_key:
            edited_api_key = edited_api_key.strip()
            if not edited_api_key:
                st.error("API key tidak boleh kosong.")
            else:
                previous_api_key = str(st.session_state.google_api_key)
                manager = _get_drive_sync_manager()
                resume_quota_sync = (
                    edited_api_key != previous_api_key
                    and (
                        manager.snapshot()["phase"] == "quota_error"
                        or manager.get_rpd_status(api_key=previous_api_key) is not None
                    )
                )
                PRIVATE_CONFIG["google_api_key"] = edited_api_key
                PRIVATE_CONFIG["google_api_key_override"] = True
                _save_private_config(PRIVATE_CONFIG)
                if resume_quota_sync and folder_id:
                    if not manager.start_sync(folder_id, edited_api_key):
                        st.session_state["_quota_resume_start_failed"] = True
                st.rerun()

        if PRIVATE_CONFIG.get("google_api_key_override") is True:
            st.caption("Aplikasi memakai API key yang disimpan dari menu ini.")
            if st.button(
                "Gunakan key dari konfigurasi server",
                key="use_server_google_api_key",
            ):
                PRIVATE_CONFIG.pop("google_api_key", None)
                PRIVATE_CONFIG.pop("google_api_key_override", None)
                _save_private_config(PRIVATE_CONFIG)
                st.rerun()

        with st.form("appearance_settings_form", clear_on_submit=False):
            selected_theme = st.selectbox(
                "Tema aplikasi",
                options=list(APP_THEMES),
                index=list(APP_THEMES).index(APP_THEME_ID),
                format_func=lambda theme_id: APP_THEMES[theme_id]["label"],
            )
            uploaded_logo = st.file_uploader(
                "Logo aplikasi (PNG, JPG, WEBP)",
                type=["png", "jpg", "jpeg", "webp"],
            )
            save_appearance = st.form_submit_button("Simpan tampilan")

        if save_appearance:
            try:
                prepared_logo = None
                if uploaded_logo is not None:
                    if uploaded_logo.size > MAX_LOGO_UPLOAD_BYTES:
                        raise ValueError("Ukuran logo maksimal 5 MB.")
                    prepared_logo = _prepare_custom_logo(uploaded_logo.getvalue())

                if prepared_logo is not None:
                    _save_custom_logo(prepared_logo)
                PRIVATE_CONFIG["app_theme"] = selected_theme
                _save_private_config(PRIVATE_CONFIG)
            except (OSError, ValueError) as error:
                st.error(f"Tampilan tidak dapat disimpan: {error}")
            else:
                st.rerun()

        if CUSTOM_LOGO_PATH.is_file() and st.button(
            "Hapus logo unggahan",
            key="remove_custom_logo",
        ):
            try:
                CUSTOM_LOGO_PATH.unlink()
            except OSError as error:
                st.error(f"Logo tidak dapat dihapus: {error}")
            else:
                st.rerun()

if CUSTOM_LOGO_PATH.is_file():
    logo_column, title_column = st.columns([1, 8], vertical_alignment="center")
    with logo_column:
        st.image(str(CUSTOM_LOGO_PATH), width=88)
    with title_column:
        st.markdown(
            f'<h1 class="app-brand-title">{html.escape(st.session_state.app_title)}</h1>',
            unsafe_allow_html=True,
        )
else:
    st.title(st.session_state.app_title)
st.caption("JH Asisten AI - Naroskeun - Nalaah - Milari")

api_key = st.session_state.google_api_key
if not api_key:
    st.info("Atur GOOGLE_API_KEY sebelum menggunakan fitur pencarian AI.")
else:
    try:
        _get_drive_sync_manager().get_rpd_status(api_key=api_key)
    except (OSError, ValueError) as error:
        st.warning(f"Status jeda kuota Gemini tidak dapat diperbarui: {error}")
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
    except EOFError:
        st.warning(
            "Indeks FAISS belum terbaca karena berkasnya belum lengkap. "
            "Muat ulang aplikasi setelah sinkronisasi selesai."
        )
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
            _render_rpd_limit_notice(api_key)
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
        for image_data in message.get("images", []):
            st.image(base64.b64decode(image_data), width=260)
        audio_data = message.get("audio")
        if isinstance(audio_data, dict):
            st.audio(
                base64.b64decode(audio_data["data"]),
                format=audio_data["mime_type"],
            )
        if message["role"] == "assistant":
            _render_source_links(
                message.get("sources", []),
                f"history_{len(st.session_state.messages)}_{id(message)}",
            )
            _render_copy_button(
                message["content"],
                f"history_copy_{len(st.session_state.messages)}_{id(message)}",
            )

chat_submission = st.chat_input(
    "Tulis, rekam suara, atau lampirkan gambar...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
    accept_audio=True,
    max_upload_size=10,
)
if chat_submission is not None:
    typed_question = chat_submission.text.strip()
    image_uploads = chat_submission.files
    audio_upload = chat_submission.audio
    media_parts: list[dict[str, str]] = []
    image_previews: list[str] = []
    audio_preview: dict[str, str] | None = None
    transcript = ""
    processing_error: Exception | None = None
    try:
        if len(image_uploads) > MAX_CHAT_IMAGE_COUNT:
            raise ValueError(f"Maksimal {MAX_CHAT_IMAGE_COUNT} gambar per pesan.")

        total_image_bytes = 0
        for image_upload in image_uploads:
            image_bytes = image_upload.getvalue()
            total_image_bytes += len(image_bytes)
            if total_image_bytes > MAX_CHAT_IMAGE_TOTAL_BYTES:
                raise ValueError("Total ukuran gambar per pesan maksimal 10 MB.")
            prepared_image, mime_type = _prepare_chat_image(image_bytes)
            encoded_image = base64.b64encode(prepared_image).decode("ascii")
            media_parts.append(
                {
                    "type": "file",
                    "source_type": "base64",
                    "mime_type": mime_type,
                    "data": encoded_image,
                }
            )
            image_previews.append(encoded_image)

        if audio_upload is not None:
            audio_bytes = audio_upload.getvalue()
            transcript = transcribe_audio(
                audio_bytes,
                audio_upload.type or "audio/wav",
                api_key=api_key,
            )
            audio_preview = {
                "data": base64.b64encode(audio_bytes).decode("ascii"),
                "mime_type": audio_upload.type or "audio/wav",
            }

        question_parts = [part for part in (typed_question, transcript) if part]
        question = "\n\n".join(question_parts)
        if not question and image_previews:
            question = "Jelaskan gambar yang saya lampirkan."
        if not question:
            raise ValueError("Tulis pertanyaan, rekam suara, atau lampirkan gambar.")
    except (GoogleAPIError, GoogleGenerativeAIError, ValueError) as error:
        processing_error = error
        question = typed_question or "Pesan suara atau gambar terlampir."

    displayed_question = question
    if transcript and typed_question:
        displayed_question = f"{typed_question}\n\n🎙️ {transcript}"
    elif transcript:
        displayed_question = f"🎙️ {transcript}"

    user_message: dict[str, object] = {
        "role": "user",
        "content": displayed_question,
        "images": image_previews,
    }
    if audio_preview is not None:
        user_message["audio"] = audio_preview
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        _render_message_content(displayed_question)
        for image_data in image_previews:
            st.image(base64.b64decode(image_data), width=260)
        if audio_preview is not None:
            st.audio(
                base64.b64decode(audio_preview["data"]),
                format=audio_preview["mime_type"],
            )

    with st.chat_message("assistant"):
        index: FAISS | None = st.session_state.get("search_index")
        response_sources: list[dict[str, str]] = []
        if processing_error is not None:
            response_text = f"Pesan tidak dapat diproses: {processing_error}"
            _render_message_content(response_text)
        elif is_source_link_request(question) and not media_parts:
            response_sources = find_source_link_matches(question, index)
            if response_sources:
                response_text = "Berikut tautan sumber yang paling cocok:"
            else:
                response_text = "Belum ditemukan nama sumber yang cocok di daftar Drive."
            _render_message_content(response_text)
            _render_source_links(response_sources, "requested_source")
        elif index is None and not media_parts:
            response_text = (
                "Indeks awal sedang disiapkan di latar belakang. "
                "Coba kirim pertanyaan lagi setelah status memori siap."
            )
            _render_message_content(response_text)
        else:
            manager = _get_drive_sync_manager()
            is_index_busy = getattr(manager, "is_index_busy", None)
            force_local_search = (
                bool(is_index_busy()) if callable(is_index_busy) else False
            )
            answer_signature = signature(answer_question)
            answer_parameters = answer_signature.parameters
            supports_search_progress = "on_progress" in answer_parameters
            supports_answer_streaming = "on_token" in answer_parameters
            supports_local_search = "force_local_search" in answer_parameters
            if not supports_search_progress or not supports_answer_streaming:
                st.warning(
                    "Layanan pencarian yang sedang berjalan masih versi lama. "
                    "Jawaban tetap diproses, tetapi mulai ulang proses Streamlit "
                    "untuk mengaktifkan status progres."
                )

            answer_message = st.empty()
            streamed_answer: list[str] = []

            search_status = st.status(
                "Mencari di indeks lokal..."
                if force_local_search
                else "Menghubungi Gemini untuk embedding pertanyaan...",
                expanded=False,
            )

            def update_search_progress(message: str) -> None:
                search_status.update(label=message, state="running")

            def show_answer_token(token: str | None) -> None:
                if token is None:
                    streamed_answer.clear()
                    answer_message.empty()
                    return
                streamed_answer.append(token)
                update_search_progress("Menerima jawaban dari Gemini...")
                answer_message.markdown("".join(streamed_answer))

            local_search_options = (
                {"force_local_search": force_local_search}
                if supports_local_search
                else {}
            )
            try:
                if supports_search_progress and supports_answer_streaming:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        on_progress=update_search_progress,
                        on_token=show_answer_token,
                        **local_search_options,
                    )
                elif supports_search_progress:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        on_progress=update_search_progress,
                        **local_search_options,
                    )
                else:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        **local_search_options,
                    )
            except (GoogleAPIError, GoogleGenerativeAIError, ValueError) as error:
                answer_message.empty()
                response_text = f"Pencarian gagal: {error}"
                search_status.update(label="Pencarian gagal", state="error")
                st.error(response_text)
            else:
                answer_message.empty()
                search_status.update(label="Selesai", state="complete")
                _render_message_content(response_text)
                _render_source_links(response_sources, "answer_source")
        _render_copy_button(response_text, "current_answer_copy")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": response_sources,
        }
    )
