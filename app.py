"""Streamlit UI for the personal AI assistant."""

from __future__ import annotations

import ast
import base64
from contextlib import contextmanager
import html
import json
import os
import time
import uuid
from datetime import datetime, timezone
from importlib import import_module
from inspect import signature
from io import BytesIO
from math import ceil
import re
from pathlib import Path

from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_javascript import st_javascript
from streamlit.errors import StreamlitSecretNotFoundError
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
    ChatKeyQuotaExceeded,
    answer_question,
    find_source_link_matches,
    get_available_chat_models,
    is_source_link_request,
    start_model_warmup,
    transcribe_audio,
)
from services.gdrive_service import (
    create_oauth_authorization,
    load_cached_credentials,
    save_oauth_callback,
)
from services.device_identity import get_device_uid
from services.admin_logic import (
    process_guest_login,
    process_admin_login,
    question_is_allowed,
    register_admin_button_click,
    verify_admin_code,
)
from services.admin_session import (
    build_admin_session_store_script,
    create_admin_session_token,
    get_admin_session_epoch,
    verify_admin_session_token,
)
from services.firestore_store import (
    HISTORY_WARNING_CHUNKS,
    HISTORY_WARNING_MESSAGES,
    MAX_MESSAGES_PER_CHUNK,
    SAVED_CHAT_LIMIT,
    ChatStoreError,
    SavedChatLimitError,
    credentials_json_from_secrets,
    delete_all_user_chats,
    delete_user_chat,
    enqueue_guest_question,
    enqueue_question_answer,
    get_saved_chats,
    get_active_chat_id,
    get_firestore_client,
    get_device_firestore_client,
    get_runtime_config,
    load_chat_history,
    save_named_chat,
    save_user_name,
    submit_answer_feedback,
)
from services.key_pool import (
    KeyPoolError,
    add_key,
    clear_session_chat_override,
    get_available_keys,
    get_model_order,
    models_for_key,
    ordered_models,
    record_quota,
    redact_secrets,
    set_session_chat_override,
)
from services.perf_monitor import record_run as record_perf_run
from services.sync_manager import DriveSyncManager
from services.user_profile import USER_NAME_ERROR, needs_user_name, validate_user_name


APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app_icon.png"
st.set_page_config(
    page_title="JH Asisten AI",
    page_icon=APP_ICON_PATH,
    layout="centered",
)

_PERF_ENABLED = (
    st.query_params.get("debug") == "1"
    or bool(st.session_state.get("admin_authenticated"))
)
_PERF_RUN_STARTED = time.perf_counter()
_PERF_BLOCKS: dict[str, float] = {}


@contextmanager
def _measure_perf(block_name: str):
    started = time.perf_counter()
    try:
        yield
    finally:
        if _PERF_ENABLED:
            _PERF_BLOCKS[block_name] = _PERF_BLOCKS.get(block_name, 0.0) + (
                time.perf_counter() - started
            )


def _start_perf_block() -> float | None:
    return time.perf_counter() if _PERF_ENABLED else None


def _finish_perf_block(block_name: str, started: float | None) -> None:
    if started is not None:
        _PERF_BLOCKS[block_name] = _PERF_BLOCKS.get(block_name, 0.0) + (
            time.perf_counter() - started
        )


def _guest_emergency_seconds(runtime: dict[str, object]) -> int:
    if not runtime.get("guest_emergency_enabled"):
        return 0
    try:
        expiry = datetime.fromisoformat(
            str(runtime.get("guest_emergency_expires_at", ""))
        )
    except ValueError:
        return 0
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return max(0, ceil((expiry - datetime.now(timezone.utc)).total_seconds()))


def _enable_guest_session(emergency: bool = False) -> None:
    if st.session_state.get("admin_authenticated"):
        return
    st.session_state["guest_mode"] = True
    st.session_state["guest_mode_via_emergency"] = emergency
    st.session_state["guest_questions_day"] = ""
    st.session_state["guest_questions_count"] = 0
    st.session_state["guest_chat_id"] = uuid.uuid4().hex[:12]
    st.session_state["chat_id"] = st.session_state["guest_chat_id"]
    st.session_state["messages"] = []
    st.session_state["history_stats"] = {}
    st.session_state["loaded_chat_id"] = st.session_state["guest_chat_id"]
    st.session_state["_guest_loaded"] = True
    st.session_state.pop("_admin_login_prompt_open", None)
    st.session_state.pop("_admin_session_store_pending", None)
    st.session_state.pop("delete_all_chats_confirmation", None)
    st.session_state.pop("clear_active_chat_confirmation", None)
    for key in (
        "_loaded_chat_uid",
        "_saved_chats_uid",
        "_saved_chats",
        "_saved_chats_error",
    ):
        st.session_state.pop(key, None)


def _submit_guest_code(input_key: str, encoded_hash: str, uid: str) -> None:
    code = str(st.session_state.pop(input_key, "") or "")
    if process_guest_login(
        code,
        encoded_hash,
        st.session_state,
        time.monotonic(),
        uid,
    ):
        _enable_guest_session()
    else:
        st.session_state["_guest_code_form_open"] = True


def _render_guest_access(
    runtime: dict[str, object],
    *,
    key_prefix: str,
    guest_code_hash: str,
) -> None:
    if st.session_state.get("admin_authenticated"):
        return
    emergency_seconds = _guest_emergency_seconds(runtime)
    if emergency_seconds:
        if st.button(
            "Mode tamu",
            key=f"{key_prefix}_emergency_guest",
            type="tertiary",
            help="Katup darurat dibuka ku admin.",
            on_click=_enable_guest_session,
            args=(True,),
        ):
            return
    elif guest_code_hash:
        if not st.session_state.get("_guest_code_form_open"):
            st.button(
                "Mode tamu",
                key=f"{key_prefix}_open_guest",
                type="tertiary",
                on_click=lambda: st.session_state.__setitem__(
                    "_guest_code_form_open", True
                ),
            )
        else:
            input_key = f"{key_prefix}_guest_code"
            with st.form(f"{key_prefix}_guest_code_form", clear_on_submit=True):
                st.text_input(
                    "Sandi mode tamu",
                    type="password",
                    key=input_key,
                )
                st.form_submit_button(
                    "Buka mode tamu",
                    on_click=_submit_guest_code,
                    args=(input_key, guest_code_hash, str(st.session_state.device_uid)),
                )
    notice = st.session_state.pop("_guest_login_notice", None)
    if notice:
        st.error(notice)
    if emergency_seconds:
        total_minutes = ceil(emergency_seconds / 60)
        hours, minutes = divmod(total_minutes, 60)
        st.caption(
            f"Katup darurat aktif; sésana {hours} jam {minutes} menit."
        )


def _record_app_perf() -> None:
    if not _PERF_ENABLED:
        return
    total_seconds = time.perf_counter() - _PERF_RUN_STARTED
    measured_seconds = sum(_PERF_BLOCKS.values())
    durations = {
        name: _PERF_BLOCKS.get(name, 0.0)
        for name in (
            "identity",
            "configuration",
            "key_pool",
            "index_loading",
            "history_loading",
        )
    }
    durations["render"] = max(0.0, total_seconds - measured_seconds)
    record_perf_run(durations, total_seconds)


PRIVATE_CONFIG_PATH = Path(__file__).resolve().parent / "private_config.json"
CUSTOM_LOGO_PATH = PRIVATE_CONFIG_PATH.parent / ".app_custom_logo.png"
MAX_LOGO_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_LOGO_IMAGE_PIXELS = 16_000_000
MAX_CHAT_IMAGE_COUNT = 4
MAX_CHAT_IMAGE_TOTAL_BYTES = 10 * 1024 * 1024
TEKS = {
    "history_warning": (
        "Riwayat obrolan parantos panjang; tiasa janten rada lambat. "
        "Mugi dipertimbangkeun pikeun ngabersihan riwayat."
    ),
    "clear_history": "Bersihkan riwayat",
    "clear_history_confirm": (
        "Yakin badé ngahapus paguneman anu keur dibuka? "
        "Paguneman séjén moal dipupus."
    ),
    "clear_history_confirm_button": "Muhun, hapus",
    "clear_history_cancel": "Ulah ayeuna",
    "clear_history_done": "Paguneman aktip parantos dipupus.",
    "delete_all_history": "Hapus Sadaya Paguneman",
    "delete_all_history_confirm": (
        "Yakin badé ngahapus sadaya paguneman dina alat ieu, kaasup chat simpen? "
        "Data anu parantos dipupus moal tiasa dipulangkeun."
    ),
    "delete_all_history_done": "Sadaya paguneman parantos dipupus.",
}

# --------------------------------------------------------------------------
# Tema (10 total). Kunci opsional "gradient" = CSS background-image halaman.
# --------------------------------------------------------------------------
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
    "cyberpunk": {
        "label": "Cyberpunk Neon Glow",
        "background": "#0B0714",
        "surface": "#150E26",
        "sidebar": "#0F0A1E",
        "text": "#EDE7FF",
        "primary": "#00E5F5",
        "primary_text": "#03141A",
        "accent": "#FF2BD6",
        "chat": "#1B1133",
        "border": "#3A2468",
        "gradient": (
            "radial-gradient(900px 520px at 90% -8%, rgba(255,43,214,0.22), transparent 60%),"
            "radial-gradient(800px 480px at -10% 8%, rgba(0,229,245,0.16), transparent 58%)"
        ),
    },
    "nordic": {
        "label": "Nordic Aurora Glass",
        "background": "#0E1B24",
        "surface": "#142733",
        "sidebar": "#0F202B",
        "text": "#E6F3F1",
        "primary": "#5EEAD4",
        "primary_text": "#052B27",
        "accent": "#8BA7FF",
        "chat": "#17303D",
        "border": "#2B4A5A",
        "gradient": (
            "radial-gradient(820px 500px at 8% -8%, rgba(94,234,212,0.20), transparent 60%),"
            "radial-gradient(860px 560px at 100% 6%, rgba(139,167,255,0.20), transparent 60%)"
        ),
    },
    "sunset": {
        "label": "Warm Sunset Terracotta",
        "background": "#FFF1E3",
        "surface": "#FFFAF4",
        "sidebar": "#FCE5D0",
        "text": "#4A2418",
        "primary": "#C94D28",
        "primary_text": "#FFFFFF",
        "accent": "#E8923A",
        "chat": "#FDE3CB",
        "border": "#F0C9A6",
        "gradient": "linear-gradient(180deg, #FFE9D3 0%, #FFD8C0 55%, #FFF1E3 100%)",
    },
    "amoled": {
        "label": "Midnight AMOLED Gold",
        "background": "#000000",
        "surface": "#0D0D0D",
        "sidebar": "#050505",
        "text": "#F2EBDD",
        "primary": "#E7C15A",
        "primary_text": "#1A1300",
        "accent": "#B8902F",
        "chat": "#141414",
        "border": "#2A2418",
        "gradient": "radial-gradient(760px 380px at 50% -8%, rgba(231,193,90,0.10), transparent 62%)",
    },
    "matcha": {
        "label": "Matcha Zen Minimalist",
        "background": "#F2F4EA",
        "surface": "#FBFCF6",
        "sidebar": "#E8ECDC",
        "text": "#2B3326",
        "primary": "#566F34",
        "primary_text": "#FFFFFF",
        "accent": "#A8B67D",
        "chat": "#E9EEDC",
        "border": "#D5DCC0",
    },
}

# --------------------------------------------------------------------------
# Font (5 pilihan Google Fonts).
# --------------------------------------------------------------------------
APP_FONTS: dict[str, dict[str, str]] = {
    "jakarta": {
        "label": "Plus Jakarta Sans - modern & profesional",
        "import": "Plus+Jakarta+Sans:wght@400;500;600;700",
        "stack": "'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "line_height": "1.65",
        "letter_spacing": "-0.003em",
    },
    "poppins": {
        "label": "Poppins - geometris, ramah & kekinian",
        "import": "Poppins:wght@400;500;600;700",
        "stack": "'Poppins', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "line_height": "1.68",
        "letter_spacing": "0em",
    },
    "inter": {
        "label": "Inter - tajam untuk teks panjang",
        "import": "Inter:wght@400;500;600;700",
        "stack": "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "line_height": "1.7",
        "letter_spacing": "-0.006em",
    },
    "lora": {
        "label": "Lora - serif editorial berkelas",
        "import": "Lora:wght@400;500;600;700",
        "stack": "'Lora', Georgia, 'Times New Roman', serif",
        "line_height": "1.78",
        "letter_spacing": "0em",
    },
    "space": {
        "label": "Space Grotesk - futuristik & unik",
        "import": "Space+Grotesk:wght@400;500;600;700",
        "stack": "'Space Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
        "line_height": "1.62",
        "letter_spacing": "-0.008em",
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
_PRIVATE_CONFIG_CONTAINS_API_KEYS = any(
    name in PRIVATE_CONFIG
    for name in (
        "google_api_key",
        "google_api_key_override",
        "GOOGLE_API_KEY",
        "GOOGLE_API_KEY_INDEX",
        "GOOGLE_API_KEY_ADMIN",
        "ENCRYPTION_KEY",
    )
)

try:
    for _k, _v in st.secrets.items():
        if (
            _k not in {
                "GOOGLE_API_KEY",
                "GOOGLE_API_KEY_INDEX",
                "GOOGLE_API_KEY_ADMIN",
                "ENCRYPTION_KEY",
            }
            and _k not in PRIVATE_CONFIG
            and isinstance(_v, (str, int, float, bool))
        ):
            PRIVATE_CONFIG[_k] = _v
except Exception:
    pass
for _secret_key in (
    "google_api_key",
    "google_api_key_override",
    "GOOGLE_API_KEY",
    "GOOGLE_API_KEY_INDEX",
    "GOOGLE_API_KEY_ADMIN",
    "ENCRYPTION_KEY",
):
    PRIVATE_CONFIG.pop(_secret_key, None)


DEFAULT_FOLDER_ID = (
    os.getenv("GOOGLE_DRIVE_FOLDER_ID", "").strip()
    or str(PRIVATE_CONFIG.get("google_drive_folder_id", "")).strip()
)
DEFAULT_THEME_ID = "amoled"
DEFAULT_FONT_ID = "lora"
APP_THEME_ID = str(PRIVATE_CONFIG.get("app_theme", DEFAULT_THEME_ID))
if APP_THEME_ID not in APP_THEMES:
    APP_THEME_ID = DEFAULT_THEME_ID
APP_FONT_ID = str(PRIVATE_CONFIG.get("app_font", DEFAULT_FONT_ID))
if APP_FONT_ID not in APP_FONTS:
    APP_FONT_ID = DEFAULT_FONT_ID
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")


def _safe_chat_id(raw: object) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", str(raw or ""))
    return cleaned[:32]


def _get_or_create_chat_id(uid: str, firestore_client: object | None) -> str:
    param_chat_id = _safe_chat_id(st.query_params.get("chat"))
    if param_chat_id:
        return param_chat_id

    saved_id = _safe_chat_id(get_active_chat_id(firestore_client, uid) or "")
    if saved_id:
        st.query_params["chat"] = saved_id
        return saved_id

    new_id = uuid.uuid4().hex[:12]
    st.query_params["chat"] = new_id
    return new_id


def _start_new_chat() -> None:
    new_id = uuid.uuid4().hex[:12]
    st.query_params["chat"] = new_id
    st.session_state.chat_id = new_id
    st.session_state.messages = []
    st.session_state.loaded_chat_id = new_id
    st.session_state.history_stats = {
        **st.session_state.get("history_stats", {}),
        "active_chunk_count": 0,
        "active_chunk_number": 0,
    }


def _save_private_config(config: dict[str, object]) -> None:
    PRIVATE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    PRIVATE_CONFIG_PATH.chmod(0o600)


if _PRIVATE_CONFIG_CONTAINS_API_KEYS:
    try:
        _save_private_config(PRIVATE_CONFIG)
    except OSError:
        st.error(
            "Kredensial API lama tidak dapat dihapus otomatis dari private_config.json. "
            "Hapus entri API key secara manual."
        )


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
            <button type="button">Salin Jawaban</button>
            <span role="status"></span>
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
    with st.expander(f"Sumber ({len(sources)})", expanded=False):
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


def _render_answer_feedback(
    *,
    answer_id: str | None,
    question: str,
    answer: str,
    chat_id: str,
    quality_metrics: dict[str, object],
    sources: list[dict[str, object]],
) -> None:
    if not isinstance(answer_id, str) or not re.fullmatch(r"[a-f0-9]{32}", answer_id):
        return
    state_key = f"_feedback_submitted_{answer_id}"
    if st.session_state.get(state_key):
        st.caption("Hatur nuhun kana peunteunna.")
        return

    pending_key = f"_feedback_pending_{answer_id}"
    pending_feedback = bool(st.session_state.get(pending_key))
    positive_column, negative_column = st.columns([1, 1])
    if positive_column.button(
        "Cocog",
        key=f"feedback_positive_{answer_id}",
        disabled=pending_feedback,
    ):
        try:
            saved = submit_answer_feedback(
                DEVICE_FIRESTORE_CLIENT,
                uid=DEVICE_UID,
                name=str(st.session_state.get("history_stats", {}).get("name", "")),
                chat_id=chat_id,
                answer_id=answer_id,
                question=question,
                answer=answer,
                rating="Cocog",
                reason=None,
                quality_metrics=quality_metrics,
                sources=sources,
            )
        except (RuntimeError, ValueError) as error:
            st.warning(f"Eupan balik teu tiasa disimpen: {error}")
        else:
            st.session_state[state_key] = True
            st.caption(
                "Peunteun parantos kacatet."
                if saved
                else "Jawaban ieu parantos kantos dipeunteun."
            )
    if negative_column.button(
        "Teu cocog",
        key=f"feedback_negative_{answer_id}",
        disabled=pending_feedback,
    ):
        st.session_state[pending_key] = True

    if st.session_state.get(pending_key):
        with st.form(f"feedback_reason_form_{answer_id}"):
            reason_label = st.selectbox(
                "Alesan (opsional)",
                ("Tanpa alesan", "Kaluar tina kontéks", "Salah", "Teu lengkep"),
                key=f"feedback_reason_{answer_id}",
            )
            submit = st.form_submit_button("Kirim peunteun")
        if submit:
            try:
                saved = submit_answer_feedback(
                    DEVICE_FIRESTORE_CLIENT,
                    uid=DEVICE_UID,
                    name=str(st.session_state.get("history_stats", {}).get("name", "")),
                    chat_id=chat_id,
                    answer_id=answer_id,
                    question=question,
                    answer=answer,
                    rating="Teu cocog",
                    reason=None if reason_label == "Tanpa alesan" else reason_label,
                    quality_metrics=quality_metrics,
                    sources=sources,
                )
            except (RuntimeError, ValueError) as error:
                st.warning(f"Eupan balik teu tiasa disimpen: {error}")
            else:
                st.session_state[state_key] = True
                st.session_state.pop(f"_feedback_pending_{answer_id}", None)
                st.caption(
                    "Peunteun parantos kacatet."
                    if saved
                    else "Jawaban ieu parantos kantos dipeunteun."
                )


@st.cache_resource
def _get_drive_sync_manager() -> DriveSyncManager:
    return DriveSyncManager()


@st.cache_resource(show_spinner=False)
def _load_cached_local_index(
    folder_id: str,
    api_key: str,
    sync_generation: int,
) -> FAISS | None:
    """Simpan indeks FAISS di RAM agar reconnect di HP tidak baca ulang dari disk."""
    _ = sync_generation
    return _get_drive_sync_manager().load_local_index(folder_id, api_key)


@st.cache_resource(show_spinner=False)
def _warm_up_models_once(api_key: str) -> bool:
    try:
        start_model_warmup(api_key)
    except Exception:
        pass
    return True


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
def _render_rpd_limit_notice_live(api_key: str) -> None:
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


def _render_rpd_limit_notice(api_key: str) -> None:
    status = _get_drive_sync_manager().get_rpd_status(api_key=api_key)
    if status is not None:
        _render_rpd_limit_notice_live(api_key)


def _render_drive_sync_status_body(
    folder_id: str,
    api_keys: list[str],
) -> None:
    api_key = api_keys[0] if api_keys else ""
    manager = _get_drive_sync_manager()
    if st.session_state.pop("_quota_resume_start_failed", False):
        st.warning(
            "API key tersimpan, tetapi sinkronisasi otomatis belum dapat dimulai. "
            "Tekan LEBETKEUN KITAB untuk melanjutkan indeks."
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
                    index = _load_cached_local_index(folder_id, api_key, generation)
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

    st.caption(f"Kitab nu disiapkeun: {file_count}")
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
        "ready": "Sadia",
        "downloaded": "Diunduh, Ngantosan dilebetkeun",
        "queued": "Teu acan dilebetkeun",
        "downloading": "nuju diunduh",
        "embedding": "nuju dilebetkeun",
        "failed": "Gagal",
    }
    counts = {
        state: sum(file_status["state"] == state for file_status in files)
        for state in status_labels
    }
    in_progress_count = (
        counts["downloading"] + counts["downloaded"] + counts["embedding"]
    )
    st.markdown(
        '<div class="jh-chips">'
        f'<span class="jh-chip ok"><b>{counts["ready"]}</b> Sadia</span>'
        f'<span class="jh-chip run"><b>{in_progress_count}</b> nuju dilebetkeun</span>'
        f'<span class="jh-chip"><b>{counts["queued"]}</b> Teu acan dilebetkeun</span>'
        f'<span class="jh-chip bad"><b>{counts["failed"]}</b> Gagal</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    with st.expander(f"Kitab nu tos sayogi ({len(files)})"):
        failed_files = [
            file_status for file_status in files if file_status["state"] == "failed"
        ]
        if failed_files:
            st.markdown(f"**File gagal ({len(failed_files)})**")
            for file_status in failed_files:
                st.write(file_status["name"])
                st.caption(
                    file_status.get("error")
                    or "Teks kosong atau rusak, silakan konversi ke Google Docs"
                )
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
            state = file_status["state"]
            columns = st.columns([4, 1])
            with columns[0]:
                st.write(f"{file_status['name']} · {status_labels.get(state, state)}")
                if file_status.get("error"):
                    st.caption(file_status["error"])

    ready_files = [file_status for file_status in files if file_status["state"] == "ready"]
    with st.expander(f"Ningalan eusi  kitab nu aya ({len(ready_files)})"):
        if not ready_files:
            st.info("teu acan aya Kitab nu tiasa di tingal.")
        else:
            source_names = {
                file_status["id"]: file_status["name"]
                for file_status in ready_files
            }
            selected_file_id = st.selectbox(
                "Pilih kitab",
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
                st.success(f"Tos lebet Muthola'aheun· {len(indexed_chunks)} potongan")
                extracted_text = manager.load_source_text(selected_file_id)
                if extracted_text is None:
                    extracted_text = _reconstruct_source_text(indexed_chunks)
                    st.caption(
                        "Teks lengkap asli belum tersimpan untuk sumber lama; "
                        "isi berikut dirangkai dari seluruh chunk indeks."
                    )
                preview_tab, read_tab = st.tabs(["Sakilas", "Baca Kitab"])
                with preview_tab:
                    st.text_area(
                        "Sakilas kitab nu tos sadia",
                        value=extracted_text[:1800],
                        height=220,
                        disabled=True,
                        key=f"drive_source_preview_text_{selected_file_id}",
                    )
                with read_tab:
                    section_size = 6000
                    section_count = max(1, (len(extracted_text) + section_size - 1) // section_size)
                    section_number = st.selectbox(
                        "Bagian Kitab",
                        options=range(section_count),
                        format_func=lambda number: f"Bagian {number + 1} dari {section_count}",
                        key=f"drive_source_section_{selected_file_id}",
                    )
                    start = section_number * section_size
                    st.text_area(
                        "Eusi Kitabb",
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


@st.fragment(run_every="3s")
def _render_drive_sync_status_live(folder_id: str, api_keys: list[str]) -> None:
    _render_drive_sync_status_body(folder_id, api_keys)


def _render_drive_sync_status(folder_id: str, api_keys: list[str]) -> None:
    if _get_drive_sync_manager().snapshot()["running"]:
        _render_drive_sync_status_live(folder_id, api_keys)
    else:
        _render_drive_sync_status_body(folder_id, api_keys)


@st.fragment(run_every="1s")
def _render_device_uid_wait() -> None:
    get_device_uid(polling=True)
    if st.session_state.get("_device_uid_browser_resolved"):
        st.rerun(scope="app")
    if st.session_state.get("_device_uid_wait_timed_out"):
        st.caption("ID alat teu acan kapendak; riwayat can tiasa disimpen.")
        if st.button("Coba deui", key="device_uid_retry"):
            for key in (
                "_device_uid_wait_started",
                "_device_uid_wait_attempts",
                "_device_uid_waiting",
                "_device_uid_wait_timed_out",
            ):
                st.session_state.pop(key, None)
            st.rerun()
        return
    st.caption("Antosan, nuju mariksa ID alat supados riwayat tiasa disimpen.")


st.session_state.setdefault("chat_api_key_override", "")
st.session_state["google_drive_folder_id"] = DEFAULT_FOLDER_ID
with _measure_perf("identity"):
    DEVICE_UID = get_device_uid()
    DEVICE_ID_FINAL = bool(
        st.session_state.get("_device_uid_browser_resolved")
    )
IS_GUEST = bool(st.session_state.get("guest_mode"))
GUEST_CODE_HASH = ""
try:
    GUEST_CODE_HASH = str(st.secrets.get("GUEST_CODE_HASH", "") or "")
except StreamlitSecretNotFoundError:
    pass
if not DEVICE_ID_FINAL and not IS_GUEST:
    _render_guest_access(
        get_runtime_config(None),
        key_prefix="guest_unrecognized_id",
        guest_code_hash=GUEST_CODE_HASH,
    )
DEVICE_FIRESTORE_CLIENT = None
if not IS_GUEST and not DEVICE_ID_FINAL:
    _render_device_uid_wait()
if _PERF_ENABLED and not IS_GUEST:
    status_label = "final" if DEVICE_ID_FINAL else "sementara"
    source_label = st.session_state.get("_device_uid_source", "temporary")
    uid_prefix = st.session_state.get("_device_uid_prefix", "")
    st.caption(
        f"ID alat · sumber: {source_label} · status: {status_label} · "
        f"awalan: {uid_prefix}"
    )
with _measure_perf("configuration"):
    try:
        firestore_credentials_json = credentials_json_from_secrets(st.secrets)
    except StreamlitSecretNotFoundError:
        firestore_credentials_json = None
    if IS_GUEST:
        GUEST_STATS_CLIENT = get_firestore_client(firestore_credentials_json)
        FIRESTORE_CLIENT = None
    else:
        FIRESTORE_CLIENT = get_firestore_client(firestore_credentials_json)
        GUEST_STATS_CLIENT = None
    DEVICE_FIRESTORE_CLIENT = (
        None
        if IS_GUEST
        else get_device_firestore_client(FIRESTORE_CLIENT)
    )
    try:
        KEY_SECRETS = dict(st.secrets)
    except StreamlitSecretNotFoundError:
        KEY_SECRETS = {}
try:
    SETTINGS_PASSWORD = str(st.secrets.get("SETTINGS_PASSWORD", "") or "")
    ADMIN_CODE_HASH = str(st.secrets.get("ADMIN_CODE_HASH", "") or "")
except StreamlitSecretNotFoundError:
    SETTINGS_PASSWORD = ""
    ADMIN_CODE_HASH = ""


def _persist_admin_session() -> None:
    if not DEVICE_ID_FINAL:
        st.error("Sesi admin teu tiasa disimpen sateuacan ID alat final.")
        return
    epoch = get_admin_session_epoch(FIRESTORE_CLIENT)
    if epoch is None:
        st.session_state.pop("_admin_session_token", None)
        st.session_state["_admin_auth_method"] = "code"
        st.warning(
            "Kode leres, nanging sesi admin teu tiasa dipupuskeun ka alat ieu "
            "kumargi Firestore teu sayogi."
        )
        return
    try:
        token = create_admin_session_token(DEVICE_UID, epoch, KEY_SECRETS)
    except (ValueError, RuntimeError) as error:
        st.session_state.pop("_admin_session_token", None)
        st.session_state["_admin_auth_method"] = "code"
        st.warning(
            "Kode leres, nanging sesi admin teu tiasa disimpen "
            f"({type(error).__name__})."
        )
        return
    st.session_state["_admin_session_token"] = token
    st.session_state["_admin_auth_method"] = "token"
    st.session_state["_admin_session_store_pending"] = True


def _render_admin_session_storage() -> None:
    token = st.session_state.get("_admin_session_token")
    if not isinstance(token, str):
        st.session_state.pop("_admin_session_store_pending", None)
        return
    try:
        result = st_javascript(
            build_admin_session_store_script(token),
            key="write_admin_session_token",
        )
    except (ValueError, RuntimeError) as error:
        result = "error"
        error_class = type(error).__name__
    else:
        error_class = ""
    if result == "stored":
        st.session_state.pop("_admin_session_store_pending", None)
    elif result == "error":
        st.session_state.pop("_admin_session_store_pending", None)
        st.warning(
            "Kode leres. Panyimpenan sesi dina panyungsi teu sayogi; "
            "anjeun tiasa kedah lebet deui saatos muka ulang aplikasi."
        )
        if error_class:
            st.caption(f"Rincian: {error_class}")


def _restore_admin_session() -> None:
    if not DEVICE_ID_FINAL:
        return
    session_token = st.session_state.get("_admin_session_token")
    auth_method = st.session_state.get("_admin_auth_method")
    browser_token = st.context.cookies.get("jh_adm", "")
    token = session_token if isinstance(session_token, str) else browser_token
    if not token:
        return
    epoch = get_admin_session_epoch(FIRESTORE_CLIENT)
    valid = epoch is not None and verify_admin_session_token(
        token,
        DEVICE_UID,
        epoch,
        KEY_SECRETS,
    )
    if valid:
        st.session_state["admin_authenticated"] = True
        st.session_state["_admin_session_token"] = token
        st.session_state["_admin_auth_method"] = "token"
        return
    if auth_method == "token" or not st.session_state.get("admin_authenticated"):
        st.session_state["admin_authenticated"] = False
        st.session_state.pop("_admin_session_token", None)
        st.session_state.pop("_admin_auth_method", None)


if not IS_GUEST:
    _restore_admin_session()
if st.session_state.get("_admin_session_store_pending"):
    _render_admin_session_storage()
if st.session_state.get("admin_authenticated"):
    _PERF_ENABLED = True


def _process_admin_command(command: str) -> bool:
    """Consume admin login commands before chat history, AI, or statistics."""
    if not re.match(r"^/admin(?:\s|$)", command, flags=re.IGNORECASE):
        return False

    if not DEVICE_ID_FINAL:
        st.session_state["_admin_login_notice"] = (
            "ID alat can final; cobian deui saatos ID alat dikenal."
        )
        return True
    code = command[len("/admin") :].strip()
    authenticated = process_admin_login(
        code,
        ADMIN_CODE_HASH,
        st.session_state,
        time.monotonic(),
        DEVICE_UID,
        empty_code_notice="Gunakan format /admin <kode>.",
    )
    if authenticated:
        _persist_admin_session()
    return True


if "app_title" not in st.session_state:
    st.session_state.app_title = str(
        PRIVATE_CONFIG.get("app_title", "JH Asisten AI ")
    )


# ==========================================================================
# Sistem tampilan: warna, font, dan CSS (ringan di Android).
# ==========================================================================
def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _rgba(color: str, alpha: float) -> str:
    red, green, blue = _hex_to_rgb(color)
    return f"rgba({red},{green},{blue},{alpha})"


def _mix(first: str, second: str, weight: float) -> str:
    a = _hex_to_rgb(first)
    b = _hex_to_rgb(second)
    mixed = tuple(round(a[i] * weight + b[i] * (1 - weight)) for i in range(3))
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _is_dark(color: str) -> bool:
    red, green, blue = _hex_to_rgb(color)
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) < 128


_USER_MESSAGE_SELECTORS = (
    '[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])',
    '[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])',
)
_USER_TEXT_TAGS = ("p", "li", "span", "a", "strong", "em", "code")

_APP_CSS = """
<style>
__FONT_IMPORT__

:root, html, body, .stApp, [data-testid="stAppViewContainer"] {
    color-scheme: __SCHEME__;
}
html, body {
    background: __BG__;
    -webkit-text-size-adjust: 100%;
    overscroll-behavior-y: contain;
}

.stApp, [data-testid="stAppViewContainer"] {
    background-color: __BG__;
    background-image: __BG_IMAGE__;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-size: cover;
    color: __TEXT__;
    -webkit-font-smoothing: antialiased;
}
[data-testid="stMain"], [data-testid="stMainBlockContainer"] { background: transparent; }

.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label, .stApp summary,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stWidgetLabel"],
.stApp [data-baseweb="select"], .stApp [data-baseweb="menu"],
.stApp [data-baseweb="tab"], .stApp .jh-chip {
    font-family: __FONT_STACK__;
    letter-spacing: __LETTER__;
}
.stApp p, .stApp li, .stApp [data-testid="stMarkdownContainer"] { line-height: __LINE__; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { line-height: 1.2; letter-spacing: -0.015em; }
.stApp .arabic-chat-line {
    direction: rtl;
    text-align: right;
    font-family: "Noto Naskh Arabic", "Amiri", "Traditional Arabic", serif;
    letter-spacing: 0;
    font-size: 1.3rem;
    line-height: 2.1;
    margin: 0.35rem 0;
}

.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label, .stApp summary, .stApp summary span,
.stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-baseweb="tab"] { color: __TEXT__; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p { color: __MUTED__; }
a { color: __PRIMARY__ !important; text-underline-offset: 3px; }

.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 820px;
    padding-top: calc(3.6rem + env(safe-area-inset-top, 0px)) !important;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: calc(9rem + env(safe-area-inset-bottom, 0px)) !important;
}

#MainMenu, .stDeployButton, [data-testid="stAppDeployButton"],
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { visibility: hidden !important; }
footer { display: none !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
}
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] { color: __TEXT__; }

.app-brand-title {
    font-size: clamp(1.7rem, 6.4vw, 2.6rem);
    line-height: 1.1;
    margin: 0;
    padding: 0;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: __TEXT__;
}
.app-brand-title::after {
    content: "";
    display: block;
    width: 3rem;
    height: 3px;
    margin-top: 0.55rem;
    border-radius: 3px;
    background: linear-gradient(90deg, __PRIMARY__, __ACCENT__);
}
.app-brand-sub {
    margin: 0.5rem 0 1.1rem;
    font-size: 0.92rem;
    color: __MUTED__ !important;
}

[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
    background: __SIDEBAR__;
    border-right: 1px solid __BORDER__;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-size: 1.05rem; font-weight: 700; letter-spacing: -0.01em;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: 0.8rem; }
[data-testid="stSidebar"] hr { border-color: __BORDER__; }

[data-testid="stAlert"] {
    background: __SURFACE__;
    border: 1px solid __BORDER__;
    border-left: 4px solid __ACCENT__;
    border-radius: 0.9rem;
    font-size: 0.88rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: __TEXT__; }
.jh-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.45rem 0 0.9rem; }
.jh-chip {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.28rem 0.7rem; border-radius: 999px;
    font-size: 0.78rem; line-height: 1.2;
    background: __SURFACE__; color: __TEXT__; border: 1px solid __BORDER__;
}
.jh-chip b { font-weight: 700; }
.jh-chip::before { content: ""; width: 0.5rem; height: 0.5rem; border-radius: 50%; background: #8a94a6; }
.jh-chip.ok::before { background: #2e9e6b; }
.jh-chip.run::before { background: __ACCENT__; }
.jh-chip.bad::before { background: #d6504a; }
.stProgress > div > div > div > div, [data-testid="stProgress"] [role="progressbar"] > div {
    background-color: __PRIMARY__;
}

[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"],
[data-testid^="stChatMessageAvatar"], [data-testid^="chatAvatarIcon"] {
    display: none !important;
}
[data-testid="stChatMessage"] > div:has([data-testid^="stChatMessageAvatar"]),
[data-testid="stChatMessage"] > div:has([data-testid^="chatAvatarIcon"]) {
    display: none !important;
}

[data-testid="stChatMessage"] {
    display: flex;
    gap: 0;
    width: 100%;
    margin: 0 0 0.75rem;
    padding: 0.9rem 1.05rem;
    background: __CHAT_GLASS__;
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid __BORDER__;
    border-left: 3px solid __ACCENT__;
    border-radius: 6px 20px 20px 20px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05), 0 14px 30px -18px __SHADOW__;
}
[data-testid="stChatMessageContent"] { width: 100%; min-width: 0; }
[data-testid="stChatMessage"] p { line-height: __LINE__; margin-bottom: 0.5rem; }
[data-testid="stChatMessage"] p:last-child { margin-bottom: 0; }

__U__ {
    width: fit-content;
    max-width: 88%;
    margin-left: auto;
    background: __USER_BG__;
    border: 1px solid transparent;
    border-radius: 20px 20px 6px 20px;
    box-shadow: 0 10px 24px -14px __SHADOW__;
}
__U_TXT__ { color: __PRIMARY_TEXT__ !important; }

.copy-answer-control { margin-top: 0.4rem; }
.copy-answer-control button {
    background: transparent;
    color: __TEXT__;
    border: 1px solid __BORDER__;
    border-radius: 999px;
    padding: 0.3rem 0.85rem;
    font: inherit;
    font-size: 0.78rem;
    cursor: pointer;
}
.copy-answer-control button:active { border-color: __PRIMARY__; }
.copy-answer-control [role=status] { margin-left: 0.5rem; font-size: 0.78rem; color: __MUTED__; }

[data-testid="stBottom"] { background: transparent !important; }
[data-testid="stBottom"] > div {
    background: linear-gradient(to top, __BG__ 68%, transparent) !important;
    width: 100%;
}
[data-testid="stBottomBlockContainer"] {
    width: 100%;
    max-width: 820px;
    margin: 0 auto;
    padding: 0.6rem 1rem calc(0.9rem + env(safe-area-inset-bottom, 0px));
    box-sizing: border-box;
}
[data-testid="stChatInput"] {
    width: 100%;
    background: __SURFACE__;
    border: 1.5px solid __BORDER__;
    border-radius: 26px;
    box-shadow: 0 8px 28px -12px __SHADOW__;
    transition: border-color 140ms ease, box-shadow 140ms ease;
}
[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    border-radius: 26px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: __PRIMARY__;
    box-shadow: 0 0 0 3px __FOCUS__, 0 8px 28px -12px __SHADOW__;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: __TEXT__ !important;
    font-size: 1rem;
    line-height: 1.5;
    caret-color: __PRIMARY__;
}
[data-testid="stChatInput"] textarea::placeholder { color: __MUTED__; opacity: 1; }
[data-testid="stChatInput"] button { color: __TEXT__; }
[data-testid="stChatInputSubmitButton"] {
    background: __PRIMARY__ !important;
    color: __PRIMARY_TEXT__ !important;
    border-radius: 50%;
}

div.stButton > button, div.stDownloadButton > button, [data-testid="stLinkButton"] a,
[data-testid="stFormSubmitButton"] button {
    background: __SURFACE__;
    border: 1px solid __ACCENT__;
    border-radius: 0.8rem;
    color: __TEXT__ !important;
    font-weight: 600;
    min-height: 2.6rem;
    transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease;
}
div.stButton > button:hover, [data-testid="stLinkButton"] a:hover,
[data-testid="stFormSubmitButton"] button:hover {
    border-color: __PRIMARY__;
    box-shadow: 0 6px 16px -8px __SHADOW__;
    transform: translateY(-1px);
}
div.stButton > button[kind="primary"] {
    background: __PRIMARY__; color: __PRIMARY_TEXT__ !important; border-color: __PRIMARY__;
}
[data-testid="stExpander"] {
    border: 1px solid __BORDER__;
    border-radius: 0.9rem;
    background: __SURFACE__;
}
[data-testid="stFileUploader"] { background: __SURFACE__; border: 1px dashed __ACCENT__; border-radius: 0.9rem; }
[data-testid="stFileUploaderDropzone"] { background: __CHAT__; border-radius: 0.8rem; }
[data-testid="stFileUploaderDropzone"] * { color: __TEXT__; }
input, textarea {
    background-color: __SURFACE__ !important;
    color: __TEXT__ !important;
    border-color: __BORDER__ !important;
}
textarea:disabled { -webkit-text-fill-color: __TEXT__ !important; opacity: 0.92; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"],
[data-baseweb="select"] > div {
    background-color: __SURFACE__ !important;
    border-color: __BORDER__ !important;
    color: __TEXT__ !important;
}
[data-baseweb="select"] span, [data-baseweb="select"] svg { color: __TEXT__ !important; fill: __TEXT__ !important; }
[data-baseweb="popover"] > div, [data-baseweb="popover"] ul, [data-baseweb="menu"] {
    background: __SURFACE__ !important;
    color: __TEXT__ !important;
}
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] { color: __TEXT__ !important; background: transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] { background: __CHAT__ !important; }
[data-baseweb="tab-highlight"] { background-color: __PRIMARY__ !important; }
[data-testid="stStatus"], [data-testid="stStatusWidget"] + div { color: __TEXT__; }

/* ---------- Optimasi GPU & scroll HP Android ---------- */
@media (max-width: 900px) {
    .stApp, [data-testid="stAppViewContainer"] { background-attachment: scroll; }
    [data-testid="stChatMessage"] {
        background: __CHAT__;
        -webkit-backdrop-filter: none;
        backdrop-filter: none;
        box-shadow: none;
    }
    [data-testid="stBottomBlockContainer"] { max-width: 100%; }
    .block-container [data-testid="stImage"] img { width: 40px !important; }
}
@media (max-width: 640px) {
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-left: 0.85rem;
        padding-right: 0.85rem;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-left: 0.7rem;
        padding-right: 0.7rem;
    }
    [data-testid="stChatMessage"] { padding: 0.75rem 0.9rem; }
    __U__ { max-width: 92%; }
    .block-container [data-testid="stImage"] img { width: 34px !important; }
}
</style>
"""


def _build_app_css(theme: dict[str, str], font: dict[str, str]) -> str:
    dark = _is_dark(theme["background"])
    shadow = _rgba("#000000", 0.55) if dark else _rgba(theme["primary"], 0.38)
    user_bg = (
        f"linear-gradient(135deg, {theme['primary']}, "
        f"{_mix(theme['primary'], theme['accent'], 0.78)})"
    )
    gradient = theme.get("gradient", "none")
    user_selectors = ", ".join(_USER_MESSAGE_SELECTORS)
    user_text_selectors = ", ".join(
        f"{prefix} {tag}"
        for prefix in _USER_MESSAGE_SELECTORS
        for tag in _USER_TEXT_TAGS
    )
    replacements = {
        "__FONT_IMPORT__": (
            f"@import url('https://fonts.googleapis.com/css2?family={font['import']}&display=swap');"
        ),
        "__FONT_STACK__": font["stack"],
        "__LINE__": font["line_height"],
        "__LETTER__": font["letter_spacing"],
        "__SCHEME__": "dark" if dark else "only light",
        "__BG__": theme["background"],
        "__BG_IMAGE__": gradient,
        "__SURFACE__": theme["surface"],
        "__SIDEBAR__": theme["sidebar"],
        "__TEXT__": theme["text"],
        "__MUTED__": _mix(theme["text"], theme["background"], 0.72),
        "__PRIMARY__": theme["primary"],
        "__PRIMARY_TEXT__": theme["primary_text"],
        "__ACCENT__": theme["accent"],
        "__CHAT__": theme["chat"],
        "__CHAT_GLASS__": _rgba(theme["chat"], 0.78),
        "__BORDER__": theme["border"],
        "__SHADOW__": shadow,
        "__FOCUS__": _rgba(theme["primary"], 0.28),
        "__USER_BG__": user_bg,
        "__U__": user_selectors,
        "__U_TXT__": user_text_selectors,
    }
    return re.sub(
        r"__[A-Z_]+?__",
        lambda match: replacements.get(match.group(0), match.group(0)),
        _APP_CSS,
    )


@st.cache_data(show_spinner=False, max_entries=50)
def _build_selected_app_css(theme_id: str, font_id: str) -> str:
    return _build_app_css(APP_THEMES[theme_id], APP_FONTS[font_id])


theme = APP_THEMES[APP_THEME_ID]
st.markdown(
    _build_selected_app_css(APP_THEME_ID, APP_FONT_ID),
    unsafe_allow_html=True,
)
with _measure_perf("configuration"):
    runtime_config = get_runtime_config(FIRESTORE_CLIENT)
settings_password_hash = str(runtime_config.get("settings_password_hash", ""))
settings_password_valid = False

with st.sidebar:
    if IS_GUEST:
        st.success("Tamu")
    st.divider()
    st.subheader("KITAB RUJUKAN")
    folder_id = st.session_state.google_drive_folder_id
    if not folder_id:
        st.warning("Atur GOOGLE_DRIVE_FOLDER_ID untuk menghubungkan folder Drive.")
    st.caption(
        "KITAB NU SIAPKEUNEUN: "
        f"{st.session_state.get('document_count', 0)}"
    )

    st.divider()
    st.subheader("Pangaturan JH")
    if not settings_password_hash and not SETTINGS_PASSWORD:
        st.info(
            "Menu Pengaturan terkunci. Admin tiasa ngatur sandina dina dashboard Admin."
        )
    else:
        settings_password = st.text_input(
            "Kata sandi pengaturan",
            type="password",
            key="settings_password",
        )
        settings_password_valid = (
            verify_admin_code(settings_password, settings_password_hash)
            if settings_password_hash
            else settings_password == SETTINGS_PASSWORD
        )
    if settings_password_valid:
        if "app_title_input" not in st.session_state:
            st.session_state.app_title_input = st.session_state.app_title

        edited_title = st.text_input(
            "Nami Aplikasi (teu kenging di robah)",
            key="app_title_input",
        )
        if edited_title != st.session_state.app_title:
            st.session_state.app_title = edited_title
            PRIVATE_CONFIG["app_title"] = edited_title
            _save_private_config(PRIVATE_CONFIG)

        with st.form("chat_api_key_override_form", clear_on_submit=True):
            edited_api_key = st.text_input(
                "API key chat khusus sesi",
                type="password",
                placeholder="Tempel API key pikeun sesi ieu",
            )
            save_api_key = st.form_submit_button("Gunakan API key sesi")

        if save_api_key:
            try:
                set_session_chat_override(st.session_state, edited_api_key)
            except KeyPoolError as error:
                st.error(str(error))
            else:
                st.rerun()

        if st.session_state.get("chat_api_key_override"):
            st.caption(
                "API key chat khusus ngan ukur aktip dina sesi ieu; "
                "teu dipaké pikeun indexing atanapi admin."
            )
            if st.button(
                "Kembali ke API key konfigurasi",
                key="clear_chat_api_key_override",
            ):
                clear_session_chat_override(st.session_state)
                st.rerun()

        with st.form("appearance_settings_form", clear_on_submit=False):
            selected_theme = st.selectbox(
                "Warna aplikasi",
                options=list(APP_THEMES),
                index=list(APP_THEMES).index(APP_THEME_ID),
                format_func=lambda theme_id: APP_THEMES[theme_id]["label"],
            )
            selected_font = st.selectbox(
                "Gaya Tulisan",
                options=list(APP_FONTS),
                index=list(APP_FONTS).index(APP_FONT_ID),
                format_func=lambda font_id: APP_FONTS[font_id]["label"],
            )
            uploaded_logo = st.file_uploader(
                "Logo aplikasi (Nging di gentos)",
                type=["png", "jpg", "jpeg", "webp"],
            )
            save_appearance = st.form_submit_button("Simpen tampilan")

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
                PRIVATE_CONFIG["app_font"] = selected_font
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

    st.divider()
    if not IS_GUEST:
        if st.button("Admin", key="admin_access_trigger", use_container_width=True):
            register_admin_button_click(st.session_state)
        remaining_admin_clicks = 3 - int(
            st.session_state.get("_admin_button_clicks", 0)
        )
        if remaining_admin_clicks < 3 and not st.session_state.get(
            "_admin_login_prompt_open"
        ):
            st.caption(
                f"Klik Admin {remaining_admin_clicks} kali deui pikeun muka login."
            )


def _render_header() -> None:
    title_html = (
        f'<h1 class="app-brand-title">{html.escape(st.session_state.app_title.strip())}</h1>'
        '<p class="app-brand-sub">JH Asisten AI - Naroskeun - Nalaah - Milari</p>'
    )
    if CUSTOM_LOGO_PATH.is_file():
        logo_column, title_column = st.columns([1, 6], vertical_alignment="center")
        with logo_column:
            st.image(str(CUSTOM_LOGO_PATH), width=56)
        with title_column:
            st.markdown(title_html, unsafe_allow_html=True)
    else:
        st.markdown(title_html, unsafe_allow_html=True)


def _render_admin_login_prompt() -> None:
    with st.sidebar:
        st.subheader("Login dashboard admin")
        with st.form("admin_sidebar_login_form", clear_on_submit=True):
            code = st.text_input("Kode admin", type="password")
            submitted = st.form_submit_button("Masuk", disabled=not DEVICE_ID_FINAL)

        if submitted:
            authenticated = process_admin_login(
                code.strip(),
                ADMIN_CODE_HASH,
                st.session_state,
                time.monotonic(),
                DEVICE_UID,
            )
            if authenticated:
                _persist_admin_session()
                st.rerun()

            notice = st.session_state.pop("_admin_login_notice", None)
            if notice:
                st.error(notice)

        if st.button("Tutup login admin", key="close_admin_sidebar_login"):
            st.session_state["_admin_login_prompt_open"] = False
            st.session_state["_admin_button_clicks"] = 0
            st.rerun()


_render_header()
if (
    st.session_state.get("_admin_login_prompt_open")
    and not st.session_state.get("admin_authenticated")
    and not IS_GUEST
):
    _render_admin_login_prompt()

with _measure_perf("key_pool"):
    CHAT_KEY_RECORDS = get_available_keys("chat", FIRESTORE_CLIENT, KEY_SECRETS)
    INDEX_KEY_RECORDS = get_available_keys("indexing", FIRESTORE_CLIENT, KEY_SECRETS)
    chat_api_key_override = str(
        st.session_state.get("chat_api_key_override", "")
    ).strip()
    chat_api_keys = (
        [chat_api_key_override]
        if chat_api_key_override
        else [item.key for item in CHAT_KEY_RECORDS]
    )
    index_api_keys = [item.key for item in INDEX_KEY_RECORDS]
    api_key = chat_api_keys[0] if chat_api_keys else ""
    index_api_key = index_api_keys[0] if index_api_keys else ""
    index_cache_key = index_api_key or "indexing-key-unavailable"
    model_order_settings = get_model_order(FIRESTORE_CLIENT, KEY_SECRETS)
    chat_model_names = (
        ordered_models(
            get_available_chat_models(api_key, background=True),
            model_order_settings,
        )
        if api_key
        else []
    )
    _chat_key_by_value = {item.key: item for item in CHAT_KEY_RECORDS}


def _chat_model_filter(key: str, models: list[str] | tuple[str, ...]) -> list[str]:
    record = _chat_key_by_value.get(key)
    return models_for_key(record, models) if record is not None else list(models)


def _record_chat_model_quota(
    key: str,
    model: str,
    error: BaseException,
) -> None:
    record_quota(
        "chat",
        FIRESTORE_CLIENT,
        KEY_SECRETS,
        key,
        error,
        model=model,
    )


if not api_key:
    st.info("Kelompok API key chat belum memiliki key aktif.")
else:
    _warm_up_models_once(api_key)
if not folder_id:
    st.info("Atur GOOGLE_DRIVE_FOLDER_ID untuk menghubungkan memori Google Drive.")

_index_loading_started = _start_perf_block()
drive_authorized = False
if folder_id:
    try:
        manager = _get_drive_sync_manager()
        current_generation = int(manager.snapshot()["generation"])
        if (
            st.session_state.get("index_folder_id") != folder_id
            or st.session_state.get("index_api_key") != index_cache_key
            or st.session_state.get("search_index") is None
        ):
            st.session_state.search_index = _load_cached_local_index(
                folder_id,
                index_cache_key,
                current_generation,
            )
            st.session_state.index_folder_id = folder_id
            st.session_state.index_api_key = index_cache_key
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

_finish_perf_block("index_loading", _index_loading_started)

if index_api_keys and folder_id and drive_authorized:
    try:
        with st.sidebar:
            _render_rpd_limit_notice(index_api_key)
            _render_drive_sync_status(folder_id, index_api_keys)
    except (GoogleAPIError, GoogleAuthError, HttpError, OSError, ValueError) as error:
        st.warning(f"Sinkronisasi Drive berjalan di latar belakang; indeks lokal tetap digunakan bila tersedia. {error}")

_history_loading_started = _start_perf_block()
requested_chat_id = _safe_chat_id(st.query_params.get("chat"))
if DEVICE_ID_FINAL:
    if IS_GUEST:
        st.session_state.setdefault("guest_chat_id", uuid.uuid4().hex[:12])
        st.session_state.chat_id = st.session_state["guest_chat_id"]
    else:
        if (
            "chat_id" not in st.session_state
            or st.session_state.get("_chat_uid") != DEVICE_UID
            or (requested_chat_id and requested_chat_id != st.session_state.chat_id)
        ):
            st.session_state.chat_id = _get_or_create_chat_id(
                DEVICE_UID,
                DEVICE_FIRESTORE_CLIENT,
            )
        st.session_state["_chat_uid"] = DEVICE_UID
elif "chat_id" not in st.session_state:
    st.session_state.chat_id = (
        st.session_state.get("guest_chat_id")
        if IS_GUEST
        else uuid.uuid4().hex[:12]
    )
chat_id = st.session_state.chat_id

if IS_GUEST:
    if not st.session_state.get("_guest_loaded"):
        st.session_state.messages = []
        st.session_state.history_stats = {}
        st.session_state["_guest_loaded"] = True
    st.session_state.loaded_chat_id = chat_id
elif not DEVICE_ID_FINAL:
    st.session_state.messages = []
    st.session_state.history_stats = {}
    st.session_state.loaded_chat_id = chat_id
    st.session_state.pop("_loaded_chat_uid", None)
elif (
    st.session_state.get("loaded_chat_id") != chat_id
    or st.session_state.get("_loaded_chat_uid") != DEVICE_UID
):
    loaded_messages, history_stats = load_chat_history(
        DEVICE_FIRESTORE_CLIENT,
        DEVICE_UID,
        chat_id,
    )
    st.session_state.messages = loaded_messages
    st.session_state.history_stats = history_stats
    st.session_state.loaded_chat_id = chat_id
    st.session_state["_loaded_chat_uid"] = DEVICE_UID
_finish_perf_block("history_loading", _history_loading_started)
if runtime_config["announcement"]:
    st.info(runtime_config["announcement"])

admin_login_notice = st.session_state.pop("_admin_login_notice", None)
if admin_login_notice:
    st.warning(admin_login_notice)
admin_session_warning = st.session_state.pop("_admin_session_clear_warning", None)
if admin_session_warning:
    st.warning(admin_session_warning)

if st.session_state.get("admin_authenticated"):
    _record_app_perf()
    admin_dashboard = import_module("services.admin_dashboard")
    admin_dashboard.render_dashboard(
        FIRESTORE_CLIENT,
        _get_drive_sync_manager(),
        folder_id,
        index_api_keys,
        drive_authorized,
        KEY_SECRETS,
    )
    st.stop()

if IS_GUEST:
    st.markdown(
        '<div style="background:#5b2c83;color:#fff;padding:0.85rem 1rem;'
        'border-radius:0.6rem;font-weight:700;">'
        "MODE TAMU - riwayat teu disimpen</div>",
        unsafe_allow_html=True,
    )
    emergency_seconds = _guest_emergency_seconds(runtime_config)
    if emergency_seconds:
        total_minutes = ceil(emergency_seconds / 60)
        hours, minutes = divmod(total_minutes, 60)
        st.error(
            "Katup darurat mode tamu keur dibuka "
            f"(sésana {hours} jam {minutes} menit)."
        )

history_stats = st.session_state.get("history_stats", {})
profile_name = str(history_stats.get("name", "") or "")
if not IS_GUEST and needs_user_name(DEVICE_ID_FINAL, profile_name):
    st.subheader("Sampurasun! Mangga tulis nami heula")
    st.text(
        "Nami dipaké supados gampil dipaluruh upami aya kasalahan atanapi pamundut "
        "bantosan. Cekap nami panggilan."
    )
    with st.form("required_user_name_form", clear_on_submit=True):
        requested_name = st.text_input("Nami", max_chars=30)
        submit_name = st.form_submit_button("Simpen nami")
    if submit_name:
        try:
            clean_name = validate_user_name(requested_name)
            save_user_name(DEVICE_FIRESTORE_CLIENT, DEVICE_UID, clean_name)
        except (ChatStoreError, ValueError) as error:
            st.error(str(error) or USER_NAME_ERROR)
        else:
            updated_history_stats = dict(history_stats)
            updated_history_stats["name"] = clean_name
            st.session_state.history_stats = updated_history_stats
            st.rerun()
    _render_guest_access(
        runtime_config,
        key_prefix="guest_missing_name",
        guest_code_hash=GUEST_CODE_HASH,
    )

if DEVICE_ID_FINAL and profile_name:
    if st.session_state.get("_saved_chats_uid") != DEVICE_UID:
        try:
            cached_saved_chats = get_saved_chats(DEVICE_FIRESTORE_CLIENT, DEVICE_UID)
            cached_saved_chats_error = ""
        except ChatStoreError as error:
            cached_saved_chats = []
            cached_saved_chats_error = str(error)
        st.session_state["_saved_chats_uid"] = DEVICE_UID
        st.session_state["_saved_chats"] = cached_saved_chats
        st.session_state["_saved_chats_error"] = cached_saved_chats_error
    saved_chats = st.session_state.get("_saved_chats", [])
    saved_chats_error = str(st.session_state.get("_saved_chats_error", ""))
else:
    saved_chats = []
    saved_chats_error = ""

if (
    int(history_stats.get("total_messages", 0)) > HISTORY_WARNING_MESSAGES
    or int(history_stats.get("total_chunks", 0)) > HISTORY_WARNING_CHUNKS
):
    st.sidebar.warning(TEKS["history_warning"])

with st.sidebar:
    st.divider()
    st.subheader("Riwayat Paguneman")
    if saved_chats_error:
        st.warning(saved_chats_error)
    if st.session_state.pop("_history_cleared_notice", False):
        st.success(TEKS["clear_history_done"])
    if st.session_state.pop("_chat_saved_notice", False):
        st.success("Paguneman disimpen. Paguneman anyar parantos dibuka.")
    if st.session_state.pop("_all_chats_deleted_notice", False):
        st.success(TEKS["delete_all_history_done"])

    if settings_password_valid and DEVICE_ID_FINAL and profile_name:
        st.text("Nami ayeuna:")
        st.text(profile_name)
        with st.form("edit_user_name_form", clear_on_submit=True):
            replacement_name = st.text_input("Ganti nami", max_chars=30)
            save_replacement_name = st.form_submit_button("Simpen nami")
        if save_replacement_name:
            try:
                clean_name = validate_user_name(replacement_name)
                save_user_name(DEVICE_FIRESTORE_CLIENT, DEVICE_UID, clean_name)
            except (ChatStoreError, ValueError) as error:
                st.error(str(error) or USER_NAME_ERROR)
            else:
                updated_history_stats = dict(
                    st.session_state.get("history_stats", {})
                )
                updated_history_stats["name"] = clean_name
                st.session_state.history_stats = updated_history_stats
                st.rerun()

    current_chat_is_saved = any(
        item["chat_id"] == chat_id for item in saved_chats
    )
    save_disabled = IS_GUEST or (
        not st.session_state.messages
        or (len(saved_chats) >= SAVED_CHAT_LIMIT and not current_chat_is_saved)
    )
    with st.form("save_named_chat_form", clear_on_submit=True):
        saved_chat_name = st.text_input(
            "Ngaran riwayat",
            max_chars=60,
            placeholder="Conto: Patarosan fikih",
        )
        st.caption("Simpen paguneman ieu kalayan nami, tuluy muka obrolan anyar.")
        save_chat = st.form_submit_button(
            "Simpan Paguneman",
            disabled=save_disabled or IS_GUEST,
            use_container_width=True,
        )
    if save_disabled and not IS_GUEST:
        if not st.session_state.messages:
            st.caption("Ajukan patarosan heula saméméh nyimpen paguneman.")
        elif len(saved_chats) >= SAVED_CHAT_LIMIT and not current_chat_is_saved:
            st.caption(
                f"Maksimal {SAVED_CHAT_LIMIT} paguneman. Pupus heula salah sahiji "
                "paguneman simpen."
            )
    if save_chat:
        try:
            updated_saved_chats = save_named_chat(
                DEVICE_FIRESTORE_CLIENT,
                DEVICE_UID,
                chat_id,
                saved_chat_name,
            )
        except SavedChatLimitError as error:
            st.warning(str(error))
        except (ChatStoreError, ValueError) as error:
            st.error(f"Paguneman teu tiasa disimpen: {error}")
        else:
            st.session_state._chat_saved_notice = True
            st.session_state["_saved_chats"] = updated_saved_chats
            _start_new_chat()
            st.rerun()

    if IS_GUEST:
        st.caption("Mode tamu: riwayat paguneman teu disimpen.")
    else:
        st.markdown(
            f"**Paguneman disimpen ({len(saved_chats)}/{SAVED_CHAT_LIMIT})**"
        )
        st.caption("Pencét ngaran pikeun muka deui.")
    for saved_chat in saved_chats:
        saved_chat_id = saved_chat["chat_id"]
        label = saved_chat["name"]
        if saved_chat_id == chat_id:
            label = f"✓ {label}"
        if st.button(
            label,
            key=f"open_saved_chat_{saved_chat_id}",
            use_container_width=True,
        ):
            st.query_params["chat"] = saved_chat_id
            st.rerun()

    if st.button(
        TEKS["delete_all_history"],
        key="delete_all_chats",
        disabled=IS_GUEST,
    ):
        st.session_state.delete_all_chats_confirmation = True
    if st.session_state.get("delete_all_chats_confirmation"):
        st.warning(TEKS["delete_all_history_confirm"])
        confirm_columns = st.columns(2)
        if confirm_columns[0].button(
            TEKS["clear_history_confirm_button"],
            key="confirm_delete_all_chats",
        ):
            try:
                delete_all_user_chats(DEVICE_FIRESTORE_CLIENT, DEVICE_UID)
            except ChatStoreError as error:
                st.error(f"Sadaya paguneman teu tiasa dipupus: {error}")
            else:
                history_stats = dict(st.session_state.get("history_stats", {}))
                history_stats.update(
                    {
                        "total_messages": 0,
                        "total_chunks": 0,
                        "active_chunk_count": 0,
                        "active_chunk_number": 0,
                    }
                )
                st.session_state.history_stats = history_stats
                st.session_state["_saved_chats"] = []
                st.session_state.delete_all_chats_confirmation = False
                st.session_state._all_chats_deleted_notice = True
                _start_new_chat()
                st.rerun()
        if confirm_columns[1].button(
            TEKS["clear_history_cancel"],
            key="cancel_delete_all_chats",
        ):
            st.session_state.delete_all_chats_confirmation = False
            st.rerun()

for message_index, message in enumerate(st.session_state.messages):
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
            previous_question = next(
                (
                    str(item.get("content", ""))
                    for item in reversed(st.session_state.messages[:message_index])
                    if item.get("role") == "user"
                ),
                "",
            )
            stored_metrics = message.get("quality_metrics", {})
            feedback_sources = message.get("source_context", [])
            if not isinstance(feedback_sources, list):
                feedback_sources = message.get("sources", [])
            if not IS_GUEST:
                _render_answer_feedback(
                    answer_id=message.get("answer_id"),
                    question=previous_question,
                    answer=str(message.get("content", "")),
                    chat_id=chat_id,
                    quality_metrics=(
                        stored_metrics if isinstance(stored_metrics, dict) else {}
                    ),
                    sources=feedback_sources,
                )

if not IS_GUEST and st.session_state.pop("_chat_deleted_notice", False):
    st.success(TEKS["clear_history_done"])

if st.button(
    TEKS["clear_history"],
    key="clear_active_chat",
    disabled=IS_GUEST,
):
    st.session_state.clear_active_chat_confirmation = True
if st.session_state.get("clear_active_chat_confirmation"):
    st.warning(TEKS["clear_history_confirm"])
    confirm_columns = st.columns(2)
    if confirm_columns[0].button(
        TEKS["clear_history_confirm_button"],
        key="confirm_clear_active_chat",
    ):
        try:
            deleted_stats = delete_user_chat(
                DEVICE_FIRESTORE_CLIENT,
                DEVICE_UID,
                chat_id,
            )
        except (ChatStoreError, ValueError) as error:
            st.error(f"Paguneman teu tiasa dipupus: {error}")
        else:
            history_stats = dict(st.session_state.get("history_stats", {}))
            history_stats.update(deleted_stats)
            st.session_state.history_stats = history_stats
            st.session_state.clear_active_chat_confirmation = False
            st.session_state._chat_deleted_notice = True
            _start_new_chat()
            st.rerun()
    if confirm_columns[1].button(
        TEKS["clear_history_cancel"],
        key="cancel_clear_active_chat",
    ):
        st.session_state.clear_active_chat_confirmation = False
        st.rerun()

def _santri_status(message: str) -> str:
    """Ubah pesan status teknis menjadi gaya santri (Sunda)."""
    m = message.strip()
    name = re.search(r"(gemini[\w.\-]*)", m, re.IGNORECASE)
    name = name.group(1) if name else "JH"

    if m.startswith("Mencari di indeks lokal"):
        return "Sakedap, urang muthola'ah kitab heula..."
    if m.startswith("Menghubungi Gemini"):
        return "Bismillah, urang nala'ah patarosan heula..."
    found = re.match(r"Mencari (\d+) bagian", m)
    if found:
        return f"Urang milari {found.group(1)} ibarot nu patali dina kitab..."
    if m.startswith("Menyusun jawaban"):
        return "Urang nyusun jawaban tina kitab, mugi berkah..."
    if m.startswith("Menyiapkan Gemini"):
        return "Urang muthola'ah lampiran heula..."
    if m.startswith("Menerima jawaban"):
        return "Jawaban nuju diserat, mangga sakedap ngantosan..."
    if m.startswith("Menunggu jawaban dari"):
        return f"Sakedap, urang ngantosan jawaban ti {name}..."
    if m.startswith("Menunggu"):
        return "Sakedap, urang deres heula..."
    if m.startswith("Dijawab oleh"):
        return f"Alhamdulillah, jawaban parantos sumping tina {name}."
    if "membatasi" in m or "bermasalah" in m:
        return f"{name} nuju repot; antosan..."
    if "terputus" in m:
        return f"aduh {name} ngalenyep sakedap; urang neruskeun jawaban..."
    if m.startswith("Semua model"):
        return "Punten, JH nuju teu sayogi; urang buka kitab langsung..."
    if m == "Selesai":
        return "Alhamdulillah, tos réngsé."
    if m == "Pencarian gagal":
        return "Punten, KITABNA TEU KAPENDAK."
    return message


guest_today = datetime.now(_WIB_TIMEZONE).strftime("%Y%m%d")
if IS_GUEST and st.session_state.get("guest_questions_day") != guest_today:
    st.session_state["guest_questions_day"] = guest_today
    st.session_state["guest_questions_count"] = 0
guest_question_limit = max(
    1, int(runtime_config.get("guest_daily_question_limit", 10))
)
guest_limit_reached = IS_GUEST and int(
    st.session_state.get("guest_questions_count", 0)
) >= guest_question_limit
if guest_limit_reached:
    st.warning(
        f"Mode tamu parantos ngahontal wates {guest_question_limit} "
        "patarosan sapoé dina sesi ieu."
    )

chat_submission = st.chat_input(
    "Taroskeun ku Seratan, rekam suanten, tanapi lampirkan gambar...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
    accept_audio=True,
    max_upload_size=10,
    disabled=(
        guest_limit_reached
        or (
            not IS_GUEST
            and (
                not DEVICE_ID_FINAL
                or needs_user_name(
                    DEVICE_ID_FINAL,
                    st.session_state.get("history_stats", {}).get("name", ""),
                )
            )
        )
    ),
)
if chat_submission is not None:
    if not IS_GUEST and (
        not DEVICE_ID_FINAL
        or needs_user_name(
            DEVICE_ID_FINAL,
            st.session_state.get("history_stats", {}).get("name", ""),
        )
    ):
        st.warning("Mangga simpen nami heula sateuacan ngamimitian paguneman.")
        st.stop()
    if guest_limit_reached:
        st.stop()
    typed_question = chat_submission.text.strip()
    if not IS_GUEST and _process_admin_command(typed_question):
        st.rerun()
    if runtime_config["maintenance"]:
        st.warning(
            runtime_config["announcement"]
            or "Aplikasi sedang dalam mode perawatan. Silakan coba kembali nanti."
        )
        st.stop()
    allowed, policy_notice = (
        (True, "")
        if IS_GUEST or chat_api_key_override
        else question_is_allowed(
            st.session_state.get("history_stats", {}),
            int(runtime_config["daily_question_limit"]),
            datetime.now(_WIB_TIMEZONE),
        )
    )
    if not allowed:
        st.warning(policy_notice)
        st.stop()

    image_uploads = chat_submission.files
    audio_upload = chat_submission.audio
    media_parts: list[dict[str, str]] = []
    image_previews: list[str] = []
    audio_preview: dict[str, str] | None = None
    transcript = ""
    processing_error: Exception | None = None
    try:
        if len(image_uploads) > MAX_CHAT_IMAGE_COUNT:
            raise ValueError(f"Maksimal {MAX_CHAT_IMAGE_COUNT} ukuran gambar .")

        total_image_bytes = 0
        for image_upload in image_uploads:
            image_bytes = image_upload.getvalue()
            total_image_bytes += len(image_bytes)
            if total_image_bytes > MAX_CHAT_IMAGE_TOTAL_BYTES:
                raise ValueError("Total ukuran gambar maksimal 10 MB.")
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
                session_key_override=bool(chat_api_key_override),
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
            raise ValueError("Taroskeun ku Seratan, rekam suanten, tanapi lampirkan gambar...")
    except (GoogleAPIError, GoogleGenerativeAIError, ValueError) as error:
        processing_error = error
        question = typed_question or "Pesan suanten tanapi gambar terlampir."

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

    response_started = time.perf_counter()
    answer_id = uuid.uuid4().hex
    quality_metrics: dict[str, object] = {
        "latency_ms": 0,
        "source_count": 0,
        "top_score": None,
        "fallback": False,
        "model_used": None,
        "source_context": [],
    }
    quality_metadata_state = {"captured": False}

    def capture_quality_metadata(metadata: dict[str, object]) -> None:
        quality_metrics.update(metadata)
        quality_metadata_state["captured"] = True

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
            response_text = (
                "Pesan tidak dapat diproses: "
                f"{redact_secrets(processing_error, chat_api_keys)}"
            )
            _render_message_content(response_text)
        elif is_source_link_request(question) and not media_parts:
            response_sources = find_source_link_matches(question, index)
            if response_sources:
                response_text = "ieu tautan sumber yang paling cocok:"
            else:
                response_text = "hapunten teu kapendak kitabna."
            _render_message_content(response_text)
            _render_source_links(response_sources, "requested_source")
        elif index is None and not media_parts:
            response_text = (
                "kitab nuju di bereskeun. "
                "taroskeun deui kin pami kitab tos beres."
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

            _prev_user = [
                m["content"]
                for m in st.session_state.messages[:-1]
                if m["role"] == "user"
            ]
            if _prev_user and len(question.split()) <= 12:
                question = f"{_prev_user[-1][:300]} {question}"
            answer_message = st.empty()
            streamed_answer: list[str] = []

            search_status = st.status(
                _santri_status(
                    "Mencari di indeks lokal..."
                    if force_local_search
                    else "Menghubungi Gemini untuk embedding pertanyaan..."
                ),
                expanded=False,
            )

            def update_search_progress(message: str) -> None:
                                search_status.update(label=_santri_status(message), state="running")

            def show_answer_token(token: str | None) -> None:
                if token is None:
                    streamed_answer.clear()
                    answer_message.empty()
                    return
                is_first_token = not streamed_answer
                streamed_answer.append(token)
                if is_first_token:
                    update_search_progress("Menerima jawaban dari Gemini...")
                answer_message.markdown("".join(streamed_answer))

            local_search_options = (
                {"force_local_search": force_local_search}
                if supports_local_search
                else {}
            )
            key_rotation_options = {}
            if "api_key_candidates" in answer_parameters:
                key_rotation_options["api_key_candidates"] = chat_api_keys[1:]
            if "model_names" in answer_parameters:
                key_rotation_options["model_names"] = chat_model_names
            if "model_filter" in answer_parameters:
                key_rotation_options["model_filter"] = _chat_model_filter
            if "on_model_quota" in answer_parameters:
                key_rotation_options["on_model_quota"] = _record_chat_model_quota
            if "session_key_override" in answer_parameters:
                key_rotation_options["session_key_override"] = bool(
                    chat_api_key_override
                )
            if "on_metadata" in answer_parameters:
                key_rotation_options["on_metadata"] = capture_quality_metadata
            try:
                if supports_search_progress and supports_answer_streaming:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        on_progress=update_search_progress,
                        on_token=show_answer_token,
                        **key_rotation_options,
                        **local_search_options,
                    )
                elif supports_search_progress:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        on_progress=update_search_progress,
                        **key_rotation_options,
                        **local_search_options,
                    )
                else:
                    response_text, response_sources = answer_question(
                        index,
                        question,
                        api_key=api_key,
                        media_parts=media_parts,
                        **key_rotation_options,
                        **local_search_options,
                    )
            except ChatKeyQuotaExceeded as error:
                answer_message.empty()
                response_text = str(error)
                search_status.update(label=_santri_status("Pencarian gagal"), state="error")
                st.warning(response_text)
            except (GoogleAPIError, GoogleGenerativeAIError, ValueError) as error:
                answer_message.empty()
                safe_error = redact_secrets(error, chat_api_keys)
                response_text = f"Pencarian gagal: {safe_error}"
                search_status.update(label=_santri_status("Pencarian gagal"), state="error")
                st.error(response_text)
            else:
                answer_message.empty()
                search_status.update(label=_santri_status("Selesai"), state="complete")
                _render_message_content(response_text)
                _render_source_links(response_sources, "answer_source")
        if not quality_metadata_state["captured"]:
            quality_metrics["latency_ms"] = max(
                0, round((time.perf_counter() - response_started) * 1000)
            )
            quality_metrics["source_count"] = len(response_sources)
            quality_metrics["source_context"] = [
                {
                    "name": source.get("name", "Dokumen"),
                    "excerpt": "",
                }
                for source in response_sources[:3]
            ]
        if not IS_GUEST:
            _render_answer_feedback(
                answer_id=answer_id,
                question=displayed_question,
                answer=response_text,
                chat_id=chat_id,
                quality_metrics=quality_metrics,
                sources=quality_metrics.get("source_context", [])
                if isinstance(quality_metrics.get("source_context"), list)
                else [],
            )
        _render_copy_button(response_text, "current_answer_copy")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": response_sources,
            "answer_id": answer_id,
            "quality_metrics": quality_metrics,
            "source_context": quality_metrics.get("source_context", []),
        }
    )
    if IS_GUEST:
        enqueue_guest_question(GUEST_STATS_CLIENT)
        st.session_state["guest_questions_day"] = guest_today
        st.session_state["guest_questions_count"] = int(
            st.session_state.get("guest_questions_count", 0)
        ) + 1
    else:
        enqueue_question_answer(
            DEVICE_FIRESTORE_CLIENT,
            DEVICE_UID,
            chat_id,
            displayed_question,
            response_text,
            response_sources,
            count_for_daily_limit=not bool(chat_api_key_override),
            answer_id=answer_id,
            quality_metrics=quality_metrics,
            model_used=(
                str(quality_metrics["model_used"])
                if quality_metrics.get("model_used")
                else None
            ),
        )
        updated_stats = dict(st.session_state.get("history_stats", {}))
        active_chunk_count = int(updated_stats.get("active_chunk_count", 0))
        if active_chunk_count + 2 > MAX_MESSAGES_PER_CHUNK:
            updated_stats["active_chunk_count"] = 2
            updated_stats["total_chunks"] = (
                int(updated_stats.get("total_chunks", 0)) + 1
            )
        else:
            updated_stats["active_chunk_count"] = active_chunk_count + 2
            updated_stats.setdefault("total_chunks", 0)
        updated_stats["total_messages"] = (
            int(updated_stats.get("total_messages", 0)) + 2
        )
        if not chat_api_key_override:
            today_wib = datetime.now(_WIB_TIMEZONE).strftime("%Y%m%d")
            current_day_count = (
                int(updated_stats.get("day_count", 0))
                if updated_stats.get("day") == today_wib
                else 0
            )
            updated_stats["day"] = today_wib
            updated_stats["day_count"] = current_day_count + 1
        st.session_state.history_stats = updated_stats


_record_app_perf()
