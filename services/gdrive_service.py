"""Read DOCX files from a Google Drive folder."""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit

from docx import Document as DocxDocument
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow

DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OAUTH_CLIENT_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"
DOCX_MIME_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
DOC_MIME_TYPE = "application/msword"
TEXT_MIME_TYPE = "text/plain"
GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
PDF_MIME_TYPE = "application/pdf"
FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SUPPORTED_FILE_MIME_TYPES = {
    DOCX_MIME_TYPE,
    DOC_MIME_TYPE,
    TEXT_MIME_TYPE,
    GOOGLE_DOC_MIME_TYPE,
    PDF_MIME_TYPE,
}
FILE_TYPE_PRIORITY = {
    DOCX_MIME_TYPE: 0,
    DOC_MIME_TYPE: 1,
    TEXT_MIME_TYPE: 2,
    GOOGLE_DOC_MIME_TYPE: 3,
    PDF_MIME_TYPE: 4,
}
PDF_EXTRACTION_TIMEOUT_SECONDS = 45
_PDF_EXTRACTION_SCRIPT = """
import io
import json
import sys
from pypdf import PdfReader

reader = PdfReader(io.BytesIO(sys.stdin.buffer.read()))
try:
    page_labels = reader.page_labels
except Exception:
    page_labels = []
pages = []
for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text()
    if page_text and page_text.strip():
        label = (
            page_labels[page_number - 1]
            if page_number <= len(page_labels)
            else page_number
        )
        if label is None or not str(label).strip():
            label = page_number
        pages.append({"page_label": str(label), "text": page_text.strip()})
sys.stdout.write(json.dumps(pages, ensure_ascii=False))
"""


class PdfExtractionTimeout(TimeoutError):
    """Raised when extracting a PDF exceeds its per-file time limit."""


def drive_file_priority(file_info: dict[str, str]) -> tuple[int, str]:
    """Sort text sources ahead of PDFs while keeping names deterministic."""
    return (
        FILE_TYPE_PRIORITY.get(file_info.get("mimeType", ""), 99),
        file_info.get("name", "").casefold(),
    )


@dataclass(frozen=True)
class DriveDocument:
    """Text and identifying metadata for a file in Drive."""

    file_id: str
    name: str
    text: str
    modified_time: str
    web_view_link: str = ""
    download_url: str = ""
    page_sections: tuple[tuple[str, str], ...] = ()


def drive_download_url(file_info: dict[str, str]) -> str:
    file_id = file_info["id"]
    if file_info.get("mimeType") == GOOGLE_DOC_MIME_TYPE:
        return f"https://docs.google.com/document/d/{file_id}/export?format=docx"
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _get_credentials() -> Credentials:
    """Load cached user credentials or require the app's OAuth consent flow."""
    credentials = load_cached_credentials()
    if credentials is None:
        raise ValueError(
            "Google Drive belum diotorisasi. Selesaikan izin Google dari halaman aplikasi."
        )
    return credentials


def load_cached_credentials() -> Credentials | None:
    """Load and refresh the locally cached OAuth credentials, if available."""
    if not OAUTH_CLIENT_FILE.is_file():
        raise ValueError("File credentials.json tidak ditemukan di root proyek.")

    if not TOKEN_FILE.is_file():
        return None

    TOKEN_FILE.chmod(0o600)
    credentials = Credentials.from_authorized_user_file(
        str(TOKEN_FILE),
        scopes=[DRIVE_READONLY_SCOPE],
    )

    if credentials and credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
        _save_credentials(credentials)

    return credentials if credentials and credentials.valid else None


def create_oauth_authorization() -> tuple[InstalledAppFlow, str, str]:
    """Create a browser authorization URL whose code can be pasted back to the app."""
    if not OAUTH_CLIENT_FILE.is_file():
        raise ValueError("File credentials.json tidak ditemukan di root proyek.")

    flow = InstalledAppFlow.from_client_secrets_file(
        str(OAUTH_CLIENT_FILE),
        scopes=[DRIVE_READONLY_SCOPE],
    )
    flow.redirect_uri = "http://localhost:8765/"
    authorization_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
    )
    return flow, authorization_url, state


def save_oauth_callback(
    flow: InstalledAppFlow,
    callback_url: str,
    expected_state: str,
) -> None:
    """Validate a pasted loopback callback, exchange its code, and cache credentials."""
    callback = urlsplit(callback_url.strip())
    callback_params = parse_qs(callback.query)
    if (
        callback.scheme != "http"
        or callback.hostname not in {"localhost", "127.0.0.1"}
        or callback.port != 8765
        or callback.path not in {"", "/"}
    ):
        raise ValueError("URL callback harus berasal dari localhost:8765.")
    if callback_params.get("state", [None])[0] != expected_state:
        raise ValueError("State OAuth tidak cocok. Mulai ulang proses izin Google.")
    if callback_params.get("error"):
        raise ValueError("Izin Google ditolak atau dibatalkan.")
    code = callback_params.get("code", [None])[0]
    if not code:
        raise ValueError("Kode OAuth tidak ditemukan pada URL callback.")

    flow.fetch_token(code=code)
    _save_credentials(flow.credentials)


def _save_credentials(credentials: Credentials) -> None:
    TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    TOKEN_FILE.chmod(0o600)




def _extract_docx_text(content: bytes) -> str:
    """Extract paragraph and table text from a DOCX payload."""
    document = DocxDocument(io.BytesIO(content))
    parts = [paragraph.text.strip() for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            parts.append(" | ".join(cell for cell in cells if cell))
    return "\n".join(part for part in parts if part)


def _extract_pdf_pages(
    content: bytes,
    timeout_seconds: float = PDF_EXTRACTION_TIMEOUT_SECONDS,
) -> list[tuple[str, str]]:
    """Extract text with original PDF page labels in a killable subprocess."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PDF_EXTRACTION_SCRIPT],
            input=content,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise PdfExtractionTimeout(
            f"PDF extraction exceeded {timeout_seconds:g} seconds"
        ) from error
    pages = json.loads(result.stdout.decode("utf-8"))
    return [
        (str(page["page_label"]), str(page["text"]))
        for page in pages
        if isinstance(page, dict) and page.get("text")
    ]


def _extract_pdf_text(
    content: bytes,
    timeout_seconds: float = PDF_EXTRACTION_TIMEOUT_SECONDS,
) -> str:
    """Extract PDF text while preserving the original page order."""
    return "\n".join(
        page_text
        for _, page_text in _extract_pdf_pages(content, timeout_seconds)
    )


def _extract_doc_text(content: bytes) -> str:
    """Extract legacy Word .doc text using the optional antiword executable."""
    antiword = shutil.which("antiword")
    if not antiword:
        raise RuntimeError(
            "File .doc memerlukan program antiword; pasang antiword di sistem "
            "atau simpan dokumen sebagai .docx."
        )
    result = subprocess.run(
        [antiword, "-"],
        input=content,
        capture_output=True,
        check=True,
        timeout=PDF_EXTRACTION_TIMEOUT_SECONDS,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def _decode_text_file(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        return content.decode("cp1252", errors="replace").strip()


def _build_drive_service():
    return build("drive", "v3", credentials=_get_credentials(), cache_discovery=False)


def list_drive_files(folder_id: str) -> list[dict[str, str]]:
    """List supported text documents and PDFs recursively."""
    if not folder_id.strip():
        raise ValueError("Folder ID Google Drive belum diatur.")

    service = _build_drive_service()
    files: list[dict[str, str]] = []
    pending_folders = [folder_id]
    visited_folders: set[str] = set()
    visited_files: set[str] = set()

    while pending_folders:
        current_folder_id = pending_folders.pop()
        if current_folder_id in visited_folders:
            continue
        visited_folders.add(current_folder_id)

        page_token: str | None = None
        while True:
            response = (
                service.files()
                .list(
                    q=f"'{current_folder_id}' in parents and trashed=false",
                    fields=(
                        "nextPageToken, files(id, name, mimeType, size, modifiedTime, "
                        "webViewLink, webContentLink)"
                    ),
                    pageSize=1000,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for file_info in response.get("files", []):
                file_id = file_info["id"]
                mime_type = file_info.get("mimeType")
                if mime_type == FOLDER_MIME_TYPE:
                    pending_folders.append(file_id)
                    continue
                if mime_type not in SUPPORTED_FILE_MIME_TYPES or file_id in visited_files:
                    continue
                visited_files.add(file_id)
                files.append(file_info)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return sorted(files, key=drive_file_priority)


def read_drive_files(
    file_infos: Sequence[dict[str, str]],
    progress_callback: Callable[[int, int, str], None] | None = None,
    file_started_callback: Callable[[int, int, str], None] | None = None,
    file_skipped_callback: Callable[[str, str, str], None] | None = None,
    file_failed_callback: Callable[[str, str, str], None] | None = None,
    before_file_callback: Callable[[], None] | None = None,
    indexed_file_modification_times: Mapping[str, str] | None = None,
) -> list[DriveDocument]:
    """Download changed files, skipping entries already represented in the index."""
    files_to_read = [
        file_info
        for file_info in file_infos
        if indexed_file_modification_times is None
        or indexed_file_modification_times.get(file_info["id"])
        != file_info.get("modifiedTime", "")
    ]
    if not files_to_read:
        return []

    service = _build_drive_service()
    documents: list[DriveDocument] = []
    total_files = len(files_to_read)
    for file_number, file_info in enumerate(files_to_read, start=1):
        file_id = file_info["id"]
        mime_type = file_info.get("mimeType")
        if mime_type not in SUPPORTED_FILE_MIME_TYPES:
            continue
        file_name = file_info.get("name", file_id)
        if before_file_callback:
            before_file_callback()
        if file_started_callback:
            file_started_callback(file_number, total_files, file_name)

        try:
            if mime_type == GOOGLE_DOC_MIME_TYPE:
                file_content = service.files().export(
                    fileId=file_id,
                    mimeType="text/plain",
                ).execute()
            else:
                request = service.files().get_media(
                    fileId=file_id,
                    supportsAllDrives=True,
                )
                content = io.BytesIO()
                downloader = MediaIoBaseDownload(content, request)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                file_content = content.getvalue()

            page_sections: tuple[tuple[str, str], ...] = ()
            if mime_type == PDF_MIME_TYPE:
                page_sections = tuple(_extract_pdf_pages(file_content))
                text = "\n".join(page_text for _, page_text in page_sections)
            elif mime_type == DOC_MIME_TYPE:
                text = _extract_doc_text(file_content)
            elif mime_type in {TEXT_MIME_TYPE, GOOGLE_DOC_MIME_TYPE}:
                text = _decode_text_file(file_content)
            else:
                text = _extract_docx_text(file_content)
            if not text:
                raise ValueError("Tidak ada teks yang dapat diekstrak dari file.")
        except PdfExtractionTimeout as error:
            if file_skipped_callback:
                file_skipped_callback(file_id, file_name, str(error))
            if file_failed_callback:
                file_failed_callback(file_id, file_name, str(error))
            text = ""
        except Exception as error:
            text = ""
            if file_failed_callback:
                file_failed_callback(file_id, file_name, str(error))

        if text:
            documents.append(
                DriveDocument(
                    file_id=file_id,
                    name=file_name,
                    text=text,
                    modified_time=file_info.get("modifiedTime", ""),
                    web_view_link=file_info.get("webViewLink", "")
                    or f"https://drive.google.com/file/d/{file_id}/view",
                    download_url=drive_download_url(file_info),
                    page_sections=page_sections,
                )
            )
        if progress_callback:
            progress_callback(
                file_number,
                total_files,
                file_name,
            )

    return documents


def read_docx_files(folder_id: str) -> list[DriveDocument]:
    """Read all DOCX and PDF files under a Drive folder."""
    return read_drive_files(list_drive_files(folder_id))
