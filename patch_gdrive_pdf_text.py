"""Tambal services/gdrive_service.py: spasi antar kata & huruf Arab pada PDF hasil OCR.

Jalankan dari root proyek:  python patch_gdrive_pdf_text.py
Cadangan: services/gdrive_service.py.bakpdf
Membutuhkan program pdftotext (paket poppler-utils).
"""
from pathlib import Path
import sys

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "services/gdrive_service.py")

EDITS = [
("import json\nimport shutil\n", "import json\nimport re\nimport shutil\n"),
("import sys\nfrom dataclasses import dataclass\n",
 "import sys\nimport unicodedata\nfrom dataclasses import dataclass\n"),
(
'''def _extract_pdf_pages(
    content: bytes,
    timeout_seconds: float = PDF_EXTRACTION_TIMEOUT_SECONDS,
) -> list[tuple[str, str]]:
    """Extract PDF text with pypdf, falling back to PyMuPDF when extraction fails."""
''',
'''_BIDI_CONTROL_RE = re.compile("[\\u202a-\\u202e\\u200e\\u200f]")
_ARABIC_PRESENTATION_RE = re.compile("[\\ufb50-\\ufdff\\ufe70-\\ufeff]+")
_SPACE_BEFORE_MARK_RE = re.compile(" (?=[\\u064b-\\u0652])")


def normalize_pdf_text(text: str) -> str:
    """Ubah huruf Arab 'bentuk presentasi' PDF menjadi huruf Arab standar."""
    text = _BIDI_CONTROL_RE.sub("", text)
    text = text.replace("\\u0627\\ufdf2", "\\u0627\\u0644\\u0644\\u0647")
    text = text.replace("\\ufdf2", "\\u0627\\u0644\\u0644\\u0647")
    text = _ARABIC_PRESENTATION_RE.sub(
        lambda match: unicodedata.normalize("NFKC", match.group()), text
    )
    return _SPACE_BEFORE_MARK_RE.sub("", text)


def _read_pdf_page_labels(content: bytes) -> list[str]:
    try:
        from pypdf import PdfReader

        return [str(label) for label in PdfReader(io.BytesIO(content)).page_labels]
    except Exception:
        return []


def _extract_pdf_pages_with_pdftotext(
    content: bytes,
    timeout_seconds: float,
) -> list[tuple[str, str]]:
    """Extract text with poppler's pdftotext, which keeps spaces between words."""
    executable = shutil.which("pdftotext")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "-enc", "UTF-8", "-", "-"],
            input=content,
            capture_output=True,
            check=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise PdfExtractionTimeout(
            f"PDF extraction exceeded {timeout_seconds:g} seconds"
        ) from error
    except (subprocess.CalledProcessError, OSError):
        return []

    labels = _read_pdf_page_labels(content)
    pages: list[tuple[str, str]] = []
    for number, page_text in enumerate(
        result.stdout.decode("utf-8", errors="replace").split("\\f"), start=1
    ):
        cleaned = page_text.strip()
        if not cleaned:
            continue
        label = labels[number - 1] if number <= len(labels) else ""
        pages.append((label.strip() or str(number), cleaned))
    return pages


def _extract_pdf_pages(
    content: bytes,
    timeout_seconds: float = PDF_EXTRACTION_TIMEOUT_SECONDS,
) -> list[tuple[str, str]]:
    """Extract PDF pages: pdftotext first, then pypdf/PyMuPDF; clean Arabic text."""
    pages = _extract_pdf_pages_with_pdftotext(content, timeout_seconds)
    if not pages:
        pages = _extract_pdf_pages_legacy(content, timeout_seconds)
    cleaned_pages = [(label, normalize_pdf_text(text)) for label, text in pages]
    return [(label, text) for label, text in cleaned_pages if text.strip()]


def _extract_pdf_pages_legacy(
    content: bytes,
    timeout_seconds: float = PDF_EXTRACTION_TIMEOUT_SECONDS,
) -> list[tuple[str, str]]:
    """Extract PDF text with pypdf, falling back to PyMuPDF when extraction fails."""
''',
),
]

def main() -> int:
    if not TARGET.is_file():
        print(f"File tidak ditemukan: {TARGET}"); return 1
    text = TARGET.read_text(encoding="utf-8")
    if "normalize_pdf_text" in text:
        print("Sudah pernah ditambal. Tidak ada yang diubah."); return 0
    for number, (old, new) in enumerate(EDITS, start=1):
        count = text.count(old)
        if count != 1:
            print(f"GAGAL pada perubahan #{number}: ditemukan {count}x (harus 1x). "
                  "Tidak ada file yang diubah."); return 1
        text = text.replace(old, new, 1)
    TARGET.with_suffix(".py.bakpdf").write_text(
        TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"Selesai: {len(EDITS)} perubahan diterapkan.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
