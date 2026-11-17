"""Pembersih teks OCR: membuang simbol sampah hasil OCR huruf Arab.

Simpan sebagai services/text_cleaner.py, lalu panggil clean_ocr_text(teks)
sebelum teks dipecah (split) dan di-embed.
"""
from __future__ import annotations

import re

ARAB_MARKER = "[teks Arab]"

# Arab asli (Unicode) dipertahankan; yang dibuang hanya sampah simbol.
_ARABIC_RE = re.compile(r"[\u0621-\u064a\u0671-\u06d3\u0750-\u077f]")
_WORDCHAR_RE = re.compile(r"[A-Za-z0-9\u00c0-\u024f]")
_EDGE_PUNCT = ".,;:!?()[]{}\"'“”‘’-–—/%"


def _is_bad_token(token: str) -> bool:
    """Token dianggap sampah jika hampir seluruhnya simbol."""
    if _ARABIC_RE.search(token):
        return False
    stripped = token.strip(_EDGE_PUNCT)
    if not stripped:
        return False  # hanya tanda baca biasa, aman
    good = len(_WORDCHAR_RE.findall(stripped))
    return good / len(stripped) < 0.6


def _is_garbage_line(tokens: list[str]) -> bool:
    if len(tokens) < 4:
        return False
    singles = sum(1 for t in tokens if len(t.strip(_EDGE_PUNCT)) == 1)
    return singles / len(tokens) > 0.5


def _clean_line(line: str) -> str:
    tokens = line.split()
    if not tokens:
        return ""
    if _is_garbage_line(tokens):
        return ARAB_MARKER

    out: list[str] = []
    run = 0  # panjang rangkaian token sampah berturut-turut

    def flush_run() -> None:
        nonlocal run
        if run >= 2:
            out.append(ARAB_MARKER)
        run = 0  # 1 token sampah tunggal dibuang begitu saja

    for tok in tokens:
        if _is_bad_token(tok):
            run += 1
        else:
            flush_run()
            out.append(tok)
    flush_run()
    return " ".join(out)


def clean_ocr_text(text: str) -> str:
    """Bersihkan teks; baris sampah diganti satu penanda [teks Arab]."""
    cleaned: list[str] = []
    for raw in text.splitlines():
        line = _clean_line(raw)
        # gabungkan penanda berurutan
        if line == ARAB_MARKER and cleaned and cleaned[-1] == ARAB_MARKER:
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    return re.sub(r"\n{3,}", "\n\n", result).strip()


if __name__ == "__main__":
    contoh = (
        "Hukum membaca doa qunut menurut mazhab Syafi'i adalah sunnah.\n"
        "~| ¢ ﻻ ، ٠ | ^ © §\n"
        "Dalilnya adalah hadits ¢ ~| ﻻ ، riwayat Imam Ahmad.\n"
        "j s y l a o i u\n"
        "Wallahu a'lam bish-shawab (QS. Al-Baqarah: 43)."
    )
    print(clean_ocr_text(contoh))
