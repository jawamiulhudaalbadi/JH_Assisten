"""Unduh FULL TEXT PISS-KTB dari archive.org, buang Arab rusak, pecah per BAB/bagian.

Cara pakai (di terminal Codespace, di folder project):
    python prepare_piss_ktb.py

Hasil: folder piss_ktb_split/ berisi file .txt siap diunggah ke Google Drive.
Butuh: pip install requests
File text_cleaner.py harus ada di folder yang sama (atau di services/).
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path

import requests

from text_cleaner import clean_ocr_text

URL = (
    "https://archive.org/download/piss-ktb-kumpulan-tanya-jawab-keagamaan-2015/"
    "PISS-KTB%20-%20Kumpulan%20Tanya-Jawab%20Keagamaan%20-%202015_djvu.txt"
)
OUT_DIR = Path("piss_ktb_split")
MAX_PART_CHARS = 250_000   # ukuran maksimal satu file; bab lebih besar dipecah lagi
MIN_CHAPTER_CHARS = 30_000  # segmen lebih kecil dari ini bukan bab sungguhan

BAB_RE = re.compile(r"^[ \t]*BAB[ \t]+(\d{1,2})\b[ \t]*(.*)$", re.M)
ENTRY_RE = re.compile(r"^[ \t]*(?:F?\d{4})\.[ \t]+\S", re.M)  # "0018. Judul"
NOISE_LINE_RE = re.compile(
    r"^\s*(?:[@©O0]\s*2015\s*www\.piss-ktb\.com.*|PUSTAKA ILMU SUNNI SALAFIYAH - KTB"
    r"|DAFTAR ISI|\|?\s*\d{1,4}\s*\|?)\s*$",
    re.I,
)


def download() -> str:
    print("Mengunduh teks dari archive.org ...")
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    r.encoding = "utf-8"
    text = r.text
    if "<html" in text[:500].lower():  # jaga-jaga kalau yang terambil halaman HTML
        text = html.unescape(re.sub(r"<[^>]+>", "", text))
    print(f"  {len(text):,} karakter")
    return text


def strip_noise(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not NOISE_LINE_RE.match(l))


def body_start(text: str) -> int:
    """Isi buku dimulai setelah 'DAFTAR ISI' terakhir (header berjalan di halaman TOC)."""
    last = 0
    for m in re.finditer(r"DAFTAR ISI", text):
        last = m.end()
    return last


def find_chapters(text: str) -> list[tuple[int, int, str]]:
    """Kembalikan [(posisi, nomor_bab, judul)], hanya BAB berurutan di badan buku."""
    found: list[tuple[int, int, str]] = []
    expected = 1
    for m in BAB_RE.finditer(text):
        n = int(m.group(1))
        if expected <= n <= expected + 2:  # toleransi OCR melewatkan satu bab
            found.append((m.start(), n, m.group(2).strip()))
            expected = n + 1
    # buang yang segmennya terlalu pendek (daftar isi tersisa / sub-bab)
    keep: list[tuple[int, int, str]] = []
    for i, (pos, n, title) in enumerate(found):
        end = found[i + 1][0] if i + 1 < len(found) else len(text)
        if end - pos >= MIN_CHAPTER_CHARS:
            keep.append((pos, n, title))
    return keep



ENTRY_ID_RE = re.compile(r"^[ \t]*(F?\d{4})\.[ \t]+\S", re.M)


def group_by_entries(text: str) -> list[tuple[str, str]]:
    """Kelompokkan entri bernomor berurutan jadi file ~MAX_PART_CHARS.
    Nama file memakai rentang nomor entri, mis. 0018-0445."""
    heads = [(m.start(), m.group(1)) for m in ENTRY_ID_RE.finditer(text)]
    if len(heads) < 200:
        return []
    out: list[tuple[str, str]] = []
    if heads[0][0] > 3000:
        out.append(("awal", text[: heads[0][0]].strip()))
    begin_i = 0
    for i in range(1, len(heads) + 1):
        end_pos = heads[i][0] if i < len(heads) else len(text)
        if end_pos - heads[begin_i][0] >= MAX_PART_CHARS or i == len(heads):
            chunk = text[heads[begin_i][0] : end_pos].strip()
            name = f"{heads[begin_i][1]}-{heads[i - 1][1]}"
            out.append((name, chunk))
            begin_i = i
    return out


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:40] or "bagian"


def split_by_size(text: str) -> list[str]:
    """Pecah teks besar di batas entri (nomor judul) atau paragraf."""
    if len(text) <= MAX_PART_CHARS:
        return [text]
    cuts = [m.start() for m in ENTRY_RE.finditer(text)]
    if len(cuts) < 3:
        cuts = [m.end() for m in re.finditer(r"\n\s*\n", text)]
    parts, begin, last_ok = [], 0, 0
    for c in cuts:
        if c - begin > MAX_PART_CHARS and last_ok > begin:
            parts.append(text[begin:last_ok])
            begin = last_ok
        last_ok = c
    parts.append(text[begin:])
    # jaga-jaga: potong paksa bila masih ada bagian raksasa
    final: list[str] = []
    for p in parts:
        while len(p) > MAX_PART_CHARS * 1.5:
            final.append(p[:MAX_PART_CHARS])
            p = p[MAX_PART_CHARS:]
        final.append(p)
    return [p.strip() for p in final if p.strip()]


def main() -> None:
    raw = download()
    print("Membersihkan header/footer dan Arab rusak ...")
    # buang front matter + daftar isi dulu (sebelum header "DAFTAR ISI" ikut terhapus)
    start = body_start(raw)
    if start < len(raw) * 0.5:  # sanity check: daftar isi ada di bagian awal buku
        raw = raw[start:]
    text = clean_ocr_text(strip_noise(raw))

    OUT_DIR.mkdir(exist_ok=True)
    for old in OUT_DIR.glob("*.txt"):
        old.unlink()

    segments = group_by_entries(text)
    if segments:
        print(f"Terdeteksi entri bernomor; dikelompokkan jadi {len(segments)} file.")
    else:
        print("Entri bernomor tidak terdeteksi; memecah berdasarkan ukuran saja.")
        segments = [(f"bagian{i:02d}", p) for i, p in enumerate(split_by_size(text), 1)]

    total_files = 0
    for name, content in segments:
        (OUT_DIR / f"PISS-KTB_{name}.txt").write_text(content, encoding="utf-8")
        total_files += 1
        print(f"  PISS-KTB_{name}.txt  ({len(content):,} karakter)")

    print(f"\nSelesai: {total_files} file di folder '{OUT_DIR}/'.")
    print("Periksa dulu 2-3 file, lalu unggah seluruh folder ke Google Drive.")


if __name__ == "__main__":
    sys.exit(main())