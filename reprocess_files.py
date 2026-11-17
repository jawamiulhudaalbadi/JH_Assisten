"""Tandai file tertentu agar diekstrak & di-embed ulang pada sinkronisasi berikutnya.

Contoh (dari root proyek):
    python reprocess_files.py "0201-0400.pdf"
    python reprocess_files.py "0201-0400.pdf" "0001-0200.pdf"
    python reprocess_files.py --list          # tampilkan nama file di manifest
Cadangan manifest: faiss_index/drive_manifest.json.bak
"""
import json
import shutil
import sys
from pathlib import Path

MANIFEST = Path("faiss_index/drive_manifest.json")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    if not MANIFEST.is_file():
        print(f"Manifest tidak ditemukan: {MANIFEST}")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    sources = manifest.get("sources", {})
    names = {fid: str(src.get("name", "")) for fid, src in sources.items() if isinstance(src, dict)}

    if args == ["--list"]:
        for name in sorted(names.values(), key=str.casefold):
            print(name)
        return 0

    wanted = {name.casefold() for name in args}
    matched = [fid for fid, name in names.items() if name.casefold() in wanted and fid in files]
    found_names = {names[fid].casefold() for fid in matched}
    for name in args:
        if name.casefold() not in found_names:
            print(f"Tidak ditemukan di manifest: {name}")
    if not matched:
        return 1

    shutil.copy2(MANIFEST, MANIFEST.with_suffix(".json.bak"))
    for fid in matched:
        files[fid] = ""  # beda dari waktu ubah Drive -> dianggap berubah
        print(f"Akan diproses ulang: {names[fid]}")
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print("Selesai. Jalankan sinkronisasi (Sinkronkan Sekarang) di Codespaces.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
