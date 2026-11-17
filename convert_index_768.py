"""Perkecil indeks FAISS dari 3072 menjadi 768 dimensi (tanpa embedding ulang).

Jalankan dari root proyek, di terminal Codespaces:
    python convert_index_768.py            # memakai folder faiss_index
    python convert_index_768.py faiss_index

Aman dijalankan: hasil ditulis ke file sementara, diperiksa dulu, dan indeks asli
disalin ke <folder>_backup_3072 sebelum diganti. index.pkl tidak berubah.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import faiss
import numpy as np

TARGET_DIMENSIONS = 768


def reduce_matrix(vectors: np.ndarray, dimensions: int = TARGET_DIMENSIONS) -> np.ndarray:
    """Ambil `dimensions` angka pertama tiap vektor, lalu normalisasi L2."""
    reduced = np.ascontiguousarray(vectors[:, :dimensions], dtype=np.float32)
    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return reduced / norms


def main() -> int:
    directory = Path(sys.argv[1] if len(sys.argv) > 1 else "faiss_index")
    source = directory / "index.faiss"
    if not source.is_file():
        print(f"File tidak ditemukan: {source}")
        return 1

    index = faiss.read_index(str(source))
    print(f"Indeks sekarang: {index.ntotal} vektor, {index.d} dimensi")
    if index.d == TARGET_DIMENSIONS:
        print("Sudah 768 dimensi. Tidak ada yang diubah.")
        return 0
    if index.d < TARGET_DIMENSIONS:
        print("Dimensi indeks lebih kecil dari target. Dibatalkan.")
        return 1

    try:
        vectors = index.reconstruct_n(0, index.ntotal)
    except Exception as error:  # indeks non-flat tidak bisa direkonstruksi
        print(f"Jenis indeks ini tidak didukung skrip ini: {error}")
        return 1

    reduced = reduce_matrix(np.asarray(vectors))
    del vectors
    if getattr(index, "metric_type", faiss.METRIC_L2) == faiss.METRIC_INNER_PRODUCT:
        new_index = faiss.IndexFlatIP(TARGET_DIMENSIONS)
    else:
        new_index = faiss.IndexFlatL2(TARGET_DIMENSIONS)
    new_index.add(reduced)

    temporary = directory / "index.faiss.new"
    faiss.write_index(new_index, str(temporary))

    # Periksa hasil tulis sebelum menggantikan indeks asli.
    check = faiss.read_index(str(temporary))
    if check.d != TARGET_DIMENSIONS or check.ntotal != index.ntotal:
        temporary.unlink(missing_ok=True)
        print("Pemeriksaan gagal (jumlah/dimensi tidak cocok). Indeks asli tidak diubah.")
        return 1
    sample_rows = np.linspace(0, check.ntotal - 1, num=min(5, check.ntotal), dtype=int)
    _, found = check.search(reduced[sample_rows], 1)
    if not np.array_equal(found[:, 0], sample_rows):
        temporary.unlink(missing_ok=True)
        print("Pemeriksaan pencarian gagal. Indeks asli tidak diubah.")
        return 1

    backup_directory = directory.parent / f"{directory.name}_backup_3072"
    backup_directory.mkdir(exist_ok=True)
    shutil.copy2(source, backup_directory / "index.faiss")
    temporary.replace(source)
    print(
        f"Selesai: {check.ntotal} vektor, sekarang {check.d} dimensi. "
        f"Cadangan asli ada di {backup_directory}/ (jangan di-commit)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
