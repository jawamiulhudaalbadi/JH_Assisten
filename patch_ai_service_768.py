"""Tambal services/ai_service.py agar embedding dipotong ke 768 dimensi.

Jalankan dari root proyek:  python patch_ai_service_768.py
Buat cadangan services/ai_service.py.bak768. Jalankan SETELAH atau BERSAMAAN dengan
convert_index_768.py (indeks dan kode harus sama-sama 768 dimensi).
"""
from pathlib import Path
import sys

TARGET = Path(sys.argv[1] if len(sys.argv) > 1 else "services/ai_service.py")

EDITS = [
(
"_CHAT_MODEL_CACHE_LOCK = Lock()\n",
'''_CHAT_MODEL_CACHE_LOCK = Lock()
EMBEDDING_DIMENSIONS = 768


def reduce_embedding_vector(vector: Sequence[float]) -> list[float]:
    """Potong vektor Gemini ke EMBEDDING_DIMENSIONS lalu normalisasi L2."""
    if len(vector) <= EMBEDDING_DIMENSIONS:
        return list(vector)
    reduced = np.asarray(vector[:EMBEDDING_DIMENSIONS], dtype=np.float32)
    norm = float(np.linalg.norm(reduced))
    if norm > 0:
        reduced = reduced / norm
    return reduced.tolist()
''',
),
(
"                vectors.extend(batch_vectors)\n",
"                vectors.extend(reduce_embedding_vector(vector) for vector in batch_vectors)\n",
),
(
"        return self._client.embed_query(text)\n",
"        return reduce_embedding_vector(self._client.embed_query(text))\n",
),
]

def main() -> int:
    if not TARGET.is_file():
        print(f"File tidak ditemukan: {TARGET}")
        return 1
    text = TARGET.read_text(encoding="utf-8")
    if "reduce_embedding_vector" in text:
        print("Sudah pernah ditambal. Tidak ada yang diubah.")
        return 0
    for number, (old, new) in enumerate(EDITS, start=1):
        count = text.count(old)
        if count != 1:
            print(f"GAGAL pada perubahan #{number}: ditemukan {count}x (harus 1x). "
                  "Tidak ada file yang diubah.")
            return 1
        text = text.replace(old, new, 1)
    TARGET.with_suffix(".py.bak768").write_text(
        TARGET.read_text(encoding="utf-8"), encoding="utf-8")
    TARGET.write_text(text, encoding="utf-8")
    print(f"Selesai: {len(EDITS)} perubahan diterapkan.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
