from pathlib import Path
import shutil

path = Path("services/ai_service.py")
shutil.copy(path, path.with_suffix(".py.bak"))
s = path.read_text(encoding="utf-8")


def rep(old, new):
    global s
    assert s.count(old) == 1, f"Potongan tidak ditemukan/tidak unik: {old[:70]!r}"
    s = s.replace(old, new)


# 1. Temperature (Gemini 3 tetap 1.0 sesuai saran Google)
rep(
    "def _create_chat_model(",
    '''CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.0") or 0.0)


def _chat_temperature(model_name: str) -> float:
    return 1.0 if "gemini-3" in model_name.casefold() else CHAT_TEMPERATURE


def _create_chat_model(''',
)
rep("        temperature=0.2,\n", "        temperature=_chat_temperature(model_name),\n")

# 2. Nilai k
rep(
    "DISCUSSION_RETRIEVAL_COUNT = 10\nDISCUSSION_MAX_CHUNKS = 12",
    "DISCUSSION_RETRIEVAL_COUNT = 6\nDISCUSSION_MAX_CHUNKS = 10",
)
rep("        retrieval_count = 8\n", "        retrieval_count = 6\n")

# 3. Aturan baca ketat (semua topik) + pengurutan chunk
rep(
    "def is_full_context_request(question: str) -> bool:",
    '''STRICT_READING_RULES = (
    "ATURAN MEMBACA SUMBER (WAJIB UNTUK SEMUA TOPIK; jika pengguna meminta menyalin "
    "teks, salin persis tanpa mengubahnya). "
    "1) Jawab HANYA dari konteks. Jangan menambah fakta, angka, hukum, atau kesimpulan "
    "yang tidak tertulis. Jika tidak ada di konteks, katakan tidak ditemukan. "
    "2) Konteks disusun berurutan sesuai dokumen. Baca dari awal sampai akhir tanpa "
    "melompati bagian. Teks asli beserta terjemahannya adalah satu kesatuan. "
    "3) Jika sumber membagi sesuatu ke dalam kategori, jenis, bab, atau poin, setiap "
    "bagian punya definisi, contoh, dan ketentuannya sendiri. Jangan memindahkan "
    "ketentuan satu bagian ke bagian lain dan jangan menggabungkan dua bagian yang berbeda. "
    "4) Pisahkan sifat/status suatu hal dari ketentuan penggunaannya atau keterangan "
    "tambahannya. Keterangan tambahan tidak mengubah statusnya. "
    "5) Sebelum menjawab, periksa tiap butir jawaban: bagian mana dari konteks yang "
    "mendukungnya, dan apakah tidak tertukar dengan bagian lain. "
    "6) Jika konteks saling bertentangan atau tidak lengkap, sebutkan apa adanya. "
    "CONTOH POLA PENALARAN: "
    "(a) Fikih: air musyammas (dipanaskan matahari dalam wadah logam selain emas dan "
    "perak) berstatus suci menyucikan, hukum penggunaan di badan makruh. Ia TIDAK masuk "
    "suci tidak menyucikan. Air musta'mal (bekas bersuci wajib) berstatus suci tidak "
    "menyucikan, itu kategori lain. Status zat (suci menyucikan, suci tidak menyucikan, "
    "najis) harus dipisah dari hukum penggunaan (mubah, makruh, haram). "
    "(b) Umum: jika dokumen menyebut produk A berstatus aktif tetapi dibatasi, jawab "
    "keduanya: A aktif, dengan pembatasannya. Jangan menulis A nonaktif. "
    "Jika redaksi konteks berbeda dari contoh, ikuti konteks. "
    "Untuk pertanyaan klasifikasi, gunakan urutan: kategori, status, ketentuan, rujukan."
)


def _order_matches_for_reading(matches: list[Document]) -> list[Document]:
    file_order: dict[str, int] = {}
    for document in matches:
        file_order.setdefault(str(document.metadata.get("file_id", "")), len(file_order))

    def key(document: Document) -> tuple[int, int]:
        try:
            chunk_index = int(document.metadata.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = 10**9
        return (file_order[str(document.metadata.get("file_id", ""))], chunk_index)

    return sorted(matches, key=key)


def is_full_context_request(question: str) -> bool:''',
)

rep(
    "    elif discussion_mode and index is not None and matches:\n"
    "        matches = _expand_discussion_neighbors(index, matches)\n",
    "    elif discussion_mode and index is not None and matches:\n"
    "        matches = _expand_discussion_neighbors(index, matches)\n"
    "    if not full_context_mode and matches:\n"
    "        matches = _order_matches_for_reading(matches)\n",
)

rep(
    "        + completeness_instruction\n"
    '        + (" " + source_line_instruction if source_line_instruction else "")\n'
    "    )",
    "        + completeness_instruction\n"
    '        + (" " + source_line_instruction if source_line_instruction else "")\n'
    '        + (" " + STRICT_READING_RULES if matches else "")\n'
    "    )",
)

path.write_text(s, encoding="utf-8")
print("Selesai. Backup: services/ai_service.py.bak")