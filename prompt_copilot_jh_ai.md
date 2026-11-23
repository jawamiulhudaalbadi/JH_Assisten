# PROMPT UNTUK COPILOT — Riwayat per pengguna + Dashboard Admin tersembunyi (JH Asisten AI)

> Sebelum ditempel: isi bagian bertanda `[ISI]`, lalu buka semua file proyek di editor
> (app.py, services/ai_service.py, services/sync_manager.py, services/gdrive_service.py,
> requirements.txt). Gunakan mode Agent/Edit supaya Copilot bisa mengubah banyak file.

---

## 0. Konteks

Aplikasi "JH Asisten AI" adalah chatbot RAG (FAISS + Gemini via LangChain) berbahasa Sunda santri /
Indonesia, dijalankan di **Streamlit Community Cloud** dan dibuka pengguna lewat **APK** (WebView).
APK dibuat dengan: `[https://freewebtoapk.com/]`.
Nama variabel yang sudah ada di Streamlit Secrets (tanpa isi): `[Nama variabel Secrets yang sudah ada: GOOGLE_API_KEY, GOOGLE_DRIVE_FOLDER_ID,
gdrive_credentials_json, gdrive_token_json, gdrive_service_account_json,
GEMINI_CHAT_MODELS, GEMINI_THINKING_LEVEL. Mungkin masih ada yang lain (misalnya
password menu pengaturan); cari di kode semua pemakaian st.secrets dan os.getenv
lalu daftar semuanya sebelum mulai bekerja.]`.

Masalah sekarang:
1. Riwayat chat tercampur untuk semua pengguna.
2. Admin harus membuka Streamlit/Codespace/Claude untuk mengelola aplikasi. Versi APK di HP tidak punya
   menu "Manage app" Streamlit.

## 1. Aturan wajib (berlaku untuk semua bagian)

- **100% gratis.** Hanya boleh memakai Streamlit Community Cloud + Firestore (Firebase paket Spark,
  **database default**, bukan database bernama karena itu butuh billing) + library pip gratis.
- **Ringan.** Tidak ada polling, tidak ada `st_autorefresh`, tidak ada timer. Import berat (firestore,
  cryptography) dilakukan malas (lazy) di dalam fungsi. Klien Firestore dibuat sekali dengan
  `@st.cache_resource`. Semua penulisan ke database berjalan di thread latar (daemon thread atau
  `ThreadPoolExecutor(max_workers=2)`) agar jawaban chat tidak tertunda.
- **Fail-soft.** Jika Firestore gagal/terkena kuota, chat TETAP jalan (riwayat jatuh ke
  `st.session_state` saja) dan tidak ada error yang tampil ke pengguna biasa. Error hanya dicatat dan
  tampil di dashboard admin.
- **Hemat kuota Firestore** (gratis: 50.000 baca & 20.000 tulis per hari). Target maksimal 3–4 tulis per
  pertanyaan. Konfigurasi dibaca dengan cache memori (TTL 5 menit; status key TTL 60 detik).
- **Jangan merusak fitur yang sudah ada** (RAG, streaming jawaban, suara, lampiran gambar, sumber,
  tombol salin, sinkron Drive, menu pengaturan). Ubah seminimal mungkin, jangan tulis ulang file besar.
- **Tanpa login, tanpa identitas.** Pengguna hanya dikenali ID perangkat acak. Admin hanya melihat ID
  pendek (misal 6 karakter pertama). Tidak ada pemberitahuan privasi tambahan.
- **Keamanan.** Tidak ada key/kode admin di dalam kode atau di APK. Semua rahasia di Streamlit Secrets.
  Aturan Firestore ditutup total (`allow read, write: if false;`) karena akses hanya lewat service account
  dari server.
- **Teks untuk pengguna biasa** memakai gaya Sunda santri yang SAMA dengan teks yang sudah ada di
  app.py (baca dulu contoh-contohnya dan tiru nadanya). Kumpulkan semua string baru di satu konstanta
  (misal `TEKS = {...}`) agar mudah diedit. Teks dashboard admin memakai bahasa Indonesia yang lugas.
- Berikan type hints dan docstring singkat berbahasa Indonesia. Jangan menambah dependensi selain yang
  disebut di bawah (`google-cloud-firestore`, `cryptography`, dan satu komponen ID perangkat jika perlu,
  versi dipin).

## 2. Fitur A — Riwayat chat per pengguna

**ID perangkat.** Urutan prioritas:
1. Parameter URL `?uid=` bila ada dan cocok regex `^[a-f0-9]{16,32}$` (APK bisa mengirimnya).
2. ID tersimpan di localStorage browser/WebView lewat satu komponen ringan (pilih yang paling kecil dan
   stabil, pin versinya). Jika belum ada, buat `uuid4().hex` lalu simpan.
3. Cadangan: ID sesi (riwayat hanya hidup selama sesi).
Baca ID sekali di awal sesi, simpan di `st.session_state`, dan jangan menunda render pertama.

**Struktur Firestore (database default):**
- `users/{uid}` : `first_seen`, `last_active`, `total_questions`, `blocked` (bool), `day`, `day_count`.
- `users/{uid}/chunks/{n}` : `messages` (list peran/teks/waktu/model_used/sumber), `count`.
  Satu chunk memuat maksimal 20 pesan. Menambah pesan = satu tulis ke chunk aktif (pertanyaan +
  jawaban sekaligus). Buat chunk baru bila penuh.
- Muat riwayat: hanya 1–2 chunk terakhir (sekitar 30 pesan) saat sesi dibuka. Tidak ada pemuatan penuh.

**Penghapusan manual oleh pengguna (tidak ada penghapusan otomatis):**
- Tombol "Bersihkan riwayat" di sidebar/menu pengaturan, dengan konfirmasi. Menghapus semua chunk
  pengguna itu dalam batch dan mereset `total_questions` riwayat. Salinan di dashboard admin juga hilang.
- Bila jumlah pesan > 200 atau chunk > 10 (konstanta yang mudah diubah), tampilkan pemberitahuan (gaya
  Sunda santri) bahwa riwayat sudah panjang dan bisa lambat, disertai tombol bersihkan.
- Hitungan "pertanyaan sering" (Fitur C) tetap tersimpan sebagai angka anonim walau riwayat dihapus.

## 3. Fitur B — Dashboard admin tersembunyi

**Cara membuka.** Tidak ada tombol/menu di tampilan biasa. Admin mengetik di kolom chat:
`/admin <kode>`. Server membandingkan dengan hash di Secrets (`ADMIN_CODE_HASH`, gunakan
`hashlib.scrypt`/PBKDF2 dengan salt, bandingkan dengan `hmac.compare_digest`).
- Teks `/admin ...` TIDAK pernah dikirim ke Gemini, TIDAK disimpan ke riwayat, TIDAK masuk statistik.
- Batasi percobaan salah: 5 kali per sesi lalu kunci 15 menit; catat percobaan gagal (hanya ID pendek).
- Status admin hanya di `st.session_state` (hilang saat sesi berakhir). Ada tombol Keluar.
- **Kode dashboard (`services/admin_dashboard.py`) di-import hanya setelah kode benar** supaya pengguna
  biasa tidak menanggung bebannya. Dashboard tidak memuat ulang otomatis; ada tombol "Segarkan".

**Menu dashboard:**
1. **Statistik**: total perangkat (pernah membuka), aktif 1 jam terakhir ("sedang online"), pengguna
   unik hari ini & 7 hari, pertanyaan per hari (grafik sederhana 14 hari). Dibaca dari dokumen ringkasan,
   bukan memindai koleksi pengguna.
2. **Riwayat chat pengguna**: daftar ID pendek berpaginasi (urut terakhir aktif), buka satu pengguna
   untuk melihat percakapannya (dimuat per halaman), pencarian kata kunci pada pengguna yang dibuka.
3. **Pertanyaan sering** (Fitur C di bawah).
4. **API Key** (Fitur D di bawah).
5. **Urutan model Gemini** (Fitur E).
6. **Sinkron Drive**: HANYA tombol "Sinkron sekarang" yang memanggil logika sinkron yang sudah ada
   (`sync_manager`). Keterangan jumlah file dan status sinkron tetap di sidebar seperti sekarang.
7. **Kontrol lain**: mode perawatan (pesan khusus untuk pengguna), pengumuman singkat di atas chat,
   blokir/buka blokir ID pengguna, batas pertanyaan per hari per pengguna (0 = tanpa batas; penimpa key
   milik pengguna tidak dihitung), bersihkan cache (`st.cache_data.clear()` / `st.cache_resource` yang
   relevan), ekspor CSV (statistik dan riwayat satu pengguna).
8. **Kesehatan sistem**: model terakhir yang menjawab, latensi dan status cooldown tiap model (dari
   `ChatModelRouter.snapshot()` bila sudah ada di ai_service.py; jangan ditulis ulang), serta 20 error
   terakhir (disimpan di satu dokumen log berukuran terbatas, bukan koleksi tak terbatas).

Catatan batasan: reboot app, log, dan edit Secrets milik Streamlit Cloud TIDAK bisa dikendalikan dari
dalam aplikasi. Jangan mencoba memalsukannya.

## 4. Fitur C — Status online & statistik hemat tulis

- "Sedang online" = aktif dalam 1 jam terakhir. Penulisan `last_active` maksimal **sekali per jam per
  perangkat**, hanya saat pengguna berinteraksi (tanpa timer). Cek dengan waktu terakhir tulis di
  `st.session_state`.
- Saat pengguna bertanya, `last_active` ikut diperbarui pada penulisan dokumen `users/{uid}` yang memang
  terjadi (tidak menambah tulis).
- Dokumen ringkasan: `stats/daily_YYYYMMDD` (`questions` via `Increment`, `uids` via `ArrayUnion`) dan
  `stats/hourly_YYYYMMDDHH` (`uids` via `ArrayUnion`). Dokumen global `stats/global` menyimpan
  `total_devices` (naik 1 saat perangkat baru pertama kali tercatat). Jaga ukuran dokumen < 1 MiB.
- **Pertanyaan sering**: koleksi `faq/{kunci_normal}` berisi `count` (via `Increment`) dan satu contoh
  teks. Normalisasi: huruf kecil, buang tanda baca, buang kata umum Indonesia/Sunda. TANPA ID pengguna.
  Dihitung dengan kode biasa (gratis). Tombol "Rangkum dengan AI" (kelompok key admin) hanya jalan saat
  diklik, hasilnya disimpan di `admin_cache/faq_summary` dan dipakai ulang tanpa memanggil Gemini lagi.

## 5. Fitur D — Tiga kelompok API key terpisah total

Tiga daftar key: **chat**, **indexing**, **admin**. Tidak ada pinjam-meminjam antar kelompok. Bila
satu kelompok habis, dua lainnya tetap jalan.

| Kelompok | Dipakai untuk | Bila semua key habis |
|---|---|---|
| chat | jawaban untuk pengguna | chat berhenti sementara dengan pesan jelas |
| indexing | embedding & sinkron Drive | sinkron tertunda, chat tidak terganggu |
| admin | fitur AI di dashboard | turun ke versi tanpa AI (hitung kata) |

- Penyimpanan: `config/keys_chat`, `config/keys_index`, `config/keys_admin`. Tiap key disimpan
  **terenkripsi Fernet** (kunci `ENCRYPTION_KEY` di Secrets). Setiap entri: `id` (sidik jari pendek),
  `label`, `enabled`, `cooldown_until`, `last_error`. Di UI key tampil tersamar (`AIza…x7Qk`).
- Dashboard: tambah, hapus, aktif/nonaktifkan, dan lihat status (aktif / kena limit sampai jam X /
  nonaktif) untuk tiap kelompok.
- **Rotasi otomatis saat limit harian (RPD).**
  - Chat: limit harian Gemini berlaku per model. Urutan: pindah ke model lain di key yang sama dulu,
    baru ganti key bila model-model teratas habis. Cooldown RPD berlangsung sampai reset harian (tengah
    malam waktu Pasifik). Gunakan penanda error yang sudah ada (`_is_chat_quota_error`,
    `get_quota_retry_delay_seconds`).
  - Indexing: pindah key saat kuota embedding harian habis; hubungkan di titik yang sudah ada
    (`is_embedding_daily_quota_error`, `is_embedding_quota_error`) di `ai_service.py`/`sync_manager.py`.
  - Admin: pindah key saat limit; bila semua habis, fitur AI dinonaktifkan sementara.
  - Key yang sedang cooldown dicoba lagi otomatis setelah waktunya lewat.
- **Key bawaan Secrets** tetap menjadi sumber awal (seed): bila daftar di database kosong, pakai key dari
  Secrets seperti sekarang.
- **Penimpa key di menu pengaturan** (sudah ada, bisa diakses pengguna) menjadi aturan khusus:
  - Hanya berlaku untuk **chat**, hanya untuk **pengguna itu**, hanya di `st.session_state`
    (TIDAK ditulis ke database, TIDAK tampil di dashboard, TIDAK dicatat di log).
  - Selama penimpa aktif, rotasi otomatis ke key default berhenti. Bila key penimpa kena limit, tampilkan
    pesan jelas (gaya Sunda santri), jangan diam-diam pindah ke key default.
  - Key default tidak terhapus/terubah; tombol "kembali ke API key konfigurasi" menghapus penimpa.
  - Pemakaian lewat penimpa tidak dihitung ke batas harian pengguna.
  - Tidak berpengaruh ke kelompok indexing dan admin.
Secrets opsional untuk key awal: GOOGLE_API_KEY (chat), GOOGLE_API_KEY_INDEX
(indexing), GOOGLE_API_KEY_ADMIN (admin). Bila yang terakhir dua itu tidak ada,
kelompoknya kosong sampai admin menambahkan key dari dashboard (jangan meminjam key chat).
## 6. Fitur E — Urutan model Gemini dari dashboard

Prioritas sumber: **pengaturan dashboard admin > variabel urutan model di Secrets (yang sudah ada) >
otomatis**.
- Mode **Otomatis**: router memilih model terbaru yang sehat dan cepat (flash di atas flash-lite, versi
  terbaru dulu, memakai latensi token pertama terukur dan cooldown bertingkat).
- Mode **Manual**: admin mengatur daftar model berurutan; model pertama selalu dipanggil lebih dulu.
  Model yang sedang gagal/cooldown tetap dilewati sementara agar tidak macet, lalu dicoba lagi otomatis.
- Daftar pilihan model diambil dari `get_available_chat_models` (hasil cache), bukan diketik tangan.
- Simpan di `config/model_order`. Bila router (`ChatModelRouter`) sudah ada di `ai_service.py`, JANGAN
  ditulis ulang. Cukup beri titik masuk agar urutan manual bisa menimpa hasil `rank()`.

## 7. Kerangka berkas yang diharapkan

- `services/firestore_store.py` — klien (cache_resource), riwayat per pengguna, statistik, faq, config,
  penulisan latar belakang, fail-soft.
- `services/key_pool.py` — tiga kelompok key, enkripsi/dekripsi, rotasi, cooldown, API `get_key(group)`,
  `report_limit(group, key_id, error)`.
- `services/admin_dashboard.py` — seluruh UI dashboard (hanya di-import setelah kode benar).
- Perubahan kecil di `app.py`, `ai_service.py`, `sync_manager.py` (hook key & urutan model, tombol sinkron
  dipindah ke dashboard).
- `requirements.txt` (tambahan dipin), `.streamlit/secrets.toml.example` (nama saja, tanpa isi),
  `PETUNJUK_SETUP.md` (langkah Firebase: buat project, aktifkan Firestore database default, buat service
  account, tempel JSON ke Secrets, set aturan tutup total, cara membuat `ADMIN_CODE_HASH` dan
  `ENCRYPTION_KEY`).

Secrets baru yang diperlukan (nama saja): `FIREBASE_SERVICE_ACCOUNT` (tabel/JSON), `ADMIN_CODE_HASH`
(+ salt), `ENCRYPTION_KEY`.

## 8. Cara kerja (penting)

Kerjakan **bertahap** dan berhenti setelah tiap fase untuk menunjukkan ringkasan perubahan serta cara
mengetesnya:
1. Fase 1: ID perangkat + `firestore_store` + riwayat per pengguna + tombol bersihkan + pemberitahuan.
2. Fase 2: pembuka dashboard (`/admin`), statistik, riwayat semua pengguna, pertanyaan sering, tombol
   sinkron, kontrol lain.
3. Fase 3: `key_pool` tiga kelompok + penimpa chat per pengguna + rotasi, lalu urutan model dari dashboard.

Sebelum mengubah apa pun: baca app.py dan ringkas dulu alur yang ada (di mana riwayat disimpan, di mana
key dipilih, di mana menu pengaturan, di mana tombol sinkron) lalu minta saya konfirmasi bila ada hal
yang meragukan. Setelah selesai tiap fase, jalankan pemeriksaan sintaks (`python -m py_compile`) dan
tulis tes kecil untuk logika murni (normalisasi pertanyaan, pemecahan chunk, rotasi key).
