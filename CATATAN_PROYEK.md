# Catatan proyek

## Status fase

- **Fase 1 — selesai:** operasi Firestore dibatasi, chat tetap berjalan saat
  Firestore gagal, statistik memakai WIB, pertanyaan FAQ dihitung tanpa UID, dan
  sandi Pengaturan dipindahkan ke Secrets.
- **Fase 2 — selesai:** login admin melalui `/admin` atau klik judul aplikasi
  7 kali dalam 3 detik, dashboard, kebijakan pemeliharaan, pengumuman, blokir
  dan batas harian, serta kontrol sinkronisasi tersedia. Keduanya memakai
  `ADMIN_CODE_HASH` dari Secrets dan perlindungan percobaan login yang sama.
  Ringkasan AI untuk FAQ sengaja belum diaktifkan sampai Fase 3.

## Berkas yang diubah atau ditambahkan

- `app.py` — login admin sebelum jalur chat, penerapan kebijakan sesi, tampilan
  pengumuman, dan pemindahan kontrol mulai/jeda/lanjut/coba ulang Drive.
- `services/firestore_store.py` — batas waktu dan circuit breaker Firestore,
  penulisan latar, ringkasan statistik WIB, FAQ, konfigurasi admin, riwayat,
  dan status kesehatan.
- `services/admin_logic.py` — verifikasi hash admin dan pemeriksaan blokir/batas
  harian yang murni, serta pengelolaan autentikasi dan lockout login.
- `services/admin_dashboard.py` — dashboard statistik, riwayat, FAQ, sinkron
  Drive, kontrol, dan kesehatan sistem; modul dimuat setelah login berhasil.
- `.streamlit/secrets.toml.example` — nama Secrets baru dan format hash admin.
- `tests/test_firestore_store.py` — uji kegagalan dan timeout riwayat Firestore,
  serta batas waktu operasi.
- `tests/test_admin_logic.py` — uji normalisasi FAQ, blokir, batas WIB, dan
  verifikasi PBKDF2.
- `CATATAN_PROYEK.md` — catatan fase, struktur data, keputusan, dan pekerjaan
  terbuka.

## Struktur data Firestore

- `users/{uid}` — aktivitas, penghitung, status blokir, hari dan jumlah
  pertanyaan, `active_chat_id`, dan daftar `chat_ids` untuk navigasi dashboard.
- `users/{uid}/chats/{chat_id}/chunks/{chunk_id}` — pesan per potongan.
- `stats/global` — jumlah perangkat.
- `stats/daily_YYYYMMDD` — jumlah pertanyaan dan UID perangkat aktif yang
  di-hash untuk hari WIB.
- `stats/hourly_YYYYMMDDHH` — UID perangkat yang di-hash dan peta
  `active_at` untuk aktivitas per jam.
- `faq/{sha256_pertanyaan_normalisasi}` — pertanyaan yang dinormalisasi,
  tampilan pertanyaan, hitungan, dan waktu terakhir; tidak menyimpan UID.
- `settings/runtime` — mode perawatan, pengumuman, dan batas pertanyaan harian.
- `system_health/errors` — maksimal 20 error terakhir dalam satu dokumen.

## Secrets

- `SETTINGS_PASSWORD` — sandi menu Pengaturan; bila kosong menu tetap terkunci.
- `ADMIN_CODE_HASH` — hash kode admin, format
  `pbkdf2_sha256$iterasi$salt$hex_digest` atau `scrypt$salt$hex_digest`.
  Jangan menyimpan kode admin teks biasa di source code atau repositori.
- `FIREBASE_SERVICE_ACCOUNT` — kredensial Firestore yang sudah digunakan.

## Keputusan desain

- Riwayat dan semua pembacaan admin memakai timeout maksimum 5 detik, tanpa
  retry otomatis. Pemuatan riwayat memiliki tenggat total 5 detik dan
  mengembalikan sesi kosong bila gagal.
- Kegagalan Firestore membuka circuit breaker memori proses selama 2 menit;
  pesan error dicatat sekali per periode. Operasi tulis dikirim ke satu worker
  latar dengan timeout RPC 10 detik agar urutan chunk tetap terjaga.
- Satu pertanyaan membuat paling banyak empat penulisan ringkasan: chunk,
  pengguna, statistik harian, dan FAQ. Ringkasan per jam ditulis terpisah paling
  banyak sekali per jam per sesi.
- Hash UID dipakai pada ringkasan unik dan FAQ menggunakan hash SHA-256 dari
  pertanyaan, sehingga FAQ tidak mengandung UID. Pemeriksaan blokir dan batas
  membaca data pengguna yang telah dimuat, tanpa baca Firestore tambahan di
  jalur pertanyaan.
- Konfigurasi runtime dicache di memori selama lima menit. Login admin
  menggunakan PBKDF2 atau scrypt dan `hmac.compare_digest`; dashboard hanya
  diimpor setelah login benar.
- Data aktif baru memakai timestamp untuk hitungan satu jam. Dokumen lama yang
  hanya berisi daftar UID ditampilkan sebagai perkiraan karena tidak memiliki
  waktu aktivitas.

## Masalah yang masih terbuka

- Statistik aktivitas historis tidak dapat direkonstruksi dengan tepat dari
  dokumen per jam lama yang belum memiliki `active_at`.
- Daftar pengguna dibatasi hingga 500 entri per pembacaan; pencarian riwayat
  dibatasi waktu operasi dan hasil tampilan maksimal 500 pesan.
- Chat lama sebelum `chat_ids` ditulis mungkin hanya dapat ditemukan lewat
  `active_chat_id`; tidak ada migrasi Firestore untuk menemukan ID chat lama.
- Penulisan chunk diserialkan oleh satu worker per proses. Dua proses aplikasi
  berbeda yang menulis ke chat yang sama secara bersamaan masih dapat
  berkompetisi; penerapan multi-worker memerlukan strategi transaksi dengan
  tenggat RPC terkontrol.
- Fitur rangkuman FAQ dengan AI menunggu kelompok key admin pada Fase 3.
- Verifikasi belum memakai proyek Firestore produksi; hasil bergantung pada
  aturan IAM dan ketersediaan service account saat deployment.
