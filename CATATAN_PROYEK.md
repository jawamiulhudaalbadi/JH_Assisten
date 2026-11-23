# Catatan proyek

## Status fase

- **Fase 1 — selesai:** operasi Firestore dibatasi, percakapan tetap berjalan saat
  Firestore gagal, statistik memakai WIB, pertanyaan FAQ dihitung tanpa UID, dan
  sandi Pengaturan dipindahkan ke Secrets.
- **Fase 2 — selesai:** login admin tersedia melalui perintah `/admin` atau
  tombol **Admin** di sidebar (klik tiga kali). Fase ini juga mencakup
  dasbor, kebijakan pemeliharaan, pengumuman, pemblokiran dan batas harian,
  serta kontrol sinkronisasi. Kedua cara login memeriksa kode yang sama,
  menggunakan `ADMIN_CODE_HASH` dari Secrets, dan menerapkan perlindungan
  percobaan gagal yang sama.
  Ringkasan AI untuk FAQ sengaja belum diaktifkan sampai Fase 3.

## Cara tes manual Fase 2

Lakukan pengujian di lingkungan uji, bukan pada data atau layanan produksi.
Pastikan `ADMIN_CODE_HASH` sudah diatur di Secrets sebelum mulai.

1. Buka aplikasi dan kirim `/admin <kode-admin>` melalui kolom percakapan.
   Pastikan dasbor admin terbuka setelah kode yang benar dimasukkan.
2. Muat ulang aplikasi. Klik tombol **Admin** di sidebar tiga kali, masukkan
   kode yang sama, lalu pastikan dasbor yang sama terbuka.
3. Uji kode yang salah sebanyak tiga kali dengan UID browser pertama. Dari
   browser lain, login dengan kode yang benar untuk mengatur ulang hitungan
   global. Kembali ke browser pertama, muat ulang halaman, lalu lakukan dua
   percobaan gagal lagi. Muat ulang sekali lagi dan coba kode yang benar; login
   seharusnya tetap terkunci selama 15 menit karena hitungan per UID bertahan.
4. Untuk menguji hitungan global secara terpisah, mulai ulang proses aplikasi
   pada lingkungan uji. Buat empat percobaan gagal dengan beberapa UID, lalu
   satu percobaan gagal lagi dari UID lain. Pastikan login semua UID terkunci
   selama 15 menit. Hitungan ini berlaku untuk seluruh proses yang sama.
5. Setelah berhasil login, uji pengumuman, mode pemeliharaan, blokir dan batas
   harian, serta kontrol sinkronisasi Drive. Pastikan perubahan berfungsi
   sesuai pengaturan dan tidak mengganggu percakapan biasa.

Penguncian memakai hitungan per UID dan hitungan global di memori proses.
Penguncian bertahan setelah halaman dimuat ulang, tetapi hitungan akan hilang
jika proses aplikasi dimulai ulang atau berpindah ke proses lain. Penguncian ini
tidak disimpan di Firestore.

## Berkas yang diubah atau ditambahkan

- `app.py` — login admin sebelum jalur percakapan, penerapan kebijakan sesi,
  tampilan pengumuman, dan pemindahan kontrol mulai/jeda/lanjut/coba ulang Drive.
- `services/firestore_store.py` — batas waktu dan circuit breaker Firestore,
  penulisan latar, ringkasan statistik WIB, FAQ, konfigurasi admin, riwayat,
  dan status kesehatan.
- `services/admin_logic.py` — verifikasi hash admin, pengelolaan autentikasi
  dan penguncian login, serta pemeriksaan blokir dan batas harian.
- `services/admin_dashboard.py` — dasbor statistik, riwayat, FAQ, sinkron
  Drive, kontrol, dan kesehatan sistem; modul dimuat setelah login berhasil.
- `.streamlit/secrets.toml.example` — nama Secrets baru dan format hash admin.
- `tests/test_firestore_store.py` — uji kegagalan dan batas waktu riwayat
  Firestore, serta batas waktu operasi.
- `tests/test_admin_logic.py` — uji normalisasi FAQ, blokir, batas WIB,
  verifikasi PBKDF2, dan penguncian login.
- `CATATAN_PROYEK.md` — catatan fase, struktur data, keputusan, dan pekerjaan
  terbuka.

## Struktur data Firestore

- `users/{uid}` — aktivitas, penghitung, status blokir, hari dan jumlah
  pertanyaan, `active_chat_id`, daftar `chat_ids`, serta `saved_chats` berisi
  maksimal lima ID dan nama percakapan tersimpan.
- `users/{uid}/chats/{chat_id}/chunks/{chunk_id}` — pesan per potongan.
- `stats/global` — jumlah perangkat.
- `stats/daily_YYYYMMDD` — jumlah pertanyaan dan UID perangkat aktif yang
  di-hash untuk hari WIB.
- `stats/hourly_YYYYMMDDHH` — UID yang di-hash dan peta `active_at` untuk
  aktivitas per jam.
- `faq/{sha256_pertanyaan_normalisasi}` — pertanyaan yang dinormalisasi,
  tampilan pertanyaan, hitungan, dan waktu terakhir; tidak menyimpan UID.
- `settings/runtime` — mode pemeliharaan, pengumuman, batas pertanyaan harian,
  serta hash sandi menu Pengaturan yang diatur admin.
- `system_health/errors` — maksimal 20 catatan kesalahan terakhir dalam satu
  dokumen.

## Secrets

- `SETTINGS_PASSWORD` — sandi lama menu Pengaturan sebagai cadangan awal.
  Sandi baru diatur dari dasbor admin dan hashnya disimpan di
  `settings/runtime`.
- `ADMIN_CODE_HASH` — hash kode admin, dengan format
  `pbkdf2_sha256$iterasi$salt$hex_digest` atau `scrypt$salt$hex_digest`.
  Jangan menyimpan kode admin dalam teks biasa di kode sumber atau repositori.
- `FIREBASE_SERVICE_ACCOUNT` — kredensial Firestore yang sudah digunakan.

Untuk membuat hash PBKDF2 secara lokal, jalankan perintah berikut di terminal,
lalu salin hasilnya ke `ADMIN_CODE_HASH` di Streamlit Cloud Secrets:

```sh
python -c 'import getpass, hashlib, secrets; p=getpass.getpass("Kode admin: ").encode(); s=secrets.token_hex(16); d=hashlib.pbkdf2_hmac("sha256", p, s.encode(), 200000).hex(); print(f"pbkdf2_sha256$200000${s}${d}")'
```

Pilih kode admin yang panjang dan unik, bukan kode angka pendek.

## Keputusan desain

- Riwayat dan semua pembacaan admin memakai batas waktu maksimum 5 detik tanpa
  percobaan ulang otomatis. Pemuatan riwayat memiliki batas waktu total 5 detik
  dan mengembalikan sesi kosong bila gagal.
- Kegagalan Firestore mengaktifkan mekanisme pemutus sementara selama 2 menit;
  pesan kesalahan dicatat sekali per periode. Operasi tulis dikirim ke satu
  tugas latar dengan batas waktu RPC 10 detik agar urutan potongan tetap terjaga.
- Satu pertanyaan membuat paling banyak empat penulisan ringkasan: potongan,
  pengguna, statistik harian, dan FAQ. Ringkasan per jam ditulis terpisah paling
  banyak sekali per jam per sesi.
- Hash UID dipakai pada ringkasan pengguna unik. FAQ menggunakan hash SHA-256
  dari pertanyaan, sehingga tidak menyimpan UID. Pemeriksaan blokir dan batas
  membaca data pengguna yang sudah dimuat, tanpa membaca Firestore lagi di
  jalur pertanyaan.
- Konfigurasi aplikasi saat berjalan disimpan sementara di memori selama lima
  menit. Login admin menggunakan PBKDF2 atau scrypt dan `hmac.compare_digest`;
  dasbor hanya diimpor setelah login berhasil.
- Penguncian login admin mencatat kegagalan per UID dan secara global di memori
  proses. Lima kegagalan mengunci semua login di proses tersebut selama 15
  menit. Hitungan global diatur ulang setelah login berhasil, sedangkan hitungan
  UID lain tetap tersimpan. Status ini bertahan setelah halaman dimuat ulang, tetapi tidak
  setelah proses dimulai ulang dan tidak dibagikan antarproses.
- Admin cukup mengetik sandi menu Pengaturan di dasbor; aplikasi membuat
  hash PBKDF2 sebelum menyimpannya. Hash konfigurasi aktif diprioritaskan
  daripada `SETTINGS_PASSWORD` lama. Pengguna biasa tetap harus memasukkan
  sandi untuk membuka menu.
- Daftar percakapan tersimpan di `users/{uid}.saved_chats`, maksimal lima.
  Daftar dan kontrol riwayat ada di sidebar. Tombol untuk menghapus percakapan
  aktif ada di halaman utama, di bawah percakapan; penghapusan semua percakapan
  merupakan tindakan terpisah di sidebar. Keduanya meminta konfirmasi.
  Percakapan yang disimpan diberi nama, lalu aplikasi membuka percakapan baru.
- Data aktivitas baru memakai timestamp untuk hitungan satu jam. Dokumen lama
  yang hanya berisi daftar UID ditampilkan sebagai perkiraan karena tidak
  memiliki `active_at`.

## Masalah yang masih terbuka

- Statistik aktivitas historis tidak dapat direkonstruksi dengan tepat dari
  dokumen per jam lama yang belum memiliki `active_at`.
- Daftar pengguna dibatasi hingga 500 entri per pembacaan; pencarian riwayat
  dibatasi waktu operasi dan hasil tampilan maksimal 500 pesan.
- Percakapan lama sebelum `chat_ids` ditulis mungkin hanya dapat ditemukan
  lewat `active_chat_id`; tidak ada migrasi Firestore untuk menemukan ID
  percakapan lama.
- Penulisan potongan diserialkan oleh satu pekerja latar per proses. Dua proses
  aplikasi berbeda yang menulis ke percakapan yang sama secara bersamaan masih
  dapat berkompetisi; penerapan banyak pekerja latar memerlukan strategi
  transaksi dengan batas waktu RPC yang terkontrol.
- Fitur ringkasan FAQ dengan AI menunggu kelompok key admin pada Fase 3.
- Verifikasi belum memakai proyek Firestore produksi; hasil bergantung pada
  aturan IAM dan ketersediaan akun layanan saat penerapan.
