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
- **Fase 3 — selesai:** key chat, indexing, dan admin terpisah; key dashboard
  dienkripsi dengan Fernet; rotasi kuota dan override chat khusus sesi
  diterapkan; dashboard dapat mengelola key dan urutan model; ringkasan FAQ AI
  berjalan hanya saat diminta admin.
- **Perbaikan Firestore produksi — selesai:** konfigurasi runtime selalu dibaca
  dari cache/nilai aman pada jalur render dan disegarkan di latar dengan batas
  5 detik. Circuit breaker hanya bereaksi pada `UNAVAILABLE`,
  `DEADLINE_EXCEEDED` RPC, dan `PermissionDenied`. Mutasi admin menggunakan
  tenggat 10 detik dan tidak bergantung pada breaker chat. Kesehatan sistem
  menampilkan breaker, timeout konfigurasi yang dilewati, dan error tulis
  terakhir per jenis.
- **Perbaikan tulis Firestore:** pasangan riwayat disimpan atomik bersama
  dokumen pengguna yang sudah dibuat. Statistik harian dan FAQ dikirim
  terpisah; statistik jam disimpan dalam transaksi pasangan chat. Antrean tulis
  memori mempertahankan pekerjaan saat sirkuit terbuka dan menjadwalkan ulang
  error sementara.
- **Perbaikan ID perangkat produksi:** penyebab UID berubah adalah respons
  kosong sementara dari komponen JavaScript dianggap sebagai UID browser yang
  rusak. Jalur perbaikan lalu membuat UID server dan menandainya final, walau
  localStorage masih berisi UID lama. Kini respons kosong hanya menghasilkan
  UID sementara sesi dan aplikasi menunggu; pembuatan UID persisten hanya
  dilakukan setelah skrip browser menyatakan penyimpanan benar-benar kosong.
  `jh_uid` dibaca dari `st.context.cookies` pada request awal dan ditulis oleh
  browser dengan masa berlaku satu tahun serta `SameSite=Lax`. Parameter
  `?uid=` tetap didukung.
- **Perbaikan eksekusi skrip UID browser:** akar masalah tambahan adalah
  newline awal pada `_READ_UID_SCRIPT` dan `_CREATE_UID_SCRIPT`.
  `streamlit_javascript` 0.1.5 membungkus kode sebagai
  `eval("(async () => {return " + js_code + "})()")`; newline setelah `return`
  memicu automatic semicolon insertion sehingga skrip tidak berjalan dan
  komponen tidak mengembalikan prop `value`. Kedua skrip kini di-trim dan
  dipastikan berbentuk ekspresi berawalan `(`. Saat balasan browser tertunda,
  fragment melakukan polling 1 detik dan berhenti setelah batas 4 detik;
  pengguna dapat menekan **Coba lagi**. UID sementara tidak disimpan di
  browser dan tidak membuka akses Firestore.
- **Pembuatan pengguna malas:** halaman yang hanya dibuka tidak lagi menulis
  aktivitas atau membuat dokumen pengguna. Setelah ID final, pengguna wajib
  menyimpan nama; transaksi itu membuat `users/{uid}` dan menaikkan
  `stats/global.total_devices` tepat satu kali untuk dokumen baru. Penulisan
  pasangan pertanyaan hanya memperbarui profil yang sudah ada.
- **Pembersihan pengguna kosong:** menu **Kontrol lain** menampilkan jumlah
  calon sebelum konfirmasi. Hanya dokumen dengan `total_questions` tepat 0 dan
  tanpa chunk yang dapat dihapus; setelahnya `total_devices` dihitung ulang
  dari dokumen pengguna tersisa.

## Fase 4A — selesai

- **Diagnosis UID:** akar masalah yang tercatat pada perubahan sebelumnya
  adalah newline setelah `return` pada skrip yang dibungkus oleh
  `streamlit_javascript` 0.1.5. Automatic semicolon insertion membuat JavaScript
  berhenti sebelum membaca/menulis localStorage dan cookie; komponen lalu
  mengembalikan nilai awal `0`. Perubahan sebelumnya memangkas kedua skrip dan
  tes Node menjalankannya melalui pembungkus `eval` yang sama. Tes memastikan
  nilai kosong tidak dianggap UID final. Kode produksi/deployment atau sesi
  Android WebView tidak tersedia di workspace ini, jadi cookie yang benar-benar
  terkirim setelah refresh belum dapat dibuktikan di deployment tersebut.
  Komponen tidak diganti: bukti yang ada menunjuk ke kesalahan sintaks/ASI pada
  payload, bukan ketidakcocokan Python 3.14; tes Node memakai pembungkus
  komponen yang sama. Ini belum menggantikan tes browser nyata.
- **Identitas:** cookie `jh_uid` dan `?uid=` diproses sebelum komponen. Setelah
  ID final tersimpan di session state, rerun tidak membaca ulang cookie/query
  dan tidak merender komponen. Komponen hanya dipakai jika belum final.
  Diagnostik kecil hanya tampil untuk `?debug=1` atau sesi admin; isinya sumber,
  status final/sementara, dan empat karakter awal. UID sementara tidak membuka
  akses operasi Firestore perangkat. Streamlit 1.67.0/Python 3.14.2 lokal
  menyediakan `st.context.cookies` dan `st.fragment`.
- **Nama pengguna:** prompt Sunda hanya muncul bagi ID final tanpa `name`.
  Pemeriksaan nama memakai data dari pembacaan dokumen pengguna yang sudah ada
  pada pemuatan riwayat. Profil baru dibuat saat tombol **Simpen nami** ditekan;
  nama divalidasi (2–30 karakter, karakter yang diizinkan, tanpa URL/kontrol)
  dan ditampilkan sebagai teks biasa. Nama kembar diizinkan dan diberi tanda
  pada tabel riwayat admin. Pengguna tanpa nama tidak dapat mengirim chat.
  Perubahan nama dari Pengaturan menulis profil satu kali.
- **Performa:** pengukuran opt-in menyimpan 30 rerun terakhir per proses dan
  mencatat rerun di atas 1,5 detik ke log tanpa UID. Kesehatan sistem
  menampilkan waktu identitas, konfigurasi, key pool, pemuatan indeks,
  pemuatan riwayat, render, dan total. AppTest lokal tanpa Secrets, Firestore,
  API key, atau indeks, dengan UID final dan debug aktif, mengukur dua rerun
  **11,9 ms lalu 5,3 ms** (rerun kedua: identity 0,0; configuration 0,4;
  key pool 0,0; index 0,0; history 0,1; render 4,8 ms). Itu bukan pembanding
  produksi. Perubahan I/O yang dapat dihitung: pemuatan
  daftar chat tersimpan berubah dari satu baca Firestore per rerun menjadi satu
  baca per sesi/UID (N rerun menjadi 1; menghemat N−1 baca); refresh key chat,
  key indexing, urutan model, dan daftar model tidak lagi menunggu I/O sinkron
  pada jalur pengguna. Cache key/urutan model tetap 60 detik; refresh saat
  kedaluwarsa berjalan di thread latar. CSS yang sama dihitung sekali per
  pasangan tema/font dalam proses.
- **Batas verifikasi:** AppTest membuktikan form nama tampil untuk ID final
  tanpa nama dan chat tetap nonaktif, sementara sesi tanpa ID final tidak
  menampilkan form. Pengujian browser produksi dan Android WebView tetap perlu
  dilakukan; hasil lokal tidak mewakili latensi jaringan Firestore atau
  perangkat.
- **Fase 4B — selesai:** sesi admin bertanda tangan berlaku 30 hari, terikat
  pada UID dan epoch pencabutan di `config/admin_session`; epoch dibaca hanya
  ketika ada token, di-cache 5 menit, dan autentikasi gagal-aman bila Firestore
  tidak tersedia. Kunci HMAC diturunkan dengan konteks khusus dari
  `ADMIN_SESSION_SECRET` atau `ENCRYPTION_KEY`, bukan dari
  `ADMIN_CODE_HASH`. Cookie `jh_adm` dan localStorage ditulis setelah proses
  login selesai dirender oleh komponen browser; token tidak masuk log.
  **Keluar** menghapus sesi lokal, sedangkan **Cabut semua sesi admin**
  menaikkan epoch agar semua token tidak berlaku.
- **Kualitas jawaban:** metadata latensi, jumlah sumber, skor teratas bila ada,
  fallback, model, dan maksimal tiga kutipan sumber disimpan bersama chunk
  jawaban. Umpan balik disimpan sekali per jawaban dan memperbarui dokumen
  ringkasan pada transaksi yang sama; penandaan admin dan penilaian AI hanya
  menulis saat tombol ditekan, dan penilaian AI dibatasi 20 item per klik.
  Ringkasan metrik sumber/fallback/latensi dihitung dari jawaban yang telah
  diberi umpan balik, bukan semua jawaban; keterangan ini ditampilkan di menu
  agar persentasenya tidak disalahartikan. Kualitas jawaban menampilkan nama,
  menyediakan filter/paginasi, dan menyimpan label tinjauan terpisah.
- **Pengguna dan performa admin:** menu **Pengguna** memakai proyeksi field,
  halaman maksimal 25 baris, pencarian nama, penanda nama ganda/pengguna tanpa
  pertanyaan/lonjakan harian, serta kontrol blokir dan batas per pengguna.
  Dashboard hanya merender menu yang dipilih di dalam `st.fragment`; data
  ringkasan/halaman memakai cache 60 detik dengan tombol **Segarkan**.
  Riwayat rinci dibaca hanya saat dibuka. Modul dashboard admin tetap tidak
  diimpor oleh pengguna biasa.
- **Pengukuran dashboard:** sebelum refaktor, waktu rerun menu penuh pada
  AppTest lokal sekitar **108,8–166 ms**. Sesudah refaktor, pengukuran badan
  fragment lokal sekitar **0,6–25,6 ms** per menu; rerun AppTest penuh sekitar
  **188–290 ms**, tetapi AppTest menjalankan ulang aplikasi secara penuh dan
  bukan simulasi interaksi fragment di browser. Karena itu angka tersebut
  tidak sebanding langsung dan bukan klaim latensi produksi. Firestore,
  API key, dan layanan produksi tidak tersedia saat pengukuran.
- **Verifikasi Fase 4B:** 84 tes terarah dan 3 tes metadata/evaluator AI lulus;
  `py_compile` dan `git diff --check` lulus. Tes menggunakan fake Firestore
  dan AppTest tanpa kredensial produksi. Peramban nyata, indeks Firestore,
  ketersediaan layanan, dan persistensi cookie Android WebView tetap harus
  diuji manual.

## Fase 4C — mode tamu berkode

- **Akses ketat:** chat tetap nonaktif bila ID belum final atau nama belum
  disimpan. Pengguna hanya dapat melewati syarat tersebut setelah memasukkan
  kode yang diverifikasi dari `GUEST_CODE_HASH`, atau ketika katup darurat admin
  sedang berlaku. Secret kosong menyembunyikan tombol. Hash PBKDF2/scrypt
  diperiksa dengan `hmac.compare_digest`; kegagalan guest memakai penguncian
  5 kali/15 menit yang terpisah dari login admin.
- **Privasi tamu:** mode hanya berada di `st.session_state`; tidak memuat atau
  menyimpan profil, riwayat, chat tersimpan, umpan balik, FAQ, statistik
  perangkat, maupun statistik pengguna. Tiap pertanyaan hanya menaikkan
  `guest_questions` pada dokumen `stats/daily_YYYYMMDD` lewat penulisan statistik
  harian yang sudah ada. Hitungan batas tamu (bawaan 10/hari) hanya di memori
  sesi dan dapat diubah dari dashboard.
- **Katup darurat:** menu **Kontrol lain** menyimpan `guest_emergency_enabled`
  dan `guest_emergency_expires_at` di `settings/runtime`. Katup bawaan mati,
  berlaku enam jam, dan dianggap mati setelah waktu kedaluwarsa. Dashboard dan
  sesi tamu menampilkan peringatan merah dengan sisa waktu saat katup aktif.
- **Verifikasi lokal:** tes unit mencakup validasi hash, penguncian yang terpisah,
  kadaluarsa katup, dan bentuk penulisan statistik tamu. AppTest memeriksa
  keadaan identitas/nama dan bahwa sesi tamu tidak membaca Firestore. Uji
  deployment dan tampilan browser tetap perlu dilakukan manual.

## Fase 4D — perbaikan sesi admin permanen & statistik guest

- **Sesi admin permanen per perangkat:** tombol **Keluar** di dashboard admin
  sekarang hanya menyembunyikan dashboard (`admin_authenticated = False`) tanpa
  menghapus token sesi di cookie/localStorage (`jh_adm`). Token TTL 30 hari
  tetap tersimpan sehingga saat reload atau masuk ulang ke mode admin, pengguna
  langsung terautentikasi tanpa diminta sandi ulang. Hanya **Cabut semua sesi
  admin** (increment epoch di `config/admin_session`) yang benar-benar mencabut
  token di semua perangkat.
- **Perbaiki bug "Keluar" yang auto-reopen:** dipisahkan status menjadi dua:
  a) `admin_device_verified` — perangkat/cookie `jh_adm` sudah terverifikasi
  valid (tetap `True` meskipun keluar dari tampilan admin, hanya dicabut oleh
  "Cabut semua sesi admin").
  b) `admin_authenticated` / `admin_panel_open` — dashboard admin sedang dibuka
  di layar.
  Saat tombol **Keluar** ditekan: set `admin_authenticated = False` dan
  `admin_exited_manually = True`, lalu `st.rerun(scope="app")` agar langsung
  kembali ke halaman utama. `_restore_admin_session()` pada reload hanya
  menyetel `admin_device_verified = True` dan **tidak** memaksa buka dashboard
  jika `admin_exited_manually = True`.
- **Masuk admin instan tanpa minta sandi ulang:** saat user menekan tombol
  **Admin** (atau `/admin`): cek `admin_device_verified` terlebih dahulu. Jika
  valid, langsung set `admin_authenticated = True`, `admin_exited_manually =
  False`, dan tampilkan dashboard **tanpa dialog sandi** dan **tanpa query
  Firestore berulang**. Validasi epoch token `jh_adm` menggunakan cache memori
  (TTL 5 menit) di `services/admin_session.py`.
- **Tampilkan nama pengguna di header:** nama pengguna aktif ditampilkan di
  bagian paling atas menu utama (di bawah judul aplikasi) setelah ID final dan
  nama disimpan.
- **Statistik guest terpisah di dashboard:** menu **Statistik** sekarang
  menampilkan dua kolom terpisah per hari — **Pertanyaan pengguna** (dari field
  `questions`) dan **Pertanyaan tamu** (dari field `guest_questions` di
  `stats/daily_YYYYMMDD`). `total_devices` tetap hanya menghitung pengguna
  terdaftar (yang memiliki nama), tidak termasuk sesi guest.
- **Pembersihan pengguna tanpa nama:** fungsi `get_users_without_name()` dan
  `delete_users_without_name_and_recount()` sekarang menargetkan **semua dokumen
  pengguna tanpa nama** (kosong/null/whitespace), bukan hanya yang
  `total_questions == 0`. Setelah hapus, `total_devices` dihitung ulang dari
  dokumen pengguna yang tersisa (yang memiliki nama asli). UI dashboard
  menampilkan preview kandidat sebelum hapus.
- **Tombol "Hapus semua tanpa nama" di tab Pengguna:** di sebelah tombol
  "Segarkan", tombol baru menampilkan konfirmasi sebelum menghapus **SEMUA**
  pengguna tanpa nama (termasuk yang sudah memiliki pertanyaan) beserta riwayat
  chat mereka, lalu menghitung ulang `total_devices`.
- **Perbarui unit test:** tambah test untuk verifikasi token persist setelah
  logout, revoke mencabut token, dan statistik guest dibaca dengan benar.
- **Percepatan identifikasi ID perangkat:** `get_device_uid()` sekarang membaca
  cookie `jh_uid` dari `st.context.cookies` pada request HTTP awal **sebelum**
  menjalankan komponen JavaScript. Jika cookie valid, UID langsung dikembalikan
  dan ditandai final (`DEVICE_ID_FINAL=True`) tanpa menunggu browser.
  Komponen JavaScript (`streamlit_javascript`) hanya dipanggil untuk pengguna
  baru yang belum memiliki cookie. Perilaku ini berlaku konsisten untuk semua
  pengguna (biasa, guest, dan admin).

### Cara tes manual Fase 4D

1. **Sesi admin permanen & perbaikan tombol Keluar:**
   - Login admin via `/admin <kode>` atau tombol Admin 3x, reload halaman → dashboard terbuka otomatis.
   - Klik **Keluar** → dashboard tertutup **seketika** (tidak auto-reopen).
   - Masuk mode admin lagi (klik Admin 1x atau `/admin`) → **langsung masuk tanpa sandi** (instan, tidak lelet).
   - Tutup browser, buka lagi → **langsung masuk** (otomatis buka dashboard karena `admin_device_verified = True` dan `admin_exited_manually = False`).
   - Klik **Keluar**, lalu reload halaman → dashboard **tidak** terbuka otomatis (karena `admin_exited_manually = True`).
   - Klik tombol **Admin** 1x → **langsung masuk tanpa sandi**.
   - Gunakan **Cabut semua sesi admin** dari perangkat lain → token tidak valid, minta sandi ulang.
2. **Tampil nama di header:** simpan nama pengguna, periksa header paling atas menampilkan "Pengguna: {nama}".
3. **Statistik guest:** masuk mode guest (via sandi atau emergency valve), ajukan beberapa pertanyaan, buka Dashboard Admin → Statistik. Verifikasi tabel harian menampilkan kolom "Pertanyaan tamu" > 0, "Total perangkat (terdaftar)" tidak berubah, dan "Pertanyaan pengguna" terpisah dari "Pertanyaan tamu".
4. **Hapus semua tanpa nama:** buka Dashboard Admin → tab **Pengguna**, klik tombol **"Hapus semua tanpa nama"** di sebelah "Segarkan". Konfirmasi dialog akan muncul menampilkan jumlah kandidat. Klik **"Konfirmasi hapus semua pengguna tanpa nama"** untuk menghapus. Verifikasi: pengguna bernama tidak terhapus, `total_devices` di Statistik terhitung ulang, dan pengguna tanpa nama (termasuk yang punya riwayat chat) terhapus.

### Cara tes manual Fase 4A

1. Pada browser uji, buka aplikasi dengan `?debug=1`. Dengan cookie `jh_uid`
   tersedia, pastikan lencana menunjukkan `sumber: cookie`, `status: final`,
   dan empat karakter awal yang sama sebelum/sesudah refresh. Pastikan refresh
   cookie tidak memanggil JavaScript (tes unit memverifikasi komponen tidak
   dipanggil pada jalur cookie).
2. Pada perangkat uji baru, hapus cookie dan localStorage situs lalu buka
   aplikasi. Pastikan identitas menjadi final dengan sumber `baru`, kemudian
   refresh dan pastikan sumber berganti menjadi `cookie` dengan awalan yang
   sama. Tanyakan sesuatu lalu refresh; riwayat harus tetap muncul.
3. Blokir JavaScript/penyimpanan browser pada lingkungan uji. Setelah empat
   detik pastikan ID berstatus sementara, keterangan riwayat belum tersimpan
   dan tombol **Coba lagi** tampil. Pastikan halaman tidak membuat dokumen
   pengguna atau mengirim UID sementara ke Firestore.
4. Untuk ID final tanpa nama, pastikan formulir nama tampil dan chat nonaktif.
   Uji nama terlalu pendek, URL, newline, dan markup; simpan nama yang valid,
   pastikan profil dan penghitung `total_devices` dibuat sekali, lalu muat
   ulang dan pastikan formulir tidak muncul. Uji nama kembar dan ganti nama
   lewat Pengaturan.
5. Di Android WebView, ulangi langkah 1–4, termasuk tutup dan buka kembali APK.
   Pastikan WebView mengizinkan cookie untuk host aplikasi. Jika awalan/sumber
   berubah, periksa kebijakan cookie dan data situs WebView; jangan
   mengasumsikan localStorage atau cookie tetap tersimpan.
6. Untuk pengukuran, buka **Kesehatan sistem** sebagai admin atau gunakan
   `?debug=1`, kemudian periksa 30 rerun terakhir. Hapus `?debug=1` setelah
   selesai.

### Cara tes manual Fase 4B

Lakukan pengujian dengan proyek Firestore uji dan akun/perangkat uji.

1. Pastikan `ADMIN_CODE_HASH` dan salah satu `ADMIN_SESSION_SECRET` atau
   `ENCRYPTION_KEY` tersedia di Secrets. Masuk dengan `/admin <kode>`, muat
   ulang, lalu tutup dan buka kembali aplikasi. Pastikan cookie `jh_adm` ada
   dan admin tidak diminta memasukkan kode lagi. Jangan menyalin token ke log,
   tangkapan layar, atau tiket.
2. Tekan **Keluar**; pastikan dashboard tertutup. Masuk kembali pada dua
   perangkat uji, lalu gunakan **Cabut semua sesi admin** dari salah satunya.
   Pastikan token lama tidak lagi memulihkan sesi. Uji juga saat Firestore
   tidak tersedia; token tidak boleh memulihkan sesi.
3. Kirim beberapa pertanyaan dengan dan tanpa sumber. Beri jawaban penilaian
   **Cocog** dan **Teu cocog**, termasuk alasan opsional. Pastikan tombol
   terkunci setelah satu penilaian, ringkasan mencerminkan penilaian, dan
   metadata jawaban tersimpan. Periksa filter, nama, sumber, paginasi, label
   tinjauan, serta tombol AI (maksimal 20 item per klik).
4. Di menu **Pengguna**, cari nama, periksa ID pendek, waktu aktivitas,
   penghitung pertanyaan, penanda nama kembar/tanpa pertanyaan/lonjakan, lalu
   uji blokir dan batas harian per pengguna.
5. Ulangi login, refresh, keluar, dan pencabutan sesi pada Android WebView.
   Cookie dan localStorage dapat dihapus oleh pembungkus APK; jika login tidak
   bertahan, periksa pengaturan cookie/WebView dan gunakan login kode. Uji
   bahwa pencabutan tetap berlaku setelah aplikasi dibuka ulang.

### Cara tes manual mode tamu

1. Atur `GUEST_CODE_HASH` di Secrets dengan hash PBKDF2/scrypt. Gunakan perintah
   berikut untuk membuat hash PBKDF2 secara lokal:

   ```sh
   python -c 'import getpass, hashlib, secrets; p=getpass.getpass("Sandi mode tamu: ").encode(); s=secrets.token_hex(16); d=hashlib.pbkdf2_hmac("sha256", p, s.encode(), 200000).hex(); print(f"pbkdf2_sha256$200000${s}${d}")'
   ```

2. Dengan ID browser belum final, pastikan chat nonaktif dan tombol **Mode
   tamu** muncul. Ulangi untuk ID final tanpa nama; formulir nama dan chat
   nonaktif, lalu gunakan sandi benar untuk masuk. Banner harus berbunyi
   **MODE TAMU - riwayat teu disimpen** dan sidebar menampilkan **Tamu**.
3. Uji sandi salah lima kali; pastikan terkunci 15 menit. Periksa log dan pesan
   galat agar sandi teks biasa tidak muncul. Pastikan penghitung admin tidak
   ikut berubah.
4. Di **Dashboard Admin → Kontrol lain**, uji katup tanpa sandi. Pastikan
   peringatan merah tampil dan sisa waktu mendekati enam jam; matikan katup dan
   pastikan akses kembali memerlukan sandi. Uji juga waktu kedaluwarsa yang
   sudah lewat.
5. Kirim pertanyaan tamu dan pastikan riwayat, profil, `total_devices`, serta
   statistik pengguna tidak berubah. Hanya `guest_questions` pada statistik
   harian boleh bertambah. Pastikan limit sesi 10 pertanyaan (atau nilai yang
   diatur admin) memblokir pertanyaan berikutnya.

## Cara tes manual Fase 3

1. Atur `ENCRYPTION_KEY` di Streamlit Secrets dengan key Fernet baru yang dibuat
   secara lokal. Tambahkan `GOOGLE_API_KEY` untuk chat; isi `GOOGLE_API_KEY_INDEX`
   dan `GOOGLE_API_KEY_ADMIN` hanya bila ingin mengaktifkan indexing dan fitur AI
   dashboard. Jangan salin key ke log, tiket, atau repositori.
2. Dari dashboard admin, buka **API Key**. Tambahkan, nonaktifkan, aktifkan, dan
   hapus key pada masing-masing kelompok. Pastikan tampilan hanya menunjukkan
   label, sidik jari pendek, dan key tersamar; tiap kelompok hanya memakai key
   miliknya.
3. Buka **Urutan model**, pilih **Manual**, atur urutan dari daftar model yang
   tersedia, lalu simpan. Ubah ke **Otomatis** untuk kembali mengikuti daftar
   otomatis atau prioritas `GEMINI_CHAT_MODELS`.
4. Di menu Pengaturan, aktifkan key chat khusus sesi. Kirim pertanyaan dan
   pastikan kuota harian pengguna tidak bertambah. Jika key sesi mencapai limit,
   aplikasi menampilkan pesan yang meminta penggantian key dan tidak pindah
   diam-diam ke key konfigurasi. Tekan **Kembali ke API key konfigurasi** untuk
   menghapus override sesi.
5. Untuk menguji rotasi, gunakan key uji dengan kuota rendah pada model teratas.
   Pastikan aplikasi mencoba model lain di key yang sama sebelum beralih ke key
   chat lain. Uji indexing dan ringkasan FAQ secara terpisah dengan kelompok
   masing-masing. Ringkasan FAQ hanya dibuat setelah tombol **Rangkum dengan AI**
   ditekan dan digunakan ulang bila daftar FAQ tidak berubah.
6. Jalankan `pytest -q tests/test_firestore_store.py tests/test_key_pool.py
   tests/test_ai_service.py tests/test_sync_manager.py`, `python -m py_compile
   app.py services/*.py`, lalu buka aplikasi dengan Streamlit AppTest. Tes tidak
   memerlukan kredensial layanan produksi; tes integrasi manual sebaiknya
   memakai proyek uji.
7. Pengujian identitas browser diperbarui pada bagian **Cara tes manual Fase
   4A** di atas.

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

- `app.py` — login admin sebelum jalur percakapan, gerbang ID/nama, mode tamu
  berkode, hitungan sesi, banner privasi, polling UID berbatas waktu, lencana
  diagnostik/performa, cache riwayat, kontrol sinkronisasi Drive, **tampilkan
  nama pengguna di header, sesi admin permanen per perangkat (logout tidak hapus
  token)**.
- `services/device_identity.py` — prioritas UID URL/cookie, pembacaan browser
  yang tidak menganggap respons kosong sebagai nilai tersimpan, skrip JavaScript
  yang aman dari ASI pada pembungkus komponen, diagnostik terbatas, serta
  penulisan cookie dan localStorage setelah konfirmasi.
- `services/user_profile.py` — validasi nama pengguna dan syarat prompt setelah
  ID final.
- `services/perf_monitor.py` — ringkasan 30 rerun terakhir di memori proses.
- `services/admin_session.py` — token admin HMAC, pemeriksaan epoch fail-closed,
  cache epoch lima menit, dan skrip tulis/hapus cookie serta localStorage.
- `services/firestore_store.py` — batas waktu dan circuit breaker Firestore,
  antrean tulis latar, ringkasan statistik WIB, FAQ, konfigurasi admin,
  riwayat, pembuatan profil/nama transaksional, statistik perangkat lazy,
  **pembersihan pengguna tanpa nama** (berdasarkan field `name` kosong/null/whitespace), serta status kesehatan. **Baca field
  `guest_questions` dari statistik harian untuk dashboard.** Pembacaan konfigurasi
  runtime tidak menunggu Firestore pada jalur render; perubahan dashboard admin
  memakai timeout 10 detik yang terpisah dari breaker chat.
- `services/key_pool.py` dan `services/ai_service.py` — refresh key, urutan
  model, dan daftar model chat di latar supaya permintaan halaman pengguna tidak
  menunggu Firestore atau penemuan model.
- `services/admin_logic.py` — verifikasi hash admin/tamu, penguncian login
  terpisah, serta pemeriksaan blokir dan batas harian.
- `services/admin_dashboard.py` — dasbor statistik, riwayat bernama, pengguna,
  kualitas jawaban, FAQ, sinkron Drive, kontrol katup darurat tamu, dan
  kesehatan sistem; menu dirender dalam fragment dan modul hanya dimuat setelah
  login berhasil. **Logout hanya sembunyikan dashboard (token persist),
  statistik harian tampilkan kolom terpisah pertanyaan pengguna vs tamu,
  pembersihan pengguna tanpa nama.**
- `services/firestore_store.py` — konfigurasi batas harian tamu/katup darurat
  dan penulisan `guest_questions` pada statistik harian.
- `.streamlit/secrets.toml.example` — nama Secrets baru dan format hash admin
  serta tamu.
- `tests/test_firestore_store.py` — uji kegagalan dan batas waktu riwayat
  Firestore, konfigurasi runtime asinkron, penolakan UID perangkat invalid,
  transaksi profil/nama, penghitung perangkat, transaksi chunk dengan retry,
  serta **pembersihan pengguna tanpa nama**. **Tambah test statistik guest.**
- `tests/test_device_identity.py` — uji respons komponen kosong sebelum UID
  localStorage tersedia, pemulihan UID melalui cookie, dukungan `?uid=`, serta
  eksekusi skrip UID melalui Node.js dengan pembungkus komponen yang sama.
- `tests/test_admin_logic.py` — uji normalisasi FAQ, blokir, batas WIB,
  verifikasi PBKDF2, dan penguncian login admin/tamu terpisah.
- `tests/test_guest_mode.py` — AppTest gerbang ID/nama, sandi/katup darurat,
  banner tamu, batas pertanyaan, dan isolasi Firestore.
- `tests/test_admin_session.py`, `tests/test_admin_firestore.py`, dan
  `tests/test_admin_dashboard_isolation.py` — uji token/epoch, umpan balik,
  paginasi/penandaan admin, serta isolasi menu dashboard. **Tambah test token
  persist setelah logout & revoke mencabut token.**
- `CATATAN_PROYEK.md` — catatan fase, struktur data, keputusan, dan pekerjaan
  terbuka.

## Struktur data Firestore

- `users/{uid}` — aktivitas, penghitung, status blokir, hari dan jumlah
  pertanyaan, nama dan `name_lower`, batas harian opsional,
  `active_chat_id`, daftar `chat_ids`, serta `saved_chats` berisi maksimal lima
  ID dan nama percakapan tersimpan.
- `users/{uid}/chats/{chat_id}/chunks/{chunk_id}` — pesan per potongan;
  jawaban menyimpan `answer_id`, `latency_ms`, `source_count`, `top_score`,
  `fallback`, `model_used`, dan maksimal tiga kutipan sumber.
- `stats/global` — jumlah perangkat.
- `stats/quality` — agregat umpan balik, sumber/fallback/latensi dan per model
  untuk jawaban yang telah dinilai.
- `stats/daily_YYYYMMDD` — jumlah pertanyaan dan UID perangkat aktif yang
  di-hash untuk hari WIB; pertanyaan tamu dihitung terpisah pada
  `guest_questions`.
- `stats/hourly_YYYYMMDDHH` — UID yang di-hash dan peta `active_at` untuk
  aktivitas per jam.
- `faq/{sha256_pertanyaan_normalisasi}` — pertanyaan yang dinormalisasi,
  tampilan pertanyaan, hitungan, dan waktu terakhir; tidak menyimpan UID.
- `config/keys_chat`, `config/keys_index`, `config/keys_admin` — daftar key per
  kelompok. Key disimpan terenkripsi Fernet, bersama ID sidik jari pendek,
  label, status aktif, cooldown, dan penanda error aman.
- `config/model_order` — mode dan daftar urutan model yang dipilih admin.
- `config/admin_session` — epoch pencabutan seluruh sesi admin.
- `admin_cache/faq_summary` — ringkasan FAQ terakhir beserta fingerprint FAQ.
- `feedback/{answer_id}` — umpan balik sekali untuk satu jawaban, nama, cuplikan
  pertanyaan/jawaban, sumber, fallback, model, alasan opsional, dan waktu.
- `reviews/{answer_id}` — label tinjauan admin dan hasil evaluasi AI.
- `settings/runtime` — mode pemeliharaan, pengumuman, batas pertanyaan harian,
  batas harian tamu, katup darurat beserta waktu kedaluwarsa, serta hash sandi
  menu Pengaturan yang diatur admin.
- `system_health/errors` — maksimal 20 catatan kesalahan terakhir dalam satu
  dokumen.

## Secrets

- `SETTINGS_PASSWORD` — sandi lama menu Pengaturan sebagai cadangan awal.
  Sandi baru diatur dari dasbor admin dan hashnya disimpan di
  `settings/runtime`.
- `ADMIN_CODE_HASH` — hash kode admin, dengan format
  `pbkdf2_sha256$iterasi$salt$hex_digest` atau `scrypt$salt$hex_digest`.
  Jangan menyimpan kode admin dalam teks biasa di kode sumber atau repositori.
- `ADMIN_SESSION_SECRET` — secret opsional khusus untuk menurunkan kunci sesi
  admin; bila tidak diatur, `ENCRYPTION_KEY` digunakan dengan konteks HMAC
  terpisah. Jangan gunakan `ADMIN_CODE_HASH` sebagai kunci tanda tangan.
- `GUEST_CODE_HASH` — hash sandi mode tamu dengan format
  `pbkdf2_sha256$iterasi$salt$hex_digest` atau `scrypt$salt$hex_digest`.
  Kosongkan untuk menyembunyikan pilihan mode tamu.
- `GOOGLE_API_KEY`, `GOOGLE_API_KEY_INDEX`, `GOOGLE_API_KEY_ADMIN` — seed key
  untuk kelompok chat, indexing, dan admin secara terpisah. Key dari satu
  kelompok tidak pernah dipinjam kelompok lain.
- `ENCRYPTION_KEY` — key Fernet yang dibutuhkan untuk menambah dan membaca key
  terenkripsi dari Firestore. Buat secara lokal dengan
  `python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'`
  dan simpan hanya di Secrets. Bila Firestore tidak tersedia, chat tetap
  memakai seed `GOOGLE_API_KEY` dari Secrets.
- `GEMINI_CHAT_MODELS` — daftar model opsional dipisahkan koma. Urutan model
  admin di Firestore mengungguli daftar ini; tanpa keduanya aplikasi memilih
  urutan otomatis dari model yang tersedia.
- `FIREBASE_SERVICE_ACCOUNT` — kredensial Firestore yang sudah digunakan.

Untuk membuat hash PBKDF2 secara lokal, jalankan perintah berikut di terminal,
lalu salin hasilnya ke `ADMIN_CODE_HASH` di Streamlit Cloud Secrets:

```sh
python -c 'import getpass, hashlib, secrets; p=getpass.getpass("Kode admin: ").encode(); s=secrets.token_hex(16); d=hashlib.pbkdf2_hmac("sha256", p, s.encode(), 200000).hex(); print(f"pbkdf2_sha256$200000${s}${d}")'
```

Pilih kode admin yang panjang dan unik, bukan kode angka pendek.
Untuk sandi mode tamu, gunakan perintah yang sama dengan prompt
`Sandi mode tamu:` lalu simpan hasilnya sebagai `GUEST_CODE_HASH`.

## Keputusan desain

- Pembacaan riwayat dan ringkasan admin memakai batas waktu maksimum 5 detik
  tanpa percobaan ulang otomatis. Pemuatan riwayat memiliki batas waktu total
  5 detik dan mengembalikan sesi kosong bila gagal; operasi konfigurasi admin
  yang menyimpan perubahan memakai tenggat terpisah 10 detik.
- Circuit breaker chat terbuka selama 2 menit hanya untuk error layanan nyata
  `UNAVAILABLE`, `DEADLINE_EXCEEDED` RPC, dan `PermissionDenied`; timeout lokal
  ketika menunggu anggaran baca tidak membuka breaker. Antrean tulis dibatasi
  200 pekerjaan; saat penuh, pekerjaan tertua dibuang dan penghitung
  `writes_dropped` bertambah. Error sementara `UNAVAILABLE` dan
  `DEADLINE_EXCEEDED` dijadwalkan ulang paling banyak tiga kali dengan jeda 1,
  3, dan 8 detik; `ABORTED` dapat dicoba ulang tetapi tidak membuka breaker.
  Setelah seluruh retry gagal, pekerjaan tetap di antrean sampai sirkuit
  tertutup. Error permanen dicatat tanpa retry. Setiap upaya tulis chat memiliki
  tenggat 10 detik dan tidak menahan pekerja antrean saat operasi melampaui
  tenggat.
- Setiap pertanyaan mengantrekan satu transaksi `@firestore.transactional`
  untuk membaca dan memperbarui chunk serta dokumen pengguna yang sudah ada.
  Retry antrean membuat objek transaksi baru; transaksi tidak dibagikan
  antar-thread. UID yang belum terkonfirmasi tidak dipakai untuk membaca atau
  menulis riwayat. Profil baru dibuat oleh transaksi simpan nama yang menaikkan
  `stats/global.total_devices` hanya jika dokumen belum ada. Transaksi
  pertanyaan mencatat UID ter-hash pada `stats/hourly_YYYYMMDDHH`; pembukaan
  halaman saja tidak membuat dokumen statistik atau pengguna.
  Penulisan `stats/daily.questions` dan FAQ terpisah (FAQ hanya jika pertanyaan
  hasil normalisasi tidak kosong).
  Menu Kesehatan sistem menampilkan `writes_ok`, `writes_failed`,
  `writes_retried`, `writes_dropped`, error terakhir, dan waktu penutupan
  circuit breaker.
- Streamlit menyediakan `st.context.cookies` setidaknya sejak versi 1.52,
  sesuai batas minimum pada dependency proyek (antarmuka hanya-baca untuk
  cookie pada request awal).
  Cookie `jh_uid` dibuat sebagai cookie first-party browser dengan atribut
  standar `Max-Age=31536000`, `Path=/`, dan `SameSite=Lax` (`Secure` pada
  HTTPS). Android WebView memiliki pengelola cookie Chromium; pemulihan tetap
  bergantung pada cookie yang diaktifkan di WebView dan cookie dikirim ke host
  aplikasi yang sama. Penerimaan aktual pada deployment Streamlit Cloud dan
  build WebView produksi perlu dites di perangkat/lingkungan tersebut; repositori
  tidak menyediakan akses runtime untuk menguji konfigurasi deployment.
- Hash UID dipakai pada ringkasan pengguna unik. FAQ menggunakan hash SHA-256
  dari pertanyaan, sehingga tidak menyimpan UID. Pemeriksaan blokir dan batas
  membaca data pengguna yang sudah dimuat, tanpa membaca Firestore lagi di
  jalur pertanyaan.
- Konfigurasi aplikasi saat berjalan disimpan sementara di memori selama lima
  menit. Nilai cache/default dikembalikan seketika; pembacaan awal dan refresh
  dilakukan di thread latar dengan batas 5 detik. Timeout lokal dicatat pada
  Kesehatan sistem tanpa membuka breaker. Login admin menggunakan PBKDF2 atau
  scrypt dan `hmac.compare_digest`; dasbor hanya diimpor setelah login berhasil.
- Mutasi dashboard admin (key, urutan model, konfigurasi runtime, sandi
  Pengaturan, dan status blokir) memakai timeout Firestore 10 detik secara
  langsung, tidak membaca status breaker chat, dan menampilkan jenis error
  tanpa mencatat isi API key. Error tulis antrean mencatat pesan yang disamarkan
  dan field penyebab; UID perangkat divalidasi sebelum dipakai sebagai ID
  dokumen.
- Penguncian login admin mencatat kegagalan per UID dan secara global di memori
  proses. Lima kegagalan mengunci semua login di proses tersebut selama 15
  menit. Hitungan global diatur ulang setelah login berhasil, sedangkan hitungan
  UID lain tetap tersimpan. Status ini bertahan setelah halaman dimuat ulang, tetapi tidak
  setelah proses dimulai ulang dan tidak dibagikan antarproses.
- Login kode tamu memakai kunci session/UID/global 5 kali/15 menit yang terpisah
  dari penghitung login admin. Guest tidak menyentuh data profil/riwayat; satu-
  satunya tulis Firestore yang diizinkan adalah increment `guest_questions`
  pada statistik harian.
- Admin cukup mengetik sandi menu Pengaturan di dasbor; aplikasi membuat
  hash PBKDF2 sebelum menyimpannya. Hash konfigurasi aktif diprioritaskan
  daripada `SETTINGS_PASSWORD` lama. Pengguna biasa tetap harus memasukkan
  sandi untuk membuka menu.
- Key chat, indexing, dan admin tidak saling meminjam. Daftar status key di-cache
  selama 60 detik; kegagalan Firestore tidak menghentikan seed chat dari Secrets.
  Cooldown kuota harian memakai tengah malam waktu Pasifik, sedangkan 429 per
  menit memakai cooldown pendek. Chat mencoba model teratas lain pada key yang
  sama sebelum pindah ke key berikutnya.
- Override chat hanya disimpan di `st.session_state` dan tidak masuk konfigurasi
  privat, dashboard, indexing, atau admin. Pemakaian lewat override tidak
  menaikkan hitungan batas harian pengguna.
- Daftar key dan perubahan status hanya disimpan saat berubah. Error yang
  disimpan hanya berupa penanda aman, dan key disamarkan pada pesan kegagalan
  sinkronisasi serta respons pengguna. Ringkasan FAQ tidak berjalan otomatis;
  cache hasil di `admin_cache/faq_summary` digunakan hanya untuk fingerprint FAQ
  yang sama.
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
- Verifikasi belum memakai proyek Firestore produksi; hasil bergantung pada
  aturan IAM dan ketersediaan akun layanan saat penerapan.
- Pengujian cookie `jh_uid`, rerun fragment, dan penyimpanan riwayat tetap perlu
  diverifikasi pada deployment Streamlit Cloud serta Android WebView produksi;
  tes otomatis lokal memeriksa eksekusi JavaScript dan batas akses Firestore,
  tetapi tidak mereproduksi siklus hidup browser pada deployment tersebut.
