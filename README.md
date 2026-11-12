# Asisten AI Pribadi

Aplikasi Streamlit untuk mencari informasi dalam dokumen `.docx`, `.doc`, `.txt`,
Google Docs, dan `.pdf` di Google Drive menggunakan LangChain, FAISS, dan Google Gemini. Aplikasi dapat membantu mencari
lirik, hukum, cerita, quotes, dan catatan berbahasa Indonesia maupun Sunda.

## Persiapan

1. Gunakan Python 3.10 atau lebih baru.
2. Buat virtual environment dan pasang dependensi:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Buat Google Cloud service account, aktifkan Google Drive API, lalu berikan
   akses **Viewer** ke folder Drive yang akan dibaca. Folder bersama juga harus
   dibagikan langsung kepada alamat email service account.
4. Siapkan API key Google Gemini dari Google AI Studio.
5. Atur variabel lingkungan berikut:

   ```bash
   export GOOGLE_API_KEY="API_KEY_GEMINI"
   export GOOGLE_DRIVE_FOLDER_ID="ID_FOLDER_GOOGLE_DRIVE"
   export GOOGLE_APPLICATION_CREDENTIALS="/path/ke/service-account.json"
   ```

   Di Windows PowerShell, gunakan `$env:GOOGLE_API_KEY="..."` dan format
   `$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\service-account.json"`.
   Sebagai alternatif untuk kredensial service account, isi
   `GOOGLE_SERVICE_ACCOUNT_JSON` dengan konten JSON kredensial. Jangan commit
   API key atau file kredensial ke Git.

## Menjalankan aplikasi

Dari direktori proyek jalankan:

```bash
streamlit run app.py
```

Buka URL lokal yang ditampilkan Streamlit. Aplikasi memuat indeks FAISS lokal
tanpa memulai sinkronisasi otomatis. Setelah izin Google Drive tersedia, tekan
**Sinkronkan Sekarang** untuk memeriksa perubahan; setiap sesi memproses maksimal
10 file dan menyimpan progres agar dapat dilanjutkan secara manual. Jika batas
kuota Gemini sementara tercapai, aplikasi menunggu lalu mencoba ulang batch
embedding yang gagal (maksimal dua kali). File yang sudah tersimpan tidak
diulang. Jika kuota tetap habis atau waktu tunggu melebihi lima menit,
sinkronisasi dijeda dengan progres file sebelumnya tetap tersimpan.
Sumber yang sudah tersedia dapat dibaca lengkap per bagian dari panel pratinjau.
File `.doc` lama memerlukan program `antiword`; jika belum tersedia, sumber
tersebut ditandai gagal dan dapat dicoba ulang setelah dependensi dipasang atau
file disimpan sebagai `.docx`.

## Catatan

- Akun service account memerlukan akses baca ke folder Drive.
- Embedding dan jawaban dikirim ke layanan Google Gemini; jangan sinkronkan
  dokumen yang tidak boleh diproses oleh layanan eksternal.
- Jawaban terkait hukum merupakan ringkasan dokumen, bukan nasihat hukum
  profesional.
