# Menyiapkan riwayat perangkat

Fase ini memakai Firestore database default dan tidak membaca atau memigrasikan
berkas lama di `chat_history/`.

1. Buat proyek Firebase dan gunakan paket Spark.
2. Dari Firebase Console, buat **Firestore Database** pada database default
   `(default)` dalam mode produksi.
   3. Buat service account di Google Cloud Console untuk proyek yang sama. Beri
      service account peran IAM yang dapat membaca dan menulis Firestore, lalu
         salin JSON key-nya.
         4. Di Streamlit Community Cloud, buka **App settings → Secrets** dan tambahkan
            `FIREBASE_SERVICE_ACCOUNT` sebagai tabel TOML, dengan isi JSON service
               account sebagai nilai-nilainya. Jangan menaruh file key di repositori.

                  Bentuk umumnya:

                     ```toml
                        [FIREBASE_SERVICE_ACCOUNT]
                           type = "service_account"
                              project_id = "ID_PROYEK_FIREBASE"
                                 private_key_id = "ID_KEY"
                                    private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
                                       client_email = "nama-service-account@ID_PROYEK_FIREBASE.iam.gserviceaccount.com"
                                          client_id = "ID_KLIEN"
                                             auth_uri = "https://accounts.google.com/o/oauth2/auth"
                                                token_uri = "https://oauth2.googleapis.com/token"
                                                   auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
                                                      client_x509_cert_url = "URL_SERTIFIKAT"
                                                         ```

                                                         5. Atur aturan Firestore agar akses klien ditutup total; aplikasi mengakses
                                                            database hanya dari server melalui service account:

                                                               ```text
                                                                  rules_version = '2';
                                                                     service cloud.firestore {
                                                                          match /databases/{database}/documents {
                                                                                 match /{document=**} {
                                                                                          allow read, write: if false;
                                                                                                 }
                                                                                                      }
                                                                                                         }
                                                                                                            ```

                                                                                                            6. Deploy ulang aplikasi. Bila secret belum diatur atau Firestore sedang gagal,
                                                                                                               percakapan tetap berjalan di sesi Streamlit saat ini, tetapi tidak tersimpan
                                                                                                                  lintas sesi.

                                                                                                                  Riwayat baru tersimpan per perangkat dan per obrolan di
                                                                                                                  `users/{uid}/chats/{chat_id}/chunks/`. Pengguna dapat menghapus semua riwayat
                                                                                                                  perangkatnya dari sidebar. Penghapusan tersebut tidak menghapus berkas lama
                                                                                                                  `chat_history/`.

## Mode tamu berkode

Mode tamu tidak terbuka otomatis. Tambahkan `GUEODE_HASH` ke Streamlit
Secrets dST_Cengan hash PBKDF2 atau scrypt; biarkan kosong untuk menyembunyikan
tombol mode tamu. Buat hash PBKDF2 secara lokal, lalu salin hasilnya ke secret:

```sh
python -c 'import getpass, hashlib, secrets; p=getpass.getpass("Sandi mode tamu: ").encode(); s=secrets.token_hex(16); d=hashlib.pbkdf2_hmac("sha256", p, s.encode(), 200000).hex(); print(f"pbkdf2_sha256$200000${s}${d}")'
```

Tambahkan `ADMIN_SESSION_SECRET` sesuai kebutuhan sesi admin; gunakan nilai
rahasia acak yang terpisah dari kode admin. Saat masuk, pengguna tanpa ID final
atau nama dapat memilih **Mode tamu** lalu memasukkan sandi tersebut. Riwayat
tidak disimpan; pertanyaan dibatasi 10 per hari dalam sesi (dapat diubah dari
Dashboard Admin → **Kontrol lain**).

Katup darurat **Buka mode tamu tanpa sandi** berada di menu yang sama. Katup
bawaan mati dan berlaku paling lama enam jam sejak dibuka. Gunakan hanya
sementara; dashboard dan banner pengguna menampilkan status serta sisa waktu.

## Nama dan akses obrolan

Nama disimpan pada profil Firestore `users/{uid}`, bukan pada masing-masing
obrolan. Setelah nama berhasil disimpan, pengguna dengan ID perangkat yang sama
tidak perlu memasukkannya lagi saat membuat obrolan baru atau memuat ulang
aplikasi. Kolom chat tetap nonaktif sampai ID perangkat terverifikasi dan nama
berhasil disimpan. Saat ID masih diperiksa, tunggu hingga proses selesai;
aplikasi terus memeriksa ID dan akan menghapus pesan coba lagi setelah ID
ditemukan.

Mode tamu tetap tersedia untuk pengguna yang mengetahui sandinya. Mode ini
tidak menyimpan nama maupun riwayat obrolan.
                                                                                                                  