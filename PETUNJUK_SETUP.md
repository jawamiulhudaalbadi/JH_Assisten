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
                                                                                                                  