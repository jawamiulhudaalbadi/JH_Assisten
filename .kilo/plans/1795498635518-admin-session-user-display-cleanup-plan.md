# Rencana Perbaikan: Sesi Admin Permanen, Tampil Nama Pengguna, Pembersihan Data & Dokumentasi

## Analisis Masalah

### 1. Sesi Admin Permanen per ID Perangkat (`admin_session.py` & `app.py`)

**Penyebab Masalah:**
- Fungsi `_restore_admin_session()` di `app.py:1319` memeriksa token dari `st.session_state` atau cookie `jh_adm`
- Tombol **"Keluar"** di `admin_dashboard.py:1320` menjalankan `build_admin_session_clear_script()` yang **menghapus cookie & localStorage** (`jh_adm`)
- Setelah logout, saat reload aplikasi, cookie sudah terhapus → `_restore_admin_session()` tidak menemukan token → admin diminta login ulang
- **Ketidaksesuaian dengan kebutuhan:** "Jika ID perangkat sudah pernah login benar, meskipun keluar dari mode admin atau reload, masuk ke mode admin lagi langsung masuk tanpa sandi"

**Solusi:**
- **Jangan hapus token saat "Keluar"** — hanya sembunyikan dashboard (`admin_authenticated = False`)
- Token di cookie/localStorage (TTL 30 hari) tetap tersimpan
- Saat reload, `_restore_admin_session()` akan menemukan token valid → `admin_authenticated = True` otomatis
- Hanya **"Cabut semua sesi admin"** (increment epoch) yang benar-benar mencabut token semua perangkat

### 2. Tampilkan Nama Pengguna di Menu Utama Paling Atas (`app.py`)

**Keadaan Saat Ini:**
- Fungsi `_render_header()` (baris 1920-1932) hanya menampilkan judul aplikasi
- Nama pengguna tersedia di `st.session_state.get("history_stats", {}).get("name", "")` setelah disimpan

**Solusi:**
- Tambahkan tampilan nama pengguna di header (misal: "Halo, {nama}" atau "Pengguna: {nama}")
- Hanya tampilkan jika `DEVICE_ID_FINAL` dan nama ada

### 3. Pembersihan & Statistik Pengguna Nyata (`firestore_store.py` & `admin_dashboard.py`)

**Analisis Mode Guest (Tamu):**
- Guest mode **tidak membuat dokumen di `users/{uid}`** — hanya menyimpan counter di session state
- Pertanyaan guest hanya increment `guest_questions` di `stats/daily_YYYYMMDD` (field terpisah dari `questions` dan `uids`)
- `total_devices` di `stats/global` **hanya menghitung pengguna terdaftar** (yang sudah simpan nama via transaksi `save_user_name`)
- Guest mode **tidak terkena pembersihan `get_empty_user_uids()`** karena tidak ada dokumen user

**Keadaan Saat Ini:**
- `get_empty_user_uids()` (baris 717-761) sudah mencari user dengan `total_questions == 0` dan tanpa chunk — **aman untuk guest**
- Tombol "Bersihkan pengguna kosong" di `_render_controls()` (baris 1137-1179) sudah menghapus dan recount `total_devices`
- `get_admin_statistics()` (baris 560-669) **HANYA membaca field `questions`** dari daily stats, **TIDAK membaca `guest_questions`**
- Dashboard menampilkan "Pertanyaan" per hari dari `questions` saja → **statistik guest tidak terlihat**

**Masalah yang Dilaporkan:**
1. "Filter/bersihkan data pengguna di Firestore agar ID lama tanpa nama dihapus" — fungsi sudah ada, perlu pastikan preview & konfirmasi jelas
2. "Statistik benar-benar menghitung dari data nyata" — **statistik guest (`guest_questions`) tidak ditampilkan di dashboard**

**Solusi:**
1. **Pembersihan:** Pastikan fungsi existing berfungsi; tambah info yang jelas di UI (jumlah kandidat, detail UID)
2. **Statistik Dashboard:** Modifikasi `get_admin_statistics()` untuk **juga membaca `guest_questions`** dari daily stats, dan tampilkan terpisah di `_render_statistics()`:
   - "Total perangkat (terdaftar)" = `total_devices` dari `stats/global`
   - "Pertanyaan pengguna" = `questions` dari daily stats
   - "Pertanyaan tamu" = `guest_questions` dari daily stats (baru)
3. **Tidak ubah** logika pembersihan user — sudah benar & aman untuk guest

### 4. Perbarui `CATATAN_PROYEK.md` & Unit Test

**Perlu Diperbarui:**
- `CATATAN_PROYEK.md`: Tambah catatan Fase 4D/perbaikan terbaru
- `tests/test_admin_session.py`: Test untuk verifikasi token persist setelah logout
- `tests/test_admin_dashboard_isolation.py`: Test menu Statistik tidak load data key/model

---

## Rincian Perubahan per File

### A. `services/admin_session.py`
**Tidak perlu diubah** — logika token/epoch sudah benar. Perubahan ada di `app.py` dan `admin_dashboard.py`.

### B. `app.py` — Perubahan Utama

#### 1. Ubah `_restore_admin_session()` (baris 1319-1344)
- **Sudah benar** — membaca token dari cookie `jh_adm` via `st.context.cookies`
- Token akan ditemukan saat reload karena cookie TTL 30 hari

#### 2. Hapus pemanggilan `_render_admin_session_storage()` setelah logout
- Saat ini `_render_admin_session_storage()` dipanggil saat `_admin_session_store_pending` True (baris 1348-1349)
- Setelah login berhasil, `_persist_admin_session()` set flag ini → script simpan token di browser
- **Tidak perlu ubah** — ini untuk simpan token pertama kali

#### 3. Tambah tampilan nama di `_render_header()` (baris 1920-1932)
```python
def _render_header() -> None:
    title_html = ...
    # Tambah: ambil nama dari history_stats
    history_stats = st.session_state.get("history_stats", {})
    profile_name = str(history_stats.get("name", "") or "")
    # Tampilkan nama jika ada
```

### C. `services/admin_dashboard.py` — Perubahan Utama

#### 1. Ubah Tombol "Keluar" di `_render_dashboard_fragment()` (baris 1320-1345)
**Sebelum:**
```python
if st.button("Keluar", key="admin_dashboard_exit"):
    st.session_state["_admin_logout_clear_pending"] = True
    st.rerun(scope="fragment")
# Lalu jalankan build_admin_session_clear_script() → HAPUS cookie & localStorage
```

**Sesudah:**
```python
if st.button("Keluar", key="admin_dashboard_exit"):
    st.session_state["admin_authenticated"] = False
    # JANGAN hapus token/cookie — biarkan persist
    st.rerun(scope="app")  # reload penuh untuk sembunyikan dashboard
```

#### 2. Hapus blok `_admin_logout_clear_pending` (baris 1323-1345)
- Hapus seluruh logika `st_javascript(build_admin_session_clear_script())` untuk logout
- Hanya set `admin_authenticated = False` dan `st.rerun(scope="app")`

#### 3. Pertahankan "Cabut semua sesi admin" (baris 1090-1115)
- Ini **harus tetap** hapus token (increment epoch) → validasi token gagal di semua perangkat
- Logika `_admin_revoke_clear_pending` dan `build_admin_session_clear_script()` dipertahankan di sini

#### 4. Modifikasi `_render_statistics()` (baris 127-155) — Tampilkan Statistik Guest Terpisah
Gunakan data `guest_questions` yang baru dari `get_admin_statistics()`:

```python
def _render_statistics(client: Any | None) -> None:
    st.subheader("Statistik")
    if st.button("Segarkan", key="admin_statistics_refresh"):
        _cached_admin_statistics.clear()
    client_key = str(id(client)) if client is not None else "unavailable"
    statistics = _cached_admin_statistics(client_key, client)
    
    # Ringkasan utama
    columns = st.columns(4)
    columns[0].metric("Total perangkat (terdaftar)", statistics["total_devices"])
    columns[1].metric("Aktif sekitar 1 jam", statistics["active_last_hour"])
    columns[2].metric("Unik hari ini", statistics["unique_today"])
    columns[3].metric("Unik 7 hari", statistics["unique_7_days"])
    st.caption(statistics["active_window_label"])
    
    # Tabel harian dengan kolom terpisah
    daily = statistics["daily_questions"]
    if daily:
        st.dataframe(
            [
                {
                    "Tanggal (WIB)": datetime.strptime(row["day"], "%Y%m%d")
                    .replace(tzinfo=_WIB)
                    .strftime("%d-%m-%Y"),
                    "Pertanyaan pengguna": row["questions"],
                    "Pertanyaan tamu": row.get("guest_questions", 0),  # BARU
                }
                for row in daily
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Ringkasan pertanyaan 14 hari belum tersedia.")
```

### D. `services/firestore_store.py` — Perubahan Baru untuk Statistik Guest

#### 1. Modifikasi `get_admin_statistics()` (baris 560-669)
Tambahkan pembacaan field `guest_questions` dari dokumen `daily_{day_key}`:

```python
# Di dalam read_statistics():
for day_key in day_keys:
    data = (
        stats_ref.document(f"daily_{day_key}")
        .get(...)
        .to_dict()
        or {}
    )
    daily_questions[day_key] = int(data.get("questions", 0))
    daily_guest_questions[day_key] = int(data.get("guest_questions", 0))  # BARU
    uids = {str(value) for value in data.get("uids", [])}
    ...
return {
    ...
    "daily_questions": [
        {"day": day_key, "questions": daily_questions[day_key], "guest_questions": daily_guest_questions[day_key]}  # BARU
        for day_key in reversed(day_keys)
    ],
}
```

#### 2. Return value tambah field `guest_questions` per hari di `daily_questions`

### E. `CATATAN_PROYEK.md`
Tambahkan entri baru di bagian "Fase 4D — perbaikan sesi admin & tampilan" (atau lanjutan Fase 4C).

### F. `tests/test_admin_session.py`
Tambah test:
- `test_token_persists_after_logout` — verifikasi token tidak terhapus saat logout biasa
- `test_revoke_all_sessions_invalidates_token` — verifikasi epoch increment mencabut token

### G. `tests/test_admin_dashboard_isolation.py`
Tambah test:
- `test_logout_does_not_clear_token` — verifikasi tombol Keluar tidak hapus cookie
- `test_revoke_clears_token` — verifikasi Cabut semua sesi hapus cookie

### H. `tests/test_firestore_store.py` (atau file test baru)
Tambah test untuk statistik guest:
- `test_get_admin_statistics_includes_guest_questions` — verifikasi `guest_questions` dibaca dari daily stats
- `test_guest_questions_not_counted_as_registered_users` — verifikasi guest tidak masuk `total_devices`

---

## Validasi & Pengujian

### Manual Testing Checklist
1. **Sesi Admin Permanen:**
   - Login admin via `/admin <kode>` atau tombol Admin 3x
   - Reload halaman → dashboard terbuka otomatis (tanpa minta sandi)
   - Klik "Keluar" → dashboard tertutup
   - Masuk ke mode admin lagi (klik Admin 3x atau `/admin`) → **langsung masuk tanpa sandi**
   - Tutup browser, buka lagi → **langsung masuk**
   - Gunakan "Cabut semua sesi admin" dari perangkat lain → token tidak valid, minta sandi ulang

2. **Tampil Nama di Header:**
   - Simpan nama pengguna
   - Periksa header paling atas menampilkan nama

3. **Pembersihan Pengguna:**
   - Buka Dashboard Admin → Kontrol lain → "Bersihkan pengguna kosong"
   - Verifikasi jumlah kandidat tampil, konfirmasi hapus, `total_devices` ter-update
   - Statistik "Total perangkat" di menu Statistik sesuai dengan data nyata

4. **Statistik Guest (BARU):**
   - Masuk mode guest (via sandi atau emergency valve)
   - Ajukan beberapa pertanyaan sebagai guest
   - Buka Dashboard Admin → Statistik
   - Verifikasi tabel harian menampilkan kolom "Pertanyaan tamu" > 0
   - Verifikasi "Total perangkat (terdaftar)" **tidak berubah** saat guest bertanya
   - Verifikasi "Pertanyaan pengguna" (regular) terpisah dari "Pertanyaan tamu"

5. **Unit Test:**
   - `pytest tests/test_admin_session.py -v`
   - `pytest tests/test_admin_dashboard_isolation.py -v`
   - `pytest tests/test_firestore_store.py -v` (termasuk test guest stats baru)
   - Semua test existing tetap lulus

---

## Urutan Implementasi

1. **`app.py`**: Tambah tampilan nama di `_render_header()`
2. **`admin_dashboard.py`**: Ubah tombol "Keluar" → jangan hapus token, hanya `admin_authenticated = False`
3. **`admin_dashboard.py`**: Hapus logika `_admin_logout_clear_pending` dan `build_admin_session_clear_script()` untuk logout biasa
4. **`admin_dashboard.py`**: Modifikasi `_render_statistics()` untuk tampilkan kolom "Pertanyaan tamu" terpisah
5. **`firestore_store.py`**: Modifikasi `get_admin_statistics()` untuk baca field `guest_questions` dari daily stats
6. **`CATATAN_PROYEK.md`**: Tambah catatan perubahan
7. **`tests/test_admin_session.py`**: Tambah test persist token
8. **`tests/test_admin_dashboard_isolation.py`**: Tambah test isolasi logout/revoke
9. **`tests/test_firestore_store.py`**: Tambah test statistik guest
10. **Jalankan test** memastikan semuanya lulus

---

## Catatan Teknis

- Cookie `jh_adm` diset dengan `Max-Age=2592000` (30 hari), `SameSite=Lax`, `Secure` di HTTPS
- `st.context.cookies` hanya membaca cookie pada **request HTTP awal** (bukan rerun)
- Token diverifikasi dengan epoch dari `config/admin_session` (cached 5 menit)
- Epoch increment via `revoke_all_admin_sessions()` → semua token invalid
- `admin_authenticated` di session_state hanya mengontrol render dashboard, bukan validasi token

### Mode Guest & Statistik (Detail Teknis)
- Guest mode **tidak pernah menulis ke koleksi `users/`** — hanya `st.session_state` lokal
- `enqueue_guest_question()` → `_write_daily_questions(guest=True)` → increment `guest_questions` di `stats/daily_YYYYMMDD`
- `stats/global.total_devices` **hanya increment saat `save_user_name()`** buat dokumen user baru (transaksional)
- `get_empty_user_uids()` hanya scan `users/` collection → **guest tidak terkena** pembersihan
- `get_admin_statistics()` sekarang baca **dua field** dari daily stats: `questions` (pengguna) + `guest_questions` (tamu)
- Dashboard menampilkan **terpisah**: "Pertanyaan pengguna" vs "Pertanyaan tamu" per hari