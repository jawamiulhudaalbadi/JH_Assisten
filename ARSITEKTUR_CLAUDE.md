# Arsitektur JH Asisten AI

Dokumen ini mendokumentasikan arsitektur lengkap aplikasi JH Asisten AI berdasarkan kode sumber di `app.py` dan folder `services/`.

---

## 1. Ringkasan Proyek (Project Stack)

| Komponen | Teknologi |
|----------|-----------|
| **Frontend/UI** | Streamlit (Python) |
| **Database** | Google Cloud Firestore (NoSQL) |
| **AI/ML** | Google Gemini (chat, embedding), FAISS (vector search) |
| **File Storage** | Google Drive API (DOCX, PDF, Google Docs, TXT) |
| **Auth** | OAuth 2.0 (Google Drive), Custom admin session (HMAC-signed token) |
| **Caching** | In-process (thread-safe), Streamlit `st.cache_data`/`st.cache_resource` |
| **Background Jobs** | `ThreadPoolExecutor` (single-worker per job type) |
| **Language** | Python 3.10+, Indonesian/Sundanese UI |

---

## 2. Device ID Identification Flow

### 2.1 Prioritas Resolusi UID (`services/device_identity.py`)

Fungsi `get_device_uid()` mengembalikan UID perangkat dengan urutan prioritas:

1. **Session State** (cepat): Jika `_device_uid_browser_resolved=True` dan `device_uid` valid di session state → pakai langsung.
2. **URL Query Param `?uid=`**: Baca dari `st.query_params.get("uid")`, hapus param, validasi regex `^[a-f0-9]{16,32}$` → simpan ke session & cookie.
3. **Cookie `jh_uid`**: Baca dari `st.context.cookies.get("jh_uid")` (HTTP request header) → validasi → simpan ke session & localStorage via JS.
4. **Browser localStorage (`jh_asisten_device_uid`)**:
   - Jalankan `st_javascript(_READ_UID_SCRIPT)` → baca localStorage, set cookie `jh_uid`.
   - Jika kosong: jalankan `_CREATE_UID_SCRIPT` → generate UUID v4 (crypto.randomUUID atau crypto.getRandomValues), simpan ke localStorage & cookie.
5. **Temporary UID** (fallback): Generate `uuid.uuid4().hex` hanya di session state (tidak persist di browser).

### 2.2 Karakteristik UID
- Format: hex 16-32 karakter (`[a-f0-9]{16,32}`)
- Disimpan di 3 tempat: session_state, cookie `jh_uid`, localStorage `jh_asisten_device_uid`
- Cookie TTL: 1 tahun (`Max-Age=31536000`)
- `st_javascript` timeout: 4 detik (`_MAX_BROWSER_WAIT_SECONDS`)

### 2.3 Optimasi Return Visit
- Cookie `jh_uid` dibaca **synchronous** dari `st.context.cookies` pada initial HTTP request (fast path)
- Hanya fallback ke JS/localStorage jika cookie tidak ada
- `_device_uid_browser_resolved` flag mencegah re-check pada rerun berikutnya

---

## 3. Chat History Storage (Firestore)

### 3.1 Koleksi Utama
```
users/{uid}/
  ├─ name, name_lower, first_seen, last_active
  ├─ total_questions, total_messages, total_chunks
  ├─ day (YYYYMMDD), day_count, daily_question_limit
  ├─ blocked (bool)
  ├─ active_chat_id, chat_ids[] (array), saved_chats[]
  └─ chunks/{chunk_number:06d}/
       ├─ messages[] (role, content, timestamp, model_used, sources, answer_id, latency_ms, source_count, fallback, top_score, source_context)
       └─ count (int)

stats/
  ├─ global: {total_devices}
  ├─ daily_YYYYMMDD: {questions, guest_questions, uids[]}
  ├─ hourly_YYYYMMDDHH: {uids[], active_at: {uid_hash: timestamp}}
  └─ quality: {feedback_count, positive_count, negative_count, without_sources_count, fallback_count, latency_total_ms, latency_count, models: {...}}

feedback/{answer_id}/ (32-char hex)
reviews/{answer_id}/ (label, ai_label, ai_rationale)
faq/{hash(normalized_question)}/ (question, display_question, count, last_seen)
settings/runtime/ (maintenance, announcement, daily_question_limit, guest_*, guest_emergency_*)
config/ (admin_session.epoch, keys_chat, keys_index, keys_admin, model_order)
system_health/errors/ (recent errors)
admin_cache/faq_summary/ (fingerprint, summary, updated_at)
```

### 3.2 Chunking & Message Pair
- `MAX_MESSAGES_PER_CHUNK = 20` (10 pasang Q&A)
- `append_message_pair()`: tambah 2 pesan (user+assistant), buat chunk baru jika >20
- Load history: ambil **2 chunk terakhir** (maks 30 pesan) via `load_chat_history()`

### 3.3 Guest Mode Storage
- Guest questions **tidak** disimpan di `users/` collection
- Hanya counter di `stats/daily_YYYYMMDD.guest_questions` (via `enqueue_guest_question()`)
- `total_devices` hanya hitung registered users (punya nama)

### 3.4 Write Path (Async, Bounded Queue)
- `enqueue_question_answer()` → `_enqueue_write()` → `_WRITE_QUEUE` (max 200)
- Single worker thread (`_WRITE_THREAD`) memproses dengan retry & circuit breaker
- `_bounded_read()` untuk reads dengan timeout & fallback

### 3.5 Circuit Breaker
- `_CIRCUIT_OPEN_UNTIL` + `FIRESTORE_CIRCUIT_SECONDS = 120s`
- Trigger: UNAVAILABLE, DEADLINE_EXCEEDED, PERMISSION_DENIED
- Auto-recover setelah 2 menit

---

## 4. Admin Authentication & Dashboard

### 4.1 Admin Session Token (`services/admin_session.py`)
- Token format: `base64url(uid|expires_at|signature)` (tanpa padding)
- Signature: HMAC-SHA256(ADMIN_SESSION_SECRET, `uid|expires_at|epoch`)
- TTL: 30 hari (`ADMIN_SESSION_TTL_SECONDS`)
- Disimpan di: cookie `jh_adm` + localStorage `jh_adm` (via JS script)

### 4.2 Epoch-based Revocation
- `config/admin_session.epoch` (integer, default 0)
- `revoke_all_admin_sessions()` → `firestore.Increment(1)` atomik
- Cache epoch 5 menit (`ADMIN_SESSION_EPOCH_CACHE_SECONDS`)
- Token valid hanya jika `token_epoch == current_epoch`

### 4.3 Session State Separation (Bug Fix)
| State | Deskripsi | Persist |
|-------|-----------|---------|
| `admin_device_verified` | Cookie/token valid, device dikenali | Ya (survive rerun) |
| `admin_authenticated` | Dashboard terbuka (password verified) | Ya |
| `admin_exited_manually` | User klik "Keluar" (hide dashboard only) | Ya |

**Flow**:
1. Sidebar "Admin" button → cek `admin_device_verified` dulu
2. Jika verified & `!admin_exited_manually` → buka dashboard **tanpa password** (instant)
3. Jika tidak verified → prompt kode admin (3 klik buka prompt)
4. "Keluar" button → set `admin_authenticated=False`, `admin_exited_manually=True` (token TETAP ada)
5. "Cabut semua sesi admin" → increment epoch → invalidate semua token → clear cookie/localStorage via JS

### 4.4 Admin Dashboard Tabs (`services/admin_dashboard.py`)
1. **Statistik**: total_devices (registered only), active_last_hour, unique_today/7d, daily_questions (user + guest columns)
2. **Riwayat chat**: cari user, buka chat, keyword search
3. **Pengguna**: list, block/unblock, daily limit per user, **Hapus semua tanpa nama** (confirm dialog)
4. **Kualitas jawaban**: metrics, AI evaluation, manual review labels
5. **Pertanyaan sering (FAQ)**: tabel + AI summary
6. **Sinkron Drive**: start/pause/resume, retry failed files
7. **Sandi Pengaturan**: hash pbkdf2_sha256
8. **API Key**: 3 groups (chat/indexing/admin), encrypted (Fernet), enable/disable/delete
9. **Urutan model**: manual/automatic
10. **Kontrol lain**: maintenance, announcement, daily limits, guest emergency toggle, **Cabut semua sesi admin**, export CSV, clear cache
11. **Kesehatan sistem**: Firestore status, write health, perf metrics, model router snapshot, recent errors

---

## 5. Session State & Cache Management

### 5.1 Key Session State Variables
```python
# Device & Auth
device_uid, _device_uid_browser_resolved, _device_uid_source
admin_device_verified, admin_authenticated, admin_exited_manually
_admin_session_token, _admin_auth_method, _admin_login_prompt_open
guest_mode, guest_mode_via_emergency, guest_chat_id, guest_questions_count

# Chat
chat_id, messages[], history_stats, loaded_chat_id
_saved_chats, _saved_chats_uid, _saved_chats_error
delete_all_chats_confirmation, clear_active_chat_confirmation

# Admin Dashboard Cache
_admin_cache_{statistics,users,faq,chats_*,history_*,faq_summary_*}
_admin_{history,users,quality}_page_* (pagination cursors)
_admin_session_store_pending, _admin_revoke_clear_pending

# Perf
_PERF_BLOCKS, _PERF_RUN_STARTED
```

### 5.2 In-Process Caches (Thread-Safe)
| Cache | TTL | Invalidate |
|-------|-----|------------|
| `_KEY_CACHE` (key_pool) | 60s | `invalidate_cache()`, admin write |
| `_MODEL_ORDER_CACHE` | 60s | `save_model_order()` |
| `_CHAT_MODEL_CACHE` (ai_service) | 1800s | - |
| `_QUERY_EMBEDDING_CACHE` | - | - |
| `_EPOCH_CACHE` (admin_session) | 300s | `revoke_all_admin_sessions()` |
| `_RUNTIME_CONFIG` (firestore_store) | 300s | `update_runtime_config()`, background refresh |
| `st.cache_data` (admin_dashboard) | 60s | Manual "Segarkan" button |

### 5.3 Background Refresh Pattern
- `get_runtime_config()`: return cached immediately, spawn background refresh jika stale
- `get_keys()`, `get_model_order()`: similar pattern dengan `_KEY_REFRESHING` flag
- Generation counters (`_KEY_REFRESH_GENERATION`, `_MODEL_ORDER_REFRESH_GENERATION`) prevent stale updates

---

## 6. API Key Management (`services/key_pool.py`)

### 6.1 Three Isolated Groups
- **chat**: user-facing chat (`GOOGLE_API_KEY`)
- **indexing**: Drive sync/embedding (`GOOGLE_API_KEY_INDEX`)
- **admin**: admin dashboard AI (`GOOGLE_API_KEY_ADMIN`)

### 6.2 Storage & Encryption
- Firestore: `config/keys_{chat,index,admin}` → encrypted dengan Fernet (`ENCRYPTION_KEY`)
- Seed key dari env/secret → auto-disabled jika custom keys exist
- `ManagedKey`: sid (fingerprint), label, enabled, cooldown_until, last_error, key (encrypted), model_cooldowns

### 6.3 Quota Handling
- `record_quota()`: set `cooldown_until` (daily → next PT midnight; rate → 5-60s)
- Per-model cooldown untuk chat group
- `get_available_keys()`: filter enabled + not in cooldown

### 6.4 Model Order
- `config/model_order`: `{mode: "automatic|manual", models: [...]}`
- `ordered_models()`: apply manual order atau secrets `GEMINI_CHAT_MODELS`

---

## 7. AI Service Pipeline (`services/ai_service.py`)

### 7.1 Embedding
- Model: `gemini-embedding-001` (768 dim, truncated & L2 normalized)
- Rate limiter: sliding window 60s (TPM, RPM, HTTP calls)
- Quota-aware: retry dengan exponential backoff, multi-key rotation
- `_QuotaAwareEmbeddings`: wraps `GoogleGenerativeAIEmbeddings`

### 7.2 Search Index (FAISS)
- Source: Google Drive files (DOCX, PDF, Google Docs, TXT)
- Chunking: `RecursiveCharacterTextSplitter` (3500/350 default)
- Manifest: `faiss_index/drive_manifest.json` (folder_id, files, has_index, sources)
- Update: `update_search_index()` → reuse unchanged chunks (fingerprint), embed only new/changed

### 7.3 Answer Generation
- **Hedged requests**: parallel model attempts (max 2, 7s hedge delay)
- **Model router**: self-learning, EWMA TTFT, cooldown on failures
- **Continuation**: auto-continue jika `MAX_TOKENS` finish reason (max 5x)
- **Fallback**: keyword search lokal jika embedding/chat gagal
- **Local fallback**: `_build_local_fallback_answer()` stitch chunks atau show top-K

### 7.4 Special Modes
- **Full context**: `is_full_context_request()` → retrieve 10, expand to 14 chunks window
- **Discussion**: `is_discussion_request()` → retrieve 9, expand neighbors ±1 (max 14)
- **Source link**: `is_source_link_request()` → match filename vs query

### 7.5 Output Formatting
- `restore_source_indentation()`: match answer indentation ke source (NBSP)
- `_StreamingMarkdownLineSanitizer`: preserve leading spaces during streaming
- References: "Sumber Rujukan: 1. Nama (Halaman X/Bagian Y)"

---

## 8. Google Drive Sync (`services/sync_manager.py`, `services/gdrive_service.py`)

### 8.1 DriveSyncManager (Single-Worker Background)
- `ThreadPoolExecutor(max_workers=1)` serialized sync
- Phases: `idle` → `checking` → `downloading` → `embedding` → `saving` → `complete`/`paused`/`quota_error`/`error`
- Pause/Resume: `_resume_event` (threading.Event)
- Session limit: `MAX_FILES_PER_SYNC_SESSION = 1000`
- Checkpoint: save index + manifest every batch

### 8.2 File Processing
1. `list_drive_files()`: recursive folder walk, filter supported MIME types
2. Compare manifest `modifiedTime` → detect changed/new/removed
3. `read_drive_files()`: download & extract text per file
   - DOCX: `python-docx`
   - PDF: `pdftotext` (poppler) → fallback `pypdf` → fallback `PyMuPDF`
   - Google Docs: export as text/plain
   - Legacy .doc: `antiword`
   - TXT: UTF-8/CP1252
4. Corruption detection: `_CORRUPTED_CHARACTER_RATIO_THRESHOLD = 0.3`
5. Embedding: `update_search_index()` dengan quota handling
6. Save: FAISS index + manifest + source_texts/{file_id}.json

### 8.3 RPD (Requests Per Day) Handling
- Daily quota → persist `sync_status.json` dengan `rpd_reset_at` (LA midnight)
- Auto-clear setelah reset time
- Per-API-key hash tracking

### 8.4 Failed File Persistence
- `sync_failures.json`: `{folder_id, files: {file_id: {name, modified_time, error}}}`
- Retry button di dashboard → re-queue file

---

## 9. Model Router (`services/model_router.py`)

### 9.1 Self-Learning Ranking
- **TTFT EWMA** (α=0.4): `record_success(model, seconds)`
- **Failures**: consecutive failures → exponential cooldown (60s, 2m, 4m... max 15m)
- **404/Not Found**: 6 jam cooldown
- **Slow losses**: `record_slow()` (lost hedge race) → deprioritize
- **Thinking unsupported**: per-model flag

### 9.2 Rank Order
1. Healthy + fast (EWMA ≤ 8s)
2. Healthy + slow (EWMA > 8s, fastest first)
3. Cooling down (soonest expiry first)
4. Dead (404, longest cooldown last)

### 9.3 Warm-up
- `warm_up(models, probe)`: parallel probe top-4 models (20s deadline)
- `first_token_timeout(model, default)`: 3× EWMA (clamped 8s..default)

---

## 10. User Profile & Policy (`services/user_profile.py`, `services/admin_logic.py`)

### 10.1 Name Validation
- 2-30 chars, alnum + space/./-
- No URL-like patterns
- `needs_user_name(uid_final, value)`: only prompt after persistent UID

### 10.2 Daily Question Limit
- Per-device: `daily_question_limit` (0 = unlimited)
- Per-user override: `users/{uid}.daily_question_limit`
- Global default: `runtime.daily_question_limit`
- Guest: `runtime.guest_daily_question_limit` (default 10)
- Reset: WIB midnight (stored `day` + `day_count`)

### 10.3 Block/Allow Logic
- `question_is_allowed()`: check `blocked` + daily limit
- Admin: `update_admin_user_policy()` (block + daily_limit)

### 10.4 Login Lockout (Process-Wide)
- Admin & Guest separate counters
- 5 failed attempts → 15 min lockout (session + UID + global)
- `_PROCESS_ADMIN_LOCK` protects shared state

---

## 11. Performance Monitoring (`services/perf_monitor.py`)

- `record_run(durations, total_seconds)`: bounded deque (30 samples)
- `record_admin_menu_run(menu, duration)`: separate deque (30 samples)
- Auto-warning log jika >1.5s
- Exposed di dashboard "Kesehatan sistem"

---

## 12. Text Cleaner (`services/text_cleaner.py`)

- OCR garbage detection untuk teks Arab
- Token-based: good chars (alnum + Latin extended) / total < 0.6 → garbage
- Line-based: >50% single-char tokens → replace dengan `[teks Arab]`
- Consecutive garbage tokens (≥2) → single marker
- Deduplicate consecutive markers, collapse 3+ newlines

---

## 13. Future Enhancement Targets

### 13.1 Local Device Cache (IndexedDB / localStorage)
**Goal**: Offline-first chat history, instant load on return visit

**Implementation Plan**:
```
browser localStorage/IndexedDB
├── chat_history_{uid}_{chat_id}  (last 30 messages, JSON)
├── saved_chats_{uid}             (named chat list)
├── user_profile_{uid}            (name, settings)
├── faq_cache                     (top 50 questions)
└── sync_metadata                 (last_sync_timestamp, etag)
```

**Sync Strategy**:
- Background sync via Service Worker atau `st_javascript` periodic
- Conflict resolution: server-wins untuk Firestore, local-wins untuk draft messages
- Incremental: hanya sync `last_active` changed chats

### 13.2 Background Sync (Service Worker / Web Worker)
**Goal**: Pre-fetch index updates, sync guest questions, warm models

**Architecture**:
```
Service Worker (navigator.serviceWorker)
├── Cache API: FAISS index chunks (if small enough), manifest
├── Background Sync API: queue guest questions when offline
├── Periodic Background Sync: daily index manifest check
└── Push API (optional): server → client index invalidation
```

**Integration Points**:
- `services/sync_manager.py`: expose `/api/sync/manifest` endpoint (ETag/Last-Modified)
- `services/ai_service.py`: model warm-up bisa di-trigger via push
- `services/firestore_store.py`: guest question queue → flush when online

### 13.3 Implementation Steps

| Phase | Task | Files to Modify |
|-------|------|-----------------|
| 1 | Add `localStorage` chat cache in `app.py` (save/load) | `app.py`, `services/device_identity.py` |
| 2 | Create `services/local_cache.py` (IndexedDB wrapper via `st_javascript`) | New file |
| 3 | Modify `load_chat_history()` → check local first, merge | `services/firestore_store.py` |
| 4 | Add Service Worker registration in Streamlit `index.html` (custom component) | Custom component |
| 5 | Background sync logic for guest questions | `services/firestore_store.py`, `services/sync_manager.py` |
| 6 | Push invalidation for FAISS index (Firestore → client) | `services/sync_manager.py`, new endpoint |

---

## 14. File Structure Summary

```
JH_Assisten/
├── app.py                      # Main Streamlit app (2914 lines)
├── services/
│   ├── ai_service.py           # Gemini chat, embedding, FAISS, search (2817 lines)
│   ├── admin_dashboard.py      # Admin UI tabs (1471 lines)
│   ├── admin_logic.py          # Auth, policy, lockout (281 lines)
│   ├── admin_session.py        # HMAC session tokens, epoch (229 lines)
│   ├── device_identity.py      # Device UID resolution (188 lines)
│   ├── firestore_store.py      # All Firestore ops, queue, circuit breaker (2526 lines)
│   ├── gdrive_service.py       # Drive API, file extraction (563 lines)
│   ├── key_pool.py             # Encrypted API key groups (797 lines)
│   ├── model_router.py         # Self-learning model ranking (257 lines)
│   ├── perf_monitor.py         # Performance timing (74 lines)
│   ├── sync_manager.py         # Background Drive sync (1183 lines)
│   ├── text_cleaner.py         # OCR garbage removal (83 lines)
│   └── user_profile.py         # Name validation (28 lines)
├── tests/                      # 104 tests (unit + integration)
├── CATATAN_PROYEK.md           # Project phase documentation
├── credentials.json            # OAuth client (gitignored)
├── token.json                  # OAuth token (gitignored)
├── service-account.json        # Service account (gitignored)
├── faiss_index/                # Vector index (gitignored)
└── assets/app_icon.png
```

---

## 15. Key Data Flows

### 15.1 User Question → Answer
```
User Question
    │
    ▼
get_device_uid() ──────► UID (cookie/localStorage/URL/temp)
    │
    ▼
get_runtime_config() ──► daily limits, maintenance, guest mode
    │
    ▼
question_is_allowed() ─► check blocked + daily limit
    │
    ▼
load_chat_history() ───► last 2 chunks (30 messages)
    │
    ▼
answer_question() ─────► FAISS similarity search
    │                         │
    │                         ▼
    │                    _QuotaAwareEmbeddings
    │                         │
    │                         ▼
    │                    Gemini embedding
    │                         │
    │                         ▼
    │                    FAISS index search
    │                         │
    │                         ▼
    │                    Retrieve top-K chunks
    │                         │
    │                         ▼
    │                    Build context + system prompt
    │                         │
    │                         ▼
    │                    Hedged model requests (router.rank)
    │                         │
    │                         ▼
    │                    Stream answer + continuation
    │                         │
    │                         ▼
    │                    restore_source_indentation()
    │                         │
    ▼                         ▼
enqueue_question_answer() ◄── Answer + sources
    │
    ▼
Firestore async write (queue)
```

### 15.2 Admin Login Flow
```
Sidebar "Admin" click
    │
    ▼
register_admin_button_click() (3 clicks → open prompt)
    │
    ▼
Check admin_device_verified (cookie jh_adm)
    │
    ├─ Yes & !admin_exited_manually → admin_authenticated=True (INSTANT)
    │
    └─ No → Prompt kode admin
              │
              ▼
         process_admin_login()
              │
              ▼
         verify_admin_code() (pbkdf2/scrypt)
              │
              ▼
         create_admin_session_token(uid, epoch)
              │
              ▼
         build_admin_session_store_script() → JS set cookie+localStorage
              │
              ▼
         admin_device_verified=True, admin_authenticated=True
```

### 15.3 Drive Sync Flow
```
Admin: "Mulai sinkronisasi"
    │
    ▼
DriveSyncManager.start_sync()
    │
    ▼
ThreadPoolExecutor.submit(_run_and_publish)
    │
    ▼
_sync():
    ├─ list_drive_files() → inventory
    ├─ Load manifest → detect changed/new/removed
    ├─ For each batch:
    │     ├─ read_drive_files() → download + extract
    │     ├─ update_search_index() → embed changed files
    │     ├─ Checkpoint: save FAISS + manifest
    │     └─ Update file_statuses
    └─ Final save
    │
    ▼
Dashboard: real-time status via snapshot()
```

---

*Dokumen ini di-generate dari analisis kode sumber lengkap. Update saat arsitektur berubah signifikan.*