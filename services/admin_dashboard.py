"""Admin-only dashboard. Import this module only after successful login."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import re
import time
from collections.abc import Sequence
from typing import Any, Callable
from zoneinfo import ZoneInfo

import streamlit as st

from services.admin_logic import hash_settings_password
from services.ai_service import get_available_chat_models, summarize_faq_questions
from services.firestore_store import (
    export_admin_users_csv,
    firestore_status,
    firestore_write_health,
    flush_recent_errors,
    get_admin_chat_history,
    get_admin_faq,
    get_admin_recent_errors,
    get_admin_statistics,
    get_admin_user_chats,
    get_admin_users,
    delete_empty_users_and_recount,
    get_empty_user_uids,
    get_runtime_config,
    save_settings_password_hash,
    set_user_blocked,
    update_runtime_config,
)
from services.key_pool import (
    KEY_GROUPS,
    KeyPoolError,
    add_key,
    get_available_keys,
    get_key_status,
    get_keys,
    get_model_order,
    mask_key,
    ordered_models,
    record_quota,
    remove_key,
    save_model_order,
    set_key_enabled,
)
from services.model_router import router

_WIB = ZoneInfo("Asia/Jakarta")
_USER_ID_RE = re.compile(r"^[a-f0-9]{16,32}$")


def _cached(key: str, loader: Callable[[], Any], ttl: float = 30) -> Any:
    state_key = f"_admin_cache_{key}"
    cached = st.session_state.get(state_key)
    if cached and time.monotonic() - cached["at"] < ttl:
        return cached["value"]
    value = loader()
    st.session_state[state_key] = {"at": time.monotonic(), "value": value}
    return value


def _render_statistics(client: Any | None) -> None:
    st.subheader("Statistik")
    statistics = _cached("statistics", lambda: get_admin_statistics(client))
    columns = st.columns(4)
    columns[0].metric("Total perangkat", statistics["total_devices"])
    columns[1].metric("Aktif sekitar 1 jam", statistics["active_last_hour"])
    columns[2].metric("Unik hari ini", statistics["unique_today"])
    columns[3].metric("Unik 7 hari", statistics["unique_7_days"])
    st.caption(statistics["active_window_label"])
    daily = statistics["daily_questions"]
    if daily:
        st.dataframe(
            [
                {
                    "Tanggal (WIB)": datetime.strptime(row["day"], "%Y%m%d")
                    .replace(tzinfo=_WIB)
                    .strftime("%d-%m-%Y"),
                    "Pertanyaan": row["questions"],
                }
                for row in daily
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Ringkasan pertanyaan 14 hari belum tersedia.")


def _render_history(client: Any | None) -> None:
    st.subheader("Riwayat chat pengguna")
    users = _cached("users", lambda: get_admin_users(client))
    if not users:
        st.info("Daftar pengguna belum tersedia.")
        return

    page_key = "_admin_user_page"
    page_count = max(1, (len(users) + 9) // 10)
    page = min(int(st.session_state.get(page_key, 0)), page_count - 1)
    start = page * 10
    columns = st.columns([1, 1, 4])
    if columns[0].button("Sebelumnya", disabled=page == 0, key="admin_users_prev"):
        st.session_state[page_key] = max(0, page - 1)
        st.rerun()
    if columns[1].button(
        "Berikutnya",
        disabled=page >= page_count - 1,
        key="admin_users_next",
    ):
        st.session_state[page_key] = min(page_count - 1, page + 1)
        st.rerun()
    columns[2].caption(f"Halaman {page + 1}/{page_count} · {len(users)} perangkat")
    st.dataframe(
        [
            {
                "ID pendek": user["short_id"],
                "Pertanyaan": user["total_questions"],
                "Pertanyaan hari ini": user["day_count"],
                "Diblokir": user["blocked"],
            }
            for user in users[start : start + 10]
        ],
        use_container_width=True,
        hide_index=True,
    )

    labels = {
        f"{user['short_id']} · {user['total_questions']} pertanyaan": user["uid"]
        for user in users
    }
    selected_label = st.selectbox("Buka satu pengguna", list(labels))
    uid = labels[selected_label]
    chats = _cached(f"chats_{uid}", lambda: get_admin_user_chats(client, uid))
    chat_options = {"Semua chat": None}
    chat_options.update(
        {
            f"{chat['short_id']}": chat["chat_id"]
            for chat in chats
        }
    )
    selected_chat = st.selectbox("Chat", list(chat_options))
    keyword = st.text_input("Cari kata kunci", key="admin_history_keyword")
    if st.button("Cari riwayat", key="admin_search_history"):
        chat_id = chat_options[selected_chat]
        result = get_admin_chat_history(client, uid, chat_id, keyword)
        st.session_state["_admin_history_result"] = result
        st.session_state["_admin_history_query"] = (uid, chat_id, keyword)
    result = st.session_state.get("_admin_history_result")
    if result is not None:
        if not result:
            st.info("Tidak ada pesan yang cocok.")
        for message in result:
            timestamp = message["timestamp"]
            st.markdown(
                f"**{message['role']} · {timestamp}**  \n{message['content']}"
            )


def _read_faq_summary(client: Any | None, fingerprint: str) -> str:
    if client is None:
        return ""
    try:
        data = (
            client.collection("admin_cache")
            .document("faq_summary")
            .get(timeout=10.0, retry=None)
            .to_dict()
            or {}
        )
    except Exception as error:
        st.warning(
            "Ringkasan FAQ gagal dibaca "
            f"({type(error).__name__}): {str(error)[:200]}"
        )
        return ""
    if data.get("fingerprint") == fingerprint:
        return str(data.get("summary", ""))
    return ""


def _write_faq_summary(
    client: Any | None,
    fingerprint: str,
    summary: str,
) -> bool:
    if client is None:
        return False
    try:
        client.collection("admin_cache").document("faq_summary").set(
            {
                "fingerprint": fingerprint,
                "summary": summary,
                "updated_at": datetime.now(_WIB),
            },
            timeout=10.0,
        )
    except Exception as error:
        st.warning(
            "Ringkasan FAQ gagal disimpan "
            f"({type(error).__name__}): {str(error)[:200]}"
        )
        return False
    return True


def _render_faq(
    client: Any | None,
    secrets: dict[str, Any],
    admin_keys: Sequence[str],
    model_names: Sequence[str],
) -> None:
    st.subheader("Pertanyaan sering")
    summary_error = st.session_state.pop("_admin_faq_summary_error", None)
    if summary_error:
        st.warning(summary_error)
    faq = _cached("faq", lambda: get_admin_faq(client))
    if faq:
        st.dataframe(
            [{"Pertanyaan": row["question"], "Jumlah": row["count"]} for row in faq],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Belum ada pertanyaan yang terkumpul.")
    fingerprint = hashlib.sha256(
        json.dumps(faq, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cached_summary = _cached(
        f"faq_summary_{fingerprint}",
        lambda: _read_faq_summary(client, fingerprint),
    )
    if cached_summary:
        st.markdown("**Ringkasan tersimpan**")
        st.write(cached_summary)
    if not admin_keys:
        st.caption("Kelompok key admin kosong atanapi sadayana nuju kena limit.")
    if st.button(
        "Rangkum dengan AI",
        disabled=not admin_keys or not faq,
        key="admin_ai_summary",
    ):
        try:
            summary = summarize_faq_questions(
                faq,
                admin_keys,
                model_names=model_names,
                on_key_quota=lambda key, error: record_quota(
                    "admin",
                    client,
                    secrets,
                    key,
                    error,
                ),
            )
        except ValueError as error:
            if not get_available_keys("admin", client, secrets):
                st.session_state["_admin_faq_summary_error"] = str(error)
                st.rerun()
            st.warning(str(error))
        else:
            st.markdown("**Ringkasan**")
            st.write(summary)
            if not _write_faq_summary(client, fingerprint, summary):
                st.caption("Ringkesan ngan disimpen dina sesi ieu; Firestore teu sayogi.")
            st.session_state[f"_admin_cache_faq_summary_{fingerprint}"] = {
                "at": time.monotonic(),
                "value": summary,
            }


def _render_api_keys(
    client: Any | None,
    secrets: dict[str, Any],
    chat_models: Sequence[str],
) -> None:
    st.subheader("Kelompok API key")
    group_labels = {
        "chat": "Chat",
        "indexing": "Indexing",
        "admin": "Admin",
    }
    tabs = st.tabs([group_labels[group] for group in KEY_GROUPS])
    for tab, group in zip(tabs, KEY_GROUPS):
        with tab:
            try:
                records = get_keys(group, client, secrets, admin=True)
            except KeyPoolError as error:
                st.error(str(error))
                continue
            for record in records:
                columns = st.columns([2, 1, 2, 1, 1])
                columns[0].write(record.label)
                columns[1].code(mask_key(record.key))
                columns[2].write(
                    get_key_status(
                        record,
                        chat_models if group == "chat" else (),
                    )
                )
                columns[3].write(record.last_error or "—")
                button_label = "Nonaktifkan" if record.enabled else "Aktifkan"
                if columns[4].button(
                    button_label,
                    key=f"key_toggle_{group}_{record.sid}",
                ):
                    try:
                        set_key_enabled(
                            group,
                            client,
                            secrets,
                            record.sid,
                            not record.enabled,
                        )
                    except KeyPoolError as error:
                        st.error(str(error))
                    else:
                        st.rerun()
                if st.button(
                    "Hapus",
                    key=f"key_remove_{group}_{record.sid}",
                ):
                    try:
                        remove_key(group, client, secrets, record.sid)
                    except KeyPoolError as error:
                        st.error(str(error))
                    else:
                        st.rerun()
            with st.form(f"add_key_form_{group}", clear_on_submit=True):
                label = st.text_input("Label", key=f"key_label_{group}")
                secret_key = st.text_input(
                    "API key anyar",
                    type="password",
                    key=f"key_value_{group}",
                )
                add = st.form_submit_button("Tambah API key")
            if add:
                try:
                    add_key(group, client, secrets, secret_key, label)
                except KeyPoolError as error:
                    st.error(str(error))
                else:
                    st.success("API key berhasil ditambahkan sacara terenkripsi.")
                    st.rerun()


def _render_model_order(
    client: Any | None,
    secrets: dict[str, Any],
    chat_keys: Sequence[str],
) -> None:
    st.subheader("Urutan model chat")
    try:
        settings = get_model_order(client, secrets, admin=True)
    except KeyPoolError as error:
        st.error(str(error))
        return
    available = (
        _cached(
            "available_chat_models",
            lambda: get_available_chat_models(chat_keys[0]),
            ttl=600,
        )
        if chat_keys
        else []
    )
    mode = st.radio(
        "Mode",
        ("Otomatis", "Manual"),
        index=0 if settings["mode"] == "automatic" else 1,
        horizontal=True,
        key="admin_model_order_mode",
    )
    selected_models = st.multiselect(
        "Daftar berurutan",
        options=available,
        default=[
            model for model in settings["models"] if model in available
        ],
        disabled=mode != "Manual",
        key="admin_model_order_models",
    )
    if st.button("Simpan urutan model", key="admin_save_model_order"):
        try:
            save_model_order(
                client,
                "manual" if mode == "Manual" else "automatic",
                selected_models,
            )
        except KeyPoolError as error:
            st.error(str(error))
        else:
            st.success("Urutan model berhasil disimpan.")
            st.rerun()


def _render_drive_sync(
    manager: Any,
    folder_id: str,
    api_keys: Sequence[str],
    drive_authorized: bool,
    client: Any | None,
    secrets: dict[str, Any],
) -> None:
    st.subheader("Sinkron Drive")
    snapshot = manager.snapshot()
    st.write(f"Status: {snapshot['status']}")
    st.caption(
        f"Fase: {snapshot['phase']} · "
        f"{snapshot['downloaded_count']}/{snapshot['download_total']} file diunduh"
    )
    if snapshot["running"]:
        label = "Lanjutkan" if snapshot["paused"] else "Jeda"
        if st.button(label, key="admin_drive_pause"):
            manager.resume() if snapshot["paused"] else manager.pause()
            st.rerun()
    elif snapshot["paused"]:
        if st.button("Lanjutkan", key="admin_drive_resume"):
            manager.resume()
            if not snapshot["running"] and folder_id and api_keys:
                manager.start_sync(
                    folder_id,
                    api_keys[0],
                    key_candidates=api_keys[1:],
                    on_key_quota=lambda key, error: record_quota(
                        "indexing", client, secrets, key, error, daily=True
                    ),
                )
            st.rerun()
    elif st.button(
        "Mulai sinkronisasi",
        key="admin_drive_start",
        disabled=not (folder_id and api_keys and drive_authorized),
    ):
        manager.start_sync(
            folder_id,
            api_keys[0],
            key_candidates=api_keys[1:],
            on_key_quota=lambda key, error: record_quota(
                "indexing", client, secrets, key, error, daily=True
            ),
        )
        st.rerun()

    failed_files = [item for item in snapshot["files"] if item["state"] == "failed"]
    if failed_files:
        st.markdown("**File gagal**")
        for item in failed_files:
            left, right = st.columns([4, 1])
            left.write(item["name"])
            left.caption(item.get("error", ""))
            if right.button("Coba ulang", key=f"admin_retry_{item['id']}"):
                if manager.retry_file(item["id"]):
                    if api_keys:
                        manager.start_sync(
                            folder_id,
                            api_keys[0],
                            key_candidates=api_keys[1:],
                            on_key_quota=lambda key, error: record_quota(
                                "indexing", client, secrets, key, error, daily=True
                            ),
                        )
                st.rerun()


def _render_controls(client: Any | None, users: list[dict[str, Any]]) -> None:
    st.subheader("Kontrol lain")
    runtime = get_runtime_config(client)
    with st.form("admin_runtime_settings"):
        maintenance = st.toggle(
            "Mode perawatan",
            value=runtime["maintenance"],
        )
        announcement = st.text_area(
            "Pengumuman untuk pengguna",
            value=runtime["announcement"],
            max_chars=1000,
        )
        daily_limit = st.number_input(
            "Batas pertanyaan per hari per pengguna (0 = tanpa batas)",
            min_value=0,
            step=1,
            value=int(runtime["daily_question_limit"]),
        )
        save_runtime = st.form_submit_button("Simpan kontrol")
    if save_runtime:
        try:
            update_runtime_config(
                client,
                maintenance=maintenance,
                announcement=announcement,
                daily_question_limit=int(daily_limit),
            )
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.success("Kontrol disimpan dan berlaku di proses ini.")

    if st.button("Bersihkan pengguna kosong", key="admin_preview_empty_users"):
        try:
            st.session_state["_empty_user_cleanup_candidates"] = (
                get_empty_user_uids(client)
            )
        except RuntimeError as error:
            st.error(f"Tidak dapat memeriksa pengguna kosong: {error}")
        else:
            st.session_state["_empty_user_cleanup_confirmation"] = True

    if st.session_state.get("_empty_user_cleanup_confirmation"):
        candidates = st.session_state.get("_empty_user_cleanup_candidates", [])
        st.warning(
            f"{len(candidates)} pengguna dengan 0 pertanyaan dan tanpa chunk "
            "akan dihapus. Pengguna yang memiliki riwayat tidak disentuh."
        )
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Konfirmasi hapus pengguna kosong",
            key="admin_confirm_empty_users",
            type="primary",
        ):
            try:
                deleted_count, remaining_count = delete_empty_users_and_recount(
                    client, candidates
                )
            except RuntimeError as error:
                st.error(f"Pembersihan pengguna kosong gagal: {error}")
            else:
                st.session_state.pop("_empty_user_cleanup_candidates", None)
                st.session_state.pop("_empty_user_cleanup_confirmation", None)
                st.session_state.pop("_admin_cache_users", None)
                st.session_state.pop("_admin_cache_statistics", None)
                st.success(
                    f"{deleted_count} pengguna dihapus; total perangkat "
                    f"dihitung ulang menjadi {remaining_count}."
                )
        if cancel_column.button(
            "Batal",
            key="admin_cancel_empty_users",
        ):
            st.session_state.pop("_empty_user_cleanup_candidates", None)
            st.session_state.pop("_empty_user_cleanup_confirmation", None)

    selectable_users = {
        user["short_id"]: user["uid"]
        for user in users
        if _USER_ID_RE.fullmatch(user["uid"])
    }
    if selectable_users:
        with st.form("admin_block_user"):
            selected = st.selectbox("ID pengguna", list(selectable_users))
            blocked = st.checkbox("Blokir perangkat ini")
            save_block = st.form_submit_button("Simpan blokir")
        if save_block:
            try:
                set_user_blocked(client, selectable_users[selected], blocked)
            except RuntimeError as error:
                st.error(str(error))
            else:
                st.success("Perubahan blokir berhasil disimpan.")

    if st.button("Bersihkan cache aplikasi", key="admin_clear_cache"):
        st.cache_data.clear()
        st.session_state.pop("_admin_cache_users", None)
        st.session_state.pop("_admin_cache_statistics", None)
        st.session_state.pop("_admin_cache_faq", None)
        st.success("Cache aplikasi dibersihkan.")
    st.download_button(
        "Ekspor pengguna ke CSV",
        data=export_admin_users_csv(users),
        file_name="pengguna_jh_asisten.csv",
        mime="text/csv",
    )


def _render_health(client: Any | None) -> None:
    st.subheader("Kesehatan sistem")
    status = firestore_status()
    write_health = firestore_write_health()
    st.metric(
        "Firestore",
        "Terkonfigurasi" if client is not None else "Tidak tersedia",
    )
    st.metric(
        "Circuit breaker chat",
        "Terbuka" if status["circuit_open"] else "Tertutup",
    )
    if status["circuit_open"]:
        st.caption(f"Sirkuit terbuka selama sekitar {status['circuit_open_seconds']} detik.")
    metric_columns = st.columns(4)
    for column, label, key in zip(
        metric_columns,
        ("Tulis berhasil", "Tulis gagal", "Percobaan ulang", "Tulis dibuang"),
        ("writes_ok", "writes_failed", "writes_retried", "writes_dropped"),
    ):
        column.metric(label, write_health[key])
    st.caption(f"Antrean tulis: {write_health['writes_queued']} dari 200")
    if write_health["last_error"]:
        st.caption(f"Kesalahan tulis terakhir: {write_health['last_error']}")
    st.caption(
        "Baca konfigurasi runtime dilewati karena timeout anggaran: "
        f"{write_health['runtime_config_skipped_timeouts']}"
    )
    if write_health["breaker_open_until"]:
        st.caption(f"Sirkuit dibuka hingga: {write_health['breaker_open_until']}")
    errors_by_kind = write_health["last_error_by_kind"]
    if errors_by_kind:
        st.markdown("**Kesalahan tulis terakhir per jenis**")
        st.dataframe(
            [
                {"Jenis": kind, "Kesalahan": message}
                for kind, message in sorted(errors_by_kind.items())
            ],
            use_container_width=True,
            hide_index=True,
        )
    st.markdown("**Snapshot model**")
    model_snapshot = router.snapshot()
    st.dataframe(model_snapshot, use_container_width=True, hide_index=True)
    flush_recent_errors(client)
    errors = _cached("recent_errors", lambda: get_admin_recent_errors(client), ttl=15)
    st.markdown("**20 error terakhir**")
    if errors:
        st.dataframe(errors[-20:], use_container_width=True, hide_index=True)
    else:
        st.caption("Belum ada error tercatat.")


def _render_settings_password(client: Any | None) -> None:
    st.subheader("Sandi menu Pengaturan")
    st.caption(
        "Sandi ieu dipénta ka pamaké saméméh muka menu Pengaturan. "
        "Admin cukup ngetik sandi anyar; aplikasi nyimpen hash-na."
    )
    with st.form("admin_settings_password", clear_on_submit=True):
        password = st.text_input(
            "Sandi Pengaturan anyar",
            type="password",
        )
        confirmation = st.text_input(
            "Ulangi sandi Pengaturan",
            type="password",
        )
        save_password = st.form_submit_button("Simpan sandi")
    if not save_password:
        return
    if not password:
        st.error("Sandi Pengaturan tidak boleh kosong.")
    elif password != confirmation:
        st.error("Konfirmasi sandi tidak sama.")
    else:
        try:
            save_settings_password_hash(client, hash_settings_password(password))
        except (RuntimeError, ValueError) as error:
            st.error(f"Sandi Pengaturan tidak tersimpan: {error}")
        else:
            st.success("Sandi Pengaturan berhasil diperbarui.")


def render_dashboard(
    client: Any | None,
    manager: Any,
    folder_id: str,
    index_keys: Sequence[str],
    drive_authorized: bool,
    secrets: dict[str, Any],
) -> None:
    st.title("Dashboard Admin")
    if st.button("Keluar", key="admin_dashboard_exit"):
        st.session_state["admin_authenticated"] = False
        st.rerun()
    chat_keys = [
        item.key for item in get_available_keys("chat", client, secrets)
    ]
    admin_keys = [
        item.key for item in get_available_keys("admin", client, secrets)
    ]
    try:
        model_settings = get_model_order(client, secrets, admin=True)
    except KeyPoolError as error:
        st.error(str(error))
        model_settings = {"mode": "automatic", "models": [], "secrets_models": []}
    available_models = (
        get_available_chat_models(chat_keys[0]) if chat_keys else []
    )
    model_names = ordered_models(available_models, model_settings)

    section = st.radio(
        "Menu dashboard",
        (
            "Statistik",
            "Riwayat chat",
            "Pertanyaan sering",
            "Sinkron Drive",
            "Sandi Pengaturan",
            "API Key",
            "Urutan model",
            "Kontrol lain",
            "Kesehatan sistem",
        ),
        horizontal=True,
        label_visibility="collapsed",
    )
    if section == "Statistik":
        _render_statistics(client)
    elif section == "Riwayat chat":
        _render_history(client)
    elif section == "Pertanyaan sering":
        _render_faq(client, secrets, admin_keys, model_names)
    elif section == "Sinkron Drive":
        _render_drive_sync(
            manager,
            folder_id,
            index_keys,
            drive_authorized,
            client,
            secrets,
        )
    elif section == "Sandi Pengaturan":
        _render_settings_password(client)
    elif section == "API Key":
        _render_api_keys(client, secrets, model_names)
    elif section == "Urutan model":
        _render_model_order(client, secrets, chat_keys)
    elif section == "Kontrol lain":
        users = _cached("users", lambda: get_admin_users(client))
        _render_controls(client, users)
    else:
        _render_health(client)
