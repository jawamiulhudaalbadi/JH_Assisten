"""Admin-only dashboard. Import this module only after successful login."""

from __future__ import annotations

from datetime import datetime
import re
import time
from typing import Any, Callable
from zoneinfo import ZoneInfo

import streamlit as st

from services.firestore_store import (
    export_admin_users_csv,
    firestore_status,
    flush_recent_errors,
    get_admin_chat_history,
    get_admin_faq,
    get_admin_recent_errors,
    get_admin_statistics,
    get_admin_user_chats,
    get_admin_users,
    get_runtime_config,
    set_user_blocked,
    update_runtime_config,
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


def _render_faq(client: Any | None) -> None:
    st.subheader("Pertanyaan sering")
    faq = _cached("faq", lambda: get_admin_faq(client))
    if faq:
        st.dataframe(
            [{"Pertanyaan": row["question"], "Jumlah": row["count"]} for row in faq],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Belum ada pertanyaan yang terkumpul.")
    st.button(
        "Rangkum dengan AI",
        disabled=True,
        help="Fitur ini tersedia setelah kelompok key admin pada Fase 3.",
        key="admin_ai_summary_disabled",
    )


def _render_drive_sync(
    manager: Any,
    folder_id: str,
    api_key: str,
    drive_authorized: bool,
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
            if not snapshot["running"] and folder_id and api_key:
                manager.start_sync(folder_id, api_key)
            st.rerun()
    elif st.button(
        "Mulai sinkronisasi",
        key="admin_drive_start",
        disabled=not (folder_id and api_key and drive_authorized),
    ):
        manager.start_sync(folder_id, api_key)
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
                    manager.start_sync(folder_id, api_key)
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
        update_runtime_config(
            client,
            maintenance=maintenance,
            announcement=announcement,
            daily_question_limit=int(daily_limit),
        )
        st.success("Kontrol disimpan dan berlaku di proses ini.")

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
            set_user_blocked(client, selectable_users[selected], blocked)
            st.success("Perubahan blokir dijadwalkan.")

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
    connected = client is not None and status["connected"]
    st.metric("Firestore", "Terkonfigurasi / siap" if connected else "Tidak tersedia")
    if not status["connected"]:
        st.caption(f"Sirkuit terbuka selama sekitar {status['circuit_open_seconds']} detik.")
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


def render_dashboard(
    client: Any | None,
    manager: Any,
    folder_id: str,
    api_key: str,
    drive_authorized: bool,
) -> None:
    st.title("Dashboard Admin")
    if st.button("Keluar", key="admin_dashboard_exit"):
        st.session_state["admin_authenticated"] = False
        st.rerun()

    section = st.radio(
        "Menu dashboard",
        (
            "Statistik",
            "Riwayat chat",
            "Pertanyaan sering",
            "Sinkron Drive",
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
        _render_faq(client)
    elif section == "Sinkron Drive":
        _render_drive_sync(manager, folder_id, api_key, drive_authorized)
    elif section == "Kontrol lain":
        users = _cached("users", lambda: get_admin_users(client))
        _render_controls(client, users)
    else:
        _render_health(client)
