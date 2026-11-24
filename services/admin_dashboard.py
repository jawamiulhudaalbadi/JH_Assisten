"""Admin-only dashboard. Import this module only after successful login."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import time
from collections.abc import Sequence
from typing import Any, Callable
from zoneinfo import ZoneInfo

import streamlit as st
from streamlit_javascript import st_javascript

from services.admin_logic import hash_settings_password
from services.ai_service import (
    evaluate_answer_samples,
    get_available_chat_models,
    summarize_faq_questions,
)
from services.admin_session import (
    build_admin_session_clear_script,
    revoke_all_admin_sessions,
)

from services.firestore_store import (
    export_admin_users_csv,
    firestore_status,
    firestore_write_health,
    flush_recent_errors,
    get_admin_user_page,
    get_admin_chat_history,
    get_admin_faq,
    get_answer_quality_page,
    get_answer_quality_summary,
    get_answer_reviews,
    get_admin_recent_errors,
    get_admin_statistics,
    get_admin_user_chats,
    get_admin_users,
    delete_users_without_name_and_recount,
    get_users_without_name,
    get_runtime_config,
    save_settings_password_hash,
    save_answer_reviews,
    update_runtime_config,
    update_admin_user_policy,
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
from services.perf_monitor import (
    record_admin_menu_run,
    recent_admin_menu_runs,
    recent_runs,
)

_WIB = ZoneInfo("Asia/Jakarta")


def _cached(key: str, loader: Callable[[], Any], ttl: float = 60) -> Any:
    state_key = f"_admin_cache_{key}"
    cached = st.session_state.get(state_key)
    if cached and time.monotonic() - cached["at"] < ttl:
        return cached["value"]
    value = loader()
    st.session_state[state_key] = {"at": time.monotonic(), "value": value}
    return value


@st.cache_data(ttl=60, show_spinner=False)
def _cached_admin_statistics(client_key: str, _client: Any) -> dict[str, Any]:
    return get_admin_statistics(_client)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_quality_summary(client_key: str, _client: Any) -> dict[str, Any]:
    return get_answer_quality_summary(_client)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_quality_page(
    client_key: str,
    filter_type: str,
    filter_value: str,
    cursor: Any | None,
    _client: Any,
) -> dict[str, Any]:
    return get_answer_quality_page(
        _client,
        filter_type=filter_type,
        filter_value=filter_value,
        start_after=cursor,
        page_size=25,
    )


@st.cache_data(ttl=60, show_spinner=False)
def _cached_user_page(
    client_key: str,
    name_query: str,
    cursor: tuple[str, str] | None,
    _client: Any,
) -> dict[str, Any]:
    return get_admin_user_page(
        _client,
        name_query=name_query,
        start_after=cursor,
        page_size=25,
    )


def _render_statistics(client: Any | None) -> None:
    st.subheader("Statistik")
    if st.button("Segarkan", key="admin_statistics_refresh"):
        _cached_admin_statistics.clear()
    client_key = str(id(client)) if client is not None else "unavailable"
    statistics = _cached_admin_statistics(client_key, client)
    columns = st.columns(4)
    columns[0].metric("Total perangkat (terdaftar)", statistics["total_devices"])
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
                    "Pertanyaan pengguna": row["questions"],
                    "Pertanyaan tamu": row.get("guest_questions", 0),
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
    if not _admin_firestore_ready(client):
        return
    client_key = str(id(client)) if client is not None else "unavailable"
    if st.button("Segarkan", key="admin_history_refresh"):
        _cached_user_page.clear()
        st.session_state.pop("_admin_history_page_query", None)
        for key in list(st.session_state):
            if key.startswith(("_admin_cache_chats_", "_admin_cache_history_")):
                st.session_state.pop(key, None)
    name_query = st.text_input(
        "Cari pangguna dumasar nami",
        key="admin_history_name_search",
    ).casefold().strip()
    if st.session_state.get("_admin_history_page_query") != name_query:
        st.session_state["_admin_history_page_query"] = name_query
        st.session_state["_admin_history_page_cursors"] = [None]
        st.session_state["_admin_history_page_number"] = 0
    cursors = st.session_state.get("_admin_history_page_cursors", [None])
    page_number = int(st.session_state.get("_admin_history_page_number", 0))
    cursor = cursors[page_number] if page_number < len(cursors) else None
    user_page = _cached_user_page(client_key, name_query, cursor, client)
    users = user_page["items"]
    if not users:
        st.info("Daftar pengguna belum tersedia.")
        return

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    page_column.caption(f"Halaman {page_number + 1}")
    if previous_column.button(
        "Sateuacanna",
        disabled=page_number == 0,
        key="admin_history_previous_users",
    ):
        st.session_state["_admin_history_page_number"] = page_number - 1
        st.rerun(scope="fragment")
    next_cursor = user_page.get("next_cursor")
    if next_column.button(
        "Salajengna",
        disabled=next_cursor is None,
        key="admin_history_next_users",
    ):
        cursors = list(cursors)
        if len(cursors) == page_number + 1:
            cursors.append(next_cursor)
        st.session_state["_admin_history_page_cursors"] = cursors
        st.session_state["_admin_history_page_number"] = page_number + 1
        st.rerun(scope="fragment")
    st.dataframe(
        [
            {
                "Nama": user["name"] or "Tanpa nama",
                "Nama ganda": user["duplicate_name"],
                "ID pendek": user["short_id"],
                "Pertanyaan": user["total_questions"],
                "Diblokir": user["blocked"],
            }
            for user in users
        ],
        use_container_width=True,
        hide_index=True,
    )

    labels = {
        f"{user['name'] or 'Tanpa nama'} · {user['short_id']}": user
        for user in users
    }
    selected_label = st.selectbox(
        "Buka hiji pangguna",
        list(labels),
        key="admin_history_selected_user",
    )
    selected_user = labels[selected_label]
    uid = selected_user["uid"]
    st.caption(
        f"Riwayat milik {selected_user['name'] or 'Tanpa nama'} · "
        f"ID {selected_user['short_id']}"
    )
    chats = _cached(f"chats_{uid}", lambda: get_admin_user_chats(client, uid))
    if not chats:
        st.info("Pamaké ieu teu acan gaduh paguneman.")
        return
    chat_options = {
        f"Paguneman · {chat['short_id']}": chat["chat_id"]
        for chat in chats
    }
    selected_chat = st.selectbox(
        "Paguneman",
        list(chat_options),
        key=f"admin_history_chat_{uid}",
    )
    keyword = st.text_input("Cari kata kunci", key="admin_history_keyword")
    if st.button("Cari riwayat", key="admin_search_history"):
        chat_id = chat_options[selected_chat]
        result = _cached(
            f"history_{uid}_{chat_id}_{keyword.casefold().strip()}",
            lambda: get_admin_chat_history(client, uid, chat_id, keyword),
            ttl=60,
        )
        st.session_state["_admin_history_result"] = result
        st.session_state["_admin_history_query"] = (
            uid,
            chat_id,
            keyword.casefold().strip(),
        )
        st.session_state["_admin_history_message_page"] = 0
    result = st.session_state.get("_admin_history_result")
    query = (uid, chat_options[selected_chat], keyword.casefold().strip())
    if st.session_state.get("_admin_history_query") != query:
        result = None
    if result is not None:
        if not result:
            st.info("Tidak ada pesan yang cocok.")
        message_page = int(st.session_state.get("_admin_history_message_page", 0))
        page_count = max(1, (len(result) + 24) // 25)
        message_page = min(message_page, page_count - 1)
        message_columns = st.columns([1, 2, 1])
        message_columns[1].caption(
            f"Halaman pesen {message_page + 1}/{page_count} · "
            f"{len(result)} pesen dimuat"
        )
        if message_columns[0].button(
            "Pesen saméméhna",
            disabled=message_page == 0,
            key="admin_history_previous_messages",
        ):
            st.session_state["_admin_history_message_page"] = message_page - 1
            st.rerun(scope="fragment")
        if message_columns[2].button(
            "Pesen salajengna",
            disabled=message_page + 1 >= page_count,
            key="admin_history_next_messages",
        ):
            st.session_state["_admin_history_message_page"] = message_page + 1
            st.rerun(scope="fragment")
        for message in result[message_page * 25 : (message_page + 1) * 25]:
            st.caption(
                f"{message['role']} · {_format_admin_timestamp(message['timestamp'])}"
            )
            st.text(message["content"])


def _format_admin_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=_WIB)
        return value.astimezone(_WIB).strftime("%d-%m-%Y %H:%M WIB")
    return str(value or "-")


def _admin_firestore_ready(client: Any | None) -> bool:
    if client is None:
        st.error("Firestore henteu sayogi; data admin teu dimuat.")
        return False
    status = firestore_status()
    if status["circuit_open"]:
        st.warning("Firestore keur teu sayogi; data nu nembongan tiasa kadaluwarsa.")
    return True


def _render_users(client: Any | None) -> None:
    st.subheader("Pengguna")
    if not _admin_firestore_ready(client):
        return
    client_key = str(id(client)) if client is not None else "unavailable"
    refresh_column, delete_column, search_column = st.columns([1, 2, 4])
    if refresh_column.button("Segarkan", key="admin_users_refresh"):
        _cached_user_page.clear()
        st.session_state.pop("_admin_users_page_query", None)
    if delete_column.button("Hapus semua tanpa nama", key="admin_delete_users_without_name"):
        st.session_state["_admin_users_delete_without_name_confirm"] = True
    name_query = search_column.text_input(
        "Cari berdasarkan nama",
        key="admin_user_name_search",
    ).casefold().strip()
    if st.session_state.get("_admin_users_page_query") != name_query:
        st.session_state["_admin_users_page_query"] = name_query
        st.session_state["_admin_users_page_cursors"] = [None]
        st.session_state["_admin_users_page_number"] = 0

    cursors = st.session_state.get("_admin_users_page_cursors", [None])
    page_number = int(st.session_state.get("_admin_users_page_number", 0))
    cursor = cursors[page_number] if page_number < len(cursors) else None
    page = _cached_user_page(client_key, name_query, cursor, client)
    users = page["items"]
    if not users:
        st.info("Teu aya pangguna anu cocog.")
        return
    today_key = datetime.now(_WIB).strftime("%Y%m%d")
    st.dataframe(
        [
            {
                "Nama": user["name"] or "Tanpa nama",
                "ID": user["short_id"],
                "First seen": _format_admin_timestamp(user["first_seen"]),
                "Last active": _format_admin_timestamp(user["last_active"]),
                "Total pertanyaan": user["total_questions"],
                "Patarosan dinten ieu": (
                    user["day_count"] if user["day"] == today_key else 0
                ),
                "Blokir": user["blocked"],
                "Batas harian": user["daily_question_limit"],
                "Tanda": " · ".join(
                    label
                    for key, label in (
                        ("duplicate_name", "Nama ganda"),
                        ("no_questions", "Belum bertanya"),
                        ("question_spike", "Lonjakan ≥20/hari"),
                    )
                    if user[key]
                ),
            }
            for user in users
        ],
        use_container_width=True,
        hide_index=True,
    )

    if st.session_state.pop("_admin_users_delete_without_name_confirm", False):
        try:
            candidates = get_users_without_name(client)
        except RuntimeError as error:
            st.error(f"Tidak dapat memeriksa pengguna tanpa nama: {error}")
        else:
            st.session_state["_admin_users_delete_without_name_candidates"] = candidates
            st.session_state["_admin_users_delete_without_name_confirm"] = True

    if st.session_state.get("_admin_users_delete_without_name_confirm"):
        candidates = st.session_state.get("_admin_users_delete_without_name_candidates", [])
        st.warning(
            f"{len(candidates)} pengguna tanpa nama (kosong/null/whitespace) "
            "akan dihapus beserta riwayat chat mereka. Pengguna yang memiliki nama tidak disentuh."
        )
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Konfirmasi hapus semua pengguna tanpa nama",
            key="admin_confirm_delete_users_without_name",
            type="primary",
        ):
            try:
                deleted_count, remaining_count = delete_users_without_name_and_recount(
                    client, candidates
                )
            except RuntimeError as error:
                st.error(f"Penghapusan pengguna tanpa nama gagal: {error}")
            else:
                st.session_state.pop("_admin_users_delete_without_name_candidates", None)
                st.session_state.pop("_admin_users_delete_without_name_confirm", None)
                st.session_state.pop("_admin_cache_users", None)
                st.session_state.pop("_admin_cache_statistics", None)
                st.success(
                    f"{deleted_count} pengguna tanpa nama dihapus; total perangkat "
                    f"dihitung ulang menjadi {remaining_count}."
                )
                st.rerun(scope="fragment")
        if cancel_column.button(
            "Batal",
            key="admin_cancel_delete_users_without_name",
        ):
            st.session_state.pop("_admin_users_delete_without_name_candidates", None)
            st.session_state.pop("_admin_users_delete_without_name_confirm", None)

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    previous_column.caption(f"Halaman {page_number + 1}")
    if previous_column.button(
        "Sateuacanna",
        key="admin_users_previous_page",
        disabled=page_number == 0,
    ):
        st.session_state["_admin_users_page_number"] = page_number - 1
        st.rerun(scope="fragment")
    next_cursor = page.get("next_cursor")
    if next_column.button(
        "Salajengna",
        key="admin_users_next_page",
        disabled=next_cursor is None,
    ):
        cursors = list(cursors)
        if len(cursors) == page_number + 1:
            cursors.append(next_cursor)
        st.session_state["_admin_users_page_cursors"] = cursors
        st.session_state["_admin_users_page_number"] = page_number + 1
        st.rerun(scope="fragment")

    labels = {
        f"{user['name'] or 'Tanpa nama'} · {user['short_id']}": user
        for user in users
    }
    selected_label = st.selectbox(
        "Atur kawijakan pangguna",
        list(labels),
        key="admin_user_policy_select",
    )
    selected_user = labels[selected_label]
    with st.form(f"admin_user_policy_form_{selected_user['uid']}"):
        blocked = st.checkbox(
            "Blokir pangguna",
            value=selected_user["blocked"],
            key=f"admin_user_blocked_{selected_user['uid']}",
        )
        daily_limit = st.number_input(
            "Batas harian (0 = turutan setelan umum)",
            min_value=0,
            max_value=10_000,
            step=1,
            value=selected_user["daily_question_limit"],
            key=f"admin_user_daily_limit_{selected_user['uid']}",
        )
        update_policy = st.form_submit_button("Simpan kawijakan")
    if update_policy:
        try:
            update_admin_user_policy(
                client,
                selected_user["uid"],
                blocked=blocked,
                daily_question_limit=int(daily_limit),
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            _cached_user_page.clear()
            st.session_state["_admin_user_policy_saved"] = True
            st.rerun(scope="fragment")
    if st.session_state.pop("_admin_user_policy_saved", False):
        st.success("Kawijakan pangguna parantos disimpen.")


def _render_answer_quality(
    client: Any | None,
    secrets: dict[str, Any],
) -> None:
    st.subheader("Kualitas jawaban")
    if not _admin_firestore_ready(client):
        return
    client_key = str(id(client)) if client is not None else "unavailable"
    if st.button("Segarkan", key="admin_quality_refresh"):
        _cached_quality_summary.clear()
        _cached_quality_page.clear()
        st.session_state.pop("_admin_quality_filter_key", None)
    summary = _cached_quality_summary(client_key, client)
    feedback_count = int(summary.get("feedback_count", 0) or 0)
    no_source_count = int(summary.get("without_sources_count", 0) or 0)
    fallback_count = int(summary.get("fallback_count", 0) or 0)
    latency_count = int(summary.get("latency_count", 0) or 0)
    latency_total = int(summary.get("latency_total_ms", 0) or 0)
    columns = st.columns(5)
    columns[0].metric("Cocog", int(summary.get("positive_count", 0) or 0))
    columns[1].metric("Teu cocog", int(summary.get("negative_count", 0) or 0))
    columns[2].metric(
        "Tanpa sumber",
        f"{100 * no_source_count / feedback_count:.1f}%"
        if feedback_count
        else "—",
    )
    columns[3].metric(
        "Fallback tanpa AI",
        f"{100 * fallback_count / feedback_count:.1f}%"
        if feedback_count
        else "—",
    )
    columns[4].metric(
        "Rata-rata latensi",
        f"{latency_total / latency_count:.0f} ms" if latency_count else "—",
    )
    st.caption(
        "Métrik sumber, fallback, jeung laténsi dumasar kana jawaban anu geus dipeunteun."
    )
    model_stats = summary.get("models", {})
    if isinstance(model_stats, dict) and model_stats:
        st.markdown("**Per model**")
        st.dataframe(
            [
                {
                    "Model": value.get("name", key),
                    "Jumlah": int(value.get("feedback_count", 0) or 0),
                    "Rata-rata latensi (ms)": (
                        round(
                            int(value.get("latency_total_ms", 0) or 0)
                            / int(value.get("latency_count", 0) or 1)
                        )
                    ),
                }
                for key, value in model_stats.items()
                if isinstance(value, dict)
            ],
            use_container_width=True,
            hide_index=True,
        )

    try:
        admin_keys = [
            item.key for item in get_available_keys("admin", client, secrets)
        ]
    except KeyPoolError as error:
        admin_keys = []
        st.warning(f"Kelompok key admin teu tiasa dibaca: {error}")
    if not admin_keys:
        st.caption("Sadaya key admin teu sayogi; penilaian AI dinonaktipkeun.")
    if st.button(
        "Nilai sampel dengan AI",
        key="admin_quality_ai_evaluate",
        disabled=not admin_keys,
    ):
        sample_page = _cached_quality_page(
            client_key,
            "Sadayana",
            "",
            None,
            client,
        )
        sample_rows = sample_page["items"][:25]
        existing_reviews = get_answer_reviews(
            client,
            [str(item.get("answer_id", "")) for item in sample_rows],
        )
        samples = [
            {
                **item,
                "sources": item.get("sources", []),
            }
            for item in sample_rows
            if not existing_reviews.get(str(item.get("answer_id", "")), {}).get(
                "ai_label"
            )
        ][:20]
        if not samples:
            st.info("Teu aya sampel anyar anu tiasa dinilai.")
        else:
            try:
                model_names = get_available_chat_models(admin_keys[0])
                evaluations = evaluate_answer_samples(
                    samples,
                    admin_keys,
                    model_names=model_names,
                    on_key_quota=lambda key, error: record_quota(
                        "admin", client, secrets, key, error
                    ),
                )
                save_answer_reviews(
                    client,
                    [
                        {
                            "answer_id": item["answer_id"],
                            "ai_label": item["label"],
                            "ai_rationale": item["rationale"],
                        }
                        for item in evaluations
                    ],
                )
            except (KeyPoolError, RuntimeError, ValueError) as error:
                st.error(f"Penilaian AI gagal: {error}")
            else:
                _cached_quality_page.clear()
                st.success(f"{len(evaluations)} sampel parantos dinilai.")
                st.rerun(scope="fragment")

    filters = (
        "Sadayana",
        "Teu cocog",
        "Tanpa sumber",
        "Fallback tanpa AI",
        "Per model",
        "Per nama",
    )
    filter_type = st.selectbox(
        "Filter kualitas",
        filters,
        key="admin_quality_filter",
    )
    filter_value = ""
    models = [
        str(value.get("name", ""))
        for value in model_stats.values()
        if isinstance(value, dict) and value.get("name")
    ] if isinstance(model_stats, dict) else []
    if filter_type == "Per model" and models:
        filter_value = st.selectbox(
            "Model",
            sorted(set(models)),
            key="admin_quality_model_filter",
        )
    elif filter_type == "Per nama":
        filter_value = st.text_input(
            "Awalan nami",
            key="admin_quality_name_filter",
        ).strip()
    normalized_filter = f"{filter_type}|{filter_value.casefold()}"
    if st.session_state.get("_admin_quality_filter_key") != normalized_filter:
        st.session_state["_admin_quality_filter_key"] = normalized_filter
        st.session_state["_admin_quality_cursors"] = [None]
        st.session_state["_admin_quality_page_number"] = 0
    cursors = st.session_state.get("_admin_quality_cursors", [None])
    page_number = int(st.session_state.get("_admin_quality_page_number", 0))
    cursor = cursors[page_number] if page_number < len(cursors) else None
    page = _cached_quality_page(
        client_key,
        filter_type,
        filter_value,
        cursor,
        client,
    )
    items = page["items"]
    reviews = get_answer_reviews(
        client, [str(item.get("answer_id", "")) for item in items]
    )
    if not items:
        st.info("Can data eupan balik anu cocog sareng filter ieu.")
    for item in items:
        answer_id = str(item.get("answer_id", ""))
        review = reviews.get(answer_id, {})
        display_name = str(item.get("name", "") or "Tanpa nama")
        with st.expander(
            f"{display_name} · {item.get('rating', '')} · "
            f"{_format_admin_timestamp(item.get('created_at'))}"
        ):
            st.caption(f"ID · {str(item.get('uid', ''))[:6]}")
            st.text("Patarosan")
            st.text(str(item.get("question", "")))
            st.text("Jawaban")
            st.text(str(item.get("answer", "")))
            st.caption(
                f"Sumber: {int(item.get('source_count', 0) or 0)} · "
                f"Fallback tanpa AI: {bool(item.get('fallback', False))} · "
                f"Model: {item.get('model_used') or 'teu aya'}"
            )
            for source in item.get("sources", [])[:3]:
                st.text(
                    f"{source.get('name', 'Dokumen')}: "
                    f"{source.get('excerpt', '')}"
                )
            if item.get("reason"):
                st.caption(f"Alesan pamaké: {item['reason']}")
            if review.get("ai_label"):
                st.text(
                    f"Penilaian AI: {review['ai_label']} · "
                    f"{review.get('ai_rationale', '')}"
                )
            current_label = review.get("label", "Sesuai")
            selected_label = st.selectbox(
                "Tandaan admin",
                ("Sesuai", "Kaluar kontéks", "Salah", "Perlu sumber"),
                index=(
                    ("Sesuai", "Kaluar kontéks", "Salah", "Perlu sumber").index(
                        current_label
                    )
                    if current_label in {
                        "Sesuai",
                        "Kaluar kontéks",
                        "Salah",
                        "Perlu sumber",
                    }
                    else 0
                ),
                key=f"admin_review_label_{answer_id}",
            )
            if st.button("Simpen tandaan", key=f"admin_save_review_{answer_id}"):
                try:
                    save_answer_reviews(
                        client,
                        [{"answer_id": answer_id, "label": selected_label}],
                    )
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))
                else:
                    st.success("Tandaan parantos disimpen.")
                    st.rerun(scope="fragment")

    previous_column, page_column, next_column = st.columns([1, 2, 1])
    previous_column.caption(f"Halaman {page_number + 1}")
    if previous_column.button(
        "Sateuacanna",
        key="admin_quality_previous_page",
        disabled=page_number == 0,
    ):
        st.session_state["_admin_quality_page_number"] = page_number - 1
        st.rerun(scope="fragment")
    next_cursor = page.get("next_cursor")
    if next_column.button(
        "Salajengna",
        key="admin_quality_next_page",
        disabled=next_cursor is None,
    ):
        cursors = list(cursors)
        if len(cursors) == page_number + 1:
            cursors.append(next_cursor)
        st.session_state["_admin_quality_cursors"] = cursors
        st.session_state["_admin_quality_page_number"] = page_number + 1
        st.rerun(scope="fragment")


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


def _render_controls(client: Any | None) -> None:
    st.subheader("Kontrol lain")
    runtime = get_runtime_config(client)
    expiry_value = str(runtime["guest_emergency_expires_at"] or "")
    try:
        expiry = datetime.fromisoformat(expiry_value) if expiry_value else None
        if expiry is not None and expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
    except ValueError:
        expiry = datetime.min.replace(tzinfo=timezone.utc)
    if (
        not runtime["guest_emergency_enabled"]
        and expiry is not None
        and expiry <= datetime.now(timezone.utc)
    ):
        st.session_state["admin_guest_emergency_toggle"] = False
    st.toggle(
        "Buka mode tamu tanpa sandi",
        value=bool(runtime["guest_emergency_enabled"]),
        key="admin_guest_emergency_toggle",
        on_change=_save_guest_emergency_toggle,
        args=(client,),
    )
    emergency_notice = st.session_state.pop("_guest_emergency_notice", None)
    if emergency_notice:
        st.error(emergency_notice)
    if runtime["guest_emergency_enabled"]:
        try:
            expires_at = datetime.fromisoformat(
                str(runtime["guest_emergency_expires_at"])
            )
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            remaining = max(
                0, int((expires_at - datetime.now(timezone.utc)).total_seconds())
            )
        except ValueError:
            remaining = 0
        total_minutes = (remaining + 59) // 60
        st.error(
            "Mode tamu tanpa sandi keur dibuka "
            f"(sésana {total_minutes // 60} jam "
            f"{total_minutes % 60} menit)."
        )
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
        guest_daily_limit = st.number_input(
            "Batas pertanyaan harian per sesi tamu",
            min_value=1,
            max_value=10_000,
            step=1,
            value=int(runtime["guest_daily_question_limit"]),
        )
        save_runtime = st.form_submit_button("Simpan kontrol")
    if save_runtime:
        try:
            update_runtime_config(
                client,
                maintenance=maintenance,
                announcement=announcement,
                daily_question_limit=int(daily_limit),
                guest_daily_question_limit=int(guest_daily_limit),
            )
        except (RuntimeError, ValueError) as error:
            st.error(str(error))
        else:
            st.success("Kontrol disimpan dan berlaku di proses ini.")

    if st.button("Cabut semua sesi admin", key="revoke_all_admin_sessions"):
        try:
            revoke_all_admin_sessions(client)
        except RuntimeError as error:
            st.error(str(error))
        else:
            st.session_state["_admin_revoke_clear_pending"] = True
            st.rerun(scope="fragment")

    if st.session_state.get("_admin_revoke_clear_pending"):
        clear_result = st_javascript(
            build_admin_session_clear_script(),
            key="clear_admin_session_after_revoke",
        )
        if clear_result in {"cleared", "cookie-cleared", "error"}:
            if clear_result != "cleared":
                st.session_state["_admin_session_clear_warning"] = (
                    "Token sesi parantos dicabut. Panyimpenan lokal panyungsi "
                    "teu tiasa dipupus sadayana."
                )
            st.session_state["admin_authenticated"] = False
            st.session_state.pop("_admin_session_token", None)
            st.session_state.pop("_admin_auth_method", None)
            st.session_state.pop("_admin_revoke_clear_pending", None)
            st.rerun(scope="app")
        st.caption("Nuju mupus sesi admin dina alat ieu...")


def _save_guest_emergency_toggle(client: Any | None) -> None:
    enabled = bool(st.session_state.get("admin_guest_emergency_toggle", False))
    expires_at = (
        (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
        if enabled
        else ""
    )
    try:
        update_runtime_config(
            client,
            guest_emergency_enabled=enabled,
            guest_emergency_expires_at=expires_at,
        )
    except (RuntimeError, ValueError) as error:
        st.session_state["admin_guest_emergency_toggle"] = not enabled
        st.session_state["_guest_emergency_notice"] = (
            f"Katup darurat teu tiasa dirobah: {error}"
        )

    if st.button("Bersihkan pengguna tanpa nama", key="admin_preview_users_without_name"):
        try:
            st.session_state["_users_without_name_candidates"] = (
                get_users_without_name(client)
            )
        except RuntimeError as error:
            st.error(f"Tidak dapat memeriksa pengguna tanpa nama: {error}")
        else:
            st.session_state["_users_without_name_confirmation"] = True

    if st.session_state.get("_users_without_name_confirmation"):
        candidates = st.session_state.get("_users_without_name_candidates", [])
        st.warning(
            f"{len(candidates)} pengguna tanpa nama (kosong/null/whitespace) "
            "akan dihapus. Pengguna yang memiliki nama tidak disentuh."
        )
        confirm_column, cancel_column = st.columns(2)
        if confirm_column.button(
            "Konfirmasi hapus pengguna tanpa nama",
            key="admin_confirm_users_without_name",
            type="primary",
        ):
            try:
                deleted_count, remaining_count = delete_users_without_name_and_recount(
                    client, candidates
                )
            except RuntimeError as error:
                st.error(f"Pembersihan pengguna tanpa nama gagal: {error}")
            else:
                st.session_state.pop("_users_without_name_candidates", None)
                st.session_state.pop("_users_without_name_confirmation", None)
                st.session_state.pop("_admin_cache_users", None)
                st.session_state.pop("_admin_cache_statistics", None)
                st.success(
                    f"{deleted_count} pengguna tanpa nama dihapus; total perangkat "
                    f"dihitung ulang menjadi {remaining_count}."
                )
        if cancel_column.button(
            "Batal",
            key="admin_cancel_users_without_name",
        ):
            st.session_state.pop("_users_without_name_candidates", None)
            st.session_state.pop("_users_without_name_confirmation", None)

    if st.button("Bersihkan cache aplikasi", key="admin_clear_cache"):
        st.cache_data.clear()
        st.session_state.pop("_admin_cache_users", None)
        st.session_state.pop("_admin_cache_statistics", None)
        st.session_state.pop("_admin_cache_faq", None)
        st.success("Cache aplikasi dibersihkan.")
    if st.button("Siapkan ekspor pengguna", key="admin_prepare_users_export"):
        st.session_state["_admin_users_export"] = _cached(
            "users_export",
            lambda: export_admin_users_csv(get_admin_users(client)),
            ttl=60,
        )
    csv_export = st.session_state.get("_admin_users_export")
    if isinstance(csv_export, str):
        st.download_button(
            "Unduh CSV pengguna",
            data=csv_export,
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
    performance_runs = recent_runs()
    st.markdown("**Performa halaman utama (30 rerun terakhir)**")
    if performance_runs:
        st.dataframe(
            list(reversed(performance_runs)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption(
            "Belum aya pangukuran. Tambahkan ?debug=1 atawa buka aplikasi salaku admin."
        )
    menu_runs = recent_admin_menu_runs()
    st.markdown("**Performa menu dashboard admin (30 interaksi terakhir)**")
    if menu_runs:
        st.dataframe(
            list(reversed(menu_runs)),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Can aya interaksi menu admin anu diukur.")
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


@st.fragment
def _render_dashboard_fragment(
    client: Any | None,
    manager: Any,
    folder_id: str,
    index_keys: Sequence[str],
    drive_authorized: bool,
    secrets: dict[str, Any],
) -> None:
    started_at = time.perf_counter()
    st.title("Dashboard Admin")
    if st.button("Keluar", key="admin_dashboard_exit"):
        st.session_state["admin_authenticated"] = False
        st.rerun(scope="app")

    section = st.radio(
        "Menu dashboard",
        (
            "Statistik",
            "Riwayat chat",
            "Pengguna",
            "Kualitas jawaban",
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
        key="admin_dashboard_section",
    )
    if section == "Statistik":
        _render_statistics(client)
    elif section == "Riwayat chat":
        _render_history(client)
    elif section == "Pengguna":
        _render_users(client)
    elif section == "Kualitas jawaban":
        _render_answer_quality(client, secrets)
    elif section == "Pertanyaan sering":
        try:
            admin_keys = [
                item.key for item in get_available_keys("admin", client, secrets)
            ]
            model_names = (
                get_available_chat_models(admin_keys[0]) if admin_keys else []
            )
        except KeyPoolError as error:
            st.error(f"Kelompok key admin teu tiasa dibaca: {error}")
        else:
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
        try:
            chat_keys = [
                item.key for item in get_available_keys("chat", client, secrets)
            ]
            available_models = (
                get_available_chat_models(chat_keys[0]) if chat_keys else []
            )
            model_settings = get_model_order(client, secrets, admin=True)
        except KeyPoolError as error:
            st.error(str(error))
        else:
            _render_api_keys(
                client,
                secrets,
                ordered_models(available_models, model_settings),
            )
    elif section == "Urutan model":
        try:
            chat_keys = [
                item.key for item in get_available_keys("chat", client, secrets)
            ]
        except KeyPoolError as error:
            st.error(str(error))
        else:
            _render_model_order(client, secrets, chat_keys)
    elif section == "Kontrol lain":
        _render_controls(client)
    else:
        _render_health(client)
    record_admin_menu_run(section, time.perf_counter() - started_at)


def render_dashboard(
    client: Any | None,
    manager: Any,
    folder_id: str,
    index_keys: Sequence[str],
    drive_authorized: bool,
    secrets: dict[str, Any],
) -> None:
    _render_dashboard_fragment(
        client,
        manager,
        folder_id,
        index_keys,
        drive_authorized,
        secrets,
    )
