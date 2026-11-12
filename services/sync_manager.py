"""Run Drive checks and FAISS updates outside Streamlit's render thread."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, time, timedelta, timezone
import json
from math import ceil
from pathlib import Path
from threading import Event, RLock
from time import monotonic
from collections.abc import Callable
from zoneinfo import ZoneInfo

from langchain_community.vectorstores import FAISS

from services.ai_service import (
    DEFAULT_INDEX_DIRECTORY,
    get_index_file_manifest,
    is_embedding_daily_quota_error,
    is_embedding_quota_error,
    load_index_manifest,
    load_search_index,
    save_index_manifest,
    update_search_index,
)
from services.gdrive_service import (
    PDF_EXTRACTION_TIMEOUT_SECONDS,
    DriveDocument,
    drive_download_url,
    drive_file_priority,
    list_drive_files,
    read_drive_files,
)

SOURCE_TEXT_DIRECTORY = DEFAULT_INDEX_DIRECTORY / "source_texts"
SYNC_STATUS_PATH = DEFAULT_INDEX_DIRECTORY / "sync_status.json"
MAX_FILES_PER_SYNC_SESSION = 10
_RPD_STATUS_LOCK = RLock()
_LOS_ANGELES_TIMEZONE = ZoneInfo("America/Los_Angeles")


class _EmbeddingQuotaExceededError(RuntimeError):
    pass


class _DailyEmbeddingQuotaExceededError(_EmbeddingQuotaExceededError):
    pass


class DriveSyncManager:
    """Share one serialized background sync and status across app sessions."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="drive-sync")
        self._lock = RLock()
        self._index_lock = RLock()
        self._config: tuple[str, str] | None = None
        self._future: Future[None] | None = None
        self._job_id = 0
        self._generation = 0
        self._status = "Sinkronisasi belum dimulai. Tekan Sinkronkan Sekarang."
        self._phase = "idle"
        self._file_count = 0
        self._downloaded_count = 0
        self._download_total = 0
        self._download_started_at = 0.0
        self._download_eta_seconds: float | None = None
        self._skipped_pdf_count = 0
        self._file_statuses: dict[str, dict[str, str]] = {}
        self._retry_file_ids: set[str] = set()
        self._resume_event = Event()
        self._resume_event.set()
        self._paused = False
        self._error: str | None = None
        self._result: dict[str, object] | None = None

    def start_sync(self, folder_id: str, api_key: str) -> bool:
        """Start one manually requested Drive sync."""
        with self._lock:
            if self.get_rpd_status() is not None:
                return False
            if self._future is not None and not self._future.done():
                return False

            config = (folder_id, api_key)
            if config != self._config:
                config_changed = self._config is not None
                self._config = config
                if config_changed:
                    self._file_statuses.clear()
                    self._retry_file_ids.clear()

            self._paused = False
            self._resume_event.set()
            self._job_id += 1
            job_id = self._job_id
            self._status = "Memeriksa perubahan Google Drive..."
            self._phase = "checking"
            self._downloaded_count = 0
            self._download_total = 0
            self._download_eta_seconds = None
            self._skipped_pdf_count = 0
            self._error = None
            self._future = self._executor.submit(
                self._run_and_publish,
                job_id,
                folder_id,
                api_key,
            )
            return True

    def record_daily_quota_limit(
        self,
        limited_at: datetime | None = None,
    ) -> dict[str, str]:
        """Persist the current RPD limit and the next Los Angeles midnight reset."""
        limited_at = limited_at or datetime.now(timezone.utc)
        if limited_at.tzinfo is None:
            raise ValueError("rpd_limited_at must include a timezone.")

        los_angeles_date = limited_at.astimezone(_LOS_ANGELES_TIMEZONE).date()
        reset_local = datetime.combine(
            los_angeles_date + timedelta(days=1),
            time.min,
            tzinfo=_LOS_ANGELES_TIMEZONE,
        )
        status = {
            "rpd_limited_at": limited_at.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "rpd_reset_at": reset_local.astimezone(timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        with _RPD_STATUS_LOCK:
            SYNC_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = SYNC_STATUS_PATH.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(status, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary_path.replace(SYNC_STATUS_PATH)
        return status

    def get_rpd_status(
        self,
        now: datetime | None = None,
    ) -> dict[str, str] | None:
        """Load a persisted RPD lockout, clearing it once its reset time passes."""
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must include a timezone.")

        with _RPD_STATUS_LOCK:
            if not SYNC_STATUS_PATH.is_file():
                return None
            try:
                status = json.loads(SYNC_STATUS_PATH.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                raise ValueError("The persisted sync status file is invalid JSON.") from error
            if not isinstance(status, dict):
                raise ValueError("The persisted sync status must be a JSON object.")

            limited_at = status.get("rpd_limited_at")
            reset_at = status.get("rpd_reset_at")
            if not isinstance(limited_at, str) or not isinstance(reset_at, str):
                raise ValueError("The persisted RPD status is missing timestamps.")
            try:
                reset_time = datetime.fromisoformat(reset_at)
                datetime.fromisoformat(limited_at)
            except ValueError as error:
                raise ValueError("The persisted RPD timestamps are invalid.") from error
            if reset_time.tzinfo is None:
                raise ValueError("The persisted RPD reset timestamp must include a timezone.")

            if now.astimezone(timezone.utc) >= reset_time.astimezone(timezone.utc):
                SYNC_STATUS_PATH.unlink(missing_ok=True)
                return None
            return {
                "rpd_limited_at": limited_at,
                "rpd_reset_at": reset_at,
            }

    def snapshot(self) -> dict[str, object]:
        """Return the latest status in a form safe to read from a UI thread."""
        with self._lock:
            return {
                "status": self._status,
                "phase": self._phase,
                "file_count": self._file_count,
                "downloaded_count": self._downloaded_count,
                "download_total": self._download_total,
                "download_eta_seconds": self._download_eta_seconds,
                "skipped_pdf_count": self._skipped_pdf_count,
                "files": [dict(file_status) for file_status in self._file_statuses.values()],
                "paused": self._paused,
                "error": self._error,
                "result": self._result,
                "generation": self._generation,
                "running": self._future is not None and not self._future.done(),
            }

    def pause(self) -> None:
        with self._lock:
            self._paused = True
            self._resume_event.clear()
            self._phase = "paused"
            self._status = "Jeda diminta; file yang sedang diproses akan diselesaikan dulu."

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._resume_event.set()
            if self._future is not None and not self._future.done():
                self._phase = "downloading"
                self._status = "Sinkronisasi dilanjutkan."

    def retry_file(self, file_id: str) -> bool:
        with self._lock:
            file_status = self._file_statuses.get(file_id)
            if not file_status or file_status["state"] != "failed":
                return False
            self._retry_file_ids.add(file_id)
            file_status["state"] = "queued"
            file_status["error"] = ""
            self._status = f"Percobaan ulang dijadwalkan: {file_status['name']}."
            return True

    def load_local_index(self, folder_id: str, api_key: str) -> FAISS | None:
        """Load a persisted index without racing the worker's disk write."""
        with self._index_lock:
            manifest = load_index_manifest(DEFAULT_INDEX_DIRECTORY)
            if manifest and manifest.get("folder_id") != folder_id:
                return None
            index = (
                None
                if manifest and not manifest.get("has_index")
                else load_search_index(api_key, DEFAULT_INDEX_DIRECTORY)
            )
            indexed_file_ids = set(get_index_file_manifest(index))

        with self._lock:
            if self._future is None or self._future.done():
                self._config = (folder_id, api_key)
                files = manifest.get("files", {}) if manifest else {}
                sources = manifest.get("sources", {}) if manifest else {}
                if not isinstance(files, dict):
                    files = {}
                if not isinstance(sources, dict):
                    sources = {}
                file_ids = set(map(str, files)) | set(map(str, sources))
                self._file_statuses = {}
                for file_id in file_ids:
                    source = sources.get(file_id, {})
                    if not isinstance(source, dict):
                        source = {}
                    modified_time = str(
                        files.get(file_id)
                        or source.get("modified_time")
                        or ""
                    )
                    self._file_statuses[file_id] = {
                        "id": file_id,
                        "name": str(source.get("name") or file_id),
                        "modified_time": modified_time,
                        "state": (
                            "ready"
                            if file_id in files and file_id in indexed_file_ids
                            else "queued"
                        ),
                        "error": "",
                    }
                self._file_count = len(file_ids)
                if self._phase == "idle":
                    self._status = (
                        "Indeks lokal siap; sinkronisasi Google Drive belum dimulai."
                        if index is not None
                        else "Belum ada indeks lokal. Tekan Sinkronkan Sekarang."
                    )
        return index

    def load_source_text(self, file_id: str) -> str | None:
        """Load the full extracted text saved for an indexed Drive file."""
        source_path = SOURCE_TEXT_DIRECTORY / f"{file_id}.json"
        if not source_path.is_file():
            return None
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(source, dict) or source.get("file_id") != file_id:
            return None
        return str(source.get("text", ""))

    def _set_status(self, job_id: int, status: str) -> None:
        with self._lock:
            if job_id == self._job_id:
                self._status = status

    def _run_and_publish(
        self,
        job_id: int,
        folder_id: str,
        api_key: str,
    ) -> None:
        try:
            result = self._sync(folder_id, api_key, job_id)
        except _DailyEmbeddingQuotaExceededError:
            try:
                self.record_daily_quota_limit()
            except OSError as error:
                with self._lock:
                    if job_id == self._job_id:
                        self._paused = True
                        self._resume_event.clear()
                        self._error = (
                            "Kuota harian Gemini habis, tetapi status limit gagal "
                            f"disimpan ke disk: {error}"
                        )
                        self._phase = "error"
                        self._status = self._error
                return
            with self._lock:
                if job_id == self._job_id:
                    self._paused = True
                    self._resume_event.clear()
                    self._error = (
                        "Gemini-ի օրական embedding քվոտան սպառվել է։ "
                        "Սինքրոնացումը կանգնեցվել է, իսկ ավարտված ֆայլերի "
                        "ինդեքսն ու manifest-ը պահպանված են։"
                    )
                    self._phase = "quota_error"
                    self._status = self._error
            return
        except _EmbeddingQuotaExceededError:
            with self._lock:
                if job_id == self._job_id:
                    self._paused = True
                    self._resume_event.clear()
                    self._error = (
                        "Batas kuota API Gemini (429) masih tercapai setelah "
                        "percobaan otomatis. Progres file yang sudah selesai "
                        "tersimpan; sinkronisasi dijeda. Lanjutkan setelah kuota pulih."
                    )
                    self._phase = "quota_error"
                    self._status = self._error
            return
        except Exception as error:
            with self._lock:
                if job_id == self._job_id:
                    self._error = str(error)
                    self._status = "Sinkronisasi Drive gagal."
                    self._phase = "error"
            return

        with self._lock:
            if job_id == self._job_id:
                self._result = result
                self._generation += 1
                session_limit_reached = bool(result["session_limit_reached"])
                self._paused = self._paused or session_limit_reached
                if self._paused:
                    self._resume_event.clear()
                    self._phase = "paused"
                else:
                    self._phase = "complete"
                self._status = (
                    f"Memori siap. {result['file_count']} file terpantau; "
                    f"{result['changed_count']} file diperbarui."
                )
                if session_limit_reached:
                    self._status = (
                        f"Sesi selesai setelah {result['processed_count']} file. "
                        "Progres tersimpan; tekan Lanjutkan muat untuk sesi berikutnya."
                    )
                elif self._paused:
                    self._status += " Jeda aktif; file yang sedang diproses sudah selesai."
                if self._skipped_pdf_count:
                    self._status += (
                        f" {self._skipped_pdf_count} PDF dilewati karena "
                        f"ekstraksi melebihi {PDF_EXTRACTION_TIMEOUT_SECONDS} detik."
                    )
                failed_count = sum(
                    file_status["state"] == "failed"
                    for file_status in self._file_statuses.values()
                )
                if failed_count:
                    self._status += f" {failed_count} sumber gagal; gunakan Coba ulang."

    def _sync(
        self,
        folder_id: str,
        api_key: str,
        job_id: int,
    ) -> dict[str, object]:
        self._set_status(job_id, "Memeriksa metadata Google Drive...")
        inventory = list_drive_files(folder_id)
        current_files = {
            str(file_info["id"]): str(file_info.get("modifiedTime", ""))
            for file_info in inventory
        }
        current_sources = {
            str(file_info["id"]): {
                "file_id": str(file_info["id"]),
                "name": str(file_info.get("name", file_info["id"])),
                "mime_type": str(file_info.get("mimeType", "")),
                "modified_time": str(file_info.get("modifiedTime", "")),
                "web_view_link": str(
                    file_info.get("webViewLink", "")
                    or f"https://drive.google.com/file/d/{file_info['id']}/view"
                ),
                "webViewLink": str(
                    file_info.get("webViewLink", "")
                    or f"https://drive.google.com/file/d/{file_info['id']}/view"
                ),
                "download_url": drive_download_url(file_info),
            }
            for file_info in inventory
        }
        with self._lock:
            if job_id == self._job_id:
                self._file_count = len(inventory)
        self._set_status(job_id, f"Drive berisi {len(inventory)} sumber dokumen.")

        with self._index_lock:
            manifest = load_index_manifest(DEFAULT_INDEX_DIRECTORY)
            needs_manifest_save = (
                manifest is None
                or manifest.get("folder_id") != folder_id
                or manifest.get("sources") != current_sources
            )
            if manifest and manifest.get("folder_id") == folder_id:
                previous_files = {
                    str(file_id): str(modified_time)
                    for file_id, modified_time in manifest["files"].items()
                }
                index = (
                    load_search_index(api_key, DEFAULT_INDEX_DIRECTORY)
                    if manifest.get("has_index")
                    else None
                )
                if manifest.get("has_index") and index is None:
                    previous_files = {}
            elif manifest is None:
                index = load_search_index(api_key, DEFAULT_INDEX_DIRECTORY)
                previous_files = get_index_file_manifest(index)
            else:
                index = None
                previous_files = {}

        indexed_file_ids = set(get_index_file_manifest(index))
        indexed_file_modification_times = {
            file_id: modified_time
            for file_id, modified_time in previous_files.items()
            if file_id in indexed_file_ids
        }
        with self._lock:
            retry_file_ids = set(self._retry_file_ids)
        pending_files = [
            file_info
            for file_info in inventory
            if (
                previous_files.get(str(file_info["id"]))
                != str(file_info.get("modifiedTime", ""))
                or str(file_info["id"]) not in indexed_file_ids
                or str(file_info["id"]) in retry_file_ids
            )
        ]
        with self._lock:
            self._retry_file_ids.difference_update(retry_file_ids)
            for file_info in inventory:
                file_id = str(file_info["id"])
                modified_time = str(file_info.get("modifiedTime", ""))
                prior_status = self._file_statuses.get(file_id)
                is_indexed = (
                    previous_files.get(file_id) == modified_time
                    and file_id in indexed_file_ids
                )
                if (
                    prior_status
                    and prior_status["state"] == "failed"
                    and file_id not in retry_file_ids
                    and prior_status.get("modified_time") == modified_time
                ):
                    state, error = "failed", prior_status.get("error", "")
                elif is_indexed and file_id not in retry_file_ids:
                    state, error = "ready", ""
                else:
                    state, error = "queued", ""
                self._file_statuses[file_id] = {
                    "id": file_id,
                    "name": file_info.get("name", file_id),
                    "modified_time": modified_time,
                    "state": state,
                    "error": error,
                }
            pending_files = [
                file_info
                for file_info in pending_files
                if self._file_statuses[str(file_info["id"])]["state"] != "failed"
                or str(file_info["id"]) in retry_file_ids
            ]
            for file_id in set(previous_files) - set(current_files):
                self._file_statuses.pop(file_id, None)
        removed_file_ids = set(previous_files) - set(current_files)
        if manifest is None or manifest.get("sources") != current_sources or needs_manifest_save:
            with self._index_lock:
                save_index_manifest(
                    folder_id,
                    previous_files,
                    index is not None,
                    DEFAULT_INDEX_DIRECTORY,
                    current_sources,
                )
        changed_file_ids = {str(file_info["id"]) for file_info in pending_files}
        changed = bool(pending_files or removed_file_ids)
        pending_files.sort(
            key=lambda file_info: (
                int(file_info.get("size", "0") or "0"),
                drive_file_priority(file_info),
            )
        )
        changed_files = pending_files[:MAX_FILES_PER_SYNC_SESSION]
        processed_file_count = 0
        updated_file_count = 0

        if changed or needs_manifest_save:
            if removed_file_ids:
                self._set_phase(job_id, "embedding")
                self._set_status(job_id, "Menghapus dokumen yang sudah dihapus dari Drive...")
                index = update_search_index(index, [], removed_file_ids, api_key)
                for file_id in removed_file_ids:
                    source_path = SOURCE_TEXT_DIRECTORY / f"{file_id}.json"
                    source_path.unlink(missing_ok=True)

            committed_files = {
                file_id: modified_time
                for file_id, modified_time in previous_files.items()
                if file_id in current_files and file_id not in changed_file_ids
            }
        else:
            committed_files = {
                file_id: modified_time
                for file_id, modified_time in previous_files.items()
                if file_id in current_files
            }

        if changed_files:
            with self._lock:
                if job_id == self._job_id:
                    self._phase = "downloading"
                    self._download_total = len(changed_files)
                    self._downloaded_count = 0
                    self._download_started_at = monotonic()
                    self._download_eta_seconds = None
                    self._status = (
                        f"Mengunduh dan mengekstrak 0/{len(changed_files)} file; "
                        "estimasi dihitung setelah file pertama."
                    )

            batch_size = 1
            for batch_start in range(0, len(changed_files), batch_size):
                batch = changed_files[batch_start : batch_start + batch_size]

                def report_download_progress(
                    done: int,
                    _batch_total: int,
                    _name: str,
                    offset: int = batch_start,
                ) -> None:
                    file_id = str(batch[done - 1]["id"])
                    self._report_download_progress(
                        job_id,
                        file_id,
                        offset + done,
                        len(changed_files),
                    )

                def report_file_started(
                    number: int,
                    _batch_total: int,
                    name: str,
                    offset: int = batch_start,
                ) -> None:
                    file_id = str(batch[number - 1]["id"])
                    self._report_file_started(
                        job_id,
                        file_id,
                        offset + number,
                        len(changed_files),
                        name,
                    )

                def report_file_skipped(file_id: str, name: str, reason: str) -> None:
                    self._report_pdf_skipped(job_id, file_id, name, reason)

                def report_file_failed(file_id: str, name: str, reason: str) -> None:
                    self._report_file_failed(job_id, file_id, name, reason)

                documents = read_drive_files(
                    batch,
                    progress_callback=report_download_progress,
                    file_started_callback=report_file_started,
                    file_skipped_callback=report_file_skipped,
                    file_failed_callback=report_file_failed,
                    before_file_callback=lambda: self._wait_until_resumed(job_id),
                    indexed_file_modification_times=indexed_file_modification_times,
                )
                processed_file_count += len(batch)
                successful_file_ids = {document.file_id for document in documents}
                with self._lock:
                    for file_id in successful_file_ids:
                        if file_id in self._file_statuses:
                            self._file_statuses[file_id]["state"] = "embedding"
                self._set_phase(job_id, "embedding")
                self._set_status(
                    job_id,
                    f"Membuat embedding batch {batch_start // batch_size + 1} "
                    f"dari {(len(changed_files) + batch_size - 1) // batch_size}...",
                )
                try:
                    index = update_search_index(
                        index,
                        documents,
                        successful_file_ids,
                        api_key,
                        on_wait=lambda seconds: self._report_embedding_wait(job_id, seconds),
                    )
                except Exception as error:
                    if is_embedding_quota_error(error):
                        for file_id in successful_file_ids:
                            self._set_file_status(
                                job_id,
                                file_id,
                                "queued",
                                "Ditunda karena batas kuota API Gemini.",
                            )
                        if is_embedding_daily_quota_error(error):
                            raise _DailyEmbeddingQuotaExceededError from error
                        raise _EmbeddingQuotaExceededError from error
                    for file_id in successful_file_ids:
                        self._set_file_status(job_id, file_id, "failed", str(error))
                    raise
                updated_file_count += len(successful_file_ids)
                committed_files.update(
                    {
                        str(file_info["id"]): str(file_info.get("modifiedTime", ""))
                        for file_info in batch
                        if str(file_info["id"]) in successful_file_ids
                    }
                )
                self._set_phase(job_id, "saving")
                self._set_status(
                    job_id,
                    f"Menyimpan batch selesai ({min(batch_start + len(batch), len(changed_files))}/"
                    f"{len(changed_files)} file)...",
                )
                with self._index_lock:
                    if index is not None:
                        index.save_local(str(DEFAULT_INDEX_DIRECTORY))
                    for document in documents:
                        self._save_source_text(document)
                    save_index_manifest(
                        folder_id,
                        committed_files,
                        index is not None,
                        DEFAULT_INDEX_DIRECTORY,
                        current_sources,
                    )
                for file_id in successful_file_ids:
                    self._set_file_status(job_id, file_id, "ready")
                self._publish_batch(
                    job_id,
                    folder_id,
                    current_files,
                    len(inventory),
                    processed_file_count,
                    index is not None,
                )

        if changed or needs_manifest_save:
            self._set_phase(job_id, "saving")
            self._set_status(job_id, "Menyimpan manifest sinkronisasi...")
            with self._index_lock:
                if index is not None:
                    index.save_local(str(DEFAULT_INDEX_DIRECTORY))
                save_index_manifest(
                    folder_id,
                    committed_files,
                    index is not None,
                    DEFAULT_INDEX_DIRECTORY,
                    current_sources,
                )

        return {
            "folder_id": folder_id,
            "files": current_files,
            "file_count": len(inventory),
            "changed_count": updated_file_count,
            "processed_count": processed_file_count,
            "session_limit_reached": (
                processed_file_count >= MAX_FILES_PER_SYNC_SESSION
            ),
            "pending_count": max(len(pending_files) - processed_file_count, 0),
            "index_updated": changed or needs_manifest_save,
            "has_index": index is not None,
        }

    @staticmethod
    def _save_source_text(document: DriveDocument) -> None:
        SOURCE_TEXT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        source_path = SOURCE_TEXT_DIRECTORY / f"{document.file_id}.json"
        temporary_path = source_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "file_id": document.file_id,
                    "name": document.name,
                    "modified_time": document.modified_time,
                    "text": document.text,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary_path.replace(source_path)

    def _publish_batch(
        self,
        job_id: int,
        folder_id: str,
        files: dict[str, str],
        file_count: int,
        changed_count: int,
        has_index: bool,
    ) -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            self._result = {
                "folder_id": folder_id,
                "files": files,
                "file_count": file_count,
                "changed_count": changed_count,
                "index_updated": True,
                "has_index": has_index,
            }
            self._generation += 1

    def _set_phase(self, job_id: int, phase: str) -> None:
        with self._lock:
            if job_id == self._job_id:
                self._phase = phase

    def _report_download_progress(
        self,
        job_id: int,
        file_id: str,
        done: int,
        total: int,
    ) -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            if (
                file_id in self._file_statuses
                and self._file_statuses[file_id]["state"] == "downloading"
            ):
                self._file_statuses[file_id]["state"] = "downloaded"
            self._downloaded_count = done
            self._download_total = total
            remaining = max(total - done, 0)
            elapsed = monotonic() - self._download_started_at
            self._download_eta_seconds = (
                elapsed / done * remaining if done and remaining else 0.0
            )
            status = f"Unduh dan ekstrak {done}/{total} selesai; {remaining} tersisa"
            if done and remaining:
                status += f" | estimasi sisa {self._format_eta(self._download_eta_seconds)}"
            self._status = status + "."

    def _report_file_started(
        self,
        job_id: int,
        file_id: str,
        current: int,
        total: int,
        name: str,
    ) -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            self._file_statuses[file_id]["state"] = "downloading"
            completed = current - 1
            self._status = (
                f"Mengunduh dan mengekstrak file {current}/{total}: {name} "
                f"({completed}/{total} selesai)."
            )

    def _report_pdf_skipped(
        self,
        job_id: int,
        file_id: str,
        name: str,
        reason: str,
    ) -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            self._skipped_pdf_count += 1
            if file_id in self._file_statuses:
                self._file_statuses[file_id]["state"] = "failed"
                self._file_statuses[file_id]["error"] = reason
            self._status = f"PDF dilewati: {name} ({reason}); sinkronisasi dilanjutkan."

    def _report_file_failed(
        self,
        job_id: int,
        file_id: str,
        name: str,
        reason: str,
    ) -> None:
        with self._lock:
            if job_id != self._job_id:
                return
            if file_id in self._file_statuses:
                self._file_statuses[file_id]["state"] = "failed"
                self._file_statuses[file_id]["error"] = reason
            self._status = f"Sumber gagal: {name}; {reason}"

    def _set_file_status(
        self,
        job_id: int,
        file_id: str,
        state: str,
        error: str = "",
    ) -> None:
        with self._lock:
            if job_id == self._job_id and file_id in self._file_statuses:
                self._file_statuses[file_id]["state"] = state
                self._file_statuses[file_id]["error"] = error

    def _wait_until_resumed(self, job_id: int) -> None:
        while not self._resume_event.wait(timeout=0.5):
            with self._lock:
                if job_id != self._job_id:
                    return

    def _report_embedding_wait(self, job_id: int, seconds: float) -> None:
        wait_seconds = ceil(seconds)
        minutes, remaining_seconds = divmod(wait_seconds, 60)
        if minutes and remaining_seconds:
            wait_label = f"{minutes} menit {remaining_seconds} detik"
        elif minutes:
            wait_label = f"{minutes} menit"
        else:
            wait_label = f"{wait_seconds} detik"
        self._set_status(
            job_id,
            f"Jeda aman embedding {wait_label} sebelum request berikutnya.",
        )

    @staticmethod
    def _format_eta(seconds: float) -> str:
        minutes = ceil(seconds / 60)
        hours, remaining_minutes = divmod(minutes, 60)
        if hours:
            return f"{hours} jam {remaining_minutes} menit" if remaining_minutes else f"{hours} jam"
        return f"{minutes} menit"