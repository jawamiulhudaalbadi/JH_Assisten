from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import Mock, patch

from services.gdrive_service import DriveDocument, TEXT_MIME_TYPE
from services.sync_manager import DriveSyncManager, is_text_corrupted


class _Index:
    def save_local(self, _directory: str) -> None:
        pass


class DriveSyncManagerTests(unittest.TestCase):
    def test_manager_instances_share_the_faiss_index_lock(self) -> None:
        first_manager = DriveSyncManager()
        second_manager = DriveSyncManager()

        self.assertIs(first_manager._index_lock, second_manager._index_lock)

    def test_is_index_busy_does_not_wait_for_a_locked_index(self) -> None:
        manager = DriveSyncManager()
        lock_acquired = Event()
        release_lock = Event()

        def hold_index_lock() -> None:
            with manager._index_lock:
                lock_acquired.set()
                release_lock.wait()

        worker = Thread(target=hold_index_lock)
        worker.start()
        try:
            self.assertTrue(lock_acquired.wait(timeout=1))
            self.assertTrue(manager.is_index_busy())
        finally:
            release_lock.set()
            worker.join(timeout=1)

    def test_is_index_busy_reports_background_embedding(self) -> None:
        manager = DriveSyncManager()
        manager._phase = "embedding"
        manager._future = Mock()
        manager._future.done.return_value = False

        self.assertTrue(manager.is_index_busy())

    def test_embedding_progress_includes_an_eta_after_a_completed_batch(self) -> None:
        manager = DriveSyncManager()
        manager._job_id = 1

        with patch(
            "services.sync_manager.monotonic",
            side_effect=(100.0, 110.0),
        ):
            manager._report_embedding_progress(1, 0, 10, 1, 2)
            self.assertIsNone(manager.snapshot()["embedding_eta_seconds"])
            manager._report_embedding_progress(1, 5, 10, 1, 2)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot["embedding_done_count"], 5)
        self.assertEqual(snapshot["embedding_total"], 10)
        self.assertEqual(snapshot["embedding_eta_seconds"], 10.0)
        self.assertIn("Membuat embedding batch 1/2", snapshot["status"])

    def test_index_checkpoint_persists_faiss_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_directory = Path(temp_dir)
            index = _Index()
            index.save_local = Mock()
            manager = DriveSyncManager()
            sources = {"file-1": {"name": "document.txt"}}

            with (
                patch(
                    "services.sync_manager.DEFAULT_INDEX_DIRECTORY",
                    index_directory,
                ),
                patch("services.sync_manager.save_index_manifest") as save_manifest,
                patch("services.sync_manager.gc.collect") as collect,
            ):
                manager._save_index_checkpoint(
                    index,
                    "folder-1",
                    {"completed-file": "modified-time"},
                    sources,
                )

        index.save_local.assert_called_once_with(str(index_directory))
        save_manifest.assert_called_once_with(
            "folder-1",
            {"completed-file": "modified-time"},
            True,
            index_directory,
            sources,
        )
        collect.assert_called_once_with()

    def test_is_text_corrupted_allows_latin_and_arabic_text(self) -> None:
        self.assertFalse(is_text_corrupted("A normal English sentence with numbers 123."))
        self.assertFalse(is_text_corrupted("هذا نص عربي واضح لاختبار قراءة المستند"))

    def test_is_text_corrupted_rejects_short_or_symbol_heavy_text(self) -> None:
        self.assertTrue(is_text_corrupted(""))
        self.assertTrue(is_text_corrupted("Too short"))
        self.assertTrue(is_text_corrupted("Readable text followed by " + "💥" * 20))

    def test_daily_quota_status_is_persisted_and_loaded(self) -> None:
        limited_at = datetime(2026, 3, 8, 8, 30, tzinfo=timezone.utc)
        expected_status = {
            "rpd_limited_at": "2026-03-08T08:30:00+00:00",
            "rpd_reset_at": "2026-03-09T07:00:00+00:00",
            "rpd_api_key_hash": (
                "62af8704764faf8ea82fc61ce9c4c390"
                "8b6cb97d463a634e9e587d7c885db0ef"
            ),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_status.json"
            with patch("services.sync_manager.SYNC_STATUS_PATH", status_path):
                writer = DriveSyncManager()
                saved_status = writer.record_daily_quota_limit(
                    limited_at,
                    api_key="test-key",
                )

                self.assertEqual(saved_status, expected_status)
                self.assertEqual(
                    json.loads(status_path.read_text(encoding="utf-8")),
                    expected_status,
                )

                reader = DriveSyncManager()
                loaded_status = reader.get_rpd_status(
                    datetime(2026, 3, 9, 6, 59, 59, tzinfo=timezone.utc),
                    api_key="test-key",
                )

        self.assertEqual(
            loaded_status,
            {
                "rpd_limited_at": expected_status["rpd_limited_at"],
                "rpd_reset_at": expected_status["rpd_reset_at"],
            },
        )

    def test_daily_quota_status_clears_and_unpauses_when_api_key_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_status.json"
            with patch("services.sync_manager.SYNC_STATUS_PATH", status_path):
                manager = DriveSyncManager()
                manager.record_daily_quota_limit(
                    datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc),
                    api_key="old-key",
                )
                manager._phase = "quota_error"
                manager._paused = True
                manager._resume_event.clear()
                manager._error = "quota exceeded"

                status = manager.get_rpd_status(
                    datetime(2026, 11, 12, 13, 0, tzinfo=timezone.utc),
                    api_key="new-key",
                )

                self.assertIsNone(status)
                self.assertFalse(status_path.exists())
                snapshot = manager.snapshot()
                future = Mock()
                future.done.return_value = False
                with patch.object(
                    manager._executor,
                    "submit",
                    return_value=future,
                ) as submit:
                    self.assertTrue(manager.start_sync("folder-1", "new-key"))
                submit.assert_called_once()

        self.assertFalse(snapshot["paused"])
        self.assertEqual(snapshot["phase"], "idle")
        self.assertIsNone(snapshot["error"])
        self.assertEqual(
            snapshot["status"],
            "Sinkronisasi belum dimulai. Tekan Sinkronkan Sekarang.",
        )

    def test_daily_quota_status_remains_for_same_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_status.json"
            with patch("services.sync_manager.SYNC_STATUS_PATH", status_path):
                manager = DriveSyncManager()
                manager.record_daily_quota_limit(
                    datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc),
                    api_key="same-key",
                )

                status = manager.get_rpd_status(
                    datetime(2026, 11, 12, 13, 0, tzinfo=timezone.utc),
                    api_key="same-key",
                )
                self.assertIsNotNone(status)
                self.assertTrue(status_path.exists())

    def test_start_sync_clears_cooldown_for_replacement_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_status.json"
            with patch("services.sync_manager.SYNC_STATUS_PATH", status_path):
                manager = DriveSyncManager()
                manager.record_daily_quota_limit(
                    datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc),
                    api_key="old-key",
                )
                manager._phase = "quota_error"
                manager._paused = True
                future = Mock()
                future.done.return_value = False

                with patch.object(
                    manager._executor,
                    "submit",
                    return_value=future,
                ) as submit:
                    self.assertTrue(manager.start_sync("folder-1", "new-key"))

                submit.assert_called_once()
                self.assertFalse(status_path.exists())


    def test_daily_quota_status_is_cleared_at_reset_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_status.json"
            with patch("services.sync_manager.SYNC_STATUS_PATH", status_path):
                manager = DriveSyncManager()
                manager.record_daily_quota_limit(
                    datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)
                )

                status = manager.get_rpd_status(
                    datetime(2026, 11, 13, 8, 0, tzinfo=timezone.utc)
                )

                self.assertIsNone(status)
                self.assertFalse(status_path.exists())

    def test_start_sync_does_not_submit_duplicate_background_work(self) -> None:
        manager = DriveSyncManager()
        future = Mock()
        future.done.return_value = False

        with (
            patch.object(manager, "get_rpd_status", return_value=None),
            patch.object(manager._executor, "submit", return_value=future) as submit,
        ):
            self.assertTrue(manager.start_sync("folder-1", "api-key"))
            self.assertFalse(manager.start_sync("folder-1", "api-key"))

        submit.assert_called_once()

    def test_loading_local_index_does_not_start_drive_sync(self) -> None:
        index = _Index()
        manager = DriveSyncManager()
        manifest = {
            "folder_id": "folder-1",
            "files": {"file-1": "modified"},
            "has_index": True,
            "sources": {
                "file-1": {"name": "document.txt", "modified_time": "modified"}
            },
        }

        with (
            patch("services.sync_manager.load_index_manifest", return_value=manifest),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch(
                "services.sync_manager.get_index_file_manifest",
                return_value={"file-1": "modified"},
            ),
            patch("services.sync_manager.list_drive_files") as list_files,
        ):
            loaded_index = manager.load_local_index("folder-1", "unused-api-key")

        self.assertIs(loaded_index, index)
        self.assertFalse(manager.snapshot()["running"])
        self.assertEqual(manager.snapshot()["files"][0]["state"], "ready")
        list_files.assert_not_called()

    def test_requeues_manifest_file_missing_from_local_index(self) -> None:
        file_info = {
            "id": "file-1",
            "name": "document.txt",
            "mimeType": TEXT_MIME_TYPE,
            "modifiedTime": "2026-11-12T00:00:00Z",
        }
        source_document = DriveDocument(
            file_id="file-1",
            name="document.txt",
            text="This document contains enough text.",
            modified_time=file_info["modifiedTime"],
        )
        index = _Index()
        manager = DriveSyncManager()
        manager._job_id = 1
        manifest = {
            "folder_id": "folder-1",
            "files": {"file-1": file_info["modifiedTime"]},
            "has_index": True,
            "sources": {},
        }

        with (
            patch("services.sync_manager.load_index_manifest", return_value=manifest),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch("services.sync_manager.get_index_file_manifest", return_value={}),
            patch("services.sync_manager.list_drive_files", return_value=[file_info]),
            patch(
                "services.sync_manager.read_drive_files",
                return_value=[source_document],
            ) as read_files,
            patch("services.sync_manager.update_search_index", return_value=index),
            patch("services.sync_manager.save_index_manifest"),
            patch.object(DriveSyncManager, "_save_source_text"),
        ):
            result = manager._sync("folder-1", "unused-api-key", 1)

        read_files.assert_called_once()
        self.assertEqual(result["changed_count"], 1)
        self.assertEqual(manager.snapshot()["files"][0]["state"], "ready")

    def test_corrupted_text_is_failed_without_embedding(self) -> None:
        file_info = {
            "id": "file-1",
            "name": "document.txt",
            "mimeType": TEXT_MIME_TYPE,
            "modifiedTime": "2026-11-12T00:00:00Z",
        }
        source_document = DriveDocument(
            file_id="file-1",
            name="document.txt",
            text="Readable text followed by " + "💥" * 20,
            modified_time=file_info["modifiedTime"],
        )
        index = _Index()
        manager = DriveSyncManager()
        manager._job_id = 1

        with (
            patch("services.sync_manager.load_index_manifest", return_value=None),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch("services.sync_manager.get_index_file_manifest", return_value={}),
            patch("services.sync_manager.list_drive_files", return_value=[file_info]),
            patch("services.sync_manager.read_drive_files", return_value=[source_document]),
            patch("services.sync_manager.update_search_index") as update_index,
            patch("services.sync_manager.save_index_manifest"),
            patch.object(DriveSyncManager, "_save_source_text") as save_source_text,
        ):
            manager._sync("folder-1", "unused-api-key", 1)

        update_index.assert_not_called()
        save_source_text.assert_not_called()
        file_status = manager.snapshot()["files"][0]
        self.assertEqual(file_status["state"], "failed")
        self.assertEqual(
            file_status["error"],
            "Teks rusak / tidak terbaca, konversi ke Google Docs",
        )

    def test_failed_file_status_survives_manager_reload(self) -> None:
        manifest = {
            "folder_id": "folder-1",
            "files": {},
            "has_index": False,
            "sources": {
                "file-1": {
                    "name": "document.txt",
                    "modified_time": "modified",
                }
            },
        }
        source_status = {
            "id": "file-1",
            "name": "document.txt",
            "modified_time": "modified",
            "state": "failed",
            "error": "Teks rusak; perbaiki file sumber.",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "sync_failures.json"
            with (
                patch("services.sync_manager.SYNC_FAILURE_STATUS_PATH", status_path),
                patch(
                    "services.sync_manager.load_index_manifest",
                    return_value=manifest,
                ),
                patch("services.sync_manager.load_search_index", return_value=None),
                patch(
                    "services.sync_manager.get_index_file_manifest",
                    return_value={},
                ),
            ):
                first_manager = DriveSyncManager()
                first_manager._config = ("folder-1", "api-key")
                first_manager._file_statuses["file-1"] = dict(source_status)
                first_manager._report_file_failed(
                    0,
                    "file-1",
                    "document.txt",
                    source_status["error"],
                )

                reloaded_manager = DriveSyncManager()
                reloaded_manager.load_local_index("folder-1", "api-key")
                reloaded_status = reloaded_manager.snapshot()["files"][0]

                self.assertEqual(reloaded_status["state"], "failed")
                self.assertEqual(reloaded_status["error"], source_status["error"])
                self.assertTrue(status_path.is_file())

                file_info = {
                    "id": "file-1",
                    "name": "document.txt",
                    "mimeType": TEXT_MIME_TYPE,
                    "modifiedTime": "modified",
                }
                with (
                    patch(
                        "services.sync_manager.list_drive_files",
                        return_value=[file_info],
                    ),
                    patch(
                        "services.sync_manager.load_index_manifest",
                        return_value=manifest,
                    ),
                    patch("services.sync_manager.load_search_index", return_value=None),
                    patch(
                        "services.sync_manager.get_index_file_manifest",
                        return_value={},
                    ),
                    patch("services.sync_manager.save_index_manifest"),
                    patch("services.sync_manager.read_drive_files") as read_files,
                ):
                    reloaded_manager._sync("folder-1", "api-key", 0)

                read_files.assert_not_called()
                self.assertEqual(
                    reloaded_manager.snapshot()["files"][0]["state"],
                    "failed",
                )

                self.assertTrue(reloaded_manager.retry_file("file-1"))
                self.assertFalse(status_path.exists())

    def test_unchanged_indexed_file_is_not_embedded_again(self) -> None:
        file_info = {
            "id": "file-1",
            "name": "document.txt",
            "mimeType": TEXT_MIME_TYPE,
            "modifiedTime": "2026-11-12T00:00:00Z",
        }
        index = _Index()
        manager = DriveSyncManager()
        manager._job_id = 1
        manifest = {
            "folder_id": "folder-1",
            "files": {"file-1": file_info["modifiedTime"]},
            "has_index": True,
            "sources": {},
        }

        with (
            patch("services.sync_manager.load_index_manifest", return_value=manifest),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch(
                "services.sync_manager.get_index_file_manifest",
                return_value={"file-1": file_info["modifiedTime"]},
            ),
            patch("services.sync_manager.list_drive_files", return_value=[file_info]),
            patch("services.sync_manager.read_drive_files") as read_files,
            patch("services.sync_manager.update_search_index") as update_index,
            patch("services.sync_manager.save_index_manifest"),
        ):
            result = manager._sync("folder-1", "unused-api-key", 1)

        read_files.assert_not_called()
        update_index.assert_not_called()
        self.assertEqual(result["processed_count"], 0)
        self.assertFalse(result["session_limit_reached"])

    def test_sync_session_processes_at_most_ten_files_and_pauses(self) -> None:
        file_infos = [
            {
                "id": f"file-{number}",
                "name": f"document-{number}.txt",
                "mimeType": TEXT_MIME_TYPE,
                "size": str(12 - number),
                "modifiedTime": "2026-11-12T00:00:00Z",
            }
            for number in range(12)
        ]
        index = _Index()
        manager = DriveSyncManager()
        manager._job_id = 1
        read_order: list[str] = []

        def read_one_file(batch, **callbacks):
            callbacks["before_file_callback"]()
            file_info = batch[0]
            read_order.append(file_info["id"])
            callbacks["file_started_callback"](1, 1, file_info["name"])
            callbacks["progress_callback"](1, 1, file_info["name"])
            return [
                DriveDocument(
                    file_id=file_info["id"],
                    name=file_info["name"],
                    text="This document contains enough text.",
                    modified_time=file_info["modifiedTime"],
                )
            ]

        with (
            patch("services.sync_manager.load_index_manifest", return_value=None),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch("services.sync_manager.get_index_file_manifest", return_value={}),
            patch("services.sync_manager.list_drive_files", return_value=file_infos),
            patch(
                "services.sync_manager.read_drive_files",
                side_effect=read_one_file,
            ) as read_files,
            patch("services.sync_manager.update_search_index", return_value=index),
            patch("services.sync_manager.save_index_manifest") as save_manifest,
            patch.object(DriveSyncManager, "_save_source_text"),
        ):
            manager._run_and_publish(1, "folder-1", "unused-api-key")

        snapshot = manager.snapshot()
        self.assertEqual(read_files.call_count, 10)
        self.assertEqual(
            read_order,
            [f"file-{number}" for number in range(11, 1, -1)],
        )
        self.assertTrue(snapshot["paused"])
        self.assertEqual(snapshot["phase"], "paused")
        self.assertEqual(snapshot["result"]["processed_count"], 10)
        self.assertEqual(snapshot["result"]["pending_count"], 2)
        self.assertGreaterEqual(save_manifest.call_count, 11)
        self.assertEqual(
            sum(file_status["state"] == "ready" for file_status in snapshot["files"]),
            10,
        )

    def test_quota_error_preserves_last_successful_file_checkpoint(self) -> None:
        file_infos = [
            {
                "id": f"file-{number}",
                "name": f"document-{number}.txt",
                "mimeType": TEXT_MIME_TYPE,
                "modifiedTime": "2026-11-12T00:00:00Z",
            }
            for number in range(2)
        ]
        index = _Index()
        manager = DriveSyncManager()
        manager._job_id = 1

        def read_one_file(batch, **callbacks):
            callbacks["before_file_callback"]()
            file_info = batch[0]
            callbacks["file_started_callback"](1, 1, file_info["name"])
            callbacks["progress_callback"](1, 1, file_info["name"])
            return [
                DriveDocument(
                    file_id=file_info["id"],
                    name=file_info["name"],
                    text="This document contains enough text.",
                    modified_time=file_info["modifiedTime"],
                )
            ]

        with (
            patch("services.sync_manager.load_index_manifest", return_value=None),
            patch("services.sync_manager.load_search_index", return_value=index),
            patch("services.sync_manager.get_index_file_manifest", return_value={}),
            patch("services.sync_manager.list_drive_files", return_value=file_infos),
            patch("services.sync_manager.read_drive_files", side_effect=read_one_file),
            patch(
                "services.sync_manager.update_search_index",
                side_effect=[
                    index,
                    RuntimeError(
                        "429 RESOURCE_EXHAUSTED: quota exceeded per day"
                    ),
                ],
            ),
            patch("services.sync_manager.save_index_manifest") as save_manifest,
            patch.object(DriveSyncManager, "_save_source_text"),
        ):
            manager._run_and_publish(1, "folder-1", "unused-api-key")

        snapshot = manager.snapshot()
        self.assertTrue(snapshot["paused"])
        self.assertEqual(snapshot["phase"], "quota_error")
        self.assertIn("poéan", snapshot["error"])
        self.assertIn("parantos disimpen", snapshot["error"])
        self.assertEqual(save_manifest.call_count, 2)
        file_states = {item["id"]: item["state"] for item in snapshot["files"]}
        self.assertEqual(file_states["file-0"], "ready")
        self.assertEqual(file_states["file-1"], "queued")


if __name__ == "__main__":
    unittest.main()
