import unittest
from unittest.mock import Mock, patch

from services.gdrive_service import DriveDocument, TEXT_MIME_TYPE
from services.sync_manager import DriveSyncManager


class _Index:
    def save_local(self, _directory: str) -> None:
        pass


class DriveSyncManagerTests(unittest.TestCase):
    def test_start_sync_does_not_submit_duplicate_background_work(self) -> None:
        manager = DriveSyncManager()
        future = Mock()
        future.done.return_value = False

        with patch.object(manager._executor, "submit", return_value=future) as submit:
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
            text="document contents",
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
                    text="document contents",
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
                    text="document contents",
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
        self.assertIn("օրական", snapshot["error"])
        self.assertIn("պահպանված են", snapshot["error"])
        self.assertEqual(save_manifest.call_count, 2)
        file_states = {item["id"]: item["state"] for item in snapshot["files"]}
        self.assertEqual(file_states["file-0"], "ready")
        self.assertEqual(file_states["file-1"], "queued")


if __name__ == "__main__":
    unittest.main()
