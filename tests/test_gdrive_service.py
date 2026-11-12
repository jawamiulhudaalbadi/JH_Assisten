import unittest
from unittest.mock import patch

from services.gdrive_service import TEXT_MIME_TYPE, read_drive_files


class ReadDriveFilesTests(unittest.TestCase):
    def test_skips_unchanged_files_before_building_drive_service(self) -> None:
        file_infos = [
            {
                "id": "file-1",
                "name": "document.txt",
                "mimeType": TEXT_MIME_TYPE,
                "modifiedTime": "2026-11-12T00:00:00Z",
            }
        ]

        with patch("services.gdrive_service._build_drive_service") as build_service:
            documents = read_drive_files(
                file_infos,
                indexed_file_modification_times={
                    "file-1": "2026-11-12T00:00:00Z"
                },
            )

        self.assertEqual(documents, [])
        build_service.assert_not_called()


if __name__ == "__main__":
    unittest.main()
