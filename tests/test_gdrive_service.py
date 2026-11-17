import subprocess
import unittest
from unittest.mock import MagicMock, Mock, patch

from services.gdrive_service import (
    PDF_EXTRACTION_TIMEOUT_SECONDS,
    TEXT_MIME_TYPE,
    _extract_pdf_pages,
    read_drive_files,
)


class ReadDriveFilesTests(unittest.TestCase):
    def test_pdf_extraction_uses_three_minute_timeout(self) -> None:
        document = MagicMock()
        document.__enter__.return_value = document
        with (
            patch(
                "services.gdrive_service.subprocess.run",
                return_value=Mock(stdout=b"[]"),
            ) as run,
            patch("services.gdrive_service.pymupdf.open", return_value=document),
        ):
            self.assertEqual(_extract_pdf_pages(b""), [])

        self.assertEqual(PDF_EXTRACTION_TIMEOUT_SECONDS, 180)
        self.assertEqual(run.call_args.kwargs["timeout"], 180)

    def test_pdf_extraction_falls_back_to_pymupdf_when_pypdf_exits_nonzero(self) -> None:
        page = Mock()
        page.get_text.return_value = "Fallback page text"
        document = MagicMock()
        document.__enter__.return_value = document
        document.__iter__.return_value = [page]

        with (
            patch(
                "services.gdrive_service.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["python", "-c"]),
            ),
            patch(
                "services.gdrive_service.pymupdf.open",
                return_value=document,
            ) as open_pdf,
        ):
            pages = _extract_pdf_pages(b"pdf bytes")

        self.assertEqual(pages, [("1", "Fallback page text")])
        open_pdf.assert_called_once_with(stream=b"pdf bytes", filetype="pdf")
        document.__exit__.assert_called_once()

    def test_pdf_extraction_falls_back_to_pymupdf_when_pypdf_text_is_empty(self) -> None:
        page = Mock()
        page.get_text.return_value = "Fallback page text"
        document = MagicMock()
        document.__enter__.return_value = document
        document.__iter__.return_value = [page]

        with (
            patch(
                "services.gdrive_service.subprocess.run",
                return_value=Mock(stdout=b"[]"),
            ),
            patch("services.gdrive_service.pymupdf.open", return_value=document),
        ):
            pages = _extract_pdf_pages(b"pdf bytes")

        self.assertEqual(pages, [("1", "Fallback page text")])

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
