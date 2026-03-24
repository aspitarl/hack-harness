"""Utilities for extracting plain text content from PDF files."""

from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(file_path: Path) -> str:
    """Extract concatenated text from all pages of a PDF file.

    Args:
        file_path: Absolute or relative PDF file path.

    Returns:
        Concatenated text content.

    Raises:
        ValueError: If the PDF cannot be parsed or has no extractable text.
    """

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to read PDF file: {file_path}") from exc

    pages_text: list[str] = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        if extracted.strip():
            pages_text.append(extracted.strip())

    if not pages_text:
        raise ValueError(
            f"No extractable text found in PDF: {file_path}. "
            "If this is a scanned PDF, use OCR before review."
        )

    return "\n\n".join(pages_text)
