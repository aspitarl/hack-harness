"""Quick diagnostic for Azure Blob URL access from Python.

Usage:
  python3 scripts/test_blob_access.py --url "https://<account>.blob.core.windows.net/<container>/<blob>"
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import re
import sys

import httpx
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient
from pypdf import PdfReader


DEFAULT_STAGE3_MIN_CHARS = 500


SOURCE_REF_FIELDS = [
    "metadata_storage_url",
    "blob_url",
    "blobUrl",
    "url",
    "uri",
    "metadata_storage_path",
    "path",
    "sourcefile",
    "sourceFile",
    "source_file",
]


def _extract_source_ref_with_field(document: dict[str, object]) -> tuple[str | None, str | None]:
    for field in SOURCE_REF_FIELDS:
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip(), field
    return None, None


def _load_search_doc_from_inputs(
    search_doc_json: str | None,
    search_doc_file: str | None,
) -> tuple[dict[str, object] | None, str | None]:
    if search_doc_json and search_doc_file:
        return None, "Provide only one of --search-doc-json or --search-doc-file."

    if not search_doc_json and not search_doc_file:
        return None, None

    try:
        if search_doc_json:
            payload = json.loads(search_doc_json)
        else:
            payload = json.loads(Path(search_doc_file).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to parse search document JSON: {exc}"

    if not isinstance(payload, dict):
        return None, "Search document JSON must be a single object."

    return payload, None


def _test_anonymous_http(url: str) -> tuple[bool, str]:
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(url)
        if response.status_code == 200:
            return True, f"HTTP 200 (downloaded {len(response.content)} bytes)"
        return False, f"HTTP {response.status_code}: {response.text[:300]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Request failed: {exc}"


def _test_identity_blob_sdk(url: str) -> tuple[bool, str]:
    try:
        credential = DefaultAzureCredential()
        blob_client = BlobClient.from_blob_url(url, credential=credential)
        properties = blob_client.get_blob_properties()
        size = getattr(properties, "size", None)
        preview = blob_client.download_blob(offset=0, length=1024).readall()
        return True, f"Authenticated read OK (size={size}, preview_bytes={len(preview)})"
    except Exception as exc:  # noqa: BLE001
        return False, f"Authenticated read failed: {exc}"


def _download_blob_bytes_with_identity(url: str) -> tuple[bool, bytes | None, str]:
    try:
        credential = DefaultAzureCredential()
        blob_client = BlobClient.from_blob_url(url, credential=credential)
        pdf_bytes = blob_client.download_blob().readall()
        return True, pdf_bytes, f"Downloaded {len(pdf_bytes)} bytes via authenticated Blob SDK"
    except Exception as exc:  # noqa: BLE001
        return False, None, f"Authenticated full download failed: {exc}"


def _diagnose_pdf_text_extraction(pdf_bytes: bytes, min_chars: int) -> tuple[bool, str]:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
    except Exception as exc:  # noqa: BLE001
        return False, f"PDF parse failed: {exc}"

    total_pages = len(reader.pages)
    pages_with_text = 0
    extracted_chunks: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_with_text += 1
            extracted_chunks.append(page_text.strip())

    raw_text = "\n\n".join(extracted_chunks)
    normalized = re.sub(r"\s+", " ", raw_text).strip()
    normalized_len = len(normalized)

    if normalized_len < min_chars:
        reason = (
            "Text extraction produced too little text for Stage 3 threshold. "
            f"normalized_chars={normalized_len}, min_required={min_chars}, "
            f"pages_with_text={pages_with_text}/{total_pages}."
        )
        if normalized_len == 0:
            reason += " Likely scanned/image-based PDF or non-extractable text layer."
        return False, reason

    preview = normalized[:220]
    return True, (
        "Text extraction diagnosis PASS. "
        f"normalized_chars={normalized_len}, pages_with_text={pages_with_text}/{total_pages}, "
        f"preview='{preview}'"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Azure Blob URL access from Python")
    parser.add_argument("--url", required=True, help="Blob URL to test")
    parser.add_argument(
        "--search-doc-json",
        help="Raw Azure Search document JSON string to diagnose source-ref selection.",
    )
    parser.add_argument(
        "--search-doc-file",
        help="Path to JSON file containing one Azure Search document object.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_STAGE3_MIN_CHARS,
        help="Minimum normalized character count to consider extraction usable (matches Stage 3 style check).",
    )
    args = parser.parse_args()

    print(f"Testing blob URL: {args.url}\n")

    search_doc, search_doc_error = _load_search_doc_from_inputs(args.search_doc_json, args.search_doc_file)
    if search_doc_error:
        print("[0/3] Search document source-ref diagnosis: FAIL")
        print(f"      {search_doc_error}\n")
        return 1

    if search_doc is not None:
        source_ref, source_field = _extract_source_ref_with_field(search_doc)
        print(f"[0/3] Search document source-ref diagnosis: {'PASS' if source_ref else 'FAIL'}")
        if source_ref:
            print(f"      Selected field: {source_field}")
            print(f"      Selected value: {source_ref}")
            if source_ref != args.url:
                print("      WARNING: --url differs from source-ref selected from search document.")
        else:
            print("      No supported source-ref fields found in search document.")
            print(f"      Checked fields: {', '.join(SOURCE_REF_FIELDS)}")
        print()

    anon_ok, anon_msg = _test_anonymous_http(args.url)
    print(f"[1/2] Anonymous HTTP access: {'PASS' if anon_ok else 'FAIL'}")
    print(f"      {anon_msg}\n")

    auth_ok, auth_msg = _test_identity_blob_sdk(args.url)
    print(f"[2/2] Authenticated Blob SDK access: {'PASS' if auth_ok else 'FAIL'}")
    print(f"      {auth_msg}\n")

    print("[3/3] Stage 3-style text extraction diagnosis:", end=" ")
    if not auth_ok:
        print("SKIP")
        print("      Cannot diagnose extraction because authenticated blob access failed.\n")
        print("Result: Python identity-based access is not working yet.")
        return 1

    dl_ok, pdf_bytes, dl_msg = _download_blob_bytes_with_identity(args.url)
    if not dl_ok or pdf_bytes is None:
        print("FAIL")
        print(f"      {dl_msg}\n")
        print("Result: Blob is reachable with identity, but full download failed for extraction diagnosis.")
        return 1

    extract_ok, extract_msg = _diagnose_pdf_text_extraction(pdf_bytes, args.min_chars)
    print("PASS" if extract_ok else "FAIL")
    print(f"      {dl_msg}")
    print(f"      {extract_msg}\n")

    if extract_ok:
        print("Result: Blob access and text extraction are both working for Stage 3-style full text loading.")
        return 0

    print("Result: Blob access works, but text extraction does not meet Stage 3 full-text criteria.")
    return 1


if __name__ == "__main__":
    sys.exit(main())