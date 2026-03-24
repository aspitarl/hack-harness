"""Quick diagnostic for Azure Blob URL access from Python.

Usage:
  python3 scripts/test_blob_access.py --url "https://<account>.blob.core.windows.net/<container>/<blob>"
"""

from __future__ import annotations

import argparse
import sys

import httpx
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobClient


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Azure Blob URL access from Python")
    parser.add_argument("--url", required=True, help="Blob URL to test")
    args = parser.parse_args()

    print(f"Testing blob URL: {args.url}\n")

    anon_ok, anon_msg = _test_anonymous_http(args.url)
    print(f"[1/2] Anonymous HTTP access: {'PASS' if anon_ok else 'FAIL'}")
    print(f"      {anon_msg}\n")

    auth_ok, auth_msg = _test_identity_blob_sdk(args.url)
    print(f"[2/2] Authenticated Blob SDK access: {'PASS' if auth_ok else 'FAIL'}")
    print(f"      {auth_msg}\n")

    if auth_ok:
        print("Result: Python can access this blob with identity-based auth.")
        return 0

    print("Result: Python identity-based access is not working yet.")
    return 1


if __name__ == "__main__":
    sys.exit(main())