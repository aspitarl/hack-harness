import hashlib
import os
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from review_to_md import (
    InvestigationArtifacts,
    generate_investigation_artifacts_sync,
    render_markdown_pdf_bytes,
)


st.set_page_config(page_title="Directive Investigation", layout="wide")
st.title("Directive Investigation")
st.caption("Upload a directive and generate investigation markdown.")


class _LiveLogWriter:
    def __init__(self, placeholder: "st.delta_generator.DeltaGenerator") -> None:
        self._placeholder = placeholder
        self._lines: list[str] = []
        self._current_line = ""
        self._last_render = 0.0

    def write(self, text: str) -> int:
        if not text:
            return 0

        for char in text:
            if char == "\r":
                self._current_line = ""
                continue
            if char == "\n":
                self._lines.append(self._current_line)
                self._current_line = ""
                continue
            self._current_line += char

        now = time.monotonic()
        if now - self._last_render > 0.1:
            self._render()
        return len(text)

    def flush(self) -> None:
        self._render()

    def _render(self) -> None:
        visible_lines = [line for line in self._lines if line.strip()]
        if self._current_line.strip():
            visible_lines.append(self._current_line)

        tail = "\n".join(visible_lines[-120:])
        self._placeholder.text(tail or "Waiting for pipeline output...")
        self._last_render = time.monotonic()


def _file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run_pipeline_from_upload(
    file_name: str,
    data: bytes,
    max_stage3_files: int,
) -> InvestigationArtifacts:
    suffix = Path(file_name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        return generate_investigation_artifacts_sync(
            tmp_path,
            max_stage3_files=max_stage3_files,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _save_markdown_to_blob(markdown: str, file_name: str, blob_prefix: str = "reports") -> str:
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL", "").strip()
    container = os.getenv("AZURE_STORAGE_CONTAINER", "").strip()
    if not account_url or not container:
        raise ValueError(
            "Set AZURE_STORAGE_ACCOUNT_URL and AZURE_STORAGE_CONTAINER to enable Blob save."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    base_name = Path(file_name).stem
    safe_prefix = blob_prefix.strip("/")
    blob_name = f"{safe_prefix}/{base_name}-{timestamp}.md" if safe_prefix else f"{base_name}-{timestamp}.md"

    credential = DefaultAzureCredential()
    client = BlobServiceClient(account_url=account_url, credential=credential)
    blob_client = client.get_blob_client(container=container, blob=blob_name)
    blob_client.upload_blob(markdown.encode("utf-8"), overwrite=False)
    return f"{account_url}/{container}/{blob_name}"


def _render_markdown_pdf_bytes(markdown: str) -> bytes:
    return render_markdown_pdf_bytes(markdown)


if "artifacts" not in st.session_state:
    st.session_state.artifacts = None
if "uploaded_digest" not in st.session_state:
    st.session_state.uploaded_digest = None
if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

uploaded_file = st.file_uploader("Drop directive file", type=["pdf", "txt", "md"])
debug_mode = st.checkbox("Debug mode (only first 3 files in Stage 3)")
run_clicked = st.button("Run Investigation", type="primary", disabled=uploaded_file is None)
run_log_panel = st.empty()

if run_clicked and uploaded_file is not None:
    data = uploaded_file.getvalue()
    digest = _file_digest(data)
    st.session_state.uploaded_name = uploaded_file.name
    log_writer = _LiveLogWriter(run_log_panel)
    run_log_panel.text("Starting investigation...")

    with st.spinner("Running investigation pipeline..."):
        try:
            stage3_file_limit = 3 if debug_mode else 20
            with contextlib.redirect_stdout(log_writer), contextlib.redirect_stderr(log_writer):
                artifacts = _run_pipeline_from_upload(
                    uploaded_file.name,
                    data,
                    max_stage3_files=stage3_file_limit,
                )
            st.session_state.artifacts = artifacts
            st.session_state.uploaded_digest = digest
            log_writer.flush()
            st.success("Investigation completed.")
        except Exception as exc:  # noqa: BLE001
            log_writer.flush()
            st.error(f"Pipeline failed: {exc}")

artifacts = st.session_state.artifacts

if artifacts is not None:
    st.subheader("Investigation Output")
    st.caption("Report is generated and available for download.")

    stage3_markdown = artifacts.stage3_report_markdown or artifacts.report_markdown
    markdown_bytes = stage3_markdown.encode("utf-8")
    download_name = (
        f"{Path(st.session_state.uploaded_name or 'investigation').stem}_stage3_updates.md"
    )
    pdf_source_markdown = stage3_markdown
    pdf_bytes = b""
    pdf_error: str | None = None
    try:
        pdf_bytes = _render_markdown_pdf_bytes(pdf_source_markdown)
    except Exception as exc:  # noqa: BLE001
        pdf_error = str(exc)
    pdf_download_name = (
        f"{Path(st.session_state.uploaded_name or 'investigation').stem}_stage3_updates.pdf"
    )
    st.download_button(
        "Download Markdown",
        data=markdown_bytes,
        file_name=download_name,
        mime="text/markdown",
    )
    if pdf_error:
        st.warning(f"PDF generation failed, markdown is still available: {pdf_error}")
    else:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=pdf_download_name,
            mime="application/pdf",
        )

    save_to_blob = st.checkbox("Also save markdown to Azure Blob Storage")
    if save_to_blob:
        blob_prefix = st.text_input("Blob folder", value="reports")
        if st.button("Save to Blob"):
            try:
                blob_url = _save_markdown_to_blob(
                    stage3_markdown,
                    file_name=(
                        f"{Path(st.session_state.uploaded_name or 'investigation').stem}_stage3_updates.md"
                    ),
                    blob_prefix=blob_prefix,
                )
                st.success(f"Saved: {blob_url}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Blob save failed: {exc}")
