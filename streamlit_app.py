import hashlib
import os
import contextlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
import re
from xml.sax.saxutils import escape

import streamlit as st
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer

from review_to_md import (
    InvestigationArtifacts,
    generate_investigation_artifacts_sync,
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
    def _inline_markdown_to_paragraph_html(text: str) -> str:
        html = escape(text)
        html = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", html)
        html = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", html)
        html = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", html)
        return html

    def _flush_paragraph(story: list, paragraph_lines: list[str], body_style: ParagraphStyle) -> None:
        if not paragraph_lines:
            return
        merged = " ".join(part.strip() for part in paragraph_lines if part.strip())
        if merged:
            story.append(Paragraph(_inline_markdown_to_paragraph_html(merged), body_style))
        paragraph_lines.clear()

    def _flush_bullets(story: list, bullet_items: list[str], body_style: ParagraphStyle) -> None:
        if not bullet_items:
            return
        list_items = [
            ListItem(Paragraph(_inline_markdown_to_paragraph_html(item), body_style))
            for item in bullet_items
        ]
        story.append(ListFlowable(list_items, bulletType="bullet", leftIndent=18))
        bullet_items.clear()

    def _flush_numbered(story: list, numbered_items: list[str], body_style: ParagraphStyle) -> None:
        if not numbered_items:
            return
        list_items = [
            ListItem(Paragraph(_inline_markdown_to_paragraph_html(item), body_style))
            for item in numbered_items
        ]
        story.append(ListFlowable(list_items, bulletType="1", leftIndent=18))
        numbered_items.clear()

    def _flush_code_block(story: list, code_lines: list[str], code_style: ParagraphStyle) -> None:
        if not code_lines:
            return
        code_text = "\n".join(code_lines)
        story.append(Preformatted(code_text, code_style))
        code_lines.clear()

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )
    heading_styles = {
        1: ParagraphStyle("H1", parent=styles["Heading1"], spaceBefore=8, spaceAfter=8),
        2: ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=6),
        3: ParagraphStyle("H3", parent=styles["Heading3"], spaceBefore=4, spaceAfter=4),
        4: ParagraphStyle("H4", parent=styles["Heading4"], spaceBefore=4, spaceAfter=4),
        5: ParagraphStyle("H5", parent=styles["Heading5"], spaceBefore=3, spaceAfter=3),
        6: ParagraphStyle("H6", parent=styles["Heading6"], spaceBefore=3, spaceAfter=3),
    }
    code_style = ParagraphStyle(
        "Code",
        parent=body_style,
        fontName="Courier",
        fontSize=8.5,
        leading=10,
        leftIndent=8,
    )

    story: list = []
    paragraph_lines: list[str] = []
    bullet_items: list[str] = []
    numbered_items: list[str] = []
    code_lines: list[str] = []
    in_code_block = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            if in_code_block:
                _flush_code_block(story, code_lines, code_style)
                story.append(Spacer(1, 6))
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if not stripped:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            story.append(Spacer(1, 6))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2)
            story.append(
                Paragraph(
                    _inline_markdown_to_paragraph_html(heading_text),
                    heading_styles.get(level, heading_styles[6]),
                )
            )
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_numbered(story, numbered_items, body_style)
            bullet_items.append(bullet_match.group(1))
            continue

        numbered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered_match:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            numbered_items.append(numbered_match.group(1))
            continue

        _flush_bullets(story, bullet_items, body_style)
        _flush_numbered(story, numbered_items, body_style)
        paragraph_lines.append(stripped)

    _flush_paragraph(story, paragraph_lines, body_style)
    _flush_bullets(story, bullet_items, body_style)
    _flush_numbered(story, numbered_items, body_style)
    if in_code_block:
        _flush_code_block(story, code_lines, code_style)

    if not story:
        story = [Paragraph("(Empty markdown report)", body_style)]

    doc.build(story)
    return pdf_buffer.getvalue()


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

    markdown_bytes = artifacts.report_markdown.encode("utf-8")
    download_name = f"{Path(st.session_state.uploaded_name or 'investigation').stem}_investigation.md"
    pdf_bytes = _render_markdown_pdf_bytes(artifacts.stage3_report_markdown)
    pdf_download_name = (
        f"{Path(st.session_state.uploaded_name or 'investigation').stem}_stage3_updates.pdf"
    )
    st.download_button(
        "Download Markdown",
        data=markdown_bytes,
        file_name=download_name,
        mime="text/markdown",
    )
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
                    artifacts.report_markdown,
                    file_name=st.session_state.uploaded_name or "directive.md",
                    blob_prefix=blob_prefix,
                )
                st.success(f"Saved: {blob_url}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Blob save failed: {exc}")
