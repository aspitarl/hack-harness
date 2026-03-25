import hashlib
import os
import contextlib
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
import re

import pandas as pd
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


def _split_stage3_file_sections(stage3_markdown: str) -> list[dict[str, str]]:
    heading_matches = list(re.finditer(r"(?m)^###\s+(.+?)\s*$", stage3_markdown))
    if not heading_matches:
        return []

    sections: list[dict[str, str]] = []
    for idx, match in enumerate(heading_matches):
        section_start = match.end()
        section_end = (
            heading_matches[idx + 1].start() if idx + 1 < len(heading_matches) else len(stage3_markdown)
        )
        file_name = match.group(1).strip()
        section_body = stage3_markdown[section_start:section_end].strip()
        if file_name:
            sections.append({"file_name": file_name, "content": section_body})
    return sections


def _extract_update_needed(section_text: str) -> str:
    match = re.search(r"(?im)^\s*-\s*Update\s+needed:\s*(Yes|No|Unclear)\b", section_text)
    if not match:
        return "Unknown"
    return match.group(1).title()


def _extract_evidence_confidence(section_text: str) -> str:
    match = re.search(r"(?im)^\s*-\s*Evidence\s+confidence:\s*(High|Medium|Low)\b", section_text)
    if not match:
        return "Unknown"
    return match.group(1).title()


def _requirement_terms(requirement: str, max_terms: int = 6) -> list[str]:
    stop_words = {
        "about",
        "after",
        "against",
        "before",
        "being",
        "between",
        "could",
        "directive",
        "from",
        "have",
        "into",
        "must",
        "order",
        "procedure",
        "shall",
        "should",
        "that",
        "their",
        "there",
        "these",
        "this",
        "those",
        "under",
        "with",
        "within",
    }
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", requirement.lower()):
        if len(token) < 5 or token in stop_words:
            continue
        if token not in terms:
            terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _requirement_file_match(requirement_idx: int, requirement: str, section_text: str) -> bool:
    normalized = section_text.lower()
    requirement_token = f"r{requirement_idx}"
    if requirement_token in normalized:
        return True

    for term in _requirement_terms(requirement):
        if term in normalized:
            return True
    return False


def _compute_visual_summary_data(
    stage3_markdown: str,
    atomic_requirements: list[str],
) -> dict[str, object]:
    sections = _split_stage3_file_sections(stage3_markdown)
    if not sections:
        return {
            "sections": [],
            "status_counts": {},
            "confidence_counts": {},
            "files_df": pd.DataFrame(),
            "coverage_df": pd.DataFrame(),
            "overall_confidence_pct": None,
        }

    file_rows: list[dict[str, object]] = []
    for section in sections:
        content = section["content"]
        update_needed = _extract_update_needed(content)
        confidence = _extract_evidence_confidence(content)
        file_rows.append(
            {
                "File": section["file_name"],
                "Update Needed": update_needed,
                "Confidence": confidence,
            }
        )

    files_df = pd.DataFrame(file_rows)

    ordered_statuses = ["Yes", "Unclear", "No", "Unknown"]
    status_counts = {
        status: int((files_df["Update Needed"] == status).sum())
        for status in ordered_statuses
        if int((files_df["Update Needed"] == status).sum()) > 0
    }

    ordered_confidence = ["High", "Medium", "Low", "Unknown"]
    confidence_counts = {
        level: int((files_df["Confidence"] == level).sum())
        for level in ordered_confidence
        if int((files_df["Confidence"] == level).sum()) > 0
    }

    confidence_weight = {"High": 3, "Medium": 2, "Low": 1}
    known_confidence_values = [
        confidence_weight[row["Confidence"]]
        for row in file_rows
        if row["Confidence"] in confidence_weight
    ]
    overall_confidence_pct = None
    if known_confidence_values:
        overall_confidence_pct = round(
            (sum(known_confidence_values) / (3 * len(known_confidence_values))) * 100,
            1,
        )

    coverage_rows: list[dict[str, object]] = []
    for req_idx, requirement in enumerate(atomic_requirements, start=1):
        row: dict[str, object] = {"Requirement": f"R{req_idx}"}
        for section in sections:
            row[section["file_name"]] = 1 if _requirement_file_match(
                req_idx,
                requirement,
                section["content"],
            ) else 0
        coverage_rows.append(row)

    coverage_df = pd.DataFrame(coverage_rows)

    return {
        "sections": sections,
        "status_counts": status_counts,
        "confidence_counts": confidence_counts,
        "files_df": files_df,
        "coverage_df": coverage_df,
        "overall_confidence_pct": overall_confidence_pct,
    }


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

    st.divider()
    st.subheader("Visual Summary")

    summary = _compute_visual_summary_data(
        stage3_markdown=stage3_markdown,
        atomic_requirements=artifacts.atomic_requirements,
    )
    sections = summary["sections"]

    if not sections:
        st.info("No Stage 3 file-level sections were found to visualize.")
    else:
        files_df = summary["files_df"]
        status_counts = summary["status_counts"]
        confidence_counts = summary["confidence_counts"]
        coverage_df = summary["coverage_df"]
        overall_confidence_pct = summary["overall_confidence_pct"]

        files_requiring_updates = int((files_df["Update Needed"] == "Yes").sum())
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Files analyzed", len(sections))
        metric_col2.metric("Updates needed", files_requiring_updates)
        metric_col3.metric(
            "Overall confidence",
            f"{overall_confidence_pct}%" if overall_confidence_pct is not None else "N/A",
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Update needed distribution")
            if status_counts:
                status_df = pd.DataFrame(
                    {
                        "Status": list(status_counts.keys()),
                        "Count": list(status_counts.values()),
                    }
                ).set_index("Status")
                st.bar_chart(status_df)
            else:
                st.info("No update-needed values found.")

        with chart_col2:
            st.caption("Evidence confidence distribution")
            if confidence_counts:
                confidence_df = pd.DataFrame(
                    {
                        "Confidence": list(confidence_counts.keys()),
                        "Count": list(confidence_counts.values()),
                    }
                ).set_index("Confidence")
                st.bar_chart(confidence_df)
            else:
                st.info("No confidence values found.")

        st.caption("Requirement-to-file coverage matrix (1 = matched, 0 = no match)")
        if not coverage_df.empty:
            st.dataframe(coverage_df, use_container_width=True, hide_index=True)
        else:
            st.info("No requirement coverage data available.")

        st.caption("Per-file triage table")
        st.dataframe(files_df, use_container_width=True, hide_index=True)
