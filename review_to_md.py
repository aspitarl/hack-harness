"""Run one-shot new-directive investigation and write markdown output to a file."""

import argparse
import asyncio
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import httpx
from azure.identity import DefaultAzureCredential
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle
from pypdf import PdfReader
from tqdm import tqdm

try:
    from azure.storage.blob import BlobClient
except Exception:  # noqa: BLE001
    BlobClient = None

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory

from chat_cli import (
    _build_investigate_user_message,
    _dual_index_retrieval_enabled,
    _extract_search_doc_name,
    _extract_search_doc_text,
    _load_review_document,
    _resolve_existing_file,
    _search_documents,
    _truncate_for_prompt,
    build_search_data_source,
    create_chat_service,
    load_agent_prompt_config,
    load_config,
    sanitize_investigation_markdown,
)


REQUIREMENT_QUERY_LIMIT = 8
STAGE2_FULL_DIRECTIVE_MAX_CHUNKS = 4
STAGE2_FULL_DIRECTIVE_CHUNK_SIZE = 2200
STAGE2_FULL_DIRECTIVE_CHUNK_OVERLAP = 300
STAGE3_MAX_FILES = 20
STAGE3_FULL_TEXT_MAX_CHARS = 16000
STAGE3_FULL_TEXT_MIN_CHARS = 500
STAGE3_FILENAME_RETRY_TOP_N = 40


@dataclass
class InvestigationArtifacts:
    report_markdown: str
    atomic_requirements: list[str]
    requirements_review_markdown: str
    stage3_report_markdown: str


def _log_stage(stage_number: int, message: str) -> None:
    print(f"[Stage {stage_number}] {message}", flush=True)


URL_SOURCE_FIELDS = [
    "metadata_storage_url",
    "blob_url",
    "blobUrl",
    "url",
    "uri",
]

PATHLIKE_SOURCE_FIELDS = [
    "metadata_storage_path",
    "path",
    "sourcefile",
    "sourceFile",
    "source_file",
]


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _collect_search_doc_source_refs(
    document: dict[str, object],
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for field in URL_SOURCE_FIELDS + PATHLIKE_SOURCE_FIELDS:
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            refs.append((field, value.strip()))
    return refs


def _extract_search_doc_source_ref(document: dict[str, object]) -> str | None:
    for _, value in _collect_search_doc_source_refs(document):
        return value
    return None


def _collect_source_ref_candidates(
    documents: list[dict[str, object]],
) -> list[tuple[str, str]]:
    seen: set[str] = set()
    url_candidates: list[tuple[str, str]] = []
    non_url_candidates: list[tuple[str, str]] = []

    for document in documents:
        for field, value in _collect_search_doc_source_refs(document):
            if value in seen:
                continue
            seen.add(value)
            if _is_http_url(value):
                url_candidates.append((field, value))
            else:
                non_url_candidates.append((field, value))

    return url_candidates + non_url_candidates


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages_text.append(page_text.strip())
    return "\n\n".join(pages_text)


async def _download_pdf_bytes(source_ref: str) -> bytes:
    parsed = urlparse(source_ref)
    if not parsed.scheme:
        raise ValueError("Source reference is not a URL.")

    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
        response = await client.get(source_ref)
        if response.status_code == 200:
            return response.content
        if response.status_code not in {401, 403, 409}:
            response.raise_for_status()

    if BlobClient is None:
        raise ValueError("Blob access requires azure-storage-blob package or SAS/public URL.")

    def _download_with_identity() -> bytes:
        credential = DefaultAzureCredential()
        blob_client = BlobClient.from_blob_url(source_ref, credential=credential)
        return blob_client.download_blob().readall()

    return await asyncio.to_thread(_download_with_identity)


async def _load_full_document_text_for_stage3(
    source_ref: str | None,
    max_chars: int = STAGE3_FULL_TEXT_MAX_CHARS,
) -> tuple[str | None, str | None]:
    if not source_ref:
        return None, "No source reference available in search metadata."

    try:
        pdf_bytes = await _download_pdf_bytes(source_ref)
        full_text = _extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        return None, f"Full document fetch failed: {exc}"

    normalized = re.sub(r"\s+", " ", full_text).strip()
    if len(normalized) < STAGE3_FULL_TEXT_MIN_CHARS:
        return None, "Full document text was empty or too short after extraction."

    return normalized[:max_chars], None


async def _load_full_document_text_from_candidates_for_stage3(
    source_ref_candidates: list[tuple[str, str]],
    max_chars: int = STAGE3_FULL_TEXT_MAX_CHARS,
) -> tuple[str | None, str | None, str | None]:
    if not source_ref_candidates:
        return None, "No source reference available in search metadata.", None

    candidate_errors: list[str] = []
    for field, source_ref in source_ref_candidates:
        full_document_text, full_document_error = await _load_full_document_text_for_stage3(
            source_ref,
            max_chars=max_chars,
        )
        if full_document_text:
            return full_document_text, None, source_ref

        normalized_error = full_document_error or "Unknown full document load failure."
        candidate_errors.append(f"{field}={source_ref} -> {normalized_error}")

    return None, "All source references failed. " + " | ".join(candidate_errors), None


def _normalize_requirement_key(text: str) -> str:
    lowered = text.lower()
    compact = re.sub(r"[^a-z0-9\s]", "", lowered)
    return re.sub(r"\s+", " ", compact).strip()


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?;:])\s+(?=[A-Z0-9\[])", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _extract_atomic_requirements(directive_text: str, max_items: int = 12) -> list[str]:
    requirement_cues = re.compile(
        r"\b(must|shall|required\s+to|is\s+required\s+to|must\s+include|shall\s+include|must\s+not|shall\s+not|will\s+include)\b",
        re.IGNORECASE,
    )

    stitched_text = re.sub(r"(?<![.!?;:])\n(?!\n)", " ", directive_text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", stitched_text) if block.strip()]

    candidates: list[str] = []
    for block in blocks:
        block_text = re.sub(r"^[-*\d\.)\s]+", "", block)
        candidates.extend(_split_sentences(block_text))

    deduped: list[str] = []
    seen: set[str] = set()

    for sentence in candidates:
        if len(sentence) < 40:
            continue
        if re.match(r"^\([A-Za-z0-9]+\)\s+", sentence) and len(sentence) < 90:
            continue
        if not requirement_cues.search(sentence):
            continue
        normalized = _normalize_requirement_key(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(sentence)
        if len(deduped) >= max_items:
            break

    if deduped:
        return deduped

    fallback: list[str] = []
    for sentence in candidates:
        if len(sentence) < 40:
            continue
        normalized = _normalize_requirement_key(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        fallback.append(sentence)
        if len(fallback) >= max_items:
            break
    return fallback


def _format_atomic_requirements_for_prompt(requirements: list[str]) -> str:
    if not requirements:
        return "No atomic requirements were extracted."
    lines = ["Atomic requirements extracted for targeted retrieval:"]
    for idx, item in enumerate(requirements, start=1):
        lines.append(f"R{idx}. {item}")
    return "\n".join(lines)


def _format_atomic_requirements_markdown(
    directive_name: str,
    requirements: list[str],
) -> str:
    lines = [
        "## Atomic Requirements Review",
        f"Directive: {directive_name}",
        "",
        "This intermediate file lists extracted atomic requirements used for per-requirement NETL retrieval.",
        "",
        "## Extracted Requirements",
    ]

    if not requirements:
        lines.append("- No requirements were extracted. Review directive text quality and extraction cues.")
        return "\n".join(lines)

    for idx, requirement in enumerate(requirements, start=1):
        lines.append(f"- R{idx}: {requirement}")

    lines.extend(
        [
            "",
            "## Review Guidance",
            "- Confirm each requirement is a single testable obligation.",
            "- Confirm no major directive section is missing representation.",
            "- Mark any requirement needing split/merge before downstream synthesis.",
        ]
    )
    return "\n".join(lines)


def _build_full_directive_chunks(
    directive_text: str,
    max_chunks: int = STAGE2_FULL_DIRECTIVE_MAX_CHUNKS,
    chunk_size: int = STAGE2_FULL_DIRECTIVE_CHUNK_SIZE,
    overlap: int = STAGE2_FULL_DIRECTIVE_CHUNK_OVERLAP,
) -> list[tuple[str, str]]:
    cleaned = re.sub(r"\s+", " ", directive_text).strip()
    if not cleaned:
        return []

    chunks: list[tuple[str, str]] = []
    start = 0
    chunk_index = 1
    text_length = len(cleaned)

    while start < text_length and len(chunks) < max_chunks:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            candidate_end = cleaned.rfind(". ", start, end)
            if candidate_end > start + (chunk_size // 2):
                end = candidate_end + 1

        snippet = cleaned[start:end].strip()
        if snippet:
            chunks.append((f"Q{chunk_index}", snippet))
            chunk_index += 1

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def _filter_docs_for_file_name(
    documents: list[dict[str, object]],
    file_name: str,
) -> list[dict[str, object]]:
    def _normalize_name_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    normalized_target = Path(file_name).name.lower()
    normalized_target_key = _normalize_name_key(normalized_target)
    normalized_target_stem_key = _normalize_name_key(Path(file_name).stem)
    filtered: list[dict[str, object]] = []

    for rank, document in enumerate(documents, start=1):
        candidate = Path(_extract_search_doc_name(document, rank)).name.lower()
        candidate_key = _normalize_name_key(candidate)
        candidate_stem_key = _normalize_name_key(Path(candidate).stem)

        if candidate == normalized_target:
            filtered.append(document)
            continue

        if candidate_key and candidate_key == normalized_target_key:
            filtered.append(document)
            continue

        if candidate_stem_key and candidate_stem_key == normalized_target_stem_key:
            filtered.append(document)

    return filtered


def _collect_affected_netl_files(
    query_results: list[dict[str, object]],
    supplemental_full_directive_results: list[dict[str, object]],
    dual_index_enabled: bool,
) -> list[str]:
    affected: set[str] = set()

    if dual_index_enabled:
        for result in query_results:
            for rank, document in enumerate(result.get("orders_documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))
            for rank, document in enumerate(result.get("procedures_documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))

        for result in supplemental_full_directive_results:
            for rank, document in enumerate(result.get("orders_documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))
            for rank, document in enumerate(result.get("procedures_documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))
    else:
        for result in query_results:
            for rank, document in enumerate(result.get("documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))

        for result in supplemental_full_directive_results:
            for rank, document in enumerate(result.get("documents", []), start=1):
                affected.add(_extract_search_doc_name(document, rank))

    return sorted(item for item in affected if item)


def _build_stage3_seed_documents_by_file(
    query_results: list[dict[str, object]],
    supplemental_full_directive_results: list[dict[str, object]],
    dual_index_enabled: bool,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    seed_map: dict[str, dict[str, list[dict[str, object]]]] = {}

    def _bucket(file_name: str) -> dict[str, list[dict[str, object]]]:
        if file_name not in seed_map:
            seed_map[file_name] = {
                "orders_documents": [],
                "procedures_documents": [],
                "documents": [],
            }
        return seed_map[file_name]

    if dual_index_enabled:
        for result in query_results + supplemental_full_directive_results:
            for rank, document in enumerate(result.get("orders_documents", []), start=1):
                file_name = _extract_search_doc_name(document, rank)
                if file_name:
                    _bucket(file_name)["orders_documents"].append(document)
            for rank, document in enumerate(result.get("procedures_documents", []), start=1):
                file_name = _extract_search_doc_name(document, rank)
                if file_name:
                    _bucket(file_name)["procedures_documents"].append(document)
    else:
        for result in query_results + supplemental_full_directive_results:
            for rank, document in enumerate(result.get("documents", []), start=1):
                file_name = _extract_search_doc_name(document, rank)
                if file_name:
                    _bucket(file_name)["documents"].append(document)

    return seed_map


async def _run_file_compliance_stage3(
    config,
    directive_text: str,
    requirements_for_retrieval: list[str],
    affected_files: list[str],
    dual_index_enabled: bool,
    seed_documents_by_file: dict[str, dict[str, list[dict[str, object]]]] | None = None,
    max_files: int = STAGE3_MAX_FILES,
) -> list[dict[str, object]]:
    if not affected_files:
        return []

    if max_files <= 0:
        return []

    requirement_seed = " ".join(requirements_for_retrieval[:3])
    directive_seed = _truncate_for_prompt(directive_text, max_chars=600)
    results: list[dict[str, object]] = []
    files_to_check = affected_files[:max_files]
    stage3_stats = {
        "files_checked": 0,
        "no_direct_match": 0,
        "source_ref_present": 0,
        "source_ref_missing": 0,
        "full_document_loaded": 0,
        "full_document_unavailable": 0,
        "matched_orders_total": 0,
        "matched_procedures_total": 0,
        "matched_documents_total": 0,
    }
    _log_stage(3, "Per-file compliance scan")
    seed_documents_by_file = seed_documents_by_file or {}

    for file_name in tqdm(
        files_to_check,
        desc="Stage 3: checking files",
        unit="file",
    ):
        stage3_stats["files_checked"] += 1
        query_text = _truncate_for_prompt(
            (
                f"{file_name} compliance update check against DOE O 458.1. "
                f"Focus requirements: {requirement_seed}. "
                f"Directive context: {directive_seed}"
            ),
            max_chars=1400,
        )

        if dual_index_enabled:
            orders_documents = await _search_documents(
                config,
                query_text,
                index_name=config.search_orders_index_name,
            )
            procedures_documents = await _search_documents(
                config,
                query_text,
                index_name=config.search_procedures_index_name,
            )

            filtered_orders = _filter_docs_for_file_name(
                orders_documents,
                file_name,
            )
            filtered_procedures = _filter_docs_for_file_name(
                procedures_documents,
                file_name,
            )
            if not filtered_orders and not filtered_procedures:
                retry_query = _truncate_for_prompt(file_name, max_chars=240)
                retry_orders = await _search_documents(
                    config,
                    retry_query,
                    index_name=config.search_orders_index_name,
                    top_n_documents=STAGE3_FILENAME_RETRY_TOP_N,
                )
                retry_procedures = await _search_documents(
                    config,
                    retry_query,
                    index_name=config.search_procedures_index_name,
                    top_n_documents=STAGE3_FILENAME_RETRY_TOP_N,
                )
                filtered_orders = _filter_docs_for_file_name(retry_orders, file_name)
                filtered_procedures = _filter_docs_for_file_name(retry_procedures, file_name)

            if not filtered_orders and not filtered_procedures:
                seed_bucket = seed_documents_by_file.get(file_name, {})
                seed_orders = _filter_docs_for_file_name(
                    seed_bucket.get("orders_documents", []),
                    file_name,
                )
                seed_procedures = _filter_docs_for_file_name(
                    seed_bucket.get("procedures_documents", []),
                    file_name,
                )
                if seed_orders or seed_procedures:
                    filtered_orders = seed_orders
                    filtered_procedures = seed_procedures
                else:
                    stage3_stats["no_direct_match"] += 1

            docs_for_file = filtered_orders + filtered_procedures
            stage3_stats["matched_orders_total"] += len(filtered_orders)
            stage3_stats["matched_procedures_total"] += len(filtered_procedures)
            source_ref_candidates = _collect_source_ref_candidates(docs_for_file)
            if source_ref_candidates:
                stage3_stats["source_ref_present"] += 1
            else:
                stage3_stats["source_ref_missing"] += 1
            full_document_text, full_document_error, source_ref = await _load_full_document_text_from_candidates_for_stage3(
                source_ref_candidates
            )
            if full_document_text:
                stage3_stats["full_document_loaded"] += 1
            else:
                stage3_stats["full_document_unavailable"] += 1

            results.append(
                {
                    "file_name": file_name,
                    "query": query_text,
                    "orders_documents": filtered_orders,
                    "procedures_documents": filtered_procedures,
                    "full_document_source": source_ref,
                    "full_document_text": full_document_text,
                    "full_document_error": full_document_error,
                }
            )
        else:
            documents = await _search_documents(config, query_text)
            filtered_documents = _filter_docs_for_file_name(documents, file_name)
            if not filtered_documents:
                retry_query = _truncate_for_prompt(file_name, max_chars=240)
                retry_documents = await _search_documents(
                    config,
                    retry_query,
                    top_n_documents=STAGE3_FILENAME_RETRY_TOP_N,
                )
                filtered_documents = _filter_docs_for_file_name(retry_documents, file_name)

            if not filtered_documents:
                seed_bucket = seed_documents_by_file.get(file_name, {})
                seed_documents = _filter_docs_for_file_name(
                    seed_bucket.get("documents", []),
                    file_name,
                )
                if seed_documents:
                    filtered_documents = seed_documents
                else:
                    stage3_stats["no_direct_match"] += 1

            stage3_stats["matched_documents_total"] += len(filtered_documents)
            source_ref_candidates = _collect_source_ref_candidates(filtered_documents)
            if source_ref_candidates:
                stage3_stats["source_ref_present"] += 1
            else:
                stage3_stats["source_ref_missing"] += 1
            full_document_text, full_document_error, source_ref = await _load_full_document_text_from_candidates_for_stage3(
                source_ref_candidates
            )
            if full_document_text:
                stage3_stats["full_document_loaded"] += 1
            else:
                stage3_stats["full_document_unavailable"] += 1

            results.append(
                {
                    "file_name": file_name,
                    "query": query_text,
                    "documents": filtered_documents,
                    "full_document_source": source_ref,
                    "full_document_text": full_document_text,
                    "full_document_error": full_document_error,
                }
            )

    _log_stage(
        3,
        (
            "Stage 3 stats: "
            f"files={stage3_stats['files_checked']}, "
            f"no_direct_match={stage3_stats['no_direct_match']}, "
            f"source_ref_present={stage3_stats['source_ref_present']}, "
            f"source_ref_missing={stage3_stats['source_ref_missing']}, "
            f"full_document_loaded={stage3_stats['full_document_loaded']}, "
            f"full_document_unavailable={stage3_stats['full_document_unavailable']}, "
            f"matched_orders_total={stage3_stats['matched_orders_total']}, "
            f"matched_procedures_total={stage3_stats['matched_procedures_total']}, "
            f"matched_documents_total={stage3_stats['matched_documents_total']}"
        ),
    )

    return results


def _build_stage3_snippet_evidence_summary(
    stage3_file_compliance: list[dict[str, object]],
) -> str:
    if not stage3_file_compliance:
        return "No Stage 3 snippet evidence available."

    lines: list[str] = ["## Stage 3 snippet evidence by file"]
    for result in stage3_file_compliance:
        file_name = str(result.get("file_name", ""))
        if not file_name:
            continue

        snippets: list[str] = []
        docs: list[dict[str, object]] = []
        docs.extend(result.get("orders_documents", []))
        docs.extend(result.get("procedures_documents", []))
        docs.extend(result.get("documents", []))

        for doc in docs:
            text = _extract_search_doc_text(doc).strip()
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text)
            if normalized in {"{}", "[]", "null"}:
                continue
            snippets.append(normalized[:280])
            if len(snippets) >= 3:
                break

        lines.append(f"- {file_name}")
        if snippets:
            for snippet in snippets:
                lines.append(f"  - \"{snippet}\"")
        else:
            lines.append("  - \"No snippet text extracted from retrieved evidence; rely on full_document_text when available.\"")

    return "\n".join(lines)


def _extract_update_needed_from_section(section_text: str) -> str:
    match = re.search(
        r"(?im)^\s*-\s*Update\s+needed:\s*(.+?)\s*$",
        section_text,
    )
    if not match:
        return "Unclear"

    normalized_value = re.sub(r"[*_`]+", "", match.group(1)).strip()
    value_match = re.match(r"^(Yes|No|Unclear)\b", normalized_value, flags=re.IGNORECASE)
    if not value_match:
        return "Unclear"

    return value_match.group(1).title()


def _extract_evidence_confidence_from_section(section_text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:-\s*)?Evidence\s+confidence:\s*(.+?)\s*$",
        section_text,
    )
    if not match:
        return "Unknown"

    normalized_value = re.sub(r"[*_`]+", "", match.group(1)).strip()
    value_match = re.match(r"^(High|Medium|Low)\b", normalized_value, flags=re.IGNORECASE)
    if not value_match:
        return "Unknown"

    return value_match.group(1).title()


def _sort_rank_for_update_needed(value: str) -> int:
    normalized = value.strip().lower()
    if normalized == "yes":
        return 0
    if normalized == "unclear":
        return 1
    if normalized == "no":
        return 2
    return 3


def _apply_update_needed_headers_and_sort(stage3_markdown: str) -> str:
    lines = stage3_markdown.splitlines()
    if not lines:
        return stage3_markdown

    top_lines: list[str] = []
    sections: list[list[str]] = []
    current_section: list[str] | None = None
    seen_first_section = False

    for line in lines:
        if line.startswith("### ") and not line.startswith("#### "):
            seen_first_section = True
            if current_section is not None:
                sections.append(current_section)
            current_section = [line]
            continue

        if current_section is None:
            if not seen_first_section:
                top_lines.append(line)
        else:
            current_section.append(line)

    if current_section is not None:
        sections.append(current_section)

    if not sections:
        return stage3_markdown

    decorated_sections: list[tuple[int, int, list[str]]] = []
    for idx, section in enumerate(sections):
        section_text = "\n".join(section)
        update_needed = _extract_update_needed_from_section(section_text)
        evidence_confidence = _extract_evidence_confidence_from_section(section_text)
        header = section[0]
        header_without_flag = re.sub(
            r"\s*[—-]\s*Update\s+needed\s*:\s*(Yes|No|Unclear)\s*(?:[—-]\s*(?:Evidence\s+)?Confidence\s*:\s*(High|Medium|Low|Unknown))?\s*$",
            "",
            header,
            flags=re.IGNORECASE,
        )
        section[0] = (
            f"{header_without_flag} — Update needed: {update_needed}"
            f" — Evidence confidence: {evidence_confidence}"
        )
        decorated_sections.append((_sort_rank_for_update_needed(update_needed), idx, section))

    decorated_sections.sort(key=lambda item: (item[0], item[1]))

    rebuilt_lines: list[str] = []
    if top_lines:
        rebuilt_lines.extend(top_lines)
        rebuilt_lines.append("")

    for section_index, (_, _, section_lines) in enumerate(decorated_sections):
        if section_index > 0:
            rebuilt_lines.append("")
        rebuilt_lines.extend(section_lines)

    return "\n".join(rebuilt_lines)


def _format_requirement_search_context_for_message(
    requirements: list[str],
    query_results: list[dict[str, object]],
    config_index_name: str | None,
    dual_index_enabled: bool,
    orders_index_name: str | None,
    procedures_index_name: str | None,
    stage2_full_directive_results: list[dict[str, object]] | None = None,
    stage3_file_compliance: list[dict[str, object]] | None = None,
) -> str:
    documents_to_investigate: list[str] = []
    payload_requirements: list[dict[str, object]] = []

    for result in query_results:
        requirement_id = result["requirement_id"]
        requirement_text = result["requirement_text"]
        requirement_query = result["query"]

        if dual_index_enabled:
            orders_documents = result["orders_documents"]
            procedures_documents = result["procedures_documents"]

            orders_context: list[dict[str, object]] = []
            for rank, doc in enumerate(orders_documents, start=1):
                name = _extract_search_doc_name(doc, rank)
                orders_context.append(
                    {
                        "rank": rank,
                        "document": name,
                        "score": doc.get("@search.score"),
                        "reranker_score": doc.get("@search.rerankerScore"),
                        "content": _extract_search_doc_text(doc),
                    }
                )
                documents_to_investigate.append(f"- [NETL order] {name} (for {requirement_id})")

            procedures_context: list[dict[str, object]] = []
            for rank, doc in enumerate(procedures_documents, start=1):
                name = _extract_search_doc_name(doc, rank)
                procedures_context.append(
                    {
                        "rank": rank,
                        "document": name,
                        "score": doc.get("@search.score"),
                        "reranker_score": doc.get("@search.rerankerScore"),
                        "content": _extract_search_doc_text(doc),
                    }
                )
                documents_to_investigate.append(
                    f"- [NETL procedure] {name} (for {requirement_id})"
                )

            payload_requirements.append(
                {
                    "requirement_id": requirement_id,
                    "requirement_text": requirement_text,
                    "query": requirement_query,
                    "sources": [
                        {
                            "index": orders_index_name,
                            "doc_type": "NETL order",
                            "documents": orders_context,
                        },
                        {
                            "index": procedures_index_name,
                            "doc_type": "NETL procedure",
                            "documents": procedures_context,
                        },
                    ],
                }
            )
        else:
            documents = result["documents"]
            single_context: list[dict[str, object]] = []
            for rank, doc in enumerate(documents, start=1):
                name = _extract_search_doc_name(doc, rank)
                single_context.append(
                    {
                        "rank": rank,
                        "document": name,
                        "score": doc.get("@search.score"),
                        "reranker_score": doc.get("@search.rerankerScore"),
                        "content": _extract_search_doc_text(doc),
                    }
                )
                documents_to_investigate.append(f"- {name} (for {requirement_id})")

            payload_requirements.append(
                {
                    "requirement_id": requirement_id,
                    "requirement_text": requirement_text,
                    "query": requirement_query,
                    "source": {
                        "index": config_index_name,
                        "documents": single_context,
                    },
                }
            )

    deduped_docs = list(dict.fromkeys(documents_to_investigate))
    docs_block = "\n".join(deduped_docs) if deduped_docs else "- (no matching documents found)"

    supplemental_payload: list[dict[str, object]] = []
    if stage2_full_directive_results:
        for result in stage2_full_directive_results:
            query_id = result["query_id"]
            query_text = result["query"]
            if dual_index_enabled:
                orders_docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["orders_documents"], start=1):
                    name = _extract_search_doc_name(doc, rank)
                    orders_docs_payload.append(
                        {
                            "rank": rank,
                            "document": name,
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                procedures_docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["procedures_documents"], start=1):
                    name = _extract_search_doc_name(doc, rank)
                    procedures_docs_payload.append(
                        {
                            "rank": rank,
                            "document": name,
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                supplemental_payload.append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "sources": [
                            {
                                "index": orders_index_name,
                                "doc_type": "NETL order",
                                "documents": orders_docs_payload,
                            },
                            {
                                "index": procedures_index_name,
                                "doc_type": "NETL procedure",
                                "documents": procedures_docs_payload,
                            },
                        ],
                    }
                )
            else:
                docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["documents"], start=1):
                    name = _extract_search_doc_name(doc, rank)
                    docs_payload.append(
                        {
                            "rank": rank,
                            "document": name,
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                supplemental_payload.append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "source": {
                            "index": config_index_name,
                            "documents": docs_payload,
                        },
                    }
                )

    stage3_payload: list[dict[str, object]] = []
    if stage3_file_compliance:
        for result in stage3_file_compliance:
            file_name = result["file_name"]
            query_text = result["query"]

            if dual_index_enabled:
                orders_docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["orders_documents"], start=1):
                    orders_docs_payload.append(
                        {
                            "rank": rank,
                            "document": _extract_search_doc_name(doc, rank),
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                procedures_docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["procedures_documents"], start=1):
                    procedures_docs_payload.append(
                        {
                            "rank": rank,
                            "document": _extract_search_doc_name(doc, rank),
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                stage3_payload.append(
                    {
                        "file_name": file_name,
                        "query": query_text,
                        "full_document_source": result.get("full_document_source"),
                        "full_document_text": result.get("full_document_text"),
                        "full_document_error": result.get("full_document_error"),
                        "sources": [
                            {
                                "index": orders_index_name,
                                "doc_type": "NETL order",
                                "documents": orders_docs_payload,
                            },
                            {
                                "index": procedures_index_name,
                                "doc_type": "NETL procedure",
                                "documents": procedures_docs_payload,
                            },
                        ],
                    }
                )
            else:
                docs_payload: list[dict[str, object]] = []
                for rank, doc in enumerate(result["documents"], start=1):
                    docs_payload.append(
                        {
                            "rank": rank,
                            "document": _extract_search_doc_name(doc, rank),
                            "score": doc.get("@search.score"),
                            "reranker_score": doc.get("@search.rerankerScore"),
                            "content": _extract_search_doc_text(doc),
                        }
                    )

                stage3_payload.append(
                    {
                        "file_name": file_name,
                        "query": query_text,
                        "full_document_source": result.get("full_document_source"),
                        "full_document_text": result.get("full_document_text"),
                        "full_document_error": result.get("full_document_error"),
                        "source": {
                            "index": config_index_name,
                            "documents": docs_payload,
                        },
                    }
                )

    payload = {
        "requirements": payload_requirements,
        "stage2_full_directive": supplemental_payload,
        "stage3_file_compliance": stage3_payload,
    }

    return (
        "Use this requirement-scoped retrieval context and cite facts conservatively.\n"
        "Treat requirement-scoped evidence as primary. Use Stage 2 full-directive retrieval only to identify gaps or missing obligations not represented in requirement-scoped evidence.\n"
        "Use stage3_file_compliance to produce a file-centric compliance result.\n"
        "For each NETL file in stage3_file_compliance, provide a clear update evaluation grounded in retrieved evidence.\n"
        "For each recommendation, reference the requirement ID (R#) and point to a NETL file section "
        "or a precise fallback string to search for in that file.\n"
        "Documents to investigate by requirement:\n"
        f"{docs_block}\n\n"
        "Requirement-scoped grounding payload:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


def _build_stage3_delta_user_message(
    directive_name: str,
    baseline_report: str,
    requirements_message: str,
    context_payload: str,
    snippet_evidence_summary: str,
    has_stage3_files: bool,
) -> str:
    no_files_instruction = (
        "If no files are available in stage3_file_compliance, output exactly: 'No Stage 3 files identified.'"
        if has_stage3_files
        else "No files are available in stage3_file_compliance. Output exactly: 'No Stage 3 files identified.'"
    )

    return "\n\n".join(
        [
            f"### DOE DIRECTIVE INPUT: {directive_name}",
            "Generate Stage 3 compliance updates only.",
            "Use stage3_file_compliance evidence as primary for this task.",
            "When full_document_text is present for a file, use it as primary evidence for that file's description and update evaluation.",
            "If full_document_text is missing, explicitly state that full text was unavailable and base evaluation only on retrieved snippets.",
            "Do NOT repeat unchanged content from baseline report.",
            "Include each NETL file represented in stage3_file_compliance.",
            "Use NETL files from the Stage 3 snippet evidence list below when available.",
            "If the Stage 3 snippet evidence list is empty or sparse, use requirement/full-directive payload context.",
            "For each NETL file, output this shape:",
            "### <NETL file name> (<Order|Procedure>)\\n#### Overall document description\\n<2-4 sentence summary of what the document governs>\\n#### Update evaluation\\n- Update needed: <Yes|No|Unclear>\\n- Rationale: <brief reason tied to directive requirements>",
            "Section 3 is optional and only include it when Update needed = Yes:",
            "#### High-level changes needed\\n- <bullet items describing needed updates at a high level, with Requirement ID(s) when possible>",
            "Section 4 is optional and only include it when concrete existing snippets are available:",
            "#### Existing text snippets vs suggested edits",
            "Under section 4, include a markdown table with exactly two columns:",
            "| Existing NETL text snippet | Suggested edit |",
            "Only add rows for snippets that actually exist in retrieved evidence; if none exist, omit section 4 for that file.",
            "Do not use placeholder text such as 'Search:' in table cells.",
            "Also include an Evidence confidence line (High/Medium/Low) per file.",
            no_files_instruction,
            "## Baseline report from Stage 1/2",
            baseline_report,
            requirements_message,
            context_payload,
            snippet_evidence_summary,
        ]
    )


def render_markdown_pdf_bytes(markdown: str) -> bytes:
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

    def _parse_table_cells(line: str) -> list[str]:
        trimmed = line.strip()
        if trimmed.startswith("|"):
            trimmed = trimmed[1:]
        if trimmed.endswith("|"):
            trimmed = trimmed[:-1]
        return [cell.strip() for cell in trimmed.split("|")]

    def _is_table_separator_row(cells: list[str]) -> bool:
        if not cells:
            return False
        for cell in cells:
            if not re.fullmatch(r":?-{3,}:?", cell.strip()):
                return False
        return True

    def _flush_table(story: list, table_rows: list[list[str]], body_style: ParagraphStyle) -> None:
        if not table_rows:
            return

        max_cols = max(len(row) for row in table_rows)
        normalized_rows = [row + [""] * (max_cols - len(row)) for row in table_rows]
        table_data = [
            [Paragraph(_inline_markdown_to_paragraph_html(cell), body_style) for cell in row]
            for row in normalized_rows
        ]
        repeat_rows = 1 if len(table_data) > 1 else 0
        table = Table(table_data, repeatRows=repeat_rows)
        table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 6))
        table_rows.clear()

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
    table_rows: list[list[str]] = []
    code_lines: list[str] = []
    in_code_block = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            _flush_table(story, table_rows, body_style)
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
            _flush_table(story, table_rows, body_style)
            story.append(Spacer(1, 6))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            _flush_table(story, table_rows, body_style)
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
            _flush_table(story, table_rows, body_style)
            bullet_items.append(bullet_match.group(1))
            continue

        numbered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if numbered_match:
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_table(story, table_rows, body_style)
            numbered_items.append(numbered_match.group(1))
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            _flush_paragraph(story, paragraph_lines, body_style)
            _flush_bullets(story, bullet_items, body_style)
            _flush_numbered(story, numbered_items, body_style)
            cells = _parse_table_cells(stripped)
            if _is_table_separator_row(cells):
                continue
            table_rows.append(cells)
            continue

        _flush_bullets(story, bullet_items, body_style)
        _flush_numbered(story, numbered_items, body_style)
        _flush_table(story, table_rows, body_style)
        paragraph_lines.append(stripped)

    _flush_paragraph(story, paragraph_lines, body_style)
    _flush_bullets(story, bullet_items, body_style)
    _flush_numbered(story, numbered_items, body_style)
    _flush_table(story, table_rows, body_style)
    if in_code_block:
        _flush_code_block(story, code_lines, code_style)

    if not story:
        story = [Paragraph("(Empty markdown report)", body_style)]

    doc.build(story)
    return pdf_buffer.getvalue()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a markdown investigation report for a new directive by searching "
            "procedures/orders in Azure AI Search."
        ),
    )
    parser.add_argument(
        "--directive",
        required=True,
        help="Path to new directive file (.pdf, .txt, .md).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output markdown file path (for example: reports/new_directive_investigation.md)",
    )
    return parser.parse_args()


async def generate_investigation_artifacts(
    directive_path: str,
    max_stage3_files: int = STAGE3_MAX_FILES,
) -> InvestigationArtifacts:
    config = load_config()
    agent_prompt_config = load_agent_prompt_config(config.agent_prompt_file)

    directive_file = _resolve_existing_file(directive_path, "Directive")
    directive_text = _truncate_for_prompt(_load_review_document(directive_file))
    user_message = _build_investigate_user_message(
        directive_name=directive_file.name,
        directive_text=directive_text,
    )
    atomic_requirements = _extract_atomic_requirements(directive_text)
    if not atomic_requirements:
        atomic_requirements = _extract_atomic_requirements(directive_text, max_items=6)
    requirements_for_retrieval = atomic_requirements[:REQUIREMENT_QUERY_LIMIT]

    kernel = Kernel()
    chat_service = create_chat_service(config)
    kernel.add_service(chat_service)

    history = ChatHistory()
    history.add_system_message(agent_prompt_config.prompt)

    settings = AzureChatPromptExecutionSettings()
    search_data_source = build_search_data_source(config)
    requirements_message = _format_atomic_requirements_for_prompt(
        requirements_for_retrieval
    )
    context_payload = ""
    stage3_file_compliance: list[dict[str, object]] = []

    if _dual_index_retrieval_enabled(config):
        _log_stage(1, "Atomic requirement retrieval (targeted stage)")
        requirement_query_results: list[dict[str, object]] = []
        for idx, requirement in enumerate(requirements_for_retrieval, start=1):
            requirement_query = _truncate_for_prompt(requirement, max_chars=800)
            orders_documents = await _search_documents(
                config,
                requirement_query,
                index_name=config.search_orders_index_name,
            )
            procedures_documents = await _search_documents(
                config,
                requirement_query,
                index_name=config.search_procedures_index_name,
            )
            requirement_query_results.append(
                {
                    "requirement_id": f"R{idx}",
                    "requirement_text": requirement,
                    "query": requirement_query,
                    "orders_documents": orders_documents,
                    "procedures_documents": procedures_documents,
                }
            )

        _log_stage(2, "Full-directive chunk retrieval (recall stage)")
        stage2_full_directive_results: list[dict[str, object]] = []
        for query_id, chunk in _build_full_directive_chunks(directive_text):
            query_text = _truncate_for_prompt(chunk, max_chars=1200)
            orders_documents = await _search_documents(
                config,
                query_text,
                index_name=config.search_orders_index_name,
            )
            procedures_documents = await _search_documents(
                config,
                query_text,
                index_name=config.search_procedures_index_name,
            )
            stage2_full_directive_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "orders_documents": orders_documents,
                    "procedures_documents": procedures_documents,
                }
            )

        affected_files = _collect_affected_netl_files(
            query_results=requirement_query_results,
            supplemental_full_directive_results=stage2_full_directive_results,
            dual_index_enabled=True,
        )
        seed_documents_by_file = _build_stage3_seed_documents_by_file(
            query_results=requirement_query_results,
            supplemental_full_directive_results=stage2_full_directive_results,
            dual_index_enabled=True,
        )
        stage3_file_compliance = await _run_file_compliance_stage3(
            config=config,
            directive_text=directive_text,
            requirements_for_retrieval=requirements_for_retrieval,
            affected_files=affected_files,
            dual_index_enabled=True,
            seed_documents_by_file=seed_documents_by_file,
            max_files=max_stage3_files,
        )

        context_payload = _format_requirement_search_context_for_message(
            requirements=requirements_for_retrieval,
            query_results=requirement_query_results,
            config_index_name=config.search_index_name,
            dual_index_enabled=True,
            orders_index_name=config.search_orders_index_name,
            procedures_index_name=config.search_procedures_index_name,
            stage2_full_directive_results=stage2_full_directive_results,
            stage3_file_compliance=stage3_file_compliance,
        )
        history.add_user_message(
            f"{user_message}\n\n{requirements_message}\n\n{context_payload}"
        )
    elif search_data_source:
        _log_stage(1, "Atomic requirement retrieval (targeted stage)")
        requirement_query_results = []
        for idx, requirement in enumerate(requirements_for_retrieval, start=1):
            requirement_query = _truncate_for_prompt(requirement, max_chars=800)
            documents = await _search_documents(config, requirement_query)
            requirement_query_results.append(
                {
                    "requirement_id": f"R{idx}",
                    "requirement_text": requirement,
                    "query": requirement_query,
                    "documents": documents,
                }
            )

        _log_stage(2, "Full-directive chunk retrieval (recall stage)")
        stage2_full_directive_results = []
        for query_id, chunk in _build_full_directive_chunks(directive_text):
            query_text = _truncate_for_prompt(chunk, max_chars=1200)
            documents = await _search_documents(config, query_text)
            stage2_full_directive_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "documents": documents,
                }
            )

        affected_files = _collect_affected_netl_files(
            query_results=requirement_query_results,
            supplemental_full_directive_results=stage2_full_directive_results,
            dual_index_enabled=False,
        )
        seed_documents_by_file = _build_stage3_seed_documents_by_file(
            query_results=requirement_query_results,
            supplemental_full_directive_results=stage2_full_directive_results,
            dual_index_enabled=False,
        )
        stage3_file_compliance = await _run_file_compliance_stage3(
            config=config,
            directive_text=directive_text,
            requirements_for_retrieval=requirements_for_retrieval,
            affected_files=affected_files,
            dual_index_enabled=False,
            seed_documents_by_file=seed_documents_by_file,
            max_files=max_stage3_files,
        )

        context_payload = _format_requirement_search_context_for_message(
            requirements=requirements_for_retrieval,
            query_results=requirement_query_results,
            config_index_name=config.search_index_name,
            dual_index_enabled=False,
            orders_index_name=None,
            procedures_index_name=None,
            stage2_full_directive_results=stage2_full_directive_results,
            stage3_file_compliance=stage3_file_compliance,
        )
        if requirement_query_results:
            history.add_user_message(f"{user_message}\n\n{context_payload}")
            history.add_user_message(
                f"{requirements_message}\n\n"
                "When no explicit section number is present in retrieved evidence, provide an exact search string the reviewer can use in the NETL file.\n"
                "Keep the final report concise and triage-focused for Stage 3.\n"
                "Under Proposed Document Updates (Stage 3 Seed), include one subsection per NETL file with short 'Proposed changes' bullets only."
            )
        else:
            history.add_user_message(user_message)
    else:
        requirements_message = _format_atomic_requirements_for_prompt(
            requirements_for_retrieval
        )
        history.add_user_message(f"{user_message}\n\n{requirements_message}")

    model_label = config.deployment_name or "unknown"
    _log_stage(4, f"LLM analysis running (deployment/model: {model_label})")

    response = await chat_service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )
    report_text = sanitize_investigation_markdown(str(response))

    stage3_history = ChatHistory()
    stage3_history.add_system_message(agent_prompt_config.prompt)
    snippet_evidence_summary = _build_stage3_snippet_evidence_summary(
        stage3_file_compliance=stage3_file_compliance,
    )
    stage3_history.add_user_message(
        _build_stage3_delta_user_message(
            directive_name=directive_file.name,
            baseline_report=report_text,
            requirements_message=requirements_message,
            context_payload=context_payload,
            snippet_evidence_summary=snippet_evidence_summary,
            has_stage3_files=bool(stage3_file_compliance),
        )
    )
    _log_stage(5, f"LLM Stage 3 refinement running (deployment/model: {model_label})")
    stage3_response = await chat_service.get_chat_message_content(
        chat_history=stage3_history,
        settings=settings,
        kernel=kernel,
    )
    stage3_report = _apply_update_needed_headers_and_sort(str(stage3_response))

    requirements_review_markdown = _format_atomic_requirements_markdown(
        directive_name=directive_file.name,
        requirements=atomic_requirements,
    )
    return InvestigationArtifacts(
        report_markdown=report_text,
        atomic_requirements=atomic_requirements,
        requirements_review_markdown=requirements_review_markdown,
        stage3_report_markdown=stage3_report,
    )


def generate_investigation_artifacts_sync(
    directive_path: str,
    max_stage3_files: int = STAGE3_MAX_FILES,
) -> InvestigationArtifacts:
    return asyncio.run(
        generate_investigation_artifacts(
            directive_path=directive_path,
            max_stage3_files=max_stage3_files,
        )
    )


def _write_output(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _write_pdf_output(output_path: Path, content: str) -> None:
    """Write a PDF from markdown content using shared markdown rendering."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(render_markdown_pdf_bytes(content))


def main() -> None:
    args = _parse_args()

    artifacts = generate_investigation_artifacts_sync(directive_path=args.directive)
    stage3_report = _apply_update_needed_headers_and_sort(
        artifacts.stage3_report_markdown
    )

    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    markdown_output = output_path.resolve()
    pdf_output = markdown_output.with_suffix(".pdf")
    requirements_output = markdown_output.with_name(
        f"{markdown_output.stem}_stage1_atomic_requirements.md"
    )
    stage3_output = markdown_output.with_name(
        f"{markdown_output.stem}_stage3_updates.md"
    )
    stage3_pdf_output = markdown_output.with_name(
        f"{markdown_output.stem}_stage3_updates.pdf"
    )

    _write_output(markdown_output, artifacts.report_markdown)
    _write_pdf_output(pdf_output, artifacts.report_markdown)
    _write_output(requirements_output, artifacts.requirements_review_markdown)
    _write_output(stage3_output, stage3_report)
    _write_pdf_output(stage3_pdf_output, stage3_report)
    print(f"Investigation markdown report written to: {markdown_output}")
    print(f"Investigation PDF report written to: {pdf_output}")
    print(f"Atomic requirements review written to: {requirements_output}")
    print(f"Stage 3 compliance updates written to: {stage3_output}")
    print(f"Stage 3 compliance PDF written to: {stage3_pdf_output}")


if __name__ == "__main__":
    main()
