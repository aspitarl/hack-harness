"""Run one-shot new-directive investigation and write markdown output to a file."""

import argparse
import asyncio
from io import BytesIO
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import httpx
from azure.identity import DefaultAzureCredential
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
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


def _log_stage(stage_number: int, message: str) -> None:
    print(f"[Stage {stage_number}] {message}", flush=True)


def _extract_search_doc_source_ref(document: dict[str, object]) -> str | None:
    source_fields = [
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
    for field in source_fields:
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
    normalized_target = Path(file_name).name.lower()
    filtered: list[dict[str, object]] = []

    for rank, document in enumerate(documents, start=1):
        candidate = Path(_extract_search_doc_name(document, rank)).name.lower()
        if candidate == normalized_target:
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


async def _run_file_compliance_stage3(
    config,
    directive_text: str,
    requirements_for_retrieval: list[str],
    affected_files: list[str],
    dual_index_enabled: bool,
) -> list[dict[str, object]]:
    if not affected_files:
        return []

    requirement_seed = " ".join(requirements_for_retrieval[:3])
    directive_seed = _truncate_for_prompt(directive_text, max_chars=600)
    results: list[dict[str, object]] = []
    files_to_check = affected_files[:STAGE3_MAX_FILES]
    _log_stage(3, "Per-file compliance scan")

    for file_name in tqdm(
        files_to_check,
        desc="Stage 3: checking files",
        unit="file",
    ):
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
                )
                retry_procedures = await _search_documents(
                    config,
                    retry_query,
                    index_name=config.search_procedures_index_name,
                )
                filtered_orders = _filter_docs_for_file_name(retry_orders, file_name)
                filtered_procedures = _filter_docs_for_file_name(retry_procedures, file_name)

            if not filtered_orders and not filtered_procedures:
                continue

            docs_for_file = filtered_orders + filtered_procedures
            _log_stage(
                3,
                (
                    f"{file_name}: matched orders={len(filtered_orders)} "
                    f"procedures={len(filtered_procedures)}"
                ),
            )
            source_ref = next(
                (
                    _extract_search_doc_source_ref(doc)
                    for doc in docs_for_file
                    if _extract_search_doc_source_ref(doc)
                ),
                None,
            )
            _log_stage(3, f"{file_name}: source_ref={'present' if source_ref else 'missing'}")
            full_document_text, full_document_error = await _load_full_document_text_for_stage3(
                source_ref
            )
            if full_document_text:
                _log_stage(
                    3,
                    f"{file_name}: full document loaded ({len(full_document_text)} chars)",
                )
            else:
                _log_stage(3, f"{file_name}: full document unavailable ({full_document_error})")

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
                retry_documents = await _search_documents(config, retry_query)
                filtered_documents = _filter_docs_for_file_name(retry_documents, file_name)

            if not filtered_documents:
                continue

            _log_stage(3, f"{file_name}: matched documents={len(filtered_documents)}")
            source_ref = next(
                (
                    _extract_search_doc_source_ref(doc)
                    for doc in filtered_documents
                    if _extract_search_doc_source_ref(doc)
                ),
                None,
            )
            _log_stage(3, f"{file_name}: source_ref={'present' if source_ref else 'missing'}")
            full_document_text, full_document_error = await _load_full_document_text_for_stage3(
                source_ref
            )
            if full_document_text:
                _log_stage(
                    3,
                    f"{file_name}: full document loaded ({len(full_document_text)} chars)",
                )
            else:
                _log_stage(3, f"{file_name}: full document unavailable ({full_document_error})")

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
            snippets.append(normalized[:280])
            if len(snippets) >= 3:
                break

        if not snippets:
            continue

        lines.append(f"- {file_name}")
        for snippet in snippets:
            lines.append(f"  - \"{snippet}\"")

    return "\n".join(lines)


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
) -> str:
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
            "If no files are available in Stage 3 evidence, output exactly: 'No Stage 3 files identified.'",
            "## Baseline report from Stage 1/2",
            baseline_report,
            requirements_message,
            context_payload,
            snippet_evidence_summary,
        ]
    )


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


async def _generate_report(
    directive_path: str,
) -> tuple[str, list[str], str, str]:
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
        stage3_file_compliance = await _run_file_compliance_stage3(
            config=config,
            directive_text=directive_text,
            requirements_for_retrieval=requirements_for_retrieval,
            affected_files=affected_files,
            dual_index_enabled=True,
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
        stage3_file_compliance = await _run_file_compliance_stage3(
            config=config,
            directive_text=directive_text,
            requirements_for_retrieval=requirements_for_retrieval,
            affected_files=affected_files,
            dual_index_enabled=False,
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
        )
    )
    stage3_response = await chat_service.get_chat_message_content(
        chat_history=stage3_history,
        settings=settings,
        kernel=kernel,
    )

    requirements_review_markdown = _format_atomic_requirements_markdown(
        directive_name=directive_file.name,
        requirements=atomic_requirements,
    )
    return report_text, atomic_requirements, requirements_review_markdown, str(stage3_response)


def _write_output(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _write_pdf_output(output_path: Path, content: str) -> None:
    """Write a simple text-rendered PDF from markdown content."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(output_path), pagesize=letter)
    page_width, page_height = letter
    left_margin = 0.75 * inch
    top_margin = 0.75 * inch
    bottom_margin = 0.75 * inch
    line_height = 14
    max_chars_per_line = 100

    y = page_height - top_margin
    lines = content.splitlines() or [""]

    for raw_line in lines:
        line = escape(raw_line)
        if not line:
            y -= line_height
            if y <= bottom_margin:
                pdf.showPage()
                y = page_height - top_margin
            continue

        wrapped_chunks = [
            line[index : index + max_chars_per_line]
            for index in range(0, len(line), max_chars_per_line)
        ]

        for chunk in wrapped_chunks:
            pdf.drawString(left_margin, y, chunk)
            y -= line_height
            if y <= bottom_margin:
                pdf.showPage()
                y = page_height - top_margin

    pdf.save()


def main() -> None:
    args = _parse_args()

    report, _, requirements_review, stage3_report = asyncio.run(
        _generate_report(directive_path=args.directive)
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

    _write_output(markdown_output, report)
    _write_pdf_output(pdf_output, report)
    _write_output(requirements_output, requirements_review)
    _write_output(stage3_output, stage3_report)
    print(f"Investigation markdown report written to: {markdown_output}")
    print(f"Investigation PDF report written to: {pdf_output}")
    print(f"Atomic requirements review written to: {requirements_output}")
    print(f"Stage 3 compliance updates written to: {stage3_output}")


if __name__ == "__main__":
    main()
