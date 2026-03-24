"""Run one-shot new-directive investigation and write markdown output to a file."""

import argparse
import asyncio
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

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
)


REQUIREMENT_QUERY_LIMIT = 8
FULL_DIRECTIVE_SECOND_PASS_MAX_CHUNKS = 4
FULL_DIRECTIVE_SECOND_PASS_CHUNK_SIZE = 2200
FULL_DIRECTIVE_SECOND_PASS_CHUNK_OVERLAP = 300


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
    max_chunks: int = FULL_DIRECTIVE_SECOND_PASS_MAX_CHUNKS,
    chunk_size: int = FULL_DIRECTIVE_SECOND_PASS_CHUNK_SIZE,
    overlap: int = FULL_DIRECTIVE_SECOND_PASS_CHUNK_OVERLAP,
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


def _format_requirement_search_context_for_message(
    requirements: list[str],
    query_results: list[dict[str, object]],
    config_index_name: str | None,
    dual_index_enabled: bool,
    orders_index_name: str | None,
    procedures_index_name: str | None,
    supplemental_full_directive_results: list[dict[str, object]] | None = None,
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
    if supplemental_full_directive_results:
        for result in supplemental_full_directive_results:
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

    payload = {
        "requirements": payload_requirements,
        "full_directive_second_pass": supplemental_payload,
    }

    return (
        "Use this requirement-scoped retrieval context and cite facts conservatively.\n"
        "Treat requirement-scoped evidence as primary. Use the full-directive second pass only to identify gaps or missing obligations not represented in requirement-scoped evidence.\n"
        "For each recommendation, reference the requirement ID (R#) and point to a NETL file section "
        "or a precise fallback string to search for in that file.\n"
        "Documents to investigate by requirement:\n"
        f"{docs_block}\n\n"
        "Requirement-scoped grounding payload:\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
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
) -> tuple[str, list[str], str]:
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
    if _dual_index_retrieval_enabled(config):
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

        supplemental_full_directive_results: list[dict[str, object]] = []
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
            supplemental_full_directive_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "orders_documents": orders_documents,
                    "procedures_documents": procedures_documents,
                }
            )

        context_payload = _format_requirement_search_context_for_message(
            requirements=requirements_for_retrieval,
            query_results=requirement_query_results,
            config_index_name=config.search_index_name,
            dual_index_enabled=True,
            orders_index_name=config.search_orders_index_name,
            procedures_index_name=config.search_procedures_index_name,
            supplemental_full_directive_results=supplemental_full_directive_results,
        )
        requirements_message = _format_atomic_requirements_for_prompt(
            requirements_for_retrieval
        )
        history.add_user_message(
            f"{user_message}\n\n{requirements_message}\n\n{context_payload}"
        )
    elif search_data_source:
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

        supplemental_full_directive_results = []
        for query_id, chunk in _build_full_directive_chunks(directive_text):
            query_text = _truncate_for_prompt(chunk, max_chars=1200)
            documents = await _search_documents(config, query_text)
            supplemental_full_directive_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "documents": documents,
                }
            )

        context_payload = _format_requirement_search_context_for_message(
            requirements=requirements_for_retrieval,
            query_results=requirement_query_results,
            config_index_name=config.search_index_name,
            dual_index_enabled=False,
            orders_index_name=None,
            procedures_index_name=None,
            supplemental_full_directive_results=supplemental_full_directive_results,
        )
        if requirement_query_results:
            requirements_message = _format_atomic_requirements_for_prompt(
                requirements_for_retrieval
            )
            history.add_user_message(f"{user_message}\n\n{context_payload}")
            history.add_user_message(
                f"{requirements_message}\n\n"
                "When no explicit section number is present in retrieved evidence, provide an exact search string the reviewer can use in the NETL file."
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
    requirements_review_markdown = _format_atomic_requirements_markdown(
        directive_name=directive_file.name,
        requirements=atomic_requirements,
    )
    return str(response), atomic_requirements, requirements_review_markdown


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

    report, _, requirements_review = asyncio.run(
        _generate_report(directive_path=args.directive)
    )

    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    markdown_output = output_path.resolve()
    pdf_output = markdown_output.with_suffix(".pdf")
    requirements_output = markdown_output.with_name(
        f"{markdown_output.stem}_atomic_requirements.md"
    )

    _write_output(markdown_output, report)
    _write_pdf_output(pdf_output, report)
    _write_output(requirements_output, requirements_review)
    print(f"Investigation markdown report written to: {markdown_output}")
    print(f"Investigation PDF report written to: {pdf_output}")
    print(f"Atomic requirements review written to: {requirements_output}")


if __name__ == "__main__":
    main()
