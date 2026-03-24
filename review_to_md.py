"""Run one-shot new-directive investigation and write markdown output to a file."""

import argparse
import asyncio
from pathlib import Path
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
    _format_dual_search_context_for_message,
    _format_search_context_for_message,
    _load_review_document,
    _resolve_existing_file,
    _search_documents,
    _truncate_for_prompt,
    build_search_data_source,
    create_chat_service,
    load_agent_prompt_config,
    load_config,
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
) -> str:
    config = load_config()
    agent_prompt_config = load_agent_prompt_config(config.agent_prompt_file)

    directive_file = _resolve_existing_file(directive_path, "Directive")
    directive_text = _truncate_for_prompt(_load_review_document(directive_file))
    user_message = _build_investigate_user_message(
        directive_name=directive_file.name,
        directive_text=directive_text,
    )
    search_query_text = _truncate_for_prompt(directive_text, max_chars=8000)

    kernel = Kernel()
    chat_service = create_chat_service(config)
    kernel.add_service(chat_service)

    history = ChatHistory()
    history.add_system_message(agent_prompt_config.prompt)

    settings = AzureChatPromptExecutionSettings()
    search_data_source = build_search_data_source(config)
    if _dual_index_retrieval_enabled(config):
        orders_documents = await _search_documents(
            config,
            search_query_text,
            index_name=config.search_orders_index_name,
        )
        procedures_documents = await _search_documents(
            config,
            search_query_text,
            index_name=config.search_procedures_index_name,
        )
        context_payload = _format_dual_search_context_for_message(
            query=search_query_text,
            orders_index_name=str(config.search_orders_index_name),
            procedures_index_name=str(config.search_procedures_index_name),
            orders_documents=orders_documents,
            procedures_documents=procedures_documents,
        )
        history.add_user_message(f"{user_message}\n\n{context_payload}")
    elif search_data_source:
        documents = await _search_documents(config, search_query_text)
        if documents:
            context_payload = _format_search_context_for_message(
                search_data_source=search_data_source,
                query=search_query_text,
                documents=documents,
            )
            history.add_user_message(f"{user_message}\n\n{context_payload}")
        else:
            history.add_user_message(user_message)
    else:
        history.add_user_message(user_message)

    response = await chat_service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )
    return str(response)


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

    report = asyncio.run(_generate_report(directive_path=args.directive))

    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    markdown_output = output_path.resolve()
    pdf_output = markdown_output.with_suffix(".pdf")

    _write_output(markdown_output, report)
    _write_pdf_output(pdf_output, report)
    print(f"Investigation markdown report written to: {markdown_output}")
    print(f"Investigation PDF report written to: {pdf_output}")


if __name__ == "__main__":
    main()
