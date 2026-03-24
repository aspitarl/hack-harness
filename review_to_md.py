"""Run a one-shot DOE directive review and write markdown output to a file."""

import argparse
import asyncio
from pathlib import Path

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory

from chat_cli import (
    _build_review_user_message,
    _load_review_document,
    _resolve_existing_file,
    _truncate_for_prompt,
    build_search_data_source,
    create_chat_service,
    load_agent_prompt_config,
    load_config,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a DOE directive review report from input files and save as markdown.",
    )
    parser.add_argument(
        "--requirements",
        required=True,
        help="Path to DOE requirements file (.pdf, .txt, .md)",
    )
    parser.add_argument(
        "--draft",
        action="append",
        required=True,
        help="Path to draft directive file (.pdf, .txt, .md). Repeat for multiple drafts.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output markdown file path (for example: reports/review.md)",
    )
    return parser.parse_args()


async def _generate_review(requirements_path: str, draft_paths: list[str]) -> str:
    config = load_config()
    agent_prompt_config = load_agent_prompt_config(config.agent_prompt_file)

    requirements_file = _resolve_existing_file(requirements_path, "Requirements")
    requirements_text = _truncate_for_prompt(_load_review_document(requirements_file))

    draft_docs: list[tuple[str, str]] = []
    for draft_path in draft_paths:
        resolved = _resolve_existing_file(draft_path, "Draft")
        draft_text = _truncate_for_prompt(_load_review_document(resolved))
        draft_docs.append((resolved.name, draft_text))

    user_message = _build_review_user_message(
        requirements_text=requirements_text,
        drafts=draft_docs,
    )

    kernel = Kernel()
    chat_service = create_chat_service(config)
    kernel.add_service(chat_service)

    history = ChatHistory()
    history.add_system_message(agent_prompt_config.prompt)
    for example in agent_prompt_config.examples:
        history.add_user_message(example.question)
        history.add_assistant_message(example.answer)
    history.add_user_message(user_message)

    settings = AzureChatPromptExecutionSettings(temperature=0.3)
    search_data_source = build_search_data_source(config)
    if search_data_source:
        settings.extra_body = {"data_sources": [search_data_source]}

    response = await chat_service.get_chat_message_content(
        chat_history=history,
        settings=settings,
        kernel=kernel,
    )
    return str(response)


def _write_output(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()

    report = asyncio.run(_generate_review(args.requirements, args.draft))

    output_path = Path(args.out).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    _write_output(output_path.resolve(), report)
    print(f"Review report written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
