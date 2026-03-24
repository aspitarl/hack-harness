import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory


@dataclass
class AppConfig:
    provider: str
    deployment_name: str
    endpoint: str
    api_key: str | None
    api_version: str | None
    search_endpoint: str | None
    search_api_key: str | None
    search_index_name: str | None
    search_semantic_configuration: str | None
    search_query_type: str
    search_in_scope: bool
    search_strictness: int
    search_top_n_documents: int


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value if value else None


def _api_version_env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.lower() == "v1":
        return default
    return value


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean value "
        "(true/false, 1/0, yes/no)."
    )


def _to_account_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.rstrip("/")


def load_config() -> AppConfig:
    load_dotenv(override=False)

    search_endpoint = _optional_env("AZURE_AI_SEARCH_ENDPOINT")
    search_index_name = _optional_env("AZURE_AI_SEARCH_INDEX_NAME")

    provider = os.getenv("CHAT_PROVIDER", "azure_openai").strip().lower()
    if provider not in {"azure_openai", "foundry"}:
        raise ValueError("CHAT_PROVIDER must be either 'azure_openai' or 'foundry'.")

    if bool(search_endpoint) != bool(search_index_name):
        raise ValueError(
            "Set both AZURE_AI_SEARCH_ENDPOINT and AZURE_AI_SEARCH_INDEX_NAME "
            "to enable Azure AI Search grounding."
        )

    if provider == "azure_openai":
        return AppConfig(
            provider=provider,
            deployment_name=_required_env("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            endpoint=_required_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_optional_env("AZURE_OPENAI_API_KEY"),
            api_version=_api_version_env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            search_endpoint=search_endpoint,
            search_api_key=_optional_env("AZURE_AI_SEARCH_API_KEY"),
            search_index_name=search_index_name,
            search_semantic_configuration=_optional_env("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION"),
            search_query_type=os.getenv("AZURE_AI_SEARCH_QUERY_TYPE", "semantic").strip().lower(),
            search_in_scope=_bool_env("AZURE_AI_SEARCH_IN_SCOPE", True),
            search_strictness=_int_env("AZURE_AI_SEARCH_STRICTNESS", 3),
            search_top_n_documents=_int_env("AZURE_AI_SEARCH_TOP_N_DOCUMENTS", 5),
        )

    # Foundry project endpoints are OpenAI-compatible for chat deployments.
    return AppConfig(
        provider=provider,
        deployment_name=_required_env("FOUNDRY_CHAT_DEPLOYMENT"),
        endpoint=_required_env("FOUNDRY_PROJECT_ENDPOINT"),
        api_key=_optional_env("FOUNDRY_API_KEY"),
        api_version=_api_version_env("FOUNDRY_API_VERSION", "2024-10-21"),
        search_endpoint=search_endpoint,
        search_api_key=_optional_env("AZURE_AI_SEARCH_API_KEY"),
        search_index_name=search_index_name,
        search_semantic_configuration=_optional_env("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION"),
        search_query_type=os.getenv("AZURE_AI_SEARCH_QUERY_TYPE", "semantic").strip().lower(),
        search_in_scope=_bool_env("AZURE_AI_SEARCH_IN_SCOPE", True),
        search_strictness=_int_env("AZURE_AI_SEARCH_STRICTNESS", 3),
        search_top_n_documents=_int_env("AZURE_AI_SEARCH_TOP_N_DOCUMENTS", 5),
    )


def build_search_data_source(config: AppConfig) -> dict[str, object] | None:
    if not config.search_endpoint or not config.search_index_name:
        return None

    parameters: dict[str, object] = {
        "endpoint": config.search_endpoint,
        "index_name": config.search_index_name,
        "query_type": config.search_query_type,
        "in_scope": config.search_in_scope,
        "strictness": config.search_strictness,
        "top_n_documents": config.search_top_n_documents,
    }

    if config.search_semantic_configuration:
        parameters["semantic_configuration"] = config.search_semantic_configuration

    if config.search_api_key:
        parameters["authentication"] = {"type": "api_key", "key": config.search_api_key}
    else:
        parameters["authentication"] = {"type": "system_assigned_managed_identity"}

    return {"type": "azure_search", "parameters": parameters}


def create_chat_service(config: AppConfig) -> ChatCompletionClientBase:
    # Foundry values may be supplied as project endpoints. SK chat completion
    # expects the account endpoint shape.
    if config.provider == "foundry":
        endpoint = _to_account_endpoint(config.endpoint)

        if config.api_key:
            return AzureChatCompletion(
                service_id="chat",
                endpoint=endpoint,
                deployment_name=config.deployment_name,
                api_key=config.api_key,
                api_version=config.api_version,
            )

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://ai.azure.com/.default")
        return AzureChatCompletion(
            service_id="chat",
            endpoint=endpoint,
            deployment_name=config.deployment_name,
            ad_token_provider=token_provider,
            api_version=config.api_version,
        )

    # Azure OpenAI endpoints use date-based api-version values.
    if config.api_key:
        return AzureChatCompletion(
            service_id="chat",
            endpoint=config.endpoint,
            deployment_name=config.deployment_name,
            api_key=config.api_key,
            api_version=config.api_version,
        )

    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
    return AzureChatCompletion(
        service_id="chat",
        endpoint=config.endpoint,
        deployment_name=config.deployment_name,
        ad_token_provider=token_provider,
        api_version=config.api_version,
    )


async def run_chat() -> None:
    config = load_config()

    kernel = Kernel()
    chat_service = create_chat_service(config)
    kernel.add_service(chat_service)
    history = ChatHistory()
    settings = AzureChatPromptExecutionSettings(temperature=0.7)
    search_data_source = build_search_data_source(config)
    if search_data_source:
        # Azure OpenAI/Foundry "On Your Data" payload.
        settings.extra_body = {"data_sources": [search_data_source]}

    kb = KeyBindings()
    should_exit = False

    @kb.add("c-c")
    def _(event) -> None:  # noqa: ANN001
        nonlocal should_exit
        should_exit = True
        event.app.exit(result="")

    @kb.add("c-x")
    def _(event) -> None:  # noqa: ANN001
        nonlocal should_exit
        should_exit = True
        event.app.exit(result="")

    session = PromptSession("you> ", key_bindings=kb)

    print("Semantic Kernel chat started.")
    if config.api_key:
        print("Auth mode: API key")
    else:
        print("Auth mode: Azure Default Credential")
    if search_data_source:
        print(f"Grounding: Azure AI Search enabled (index: {config.search_index_name})")
    else:
        print("Grounding: disabled")
    print("Press Ctrl+X or Ctrl+C to exit.\n")

    with patch_stdout():
        while True:
            if should_exit:
                break

            try:
                user_text = (await session.prompt_async()).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if should_exit:
                break

            if not user_text:
                continue

            history.add_user_message(user_text)

            try:
                response = await chat_service.get_chat_message_content(
                    chat_history=history,
                    settings=settings,
                    kernel=kernel,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"assistant> [error] {exc}")
                continue

            assistant_text = str(response)
            history.add_assistant_message(assistant_text)
            print(f"assistant> {assistant_text}\n")

    print("Exiting chat.")


if __name__ == "__main__":
    asyncio.run(run_chat())
