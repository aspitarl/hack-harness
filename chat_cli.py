"""Interactive Semantic Kernel chat CLI with optional MCP/OpenAPI tooling.

This module is designed as both an executable chat client and a learning sample.
It demonstrates how to combine:

1. Azure OpenAI or Azure AI Foundry chat completions via Semantic Kernel.
2. Optional Azure AI Search grounding ("On Your Data").
3. Optional MCP-like tool integration from an OpenAPI 3.x document.
4. Optional automatic tool invocation via Semantic Kernel function calling.

High-level flow:

1. Load runtime configuration from environment variables.
2. Build the chat completion service (API key or managed identity auth).
3. Build initial chat history from an agent prompt YAML file.
4. Optionally register OpenAPI operations as Semantic Kernel plugin functions.
5. Run an interactive REPL loop for user messages, slash commands, and replies.
"""

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
import httpx
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.openapi_plugin import OpenAPIFunctionExecutionParameters
from semantic_kernel.contents.chat_history import ChatHistory
import yaml


@dataclass
class AppConfig:
    """Runtime configuration loaded from .env/environment variables.

    The fields are grouped around three concerns:

    1. Chat provider settings (Azure OpenAI or Foundry).
    2. Optional Azure AI Search grounding settings.
    3. Optional MCP/OpenAPI tool settings.
    """

    provider: str
    deployment_name: str
    endpoint: str
    api_key: str | None
    api_version: str | None
    search_endpoint: str | None
    search_api_key: str | None
    search_api_version: str
    search_index_name: str | None
    search_semantic_configuration: str | None
    search_query_type: str
    search_in_scope: bool
    search_strictness: int
    search_top_n_documents: int
    agent_prompt_file: str
    mcp_openapi_spec_url: str | None
    mcp_base_url: str | None
    mcp_timeout_seconds: int
    mcp_default_headers: dict[str, str]
    debug_chat_messages: bool


@dataclass
class AgentPromptExample:
    """Single few-shot example pair from the agent YAML configuration."""

    question: str
    answer: str


@dataclass
class AgentPromptConfig:
    """System prompt plus optional few-shot examples for chat initialization."""

    prompt: str
    examples: list[AgentPromptExample]


@dataclass
class OpenAPIParameter:
    """Subset of OpenAPI parameter metadata used for tool invocation."""

    name: str
    location: str
    required: bool
    description: str


@dataclass
class OpenAPITool:
    """Parsed OpenAPI operation represented as a callable tool descriptor."""

    name: str
    description: str
    method: str
    path: str
    parameters: list[OpenAPIParameter]
    request_body_required: bool


def _required_env(name: str) -> str:
    """Read and validate a required environment variable.

    Raises:
        ValueError: If the value is missing or empty after trimming whitespace.
    """

    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    """Read an optional environment variable and normalize empty -> None."""

    value = os.getenv(name, "").strip()
    return value if value else None


def _api_version_env(name: str, default: str) -> str:
    """Read API version and map blank/v1 placeholder values to a default."""

    value = os.getenv(name, "").strip()
    if not value or value.lower() == "v1":
        return default
    return value


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable with validation and fallback."""

    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _json_object_env(name: str) -> dict[str, str]:
    """Read a JSON object from environment and coerce keys/values to strings.

    This helper is used for MCP default headers so callers can supply values like:
    {"x-api-key":"..."} directly in .env.
    """

    value = os.getenv(name, "").strip()
    if not value:
        return {}

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Environment variable {name} must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Environment variable {name} must be a JSON object.")

    normalized: dict[str, str] = {}
    for k, v in parsed.items():
        normalized[str(k)] = str(v)
    return normalized


def _bool_env(name: str, default: bool) -> bool:
    """Read a boolean environment variable supporting common truthy/falsy tokens."""

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
    """Reduce a URL to scheme + host, dropping any path/query/fragment.

    Example:
        https://host/api/projects/foo -> https://host
    """

    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.rstrip("/")


def _resolve_search_endpoint(endpoint: str | None, service_name: str | None) -> str | None:
    """Resolve Azure AI Search endpoint from explicit endpoint or service name.

    This mirrors the behavior in testseach.py where service name maps to:
    https://<service>.search.windows.net
    """

    if endpoint and service_name:
        raise ValueError(
            "Set either AZURE_AI_SEARCH_ENDPOINT or AZURE_AI_SEARCH_SERVICE_NAME, not both."
        )
    if endpoint:
        return endpoint.rstrip("/")
    if service_name:
        return f"https://{service_name}.search.windows.net"
    return None


def load_config() -> AppConfig:
    """Load and validate application settings from environment variables.

    Returns:
        AppConfig: Normalized runtime settings for chat, grounding, and MCP.

    Raises:
        ValueError: If provider is invalid or required environment values are missing.
    """

    load_dotenv(override=False)

    search_endpoint = _resolve_search_endpoint(
        _optional_env("AZURE_AI_SEARCH_ENDPOINT"),
        _optional_env("AZURE_AI_SEARCH_SERVICE_NAME"),
    )
    search_index_name = _optional_env("AZURE_AI_SEARCH_INDEX_NAME")
    mcp_openapi_spec_url = _optional_env("MCP_OPENAPI_SPEC_URL")
    mcp_base_url = _optional_env("MCP_BASE_URL")
    mcp_timeout_seconds = _int_env("MCP_TIMEOUT_SECONDS", 30)
    mcp_default_headers = _json_object_env("MCP_DEFAULT_HEADERS")

    provider = os.getenv("CHAT_PROVIDER", "azure_openai").strip().lower()
    if provider not in {"azure_openai", "foundry"}:
        raise ValueError("CHAT_PROVIDER must be either 'azure_openai' or 'foundry'.")

    if bool(search_endpoint) != bool(search_index_name):
        raise ValueError(
            "Set AZURE_AI_SEARCH_INDEX_NAME and one of "
            "AZURE_AI_SEARCH_ENDPOINT/AZURE_AI_SEARCH_SERVICE_NAME "
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
            search_api_version=_api_version_env("AZURE_AI_SEARCH_API_VERSION", "2024-07-01"),
            search_index_name=search_index_name,
            search_semantic_configuration=_optional_env("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION"),
            search_query_type=os.getenv("AZURE_AI_SEARCH_QUERY_TYPE", "semantic").strip().lower(),
            search_in_scope=_bool_env("AZURE_AI_SEARCH_IN_SCOPE", True),
            search_strictness=_int_env("AZURE_AI_SEARCH_STRICTNESS", 3),
            search_top_n_documents=_int_env("AZURE_AI_SEARCH_TOP_N_DOCUMENTS", 5),
            agent_prompt_file=os.getenv("AGENT_PROMPT_FILE", "agents/default.yaml").strip(),
            mcp_openapi_spec_url=mcp_openapi_spec_url,
            mcp_base_url=mcp_base_url,
            mcp_timeout_seconds=mcp_timeout_seconds,
            mcp_default_headers=mcp_default_headers,
            debug_chat_messages=_bool_env("DEBUG_CHAT_MESSAGES", False),
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
        search_api_version=_api_version_env("AZURE_AI_SEARCH_API_VERSION", "2024-07-01"),
        search_index_name=search_index_name,
        search_semantic_configuration=_optional_env("AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION"),
        search_query_type=os.getenv("AZURE_AI_SEARCH_QUERY_TYPE", "semantic").strip().lower(),
        search_in_scope=_bool_env("AZURE_AI_SEARCH_IN_SCOPE", True),
        search_strictness=_int_env("AZURE_AI_SEARCH_STRICTNESS", 3),
        search_top_n_documents=_int_env("AZURE_AI_SEARCH_TOP_N_DOCUMENTS", 5),
        agent_prompt_file=os.getenv("AGENT_PROMPT_FILE", "agents/default.yaml").strip(),
        mcp_openapi_spec_url=mcp_openapi_spec_url,
        mcp_base_url=mcp_base_url,
        mcp_timeout_seconds=mcp_timeout_seconds,
        mcp_default_headers=mcp_default_headers,
        debug_chat_messages=_bool_env("DEBUG_CHAT_MESSAGES", False),
    )


def _render_chat_history_for_debug(history: ChatHistory) -> list[dict[str, str]]:
    """Convert chat history to a compact role/content list for debug logging."""

    rendered: list[dict[str, str]] = []
    for message in history.messages:
        role_value = getattr(message, "role", "unknown")
        role = str(role_value)

        content_value = getattr(message, "content", None)
        if isinstance(content_value, str):
            content = content_value
        elif content_value is None:
            content = ""
        else:
            content = str(content_value)

        rendered.append({"role": role, "content": content})

    return rendered


def _normalize_tool_name(method: str, path: str) -> str:
    """Create a safe fallback tool name from HTTP method + path."""

    raw_name = f"{method}_{path}"
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw_name).strip("_")


def _load_openapi_document(spec_url: str, timeout_seconds: int) -> dict[str, object]:
    """Download and parse an OpenAPI 3.x document from a URL.

    The parser supports both JSON and YAML payloads.

    Raises:
        ValueError: If the document cannot be loaded, parsed, or is not OpenAPI 3.x.
    """

    try:
        with urlopen(spec_url, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except URLError as exc:
        raise ValueError(f"Unable to load MCP OpenAPI spec from {spec_url}: {exc}") from exc

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(payload)

    if not isinstance(parsed, dict):
        raise ValueError("MCP OpenAPI spec must be a JSON or YAML object.")

    version = str(parsed.get("openapi", "")).strip()
    if not version.startswith("3"):
        raise ValueError("MCP OpenAPI spec must be version 3.x.")

    return parsed


def _resolve_openapi_base_url(
    spec: dict[str, object],
    spec_url: str,
    base_url_override: str | None,
) -> str:
    """Resolve the effective base URL for OpenAPI operation calls.

    Resolution order:
    1. MCP_BASE_URL override (if provided).
    2. First non-localhost OpenAPI server URL.
    3. Spec origin (scheme + host from spec URL).
    """

    if base_url_override:
        return base_url_override.rstrip("/")

    origin = _to_account_endpoint(spec_url)
    servers = spec.get("servers", [])
    if isinstance(servers, list):
        for server in servers:
            if not isinstance(server, dict):
                continue
            raw_url = str(server.get("url", "")).strip()
            if not raw_url:
                continue

            if raw_url.startswith("/"):
                return urljoin(origin + "/", raw_url.lstrip("/")).rstrip("/")

            parsed = urlparse(raw_url)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                host = parsed.hostname or ""
                if host not in {"localhost", "127.0.0.1"}:
                    return raw_url.rstrip("/")

    return origin


class OpenAPIMCPInterface:
    """Simple MCP-like adapter around an OpenAPI specification.

    Purpose:
    - Expose OpenAPI operations as named tools.
    - Execute tool calls using a tool-name + JSON-arguments shape.

    Note:
    This class is used for manual slash-command invocation (/mcp ...). Automatic
    invocation is handled separately through Semantic Kernel plugin registration.
    """

    def __init__(
        self,
        spec_url: str,
        base_url_override: str | None,
        timeout_seconds: int,
        default_headers: dict[str, str],
    ) -> None:
        """Initialize and load the OpenAPI tool catalog."""

        self.spec_url = spec_url
        self.base_url_override = base_url_override
        self.timeout_seconds = timeout_seconds
        self.default_headers = default_headers
        self.base_url = ""
        self._tools: dict[str, OpenAPITool] = {}
        self.reload()

    def reload(self) -> None:
        """Reload OpenAPI spec and refresh parsed tools/base URL."""

        spec = _load_openapi_document(self.spec_url, self.timeout_seconds)
        self.base_url = _resolve_openapi_base_url(spec, self.spec_url, self.base_url_override)
        self._tools = self._parse_tools(spec)

    def list_tools(self) -> list[OpenAPITool]:
        """Return tools sorted by name for stable output and discoverability."""

        return sorted(self._tools.values(), key=lambda item: item.name)

    def call_tool(self, tool_name: str, arguments: dict[str, object]) -> dict[str, object]:
        """Invoke a parsed tool with JSON-like arguments.

        Args:
            tool_name: OpenAPI operationId (or generated fallback name).
            arguments: Dictionary containing path/query/header/cookie fields and
                optional body payload under the key "body".

        Returns:
            Structured dictionary with request metadata and parsed response data.
        """

        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(sorted(self._tools.keys()))
            raise ValueError(f"Unknown MCP tool '{tool_name}'. Available tools: {available}")

        request_path = tool.path
        query_items: list[tuple[str, str]] = []
        request_headers: dict[str, str] = dict(self.default_headers)
        cookie_items: list[str] = []

        # Convert user-supplied arguments into path/query/header/cookie inputs.
        for parameter in tool.parameters:
            value = arguments.get(parameter.name)
            if parameter.required and value is None:
                raise ValueError(
                    f"Missing required argument '{parameter.name}' for tool '{tool_name}'."
                )
            if value is None:
                continue

            if parameter.location == "path":
                request_path = request_path.replace(
                    "{" + parameter.name + "}",
                    quote(str(value), safe=""),
                )
            elif parameter.location == "query":
                query_items.append((parameter.name, str(value)))
            elif parameter.location == "header":
                request_headers[parameter.name] = str(value)
            elif parameter.location == "cookie":
                cookie_items.append(f"{parameter.name}={value}")

        if cookie_items:
            request_headers["Cookie"] = "; ".join(cookie_items)

        request_url = urljoin(self.base_url.rstrip("/") + "/", request_path.lstrip("/"))
        if query_items:
            request_url = f"{request_url}?{urlencode(query_items)}"

        # Request bodies are passed as arguments["body"] to avoid collisions
        # with parameter names from the OpenAPI operation.
        request_body: bytes | None = None
        if "body" in arguments:
            request_body = json.dumps(arguments["body"]).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        elif tool.request_body_required:
            raise ValueError(f"Tool '{tool_name}' requires a JSON 'body' argument.")

        request = Request(
            url=request_url,
            method=tool.method.upper(),
            headers=request_headers,
            data=request_body,
        )
        print(
            "debug> mcp manual-call: "
            f"tool={tool.name} method={tool.method.upper()} url={request_url}"
        )

        # Execute HTTP call and normalize common network/HTTP failures.
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                raw_body = response.read().decode("utf-8")
                status = response.status
                response_headers = dict(response.headers.items())
                print(
                    "debug> mcp manual-call result: "
                    f"tool={tool.name} status={status}"
                )
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            print(
                "debug> mcp manual-call result: "
                f"tool={tool.name} status={exc.code}"
            )
            raise RuntimeError(
                f"MCP tool call failed: HTTP {exc.code} on {tool.method.upper()} {request_url}"
                f"\n{error_body}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"MCP tool call failed: unable to reach {request_url}: {exc}"
            ) from exc

        # Prefer JSON parsing, but keep raw text if the response is not JSON.
        try:
            parsed_body: object = json.loads(raw_body) if raw_body.strip() else {}
        except json.JSONDecodeError:
            parsed_body = raw_body

        return {
            "tool": tool.name,
            "method": tool.method.upper(),
            "url": request_url,
            "status": status,
            "headers": response_headers,
            "data": parsed_body,
        }

    def _parse_tools(self, spec: dict[str, object]) -> dict[str, OpenAPITool]:
        """Extract callable operations from OpenAPI paths into tool descriptors."""

        paths = spec.get("paths", {})
        if not isinstance(paths, dict):
            raise ValueError("OpenAPI spec 'paths' must be an object.")

        tools: dict[str, OpenAPITool] = {}
        for path, path_item in paths.items():
            if not isinstance(path, str) or not isinstance(path_item, dict):
                continue

            for method, operation in path_item.items():
                method_name = str(method).lower()
                if method_name not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                    continue
                if not isinstance(operation, dict):
                    continue

                operation_id = str(operation.get("operationId", "")).strip()
                tool_name = operation_id if operation_id else _normalize_tool_name(method_name, path)
                description = str(
                    operation.get("description")
                    or operation.get("summary")
                    or f"{method_name.upper()} {path}"
                ).strip()

                raw_parameters = operation.get("parameters", [])
                parameters: list[OpenAPIParameter] = []
                if isinstance(raw_parameters, list):
                    for item in raw_parameters:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("name", "")).strip()
                        location = str(item.get("in", "")).strip().lower()
                        if not name or location not in {"path", "query", "header", "cookie"}:
                            continue
                        parameters.append(
                            OpenAPIParameter(
                                name=name,
                                location=location,
                                required=bool(item.get("required", False)),
                                description=str(item.get("description", "")).strip(),
                            )
                        )

                request_body_required = False
                request_body = operation.get("requestBody")
                if isinstance(request_body, dict):
                    request_body_required = bool(request_body.get("required", False))

                tools[tool_name] = OpenAPITool(
                    name=tool_name,
                    description=description,
                    method=method_name,
                    path=path,
                    parameters=parameters,
                    request_body_required=request_body_required,
                )

        if not tools:
            raise ValueError("No operations found in MCP OpenAPI spec.")
        return tools


def _format_mcp_help() -> str:
    """Return CLI help text for manual MCP slash commands."""

    return (
        "MCP commands:\n"
        "  /mcp tools\n"
        "  /mcp tools/list\n"
        "  /mcp call <tool_name> <json-args>\n"
        "  /mcp tools/call <tool_name> <json-args>\n"
        "  /mcp reload\n"
        "Example:\n"
        "  /mcp call getWeatherForecast {\"latitude\":47.6,\"longitude\":-122.3}\n"
    )


def _extract_mcp_request_details(kwargs: dict[str, object]) -> tuple[str, str, str]:
    """Best-effort extraction of operation/method/url from OpenAPI callback kwargs."""

    def _from_mapping_or_attr(obj: object, names: list[str]) -> object | None:
        if obj is None:
            return None
        for name in names:
            if isinstance(obj, dict) and name in obj:
                value = obj[name]
                if value is not None:
                    return value
            elif hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
        return None

    operation_names = [
        "operation_id",
        "operationId",
        "operation",
        "function_name",
        "functionName",
        "name",
    ]
    method_names = ["method", "http_method", "httpMethod"]
    url_names = ["url", "request_url", "requestUrl", "uri"]

    operation_value = _from_mapping_or_attr(kwargs, operation_names)
    method_value = _from_mapping_or_attr(kwargs, method_names)
    url_value = _from_mapping_or_attr(kwargs, url_names)

    # Semantic Kernel may pass request metadata nested under context/request objects.
    nested_candidates: list[object] = [
        kwargs.get("request") if isinstance(kwargs, dict) else None,
        kwargs.get("http_request") if isinstance(kwargs, dict) else None,
        kwargs.get("context") if isinstance(kwargs, dict) else None,
    ]

    if isinstance(kwargs, dict):
        operation_value = operation_value or _from_mapping_or_attr(
            kwargs.get("operation"), operation_names
        )

    for candidate in nested_candidates:
        if operation_value is None:
            operation_value = _from_mapping_or_attr(candidate, operation_names)
        if method_value is None:
            method_value = _from_mapping_or_attr(candidate, method_names)
        if url_value is None:
            url_value = _from_mapping_or_attr(candidate, url_names)

        if isinstance(candidate, dict):
            nested_request = candidate.get("request")
            if method_value is None:
                method_value = _from_mapping_or_attr(nested_request, method_names)
            if url_value is None:
                url_value = _from_mapping_or_attr(nested_request, url_names)

    operation = str(operation_value or "unknown_operation")
    method = str(method_value or "UNKNOWN").upper()
    url = str(url_value or "unknown_url")
    return operation, method, url


def _build_mcp_http_client(timeout_seconds: int) -> httpx.AsyncClient:
    """Create an HTTP client that logs MCP auto-call request/response details."""

    async def _log_request(request: httpx.Request) -> None:
        print(
            "debug> mcp auto-call request: "
            f"method={request.method.upper()} url={request.url}"
        )

    async def _log_response(response: httpx.Response) -> None:
        print(
            "debug> mcp auto-call response: "
            f"status={response.status_code} method={response.request.method.upper()} "
            f"url={response.request.url}"
        )

    return httpx.AsyncClient(
        timeout=float(timeout_seconds),
        event_hooks={"request": [_log_request], "response": [_log_response]},
    )


def _build_openapi_auth_callback(default_headers: dict[str, str]):
    """Build auth callback used by Semantic Kernel OpenAPI plugin execution.

    SK's OpenAPI runner calls this callback before each request. Returning the
    configured default headers allows simple API-key style authentication.
    """

    async def _auth_callback(**_kwargs) -> dict[str, str]:  # noqa: ANN003
        # OpenAPI runner passes request headers as kwargs here, not request
        # metadata like method/url. Keep this callback auth-only.
        return dict(default_headers)

    return _auth_callback


def _handle_mcp_command(command_text: str, mcp: OpenAPIMCPInterface | None) -> bool:
    """Handle manual MCP slash commands.

    Returns:
        bool: True if command_text was an MCP command and was handled.
    """

    if not command_text.startswith("/mcp"):
        return False

    if mcp is None:
        print("mcp> MCP is not enabled. Set MCP_OPENAPI_SPEC_URL in .env")
        return True

    text = command_text.strip()
    if text in {"/mcp", "/mcp help", "/mcp ?"}:
        print(_format_mcp_help())
        return True

    if text in {"/mcp tools", "/mcp tools/list"}:
        print("mcp> available tools:")
        for tool in mcp.list_tools():
            required_args = [p.name for p in tool.parameters if p.required]
            suffix = f" | required: {', '.join(required_args)}" if required_args else ""
            body_suffix = " | requires body" if tool.request_body_required else ""
            print(f"- {tool.name}: {tool.method.upper()} {tool.path}{suffix}{body_suffix}")
        return True

    if text == "/mcp reload":
        mcp.reload()
        print(f"mcp> reloaded spec, tools available: {len(mcp.list_tools())}")
        return True

    tokens = text.split(maxsplit=3)
    if len(tokens) >= 3 and tokens[1] in {"call", "tools/call"}:
        tool_name = tokens[2]
        arguments: dict[str, object] = {}
        if len(tokens) == 4:
            try:
                loaded = json.loads(tokens[3])
            except json.JSONDecodeError as exc:
                print(f"mcp> invalid JSON arguments: {exc}")
                return True
            if not isinstance(loaded, dict):
                print("mcp> JSON arguments must be an object.")
                return True
            arguments = loaded

        result = mcp.call_tool(tool_name, arguments)
        print("mcp> tool result:")
        print(json.dumps(result, indent=2))
        return True

    print(_format_mcp_help())
    return True


def load_agent_prompt_config(file_path: str) -> AgentPromptConfig:
    """Load and validate agent system prompt + few-shot examples from YAML.

    Expected shape:
        prompt: <string>
        example:
          - question: <string>
            answer: <string>
    """

    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"Agent prompt config file not found: {file_path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        raise ValueError(f"Agent prompt config must include a non-empty 'prompt': {file_path}")

    raw_examples = data.get("example", [])
    if raw_examples is None:
        raw_examples = []
    if not isinstance(raw_examples, list):
        raise ValueError(f"'example' must be a list in agent prompt config: {file_path}")

    examples: list[AgentPromptExample] = []
    for idx, item in enumerate(raw_examples, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"example[{idx}] must be an object in agent prompt config: {file_path}")

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if not question or not answer:
            raise ValueError(
                f"example[{idx}] must include non-empty 'question' and 'answer' in {file_path}"
            )

        examples.append(AgentPromptExample(question=question, answer=answer))

    return AgentPromptConfig(prompt=prompt, examples=examples)


def build_search_data_source(config: AppConfig) -> dict[str, object] | None:
    """Build Azure AI Search data source payload for OYD chat requests.

    Returns None if grounding is not configured.
    """

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


def _extract_search_doc_text(document: object) -> str:
    """Best-effort conversion of a search document into compact grounding text."""

    if not isinstance(document, dict):
        return str(document)

    preferred_fields = [
        "content",
        "text",
        "chunk",
        "chunkText",
        "summary",
        "description",
        "title",
    ]
    for field in preferred_fields:
        value = document.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    snippets: list[str] = []
    for key, value in document.items():
        if key.startswith("@"):
            continue
        if isinstance(value, str) and value.strip():
            snippets.append(f"{key}: {value.strip()}")
        if len(snippets) >= 3:
            break
    return " | ".join(snippets) if snippets else json.dumps(document, ensure_ascii=True)


def _format_search_context_for_message(
    search_data_source: dict[str, object],
    query: str,
    documents: list[dict[str, object]],
) -> str:
    """Build a message-level grounding payload from Azure AI Search results."""

    context_docs: list[dict[str, object]] = []
    for idx, document in enumerate(documents, start=1):
        context_docs.append(
            {
                "rank": idx,
                "score": document.get("@search.score"),
                "reranker_score": document.get("@search.rerankerScore"),
                "content": _extract_search_doc_text(document),
            }
        )

    payload = {
        "datasources": [
            {
                "source": search_data_source,
                "query": query,
                "documents": context_docs,
            }
        ]
    }

    return (
        "Use this grounding data if relevant and cite its facts conservatively.\n"
        "Grounding payload (message-level, not request-level):\n"
        f"{json.dumps(payload, ensure_ascii=True)}"
    )


async def _search_documents(
    config: AppConfig,
    user_query: str,
) -> list[dict[str, object]]:
    """Query Azure AI Search directly and return documents for grounding."""

    if not config.search_endpoint or not config.search_index_name:
        return []

    url = (
        f"{config.search_endpoint.rstrip('/')}/indexes/"
        f"{quote(config.search_index_name, safe='')}/docs/search"
        f"?api-version={quote(config.search_api_version, safe='')}"
    )

    payload: dict[str, object] = {
        "search": user_query,
        "top": config.search_top_n_documents,
    }

    if config.search_query_type in {"semantic", "simple", "full"}:
        payload["queryType"] = config.search_query_type
    if config.search_query_type == "semantic" and config.search_semantic_configuration:
        payload["semanticConfiguration"] = config.search_semantic_configuration

    headers = {"Content-Type": "application/json"}
    if config.search_api_key:
        headers["api-key"] = config.search_api_key
    else:
        credential = DefaultAzureCredential()
        token = credential.get_token("https://search.azure.com/.default")
        headers["Authorization"] = f"Bearer {token.token}"

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()

    values = body.get("value", []) if isinstance(body, dict) else []
    return [item for item in values if isinstance(item, dict)]


def create_chat_service(config: AppConfig) -> ChatCompletionClientBase:
    """Create an AzureChatCompletion service for the selected provider mode.

    Auth mode is chosen based on whether an API key is present:
    - API key auth when key is provided.
    - DefaultAzureCredential token provider when key is omitted.
    """

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
    """Run the interactive chat REPL.

    This function orchestrates initialization, optional plugin registration,
    optional grounding, slash-command handling, and assistant responses.
    """

    config = load_config()
    agent_prompt_config = load_agent_prompt_config(config.agent_prompt_file)

    kernel = Kernel()
    chat_service = create_chat_service(config)
    kernel.add_service(chat_service)
    # Seed history with system prompt and optional few-shot examples to shape
    # assistant behavior before user interaction starts.
    history = ChatHistory()
    history.add_system_message(agent_prompt_config.prompt)
    for example in agent_prompt_config.examples:
        history.add_user_message(example.question)
        history.add_assistant_message(example.answer)

    settings = AzureChatPromptExecutionSettings(temperature=0.7)
    search_data_source = build_search_data_source(config)
    search_grounding_active = bool(search_data_source)

    # mcp_interface powers manual slash commands; mcp_auto_plugin_enabled tracks
    # whether SK automatic function-calling is active for normal chat turns.
    mcp_interface: OpenAPIMCPInterface | None = None
    mcp_http_client: httpx.AsyncClient | None = None
    mcp_auto_plugin_enabled = False
    if config.mcp_openapi_spec_url:
        try:
            mcp_interface = OpenAPIMCPInterface(
                spec_url=config.mcp_openapi_spec_url,
                base_url_override=config.mcp_base_url,
                timeout_seconds=config.mcp_timeout_seconds,
                default_headers=config.mcp_default_headers,
            )

            openapi_spec = _load_openapi_document(
                spec_url=config.mcp_openapi_spec_url,
                timeout_seconds=config.mcp_timeout_seconds,
            )
            openapi_execution_settings = OpenAPIFunctionExecutionParameters(
                http_client=_build_mcp_http_client(config.mcp_timeout_seconds),
                server_url_override=mcp_interface.base_url,
                timeout=float(config.mcp_timeout_seconds),
                auth_callback=_build_openapi_auth_callback(config.mcp_default_headers),
            )
            mcp_http_client = openapi_execution_settings.http_client
            # Register OpenAPI operations as kernel functions so the model can
            # discover and call them automatically.
            kernel.add_plugin_from_openapi(
                plugin_name="mcp",
                openapi_parsed_spec=openapi_spec,
                execution_settings=openapi_execution_settings,
                description="Tools loaded from MCP OpenAPI spec",
            )
            settings.function_choice_behavior = FunctionChoiceBehavior.Auto(
                auto_invoke=True,
                filters={"included_plugins": ["mcp"]},
            )
            mcp_auto_plugin_enabled = True
        except Exception as exc:  # noqa: BLE001
            print(f"MCP init error: {exc}")

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
    if search_grounding_active:
        print(f"Grounding: Azure AI Search enabled (index: {config.search_index_name})")
    else:
        print("Grounding: disabled")
    print(f"Agent prompt config: {config.agent_prompt_file}")
    if agent_prompt_config.examples:
        print(f"Agent examples loaded: {len(agent_prompt_config.examples)}")
    if mcp_interface:
        print(f"MCP: enabled ({len(mcp_interface.list_tools())} tools from OpenAPI)")
        print("MCP commands: /mcp tools, /mcp call <tool_name> <json-args>, /mcp reload")
        if mcp_auto_plugin_enabled:
            print("MCP auto-calling: enabled via Semantic Kernel plugin")
        else:
            print("MCP auto-calling: disabled")
    elif config.mcp_openapi_spec_url:
        print("MCP: disabled due to initialization error")
    else:
        print("MCP: disabled")
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

            # Slash commands are processed before normal model turns.
            try:
                if _handle_mcp_command(user_text, mcp_interface):
                    print()
                    continue
            except Exception as exc:  # noqa: BLE001
                print(f"mcp> [error] {exc}\n")
                continue

            user_message_text = user_text
            if search_grounding_active and search_data_source is not None:
                try:
                    documents = await _search_documents(config, user_text)
                    if documents:
                        context_payload = _format_search_context_for_message(
                            search_data_source=search_data_source,
                            query=user_text,
                            documents=documents,
                        )
                        user_message_text = f"{user_text}\n\n{context_payload}"
                        print(
                            "debug> search: enabled "
                            f"query={json.dumps(user_text)} index={config.search_index_name} "
                            f"hits={len(documents)}"
                        )
                    else:
                        print(
                            "debug> search: enabled "
                            f"query={json.dumps(user_text)} index={config.search_index_name} hits=0"
                        )
                except Exception as exc:  # noqa: BLE001
                    print(f"debug> search: failed error={exc}")
            else:
                print("debug> search: disabled")

            history.add_user_message(user_message_text)

            if config.debug_chat_messages:
                debug_messages = _render_chat_history_for_debug(history)
                print("debug> azure-openai request messages:")
                print(json.dumps(debug_messages, ensure_ascii=True, indent=2))

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

    if mcp_http_client is not None:
        await mcp_http_client.aclose()

    print("Exiting chat.")


if __name__ == "__main__":
    asyncio.run(run_chat())
