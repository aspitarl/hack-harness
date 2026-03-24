# hack-harness

A Python command-line chat harness built with Semantic Kernel.

## What this does

- Uses Semantic Kernel chat completion only (no Azure AI Agent Framework).
- Supports two endpoint modes from `.env`:
	- Azure OpenAI endpoint
	- Foundry project endpoint with chat deployment
- Optional MCP-like tool interface backed by a published OpenAPI 3.x spec.
- Includes a DOE directive review command for comparing requirements and draft directive files.
- Exits on `Ctrl+C` or `Ctrl+X`.

## Files

- `chat_cli.py` - interactive CLI chat app
- `requirements.txt` - Python dependencies
- `.env` - runtime configuration

## Setup

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Edit `.env`.

### Agent Prompt Configuration

This project loads a system prompt and optional few-shot examples from an agent config file in `agents/`.

```env
AGENT_PROMPT_FILE=agents/default.yaml
```

Expected YAML shape:

```yaml
prompt: |
	You are a helpful assistant...

example:
	- question: First user message
		answer: First assistant response
	- question: Second user message
		answer: Second assistant response
```

The `prompt` is added as a system message at startup, and each `example` pair is added to chat history before interactive input begins.

### DOE Directive Review Command

Inside the interactive prompt, run:

```text
/review --requirements <requirements.pdf> --draft <draft-file> [--draft <draft-file> ...]
```

Supported file types:

- Requirements: `.pdf`, `.txt`, `.md`
- Drafts: `.pdf`, `.txt`, `.md`

For paths with spaces, wrap each path in quotes.

Example with your current files:

```text
/review --requirements "/pdfs/DOE_Directives.pdf" --draft "/pdfs/P251-1-01_DirectivesProcessing.pdf" --draft "/pdfs/O251_1_DirectivesProgram.pdf"
```

The command loads document text, sends a structured review request to the model, and returns a markdown report with:

- Executive summary
- Requirement-by-requirement findings
- Gaps and risks
- Suggested revisions
- Priority next steps

Notes:

- Very large documents are truncated before prompt submission.
- Scanned PDFs without embedded text require OCR before review.

### One-shot Review to Markdown File

Use this non-interactive script to run one review and save directly to a markdown file:

```bash
python3 review_to_md.py \
	--requirements "pdfs/DOE_Directives.pdf" \
	--draft "pdfs/P251-1-01_DirectivesProcessing.pdf" \
	--draft "pdfs/O251_1_DirectivesProgram.pdf" \
	--out "reports/doe_review.md"
```

The script prints the output file path after writing the report.

### Option A: Azure OpenAI

```env
CHAT_PROVIDER=azure_openai

AZURE_OPENAI_ENDPOINT=https://<your-openai-resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-api-key-or-leave-blank-for-default-credential>
AZURE_OPENAI_CHAT_DEPLOYMENT=<your-chat-deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
```

### Option B: Foundry project endpoint

```env
CHAT_PROVIDER=foundry

FOUNDRY_PROJECT_ENDPOINT=https://<your-project-host>.services.ai.azure.com/api/projects/<your-project-name>
FOUNDRY_API_KEY=<your-api-key-or-leave-blank-for-default-credential>
FOUNDRY_CHAT_DEPLOYMENT=<your-chat-deployment-name>
FOUNDRY_API_VERSION=2024-10-21
```

### Optional: Azure AI Search grounding

You can ground responses with an Azure AI Search index in either provider mode.

Set both of these values to enable grounding:

```env
AZURE_AI_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_AI_SEARCH_INDEX_NAME=<your-index-name>
```

Authentication for Search:

- API key mode: set `AZURE_AI_SEARCH_API_KEY`.
- Keyless mode: leave `AZURE_AI_SEARCH_API_KEY` blank and use managed identity.

Optional tuning:

```env
AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION=<semantic-config-name>
AZURE_AI_SEARCH_QUERY_TYPE=semantic
AZURE_AI_SEARCH_IN_SCOPE=true
AZURE_AI_SEARCH_STRICTNESS=3
AZURE_AI_SEARCH_TOP_N_DOCUMENTS=5
```

When enabled, the CLI prints that grounding is active and shows the selected index name at startup.

### Optional: MCP-like OpenAPI tools

You can enable an MCP-style tool surface where each OpenAPI operation is exposed as a callable tool in the CLI.
When configured, Semantic Kernel also loads the OpenAPI operations as a plugin and can automatically invoke tools during normal chat turns.

```env
MCP_OPENAPI_SPEC_URL=https://func-amdojxfludppe.azurewebsites.net/api/spec/openapi?code=<your-code>
MCP_BASE_URL=
MCP_TIMEOUT_SECONDS=30
MCP_DEFAULT_HEADERS={}
```

Notes:

- `MCP_OPENAPI_SPEC_URL` is required to enable MCP commands.
- `MCP_BASE_URL` is optional; if empty, the app resolves a base URL from the spec. If the spec server points to localhost, it falls back to the spec URL host.
- `MCP_DEFAULT_HEADERS` must be a JSON object string (for example `{"x-api-key":"..."}`).

Interactive commands:

```text
/mcp tools
/mcp tools/list
/mcp call <tool_name> <json-args>
/mcp tools/call <tool_name> <json-args>
/mcp reload
```

Automatic mode:

- Ask naturally in chat (for example, "What's the weather in Seattle?") and the model can automatically select and call MCP/OpenAPI tools.
- Slash commands remain available for manual inspection and testing.

Example:

```text
/mcp call getWeatherForecast {"latitude":47.6,"longitude":-122.3}
```

## Run

```bash
python chat_cli.py
```

Type your message at `you>`. The app exits immediately on `Ctrl+C` or `Ctrl+X`.

## Notes

- The script is structured so you can later add grounding with Azure AI Search and tools backed by your Function App MCP server.
- If your deployment requires a different API version, update the matching `*_API_VERSION` value in `.env`.
- If the API key value is blank, the app automatically falls back to Azure Default Credential.
- In `foundry` mode, keyless auth requests tokens for `https://ai.azure.com/.default`.
- In `azure_openai` mode, keyless auth requests tokens for `https://cognitiveservices.azure.com/.default`.
- In `foundry` mode, if you provide a project endpoint URL, the app automatically uses the account endpoint host for chat completion calls.
