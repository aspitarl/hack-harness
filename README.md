# netl-hack-harness

A Python command-line chat harness built with Semantic Kernel.

## What this does

- Uses Semantic Kernel chat completion only (no Azure AI Agent Framework).
- Supports two endpoint modes from `.env`:
	- Azure OpenAI endpoint
	- Foundry project endpoint with chat deployment
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
