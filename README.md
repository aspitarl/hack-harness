# DOE Directive Impact Analysis Pipeline

This repository analyzes a new DOE directive and identifies NETL orders/procedures that likely require updates.

The pipeline performs staged retrieval plus model synthesis and produces report artifacts reviewers can use for triage.

## What the pipeline does

- Extracts atomic requirements from the incoming directive.
- Runs Stage 1/2 relevance flagging using requirement-based and directive-chunk search hits.
- Runs Stage 3 file-level evidence assembly for flagged files (snippets + full PDF text when available).
- Performs model synthesis to produce update recommendations and reviewer-ready artifacts.

For the detailed flow diagram and stage notes, see `doc/pipeline_flow.md`.

## Repository entry points

- `review_to_md.py` - non-interactive pipeline run that writes markdown/PDF artifacts.
- `streamlit_app.py` - web UI for upload, run, visualization, and downloads.
- `chat_cli.py` - interactive CLI with `/investigate` command and optional MCP tooling.
- `tests/test_review_to_md_contract.py` - contract tests for requirement extraction and report formatting.

## Quick start

1. (Recommended) create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables in `.env` (see configuration section below).

## Run the pipeline (recommended)

One-shot run with file outputs:

```bash
python3 review_to_md.py \
  --directive "pdfs/new_directive.pdf" \
  --out "reports/new_directive_investigation.md"
```

Outputs written per run:

- `<out>.md` - consolidated investigation markdown.
- `<out>.pdf` - PDF rendering of the consolidated investigation.
- `<out>_stage1_atomic_requirements.md` - extracted requirements for reviewer validation.
- `<out>_stage3_updates.md` - Stage 3 file-level update assessments.
- `<out>_stage3_updates.pdf` - PDF rendering of Stage 3 updates.

Example output files already in this repo are under `reports/`.

## Run with Streamlit UI

```bash
streamlit run streamlit_app.py
```

In the app:

- Upload directive file (`.pdf`, `.txt`, `.md`).
- Run the DOE directive impact analysis pipeline.
- Download Stage 3 markdown/PDF outputs.
- View Stage 3 visual summary (status, confidence, requirement-to-file coverage).

Optional Stage 3 markdown save to Blob Storage:

```env
AZURE_STORAGE_ACCOUNT_URL=https://<storage-account>.blob.core.windows.net
AZURE_STORAGE_CONTAINER=<container-name>
```

## Configuration

The pipeline supports two chat backends.

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

### Azure AI Search retrieval configuration

Single index mode:

```env
AZURE_AI_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_AI_SEARCH_INDEX_NAME=<your-index-name>
```

Dual index mode (orders + procedures):

```env
AZURE_AI_SEARCH_ENDPOINT=https://<your-search-service>.search.windows.net
AZURE_AI_SEARCH_ORDERS_INDEX_NAME=<orders-index-name>
AZURE_AI_SEARCH_PROCEDURES_INDEX_NAME=<procedures-index-name>
```

Authentication:

- Set `AZURE_AI_SEARCH_API_KEY` for API-key auth, or
- leave it blank to use managed identity / default credential.

Optional retrieval tuning:

```env
AZURE_AI_SEARCH_SEMANTIC_CONFIGURATION=<semantic-config-name>
AZURE_AI_SEARCH_QUERY_TYPE=semantic
AZURE_AI_SEARCH_IN_SCOPE=true
AZURE_AI_SEARCH_STRICTNESS=3
AZURE_AI_SEARCH_TOP_N_DOCUMENTS=5
```

### Agent prompt configuration

```env
AGENT_PROMPT_FILE=agents/default.yaml
```

Expected shape:

```yaml
prompt: |
  You are a helpful assistant...
```

## Optional interactive CLI mode

Run:

```bash
python chat_cli.py
```

Inside the REPL, run investigation command:

```text
/investigate --directive "pdfs/new_directive.pdf"
```

This path is useful for interactive exploration, but `review_to_md.py` is the primary reproducible pipeline entry point.

## Notes and constraints

- Scanned PDFs without embedded text require OCR before analysis.
- Very large directive text may be truncated before prompt submission.
- Stage 3 performs retrieval/evidence preparation; model reasoning is in downstream synthesis.
