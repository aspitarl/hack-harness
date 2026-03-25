## DOE Directive Impact Analysis Pipeline

This workflow evaluates a new DOE directive and identifies which existing NETL orders and procedures likely need updates.

- Searches NETL order and procedure content for sections related to the new DOE directive.
- Reviews likely matches to see which documents may need updates.
- Produces a short report on what to update, why, and next steps.


## Pipeline Flow

```mermaid
flowchart LR
    ND[[New DOE Directive PDF]] -->  B[Extract atomic requirements]
    ND -. directive text context .-> G
    B --> D[Stage 1/2 relevance flagging via requirement and chunk search hits]

    D --> E[Stage 3 file-level evidence assembly from flagged files]
    E --> F[Assemble combined evidence context for model synthesis]

    F --> G[Model synthesis]
    G --> H[Write report artifacts]

    subgraph AZ[Azure services]
        direction TB
        AS[(Azure AI Search)]
        AB[(Azure Blob Storage<br>w/ Procedures and Orders PDFs)]
        AO[(Azure OpenAI<br/>GPT5.3-chat)]
    end

    AS -. search results .-> D
    AS -. search results .-> E

    AB -. full PDF content .-> E

    AO -. model responses .-> G
```

## Notes

- Stage 1 and Stage 2 perform relevance flagging of candidate NETL documents using requirement-based and full-directive chunk search, respectively.
- Stage 3 performs file-level evidence assembly for those flagged files: file-specific matching snippets plus full-PDF text when available.
- Stage 3 is retrieval and evidence preparation; the LLM reasoning happens in the downstream model synthesis step.
- Model synthesis includes directive text context (baseline pass) in addition to evidence assembled from Stages 1/2/3.