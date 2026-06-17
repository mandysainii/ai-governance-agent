# ai-governance-agent
An AI agent that helps policy analysts research and compare AI regulations across jurisdictions using retrieval augmented generation and a ReAct reasoning pattern, grounded in the NIST AI RMF, NIST AI 600-1, and the EU AI Act.

## Running order

1. **`01_data_pipeline.ipynb`** — downloads the three PDFs, chunks them, saves to a Delta table, and creates the Databricks Vector Search index (`ai_governance_endpoint` / `main.default.ai_governance_index`).
2. **`02_eda.ipynb`** — exploratory analysis (optional).
3. **`03_agent_evaluation.ipynb`** — runs the agent across two Databricks OSS models + llm.londonary.com (Qwen 3.6), with MLflow tracing, LLM-as-judge scoring, and ROI comparison.

## LLM setup

- **Primary OSS model:** `databricks-gpt-oss-120b` (override with `DATABRICKS_PRIMARY_ENDPOINT` env var)
- **Secondary OSS model:** `databricks-meta-llama-3-3-70b-instruct` (override with `DATABRICKS_SECONDARY_ENDPOINT`)
- **Self-hosted model:** Qwen 3.6 at `llm.londonary.com` via OpenAI-compatible API (override with `OPENSOURCE_BASE_URL` / `OPENSOURCE_MODEL`)
- **Anthropic (optional):** Set `ANTHROPIC_API_KEY` to add Claude Sonnet and Haiku to the comparison. If set, Claude is also used as the LLM judge; otherwise the judge falls back to `databricks-gpt-oss-120b`.

## Vector search

Notebook 03 uses **Databricks Vector Search** by default (`DatabricksVSToolbox`). The VS index was built in notebook 01 using `databricks-bge-large-en` — queries go through `query_text=` so no local embedder is needed.

`src/vector_store.py` (numpy-based `SimpleVectorStore`) is only needed for:
- **Local dev** outside Databricks — notebook 03 falls back to it automatically if VS is unreachable.
- **`02_eda.ipynb`** — the t-SNE embedding visualization loads it directly.

Step 6 of notebook 01 builds the local numpy store (`vector_store/policy_store.npz`). This step can be skipped when running entirely in Databricks.
