# ai-governance-agent
An AI agent that helps policy analysts research and compare AI regulations across jurisdictions using retrieval-augmented generation and a ReAct reasoning pattern, grounded in the NIST AI RMF 1.0, NIST AI 600-1 (Generative AI Profile), and the EU AI Act (Regulation 2024/1689).

Built for Meridian Governance Group as part of AAI-510 (Agentic AI) at the University of San Diego.

## Running order
1. **`01_data_pipeline.ipynb`** — downloads the three source PDFs, extracts and chunks 891K+ characters into 1,107 segments, saves to a Delta table, and creates the Databricks Vector Search index (`ai_governance_endpoint` / `main.default.ai_governance_index`).
2. **`02_eda.ipynb`** — exploratory data analysis of the chunked corpus (chunk length distribution, t-SNE embedding visualization, per-document breakdown).
3. **`03_agent_evaluation.ipynb`** — runs the ReAct agent across three LLMs (Databricks GPT-OSS 120B, Claude Sonnet 4.6, Claude Haiku 4.5) with MLflow tracing, a custom LLM-as-judge scorer, groundedness checks, rubric thresholds, and an ROI comparison across 10 evaluation queries.

## LLM setup
- **Primary OSS model:** `databricks-gpt-oss-120b` (override with `DATABRICKS_PRIMARY_ENDPOINT` env var)
- **Anthropic models:** Set `ANTHROPIC_API_KEY` to add Claude Sonnet 4.6 and Claude Haiku 4.5 to the comparison. When set, Claude Haiku is also used as the LLM judge; otherwise the judge falls back to `databricks-gpt-oss-120b`.
- **Self-hosted model (optional):** Qwen 3.6 at `llm.londonary.com` via OpenAI-compatible API (override with `OPENSOURCE_BASE_URL` / `OPENSOURCE_MODEL`). Excluded from final evaluation due to Cloudflare blocking from Databricks cloud egress IPs.

## Evaluation
Notebook 03 evaluates the agent on 10 queries (7 in-scope, 3 out-of-scope) using `mlflow.genai.evaluate()` with:
- **Custom `policy_judge` scorer** — LLM-as-judge scoring accuracy, relevance, completeness, and clarity (1-5 scale)
- **Groundedness check** — verifies answers trace back to retrieved source chunks, not general knowledge
- **Rubric threshold** — pass threshold of 3.0 on all dimensions
- **Built-in MLflow scorers** — `RelevanceToQuery` and `Safety`

Results are logged to the `/meridian-policy-agent` MLflow experiment and saved to `traces/evaluation_results.json`.

## Vector search
Notebook 03 uses **Databricks Vector Search** by default (`DatabricksVSToolbox`). The index was built in notebook 01 using `databricks-bge-large-en`.

`src/vector_store.py` (numpy-based `SimpleVectorStore`) is available as a fallback for local development outside Databricks.

## Source documents
| Document | Publisher | Year |
|----------|-----------|------|
| NIST AI Risk Management Framework (AI RMF 1.0) | NIST | 2023 |
| NIST AI 600-1 (Generative AI Profile) | NIST | 2024 |
| EU Artificial Intelligence Act (Regulation 2024/1689) | European Union | 2024 |
