# AI Policy Research Agent

An AI agent that helps policy analysts research and compare AI governance frameworks. Ask it a question about NIST or the EU AI Act and it retrieves the relevant passages, reasons over them, and answers with answers grounded in the source text.

Built for the fictional consulting firm Meridian Governance Group as the capstone for AAI-510 (Agentic AI) in the Applied AI master's program at the University of San Diego.

## What it does

Analysts at a governance consultancy spend a lot of time digging through long regulatory documents. This agent speeds that up by:

- Retrieving the most relevant passages from the source frameworks (RAG over a Databricks Vector Search index)
- Reasoning over them with a ReAct pattern and tool calls
- Answering with content that traces back to retrieved chunks
- Declining or flagging questions that fall outside the indexed documents

## Source documents

| Document | Year |
|---|---|
| NIST AI Risk Management Framework (AI RMF 1.0) | 2023 |
| NIST AI 600-1, Generative AI Profile | 2024 |
| EU AI Act, Regulation (EU) 2024/1689 | 2024 |

These are chunked into 1,107 segments and stored in a Delta table before indexing.

## How it works

```
PDFs -> chunking -> Delta table -> Databricks Vector Search index
                                          |
user question -> query expansion -> retrieval -> LLM (ReAct + tools) -> grounded answer
                                                       |
                                                 MLflow tracing
```

I went with a single agent that uses tools rather than a multi-agent setup. For a governance use case, one agent is easier to trace, audit, and hold accountable for its answers.

## Repository layout

| Path | What it is |
|---|---|
| `01_data_pipeline.ipynb` | Downloads the PDFs, chunks them, writes the Delta table, builds the Vector Search index |
| `02_EDA.ipynb` | Exploratory analysis of the chunks: length distribution, t-SNE visualization, per-document breakdown |
| `03_agent_evaluation.ipynb` | Runs the agent across multiple LLMs with MLflow tracing and scores the results |
| `03_agent_evaluation_clean.ipynb` | Cleaned-up version of the evaluation notebook |
| `src/` | Agent code, including `vector_store.py`, a local numpy-based fallback for retrieval |
| `traces/` | Saved evaluation output, including `evaluation_results.json` |
| `.env.example` | Template for required environment variables |

Run the notebooks in order: 01, then 02, then 03.

## Models evaluated

| Model | Role |
|---|---|
| `databricks-gpt-oss-120b` | Primary open-source model, and the judge when no Anthropic key is set |
| Claude Sonnet 4.6 | Added when `ANTHROPIC_API_KEY` is set |
| Claude Haiku 4.5 | Added when `ANTHROPIC_API_KEY` is set, and used as the judge in that case |

I also tried a self-hosted Qwen 3.6 model, but left it out of the final evaluation because Cloudflare blocked requests coming from Databricks egress IPs.

Embeddings come from `databricks-bge-large-en`.

## Evaluation

Each model is run over 10 test queries: 7 in scope and 3 out of scope, so the out-of-scope ones test whether the agent knows when to stop.

- **Custom `policy_judge`:** scores accuracy, relevance, completeness, and clarity on a 1 to 5 scale
- **Groundedness check:** confirms the answer traces back to the retrieved chunks
- **Pass threshold:** 3.0 on every dimension
- **Built-in MLflow scorers:** `RelevanceToQuery` and `Safety`
- **ROI comparison:** quality against cost across the three models

Results are logged to the `/meridian-policy-agent` MLflow experiment and saved to `traces/evaluation_results.json`.

### Results

<!-- Fill in with your real numbers before sharing -->
| Model | Accuracy | Relevance | Completeness | Clarity | Grounded | Pass rate |
|---|---|---|---|---|---|---|
| databricks-gpt-oss-120b | | | | | | |
| Claude Sonnet 4.6 | | | | | | |
| Claude Haiku 4.5 | | | | | | |

### What I learned about LLM judges

Automated judges are a useful first pass, but I would not let one make the final call in a high-stakes setting. Reading traces by hand caught failure modes the judge missed, like null outputs and error states that still got scored. Human review stayed part of the loop.

## Setup

**Requirements**

- A Databricks workspace with Vector Search and Foundation Model API access
- Python 3.10 or later
- Optional: an Anthropic API key to include the Claude models

**Steps**

1. Clone the repo.
2. Copy `.env.example` to `.env` and fill in your values.
3. Import the notebooks into Databricks.
4. Run `01_data_pipeline.ipynb`, then `02_EDA.ipynb`, then `03_agent_evaluation.ipynb`.

**Environment variables**

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Enables the Claude models (optional) |
| `DATABRICKS_PRIMARY_ENDPOINT` | Primary model serving endpoint |
| `OPENSOURCE_BASE_URL` | Base URL for the optional self-hosted model |
| `OPENSOURCE_MODEL` | Model name for the optional self-hosted model |

**Databricks resources**

- Vector Search endpoint: `ai_governance_endpoint`
- Index: `main.default.ai_governance_index`

## Limitations

- Covers three documents only. It is not a general legal research tool.
- Answers are retrieval-grounded but not legal advice, and a person should verify anything that matters.
- The test set is small (10 queries), so the scores are directional, not statistically strong.
- The judge is itself an LLM, which brings its own bias and error.

## Possible next steps

- Add more jurisdictions and frameworks (for example ISO/IEC 42001, state-level US laws)
- Grow the test set and add human-labeled reference answers
- Add citation formatting that points to specific sections and articles

## Team

Built by Mandeep Saini, Dylan, and Nicholson for USD AAI-510.

## License

MIT. See `LICENSE`.
