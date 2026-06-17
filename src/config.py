"""Central configuration for the AI Policy Research Agent.

All tunable constants live here so the notebooks and modules stay in sync.

NOTE ON AI USAGE (academic integrity): The scaffolding in this repository was
developed with the assistance of Anthropic's Claude (Claude Code). All design
decisions, model choices, and parameters were reviewed and are explained in
code comments. See README.md for the full disclosure.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
TRACES_DIR = PROJECT_ROOT / "traces"

for _d in (DATA_DIR, VECTOR_STORE_DIR, TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Source documents — the agent's regulatory knowledge base.
# All three are publicly available with no authentication required.
# --------------------------------------------------------------------------- #
DOCUMENTS = [
    {
        "doc_id": "nist_ai_rmf_1_0",
        "title": "NIST AI Risk Management Framework (AI RMF 1.0)",
        "short_name": "NIST AI RMF",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf",
        "publisher": "NIST",
        "year": 2023,
    },
    {
        "doc_id": "nist_ai_600_1",
        "title": "NIST AI 600-1: Generative AI Profile",
        "short_name": "NIST GenAI Profile",
        "url": "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf",
        "publisher": "NIST",
        "year": 2024,
    },
    {
        "doc_id": "eu_ai_act",
        "title": "EU Artificial Intelligence Act (Regulation 2024/1689)",
        "short_name": "EU AI Act",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        "publisher": "European Union",
        "year": 2024,
    },
]

# Map doc_id -> short_name for convenient lookups in tools/agent.
DOC_SHORT_NAMES = {d["doc_id"]: d["short_name"] for d in DOCUMENTS}

# --------------------------------------------------------------------------- #
# Chunking parameters (recursive character splitter).
# --------------------------------------------------------------------------- #
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # characters of overlap between adjacent chunks

# --------------------------------------------------------------------------- #
# Embeddings.
# all-MiniLM-L6-v2: 384-dim, fast, runs locally on CPU — no API cost, no key.
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Persisted vector store artifacts.
VECTOR_STORE_PATH = VECTOR_STORE_DIR / "policy_store.npz"
VECTOR_STORE_META_PATH = VECTOR_STORE_DIR / "policy_store_meta.json"

# --------------------------------------------------------------------------- #
# Anthropic models — used only if ANTHROPIC_API_KEY is set.
# --------------------------------------------------------------------------- #
PRIMARY_MODEL = "claude-sonnet-4-6"
SECONDARY_MODEL = "claude-haiku-4-5-20251001"

# USD per 1,000,000 tokens. Source: Anthropic pricing (cached 2026-05).
# Databricks and self-hosted endpoints have $0 marginal per-token cost.
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

# --------------------------------------------------------------------------- #
# Databricks endpoints — two OSS models for the head-to-head comparison.
# Override via env vars if your workspace uses different endpoint names.
# --------------------------------------------------------------------------- #
DATABRICKS_PRIMARY_ENDPOINT = os.environ.get(
    "DATABRICKS_PRIMARY_ENDPOINT", "databricks-gpt-oss-120b"
)
DATABRICKS_SECONDARY_ENDPOINT = os.environ.get(
    "DATABRICKS_SECONDARY_ENDPOINT", "databricks-meta-llama-3-3-70b-instruct"
)

# --------------------------------------------------------------------------- #
# Open-source model — Qwen 3.6 self-hosted at llm.londonary.com, reached via
# its OpenAI-compatible API. Marginal per-token cost is effectively $0.
# --------------------------------------------------------------------------- #
OPENSOURCE_BASE_URL = os.environ.get(
    "OPENSOURCE_BASE_URL", "https://llm.londonary.com/v1"
)
OPENSOURCE_MODEL = os.environ.get("OPENSOURCE_MODEL", "")
OPENSOURCE_INFRA_USD_PER_HOUR = float(
    os.environ.get("OPENSOURCE_INFRA_USD_PER_HOUR", "0.0")
)

# --------------------------------------------------------------------------- #
# Judge configuration.
# Prefer Anthropic (higher accuracy for scoring). Fall back to the primary
# Databricks endpoint if ANTHROPIC_API_KEY is not set.
# --------------------------------------------------------------------------- #
_has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

JUDGE_MODEL = PRIMARY_MODEL if _has_anthropic else DATABRICKS_PRIMARY_ENDPOINT
JUDGE_BACKEND: str = "anthropic" if _has_anthropic else "databricks"

# --------------------------------------------------------------------------- #
# LLM Registry — OSS-first. Anthropic entries added only when key is present.
#
# Each entry must have:
#   label   — human-readable name shown in tables and charts
#   backend — "anthropic" | "databricks" | "openai_compat"
#
# Backend-specific keys:
#   anthropic     → model  (Anthropic model ID)
#   databricks    → endpoint  (Databricks serving endpoint name)
#   openai_compat → (reads OPENSOURCE_BASE_URL / OPENSOURCE_MODEL from env)
# --------------------------------------------------------------------------- #
LLM_REGISTRY: dict = {
    "db_primary": {
        "label": f"Databricks / {DATABRICKS_PRIMARY_ENDPOINT}",
        "backend": "databricks",
        "endpoint": DATABRICKS_PRIMARY_ENDPOINT,
    },
    "db_secondary": {
        "label": f"Databricks / {DATABRICKS_SECONDARY_ENDPOINT}",
        "backend": "databricks",
        "endpoint": DATABRICKS_SECONDARY_ENDPOINT,
    },
    "opensource": {
        "label": "Qwen 3.6 (llm.londonary.com)",
        "backend": "openai_compat",
    },
}

if _has_anthropic:
    LLM_REGISTRY["sonnet"] = {
        "label": "Claude Sonnet 4.6",
        "backend": "anthropic",
        "model": PRIMARY_MODEL,
    }
    LLM_REGISTRY["haiku"] = {
        "label": "Claude Haiku 4.5",
        "backend": "anthropic",
        "model": SECONDARY_MODEL,
    }

# Keys used for the head-to-head trace. Both Databricks models + londonary run
# always; Claude is added when the key is available.
EVAL_HEAD_TO_HEAD_KEYS: list = ["db_primary", "db_secondary", "opensource"]
if _has_anthropic:
    EVAL_HEAD_TO_HEAD_KEYS.append("sonnet")

# --------------------------------------------------------------------------- #
# Agent behavior.
# --------------------------------------------------------------------------- #
MAX_TOKENS = 2048
MAX_AGENT_ITERATIONS = 6   # safety cap on the ReAct tool loop
TOP_K_RESULTS = 4          # chunks returned per semantic search

SYSTEM_PROMPT = """You are the AI Policy Research Assistant for Meridian \
Governance Group, a consulting firm that advises clients on AI governance and \
regulatory compliance.

Your knowledge base consists of three authoritative documents:
  1. NIST AI Risk Management Framework (AI RMF 1.0)
  2. NIST AI 600-1 (Generative AI Profile)
  3. The EU Artificial Intelligence Act (Regulation 2024/1689)

RULES OF ENGAGEMENT:
- Answer ONLY questions about AI governance, AI risk management, and the \
regulations in your knowledge base. You MUST use your tools to ground every \
substantive claim in the source documents.
- If a question is unrelated to AI policy/governance (e.g. cooking, sports, \
general coding, personal advice), politely DECLINE. Explain that you are a \
specialized AI-policy assistant and state what you CAN help with. Do not \
attempt to answer out-of-scope questions from general knowledge.
- If the documents do not contain the answer, say so plainly rather than \
guessing. Cite the source document by name when you use retrieved material.
- Be precise, cite specifics (article numbers, function names, risk \
categories), and write for a professional governance audience.

Use the available tools to search and compare the source documents before \
answering. After gathering evidence, give a clear, well-structured final \
answer."""

# --------------------------------------------------------------------------- #
# Phoenix / OpenTelemetry tracing.
# --------------------------------------------------------------------------- #
PHOENIX_PROJECT_NAME = "meridian-policy-agent"


def get_api_key() -> str:
    """Return the Anthropic API key, raising a clear error if missing."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
            "your key, then load it (e.g. `from dotenv import load_dotenv; "
            "load_dotenv()`)."
        )
    return key


def get_databricks_client():
    """Return an OpenAI-compatible client backed by Databricks Model Serving.

    The Databricks SDK reads DATABRICKS_HOST and DATABRICKS_TOKEN from the
    environment (or ~/.databrickscfg) automatically.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as exc:
        raise RuntimeError(
            "databricks-sdk is not installed. Run: pip install databricks-sdk"
        ) from exc
    wc = WorkspaceClient()
    return wc.serving_endpoints.get_open_ai_client()


def get_judge_client():
    """Return the client used by the LLM judge.

    Returns an Anthropic client when ANTHROPIC_API_KEY is set; otherwise
    returns a Databricks OpenAI-compatible client so the judge runs on
    the primary OSS endpoint with no API key required.
    """
    if JUDGE_BACKEND == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=get_api_key())
    return get_databricks_client()


def create_agent(key: str, toolbox):
    """Instantiate a PolicyResearchAgent for the given LLM registry key."""
    from .agent import PolicyResearchAgent
    entry = LLM_REGISTRY[key]
    backend = entry["backend"]

    if backend == "anthropic":
        return PolicyResearchAgent(toolbox, model=entry["model"], backend="anthropic")

    if backend == "databricks":
        return PolicyResearchAgent(
            toolbox,
            client=get_databricks_client(),
            model=entry["endpoint"],
            backend="databricks",
        )

    if backend == "openai_compat":
        from openai import OpenAI
        client = OpenAI(base_url=OPENSOURCE_BASE_URL, api_key=get_opensource_api_key())
        return PolicyResearchAgent(toolbox, client=client, backend="openai_compat")

    raise ValueError(f"Unknown backend '{backend}' for key '{key}'")


def get_opensource_api_key() -> str:
    """Return the API key for the open-source/llama.cpp server.

    Many local deployments require no key; the OpenAI SDK still needs a
    non-empty string, so we fall back to a harmless placeholder.
    """
    return (
        os.environ.get("OPENSOURCE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "sk-no-key-required"
    )


def resolve_opensource_model(client) -> str:
    """Return the open-source model id, auto-discovering it if not configured.

    If OPENSOURCE_MODEL is set we trust it. Otherwise we ask the server which
    models it serves (GET /v1/models) and take the first one.
    """
    if OPENSOURCE_MODEL:
        return OPENSOURCE_MODEL
    try:
        listed = client.models.list()
        ids = [m.id for m in listed.data]
        if ids:
            return ids[0]
    except Exception:
        pass
    return "qwen3.6"
