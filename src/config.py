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
        # Official Journal consolidated text (PDF).
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
# Models — TWO Claude models are used so the evaluation can compare them.
# Pricing (per 1M tokens) is recorded here for the ROI calculation.
# --------------------------------------------------------------------------- #
PRIMARY_MODEL = "claude-sonnet-4-6"            # higher capability, higher cost
SECONDARY_MODEL = "claude-haiku-4-5-20251001"  # faster + cheaper, for ROI compare
JUDGE_MODEL = "claude-sonnet-4-6"              # LLM-as-judge scorer

# USD per 1,000,000 tokens. Source: Anthropic pricing (cached 2026-05).
MODEL_PRICING = {
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
}

# --------------------------------------------------------------------------- #
# Open-source model (3rd comparison point) — a Qwen3 model self-hosted on a
# llama.cpp `llama-server`, reached via its OpenAI-compatible API.
# Self-hosted => marginal per-token cost is effectively $0 (you pay for GPU
# time, not per token). estimate_cost() returns 0 for any model not listed in
# MODEL_PRICING, which is exactly what we want here. Set the constants below
# (or the matching env vars) to point at your server.
# --------------------------------------------------------------------------- #
OPENSOURCE_BASE_URL = os.environ.get(
    "OPENSOURCE_BASE_URL", "https://llm.londonary.com/v1"
)
# Leave empty to auto-discover the model id from the server's /v1/models.
OPENSOURCE_MODEL = os.environ.get("OPENSOURCE_MODEL", "")
# Optional: amortized infra cost ($/hour for the GPU host) for an infra-based
# ROI angle. 0.0 => treat the open-source model's marginal cost as free.
OPENSOURCE_INFRA_USD_PER_HOUR = float(
    os.environ.get("OPENSOURCE_INFRA_USD_PER_HOUR", "0.0")
)

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


def get_opensource_api_key() -> str:
    """Return the API key for the open-source/llama.cpp server.

    Many local llama.cpp deployments require no key; the OpenAI SDK still needs
    a non-empty string, so we fall back to a harmless placeholder.
    """
    return (
        os.environ.get("OPENSOURCE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or "sk-no-key-required"
    )


def resolve_opensource_model(client) -> str:
    """Return the open-source model id, auto-discovering it if not configured.

    If OPENSOURCE_MODEL is set we trust it. Otherwise we ask the server which
    models it serves (``GET /v1/models``) and take the first one.
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
    # Last resort: a sensible default. Override via OPENSOURCE_MODEL if wrong.
    return "qwen3"
