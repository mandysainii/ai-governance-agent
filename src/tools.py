"""The agent's callable tools, plus the Anthropic tool-schema definitions.

Three tools are exposed to Claude (the rubric requires the agent to have an LLM
plus at least two tools):

    1. search_policy_documents   — semantic search across the whole corpus
    2. compare_frameworks        — search the SAME topic in two named documents
    3. summarize_policy_topic     — gather the top evidence for one topic

Tools are implemented as instance methods on ``PolicyToolbox`` so they can share
a single embedding model + vector store (instance-based, per the project's
Java-style preference applied to Python). The module also exposes the JSON tool
schemas the agent passes to the Messages API.

AI-USAGE NOTE: Tool schemas and retrieval glue were drafted with Claude Code;
the retrieval logic and schemas were reviewed by the author.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config
from .vector_store import SimpleVectorStore


class PolicyToolbox:
    """Holds the embedder + vector store and implements the agent's tools."""

    def __init__(self, store: SimpleVectorStore, embed_model: Any) -> None:
        # ``embed_model`` is a sentence_transformers.SentenceTransformer (or any
        # object with ``.encode(list[str]) -> np.ndarray``). Injected so the
        # store and tools share one model and the notebook controls its load.
        self.store = store
        self.embed_model = embed_model

    # --------------------------------------------------------------------- #
    # Internal: embed a query string to a single vector.
    # --------------------------------------------------------------------- #
    def _embed(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text])
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    @staticmethod
    def _format_hits(hits: list[dict[str, Any]]) -> str:
        """Render retrieved chunks into a compact, citable text block."""
        if not hits:
            return "No relevant passages were found in the knowledge base."
        lines: list[str] = []
        for i, h in enumerate(hits, start=1):
            src = config.DOC_SHORT_NAMES.get(h.get("doc_id", ""), h.get("doc_id", "?"))
            page = h.get("page")
            loc = f", p.{page}" if page is not None else ""
            lines.append(
                f"[{i}] Source: {src}{loc} (similarity={h['score']:.3f})\n"
                f"{h['text'].strip()}"
            )
        return "\n\n".join(lines)

    # --------------------------------------------------------------------- #
    # Tool 1: semantic search across the whole corpus.
    # --------------------------------------------------------------------- #
    def search_policy_documents(self, query: str, top_k: int | None = None) -> str:
        k = top_k or config.TOP_K_RESULTS
        hits = self.store.search(self._embed(query), top_k=k)
        return self._format_hits(hits)

    # --------------------------------------------------------------------- #
    # Tool 2: compare how two frameworks treat the same topic.
    # --------------------------------------------------------------------- #
    def compare_frameworks(self, topic: str, doc_id_a: str, doc_id_b: str) -> str:
        per_doc = max(2, config.TOP_K_RESULTS // 2)
        q = self._embed(topic)
        hits_a = self.store.search(q, top_k=per_doc, doc_id=doc_id_a)
        hits_b = self.store.search(q, top_k=per_doc, doc_id=doc_id_b)
        name_a = config.DOC_SHORT_NAMES.get(doc_id_a, doc_id_a)
        name_b = config.DOC_SHORT_NAMES.get(doc_id_b, doc_id_b)
        return (
            f"=== {name_a} on '{topic}' ===\n{self._format_hits(hits_a)}\n\n"
            f"=== {name_b} on '{topic}' ===\n{self._format_hits(hits_b)}"
        )

    # --------------------------------------------------------------------- #
    # Tool 3: gather the strongest evidence for a single topic (deeper recall).
    # --------------------------------------------------------------------- #
    def summarize_policy_topic(self, topic: str) -> str:
        hits = self.store.search(self._embed(topic), top_k=config.TOP_K_RESULTS + 2)
        return self._format_hits(hits)

    # --------------------------------------------------------------------- #
    # Dispatch: route a tool_use block from the agent to the right method.
    # --------------------------------------------------------------------- #
    def dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        try:
            if tool_name == "search_policy_documents":
                return self.search_policy_documents(
                    query=tool_input["query"],
                    top_k=tool_input.get("top_k"),
                )
            if tool_name == "compare_frameworks":
                return self.compare_frameworks(
                    topic=tool_input["topic"],
                    doc_id_a=tool_input["doc_id_a"],
                    doc_id_b=tool_input["doc_id_b"],
                )
            if tool_name == "summarize_policy_topic":
                return self.summarize_policy_topic(topic=tool_input["topic"])
            return f"Error: unknown tool '{tool_name}'."
        except KeyError as exc:  # missing required argument
            return f"Error: missing required argument {exc} for tool '{tool_name}'."
        except Exception as exc:  # defensive: never crash the agent loop
            return f"Error while running tool '{tool_name}': {exc}"


# --------------------------------------------------------------------------- #
# Tool schemas passed to the Anthropic Messages API.
# The valid doc_id enum values are generated from config so they never drift.
# --------------------------------------------------------------------------- #
_DOC_IDS = [d["doc_id"] for d in config.DOCUMENTS]
_DOC_DESC = "; ".join(f"'{d['doc_id']}' = {d['short_name']}" for d in config.DOCUMENTS)

TOOL_SCHEMAS = [
    {
        "name": "search_policy_documents",
        "description": (
            "Semantic search across the full AI-policy knowledge base (NIST AI "
            "RMF, NIST Generative AI Profile, and the EU AI Act). Returns the "
            "most relevant passages with their source and similarity score. Use "
            "this for any question about the content of the regulations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many passages to return (default 4).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_frameworks",
        "description": (
            "Compare how two specific documents address the same topic. Returns "
            "the most relevant passages from each document side by side. Use "
            "this when the user asks how two frameworks differ or align. Valid "
            f"document IDs: {_DOC_DESC}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to compare (e.g. 'risk assessment').",
                },
                "doc_id_a": {
                    "type": "string",
                    "enum": _DOC_IDS,
                    "description": "First document ID.",
                },
                "doc_id_b": {
                    "type": "string",
                    "enum": _DOC_IDS,
                    "description": "Second document ID.",
                },
            },
            "required": ["topic", "doc_id_a", "doc_id_b"],
        },
    },
    {
        "name": "summarize_policy_topic",
        "description": (
            "Gather the strongest evidence across the whole corpus for a single "
            "topic, returning extra passages for a thorough summary. Use this "
            "when the user asks you to summarize or explain a policy topic."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The policy topic to summarize.",
                },
            },
            "required": ["topic"],
        },
    },
]
