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

import json
from contextlib import contextmanager
from typing import Any

import numpy as np

from . import config
from .vector_store import SimpleVectorStore

# Optional MLflow tracing. When mlflow is present (Databricks), each retrieval
# emits a RETRIEVER span whose output is the retrieved chunks as Documents —
# this is what mlflow.genai's built-in RetrievalGroundedness scorer reads to
# check the answer is backed by the passages. When mlflow is absent (local
# runs, the self-check), this degrades to a no-op so nothing else changes.
try:
    import mlflow as _mlflow

    try:
        from mlflow.entities import Document as _MLflowDocument
    except Exception:  # older mlflow without Document
        _MLflowDocument = None
    _MLFLOW_AVAILABLE = True
except Exception:
    _mlflow = None
    _MLflowDocument = None
    _MLFLOW_AVAILABLE = False


def _to_documents(hits: list[dict[str, Any]]) -> list[Any]:
    """Render retrieved hits as MLflow Documents (or plain dicts on old mlflow)."""
    docs: list[Any] = []
    for i, h in enumerate(hits):
        text = h.get("text", "")
        meta = {k: v for k, v in h.items() if k != "text"}
        if _MLflowDocument is not None:
            docs.append(_MLflowDocument(id=str(i), page_content=text, metadata=meta))
        else:
            docs.append({"id": str(i), "page_content": text, "metadata": meta})
    return docs


@contextmanager
def _retriever_span(name: str, query: str):
    """Yield a ``record(hits)`` callback that logs hits to a RETRIEVER span.

    No-op (the callback does nothing) when mlflow is unavailable.
    """
    if not _MLFLOW_AVAILABLE:
        yield lambda hits: None
        return
    with _mlflow.start_span(name=name, span_type="RETRIEVER") as span:
        span.set_inputs({"query": query})
        yield lambda hits: span.set_outputs(_to_documents(hits))


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
        with _retriever_span("search_policy_documents", query) as record:
            hits = self.store.search(self._embed(query), top_k=k)
            record(hits)
            return self._format_hits(hits)

    # --------------------------------------------------------------------- #
    # Tool 2: compare how two frameworks treat the same topic.
    # --------------------------------------------------------------------- #
    def compare_frameworks(self, topic: str, doc_id_a: str, doc_id_b: str) -> str:
        per_doc = max(2, config.TOP_K_RESULTS // 2)
        q = self._embed(topic)
        with _retriever_span("compare_frameworks", topic) as record:
            hits_a = self.store.search(q, top_k=per_doc, doc_id=doc_id_a)
            hits_b = self.store.search(q, top_k=per_doc, doc_id=doc_id_b)
            record(hits_a + hits_b)
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
        with _retriever_span("summarize_policy_topic", topic) as record:
            hits = self.store.search(self._embed(topic), top_k=config.TOP_K_RESULTS + 2)
            record(hits)
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
# Databricks Vector Search toolbox — no local embedder or numpy store needed.
# The VS index (created in 01_data_pipeline.ipynb) handles embeddings itself.
# --------------------------------------------------------------------------- #

# Maps config doc_id values to the 'source' column written by the pipeline.
_DOC_ID_TO_SOURCE = {
    "nist_ai_rmf_1_0": "nist_ai_rmf",
    "nist_ai_600_1":   "nist_ai_600_1",
    "eu_ai_act":       "eu_ai_act",
}
_SOURCE_TO_DOC_ID = {v: k for k, v in _DOC_ID_TO_SOURCE.items()}


class DatabricksVSToolbox:
    """PolicyToolbox backed by Databricks Vector Search.

    Pass the live VS index object (from VectorSearchClient.get_index()).
    No SentenceTransformer or .npz file required — the index embeds queries
    internally using the model it was built with (databricks-bge-large-en).
    """

    def __init__(self, vs_index: Any) -> None:
        self.vs_index = vs_index

    def _search_raw(
        self, query: str, top_k: int, source_filter: str | None = None
    ) -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {
            "query_text": query,
            "columns": ["source", "text"],
            "num_results": top_k,
        }
        if source_filter:
            kwargs["filters_json"] = json.dumps({"source": source_filter})
        result = self.vs_index.similarity_search(**kwargs)
        rows = result.get("result", {}).get("data_array", [])
        return [
            {
                "doc_id": _SOURCE_TO_DOC_ID.get(r[0], r[0]),
                "text": r[1],
                "score": r[2] if len(r) > 2 else 0.0,
            }
            for r in rows
        ]

    @staticmethod
    def _format_hits(hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "No relevant passages were found in the knowledge base."
        lines: list[str] = []
        for i, h in enumerate(hits, start=1):
            src = config.DOC_SHORT_NAMES.get(h.get("doc_id", ""), h.get("doc_id", "?"))
            score = h.get("score")
            score_str = f" (similarity={score:.3f})" if score else ""
            lines.append(f"[{i}] Source: {src}{score_str}\n{h['text'].strip()}")
        return "\n\n".join(lines)

    def search_policy_documents(self, query: str, top_k: int | None = None) -> str:
        with _retriever_span("search_policy_documents", query) as record:
            hits = self._search_raw(query, top_k or config.TOP_K_RESULTS)
            record(hits)
            return self._format_hits(hits)

    def compare_frameworks(self, topic: str, doc_id_a: str, doc_id_b: str) -> str:
        per_doc = max(2, config.TOP_K_RESULTS // 2)
        src_a = _DOC_ID_TO_SOURCE.get(doc_id_a, doc_id_a)
        src_b = _DOC_ID_TO_SOURCE.get(doc_id_b, doc_id_b)
        with _retriever_span("compare_frameworks", topic) as record:
            hits_a = self._search_raw(topic, per_doc, source_filter=src_a)
            hits_b = self._search_raw(topic, per_doc, source_filter=src_b)
            record(hits_a + hits_b)
            name_a = config.DOC_SHORT_NAMES.get(doc_id_a, doc_id_a)
            name_b = config.DOC_SHORT_NAMES.get(doc_id_b, doc_id_b)
            return (
                f"=== {name_a} on '{topic}' ===\n{self._format_hits(hits_a)}\n\n"
                f"=== {name_b} on '{topic}' ===\n{self._format_hits(hits_b)}"
            )

    def summarize_policy_topic(self, topic: str) -> str:
        with _retriever_span("summarize_policy_topic", topic) as record:
            hits = self._search_raw(topic, config.TOP_K_RESULTS + 2)
            record(hits)
            return self._format_hits(hits)

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
        except KeyError as exc:
            return f"Error: missing required argument {exc} for tool '{tool_name}'."
        except Exception as exc:
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
