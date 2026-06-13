"""A minimal, dependency-light vector store using numpy cosine similarity.

Why not FAISS? faiss-cpu does not reliably ship wheels for very new Python
builds (this project was developed on Python 3.14). A numpy dot-product over a
few thousand normalized embeddings is plenty fast for a 3-document corpus and
keeps the install footprint small and portable.

AI-USAGE NOTE: This class was drafted with Claude Code assistance; the cosine
math and persistence format were reviewed by the author.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SimpleVectorStore:
    """Stores embeddings + chunk text/metadata and does cosine-similarity search.

    Embeddings are L2-normalized on add, so a cosine search reduces to a single
    matrix-vector dot product.
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        # (N, dim) float32 matrix of L2-normalized vectors.
        self._embeddings: np.ndarray = np.empty((0, embedding_dim), dtype=np.float32)
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []

    # --------------------------------------------------------------------- #
    # Build
    # --------------------------------------------------------------------- #
    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Append a batch of embeddings with parallel text + metadata lists."""
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("embeddings, texts, and metadatas must be equal length")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"expected embeddings of shape (n, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )

        normalized = self._normalize(embeddings)
        self._embeddings = np.vstack([self._embeddings, normalized])
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    # --------------------------------------------------------------------- #
    # Query
    # --------------------------------------------------------------------- #
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top_k most similar chunks.

        If ``doc_id`` is given, restrict the search to chunks from that document.
        Each result dict has: text, score, and the chunk's metadata fields.
        """
        if len(self._texts) == 0:
            return []

        q = self._normalize(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        # Cosine similarity == dot product of normalized vectors.
        scores = (self._embeddings @ q.T).ravel()

        candidate_idx = np.arange(len(self._texts))
        if doc_id is not None:
            mask = np.array(
                [m.get("doc_id") == doc_id for m in self._metadatas], dtype=bool
            )
            candidate_idx = candidate_idx[mask]
            if candidate_idx.size == 0:
                return []
            scores = scores[candidate_idx]

        # Top-k by score (descending).
        k = min(top_k, candidate_idx.size)
        top_local = np.argsort(scores)[::-1][:k]

        results: list[dict[str, Any]] = []
        for local_i in top_local:
            global_i = int(candidate_idx[local_i])
            results.append(
                {
                    "text": self._texts[global_i],
                    "score": float(scores[local_i]),
                    **self._metadatas[global_i],
                }
            )
        return results

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def save(self, npz_path: str | Path, meta_path: str | Path) -> None:
        """Persist embeddings to an .npz and texts/metadata to JSON."""
        np.savez_compressed(npz_path, embeddings=self._embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_dim": self.embedding_dim,
                    "texts": self._texts,
                    "metadatas": self._metadatas,
                },
                f,
            )

    @classmethod
    def load(cls, npz_path: str | Path, meta_path: str | Path) -> "SimpleVectorStore":
        """Reconstruct a store previously written by ``save``."""
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        store = cls(embedding_dim=meta["embedding_dim"])
        with np.load(npz_path) as data:
            store._embeddings = data["embeddings"].astype(np.float32)
        store._texts = meta["texts"]
        store._metadatas = meta["metadatas"]
        return store

    # --------------------------------------------------------------------- #
    # Introspection (used by the EDA notebook)
    # --------------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._texts)

    @property
    def embeddings(self) -> np.ndarray:
        return self._embeddings

    @property
    def texts(self) -> list[str]:
        return self._texts

    @property
    def metadatas(self) -> list[dict[str, Any]]:
        return self._metadatas

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize rows; guard against divide-by-zero on empty vectors."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
