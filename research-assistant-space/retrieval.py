"""Retrieval for the research assistant.

The corpus -- loading, normalisation, the fallback set and the embedding index
-- lives in the vendored papers.py, shared byte-for-byte with
paper-recommender-space. This module holds only the Retriever built on top.
"""

from __future__ import annotations

import numpy as np

from papers import (
    DATASET_ID,
    FALLBACK_PAPERS,
    MAX_PAPERS,
    build_index,
    load_papers,
    normalize,
    normalize_row,
)

__all__ = [
    "DATASET_ID",
    "FALLBACK_PAPERS",
    "MAX_PAPERS",
    "Retriever",
    "build_index",
    "load_papers",
    "normalize",
    "normalize_row",
]


class Retriever:
    """Embeds a corpus once and answers similarity queries against it."""

    def __init__(self, papers: list[dict], encode_fn):
        self.papers = papers
        self.encode_fn = encode_fn
        self.matrix = build_index(papers, encode_fn)

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not query.strip() or not self.papers:
            return []
        vec = np.asarray(self.encode_fn([query]), dtype="float32").reshape(-1)
        vec = vec / (np.linalg.norm(vec) or 1.0)
        sims = self.matrix @ vec
        order = np.argsort(sims)[::-1][:k]
        return [
            {**self.papers[int(i)], "similarity": float(sims[i]) * 100} for i in order
        ]
