"""Ranking logic for the paper recommender (no Gradio import) so it is testable.

The corpus itself -- loading, normalisation, the fallback set and the embedding
index -- lives in the vendored papers.py, shared byte-for-byte with
research-assistant-space. Ranking is cosine similarity over the embedding model;
the corpus is small enough that a plain NumPy matmul is fast and an ANN index
(e.g. FAISS) isn't needed.
"""

from __future__ import annotations

from typing import Optional

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
    "CATEGORIES",
    "DATASET_ID",
    "FALLBACK_PAPERS",
    "MAX_PAPERS",
    "build_index",
    "load_papers",
    "normalize",
    "normalize_row",
    "recommend",
]

CATEGORIES = {
    "All": None,
    "cs.AI - Artificial Intelligence": "cs.AI",
    "cs.CL - Computation & Language": "cs.CL",
    "cs.CV - Computer Vision": "cs.CV",
    "cs.LG - Machine Learning": "cs.LG",
    "stat.ML - Machine Learning (Stats)": "stat.ML",
}


def recommend(
    query: str,
    papers: list[dict],
    paper_matrix: np.ndarray,
    encode_fn,
    category_value: Optional[str],
    num_results: int,
) -> list[dict]:
    """Return ranked, category-filtered results for *query* (most similar first)."""
    if not query.strip():
        return []
    query_vec = np.asarray(encode_fn([query]), dtype="float32").reshape(-1)
    query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)
    sims = paper_matrix @ query_vec
    order = np.argsort(sims)[::-1]

    results = []
    for idx in order:
        if len(results) >= num_results:
            break
        paper = papers[int(idx)]
        if category_value and paper["category"] != category_value:
            continue
        results.append({**paper, "similarity": float(sims[idx]) * 100})
    return results
