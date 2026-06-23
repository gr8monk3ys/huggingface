"""Retrieval over the arXiv papers dataset for the research assistant.

Loads ``gr8monk3ys/cs-ml-academic-papers`` from the Hub (with a small
landmark-paper fallback) and provides cosine-similarity search. The embedding
function is injected so the logic is unit-testable without heavy deps, and the
dataset loader is injectable / fails soft so there is no hard network dependency.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

DATASET_ID = os.environ.get("PAPERS_DATASET", "gr8monk3ys/cs-ml-academic-papers")
MAX_PAPERS = int(os.environ.get("PAPERS_MAX", "2500"))

FALLBACK_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer, a network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions. It is more parallelizable and reaches state-of-the-art translation quality with less training time.",
        "authors": "Vaswani et al.",
        "category": "cs.CL",
        "year": "2017",
        "url": "https://arxiv.org/abs/1706.03762",
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "BERT pre-trains deep bidirectional representations by jointly conditioning on left and right context, obtaining state-of-the-art results on a wide range of NLP tasks with a single additional output layer.",
        "authors": "Devlin et al.",
        "category": "cs.CL",
        "year": "2018",
        "url": "https://arxiv.org/abs/1810.04805",
    },
    {
        "title": "Deep Residual Learning for Image Recognition (ResNet)",
        "abstract": "We present a residual learning framework that eases the training of substantially deeper networks by reformulating layers as learning residual functions, enabling accuracy gains from greatly increased depth.",
        "authors": "He et al.",
        "category": "cs.CV",
        "year": "2015",
        "url": "https://arxiv.org/abs/1512.03385",
    },
    {
        "title": "You Only Look Once (YOLO): Real-Time Object Detection",
        "abstract": "We frame object detection as a single regression problem from image pixels to bounding boxes and class probabilities, enabling extremely fast, end-to-end real-time detection.",
        "authors": "Redmon et al.",
        "category": "cs.CV",
        "year": "2016",
        "url": "https://arxiv.org/abs/1506.02640",
    },
    {
        "title": "Dropout: A Simple Way to Prevent Neural Network Overfitting",
        "abstract": "Dropout randomly drops units during training to prevent co-adaptation, acting as regularization that significantly reduces overfitting in deep neural networks.",
        "authors": "Srivastava et al.",
        "category": "cs.LG",
        "year": "2014",
        "url": "https://jmlr.org/papers/v15/srivastava14a.html",
    },
    {
        "title": "Batch Normalization",
        "abstract": "Batch normalization reduces internal covariate shift by normalizing layer inputs, allowing higher learning rates, less careful initialization, and acting as a regularizer.",
        "authors": "Ioffe, Szegedy",
        "category": "cs.LG",
        "year": "2015",
        "url": "https://arxiv.org/abs/1502.03167",
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "Adam is a computationally efficient first-order gradient-based optimizer combining momentum and adaptive per-parameter learning rates, well suited to large-scale and noisy problems.",
        "authors": "Kingma, Ba",
        "category": "cs.LG",
        "year": "2014",
        "url": "https://arxiv.org/abs/1412.6980",
    },
    {
        "title": "An Image is Worth 16x16 Words (Vision Transformer)",
        "abstract": "A pure transformer applied to sequences of image patches performs very well on image classification when pre-trained on large datasets, challenging the necessity of convolutions.",
        "authors": "Dosovitskiy et al.",
        "category": "cs.CV",
        "year": "2020",
        "url": "https://arxiv.org/abs/2010.11929",
    },
]


def _first(row: dict, *keys: str, default: str = ""):
    """Return the first present, non-empty value among *keys*."""
    for key in keys:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return default


def normalize_row(row: dict) -> dict:
    """Map a raw dataset row (varied schemas) to our paper dict."""
    category = _first(row, "primary_category", "category", "categories")
    if isinstance(category, list):
        category = category[0] if category else ""
    year = str(_first(row, "year", "published", "updated", "date"))[:4]
    authors = _first(row, "authors", "author")
    if isinstance(authors, list):
        authors = ", ".join(authors[:3]) + (", et al." if len(authors) > 3 else "")
    return {
        "title": str(_first(row, "title")).strip(),
        "abstract": str(_first(row, "abstract", "summary")).strip(),
        "authors": authors or "Unknown",
        "category": category or "",
        "year": year,
        "url": _first(row, "url", "arxiv_url", "pdf_url", "entry_id", "arxiv_id"),
    }


def load_papers(
    dataset_id: str = DATASET_ID,
    split: str = "train",
    max_papers: int = MAX_PAPERS,
    loader=None,
) -> Optional[list[dict]]:
    """Load papers from the Hub dataset, or return ``None`` on any failure."""
    try:
        if loader is None:
            from datasets import load_dataset as loader
        dataset = loader(dataset_id, split=split)
        papers = []
        for row in dataset:
            paper = normalize_row(row)
            if paper["title"] and paper["abstract"]:
                papers.append(paper)
            if len(papers) >= max_papers:
                break
        return papers or None
    except Exception:
        return None


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows, guarding against zero vectors."""
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_index(papers: list[dict], encode_fn) -> np.ndarray:
    """Embed ``"{title} {abstract}"`` for each paper and L2-normalize."""
    texts = [f"{p['title']} {p['abstract']}" for p in papers]
    return normalize(encode_fn(texts))


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
