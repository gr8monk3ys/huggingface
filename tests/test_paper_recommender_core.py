"""Tests for the paper-recommender pure logic (core.py)."""

import numpy as np

from conftest import load_local_module

core = load_local_module("paper_recommender_core", "paper-recommender-space/core.py")


def test_normalize_guards_zero_vector():
    out = core.normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(np.linalg.norm(out, axis=1), [0.0, 1.0])


def test_normalize_row_maps_dataset_schema():
    row = {
        "title": "  A Title ",
        "abstract": "Some abstract",
        "authors": ["A", "B", "C", "D", "E"],
        "primary_category": "cs.LG",
        "categories": ["cs.LG", "stat.ML"],
        "published": "2021-05-01T00:00:00",
        "url": "http://x",
    }
    paper = core.normalize_row(row)
    assert paper["title"] == "A Title"
    assert paper["category"] == "cs.LG"
    assert paper["year"] == "2021"
    assert paper["authors"].endswith("et al.")  # >4 authors -> truncated


def test_load_papers_with_injected_loader_drops_incomplete():
    rows = [
        {"title": "T1", "abstract": "A1", "authors": ["X"], "primary_category": "cs.CL", "published": "2020", "url": "u1"},
        {"title": "", "abstract": "missing title", "primary_category": "cs.CL"},
    ]
    papers = core.load_papers(loader=lambda dataset_id, split: rows)
    assert papers is not None
    assert [p["title"] for p in papers] == ["T1"]


def test_load_papers_returns_none_on_failure():
    def boom(dataset_id, split):
        raise RuntimeError("hub down")

    assert core.load_papers(loader=boom) is None


def _keyword_encoder(texts):
    vecs = []
    for text in texts:
        low = text.lower()
        vecs.append([1.0, 0.0] if ("cat" in low or "feline" in low) else [0.0, 1.0])
    return np.array(vecs, dtype="float32")


def test_recommend_ranks_and_filters():
    papers = [
        {"title": "cat", "abstract": "feline", "category": "cs.CV", "authors": "a", "year": "2020", "url": "u"},
        {"title": "dog", "abstract": "canine", "category": "cs.LG", "authors": "b", "year": "2021", "url": "u"},
    ]
    matrix = core.build_index(papers, _keyword_encoder)

    top = core.recommend("cat feline", papers, matrix, _keyword_encoder, None, 5)
    assert top[0]["title"] == "cat"
    assert top[0]["similarity"] > top[1]["similarity"]

    filtered = core.recommend("cat feline", papers, matrix, _keyword_encoder, "cs.LG", 5)
    assert [r["title"] for r in filtered] == ["dog"]

    assert core.recommend("   ", papers, matrix, _keyword_encoder, None, 5) == []
