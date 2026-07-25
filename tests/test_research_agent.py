"""Tests for the research assistant's agent + retrieval logic."""

import numpy as np

from conftest import load_local_module

agent = load_local_module("research_agent", "research-assistant-space/agent.py")
retrieval = load_local_module(
    "research_retrieval", "research-assistant-space/retrieval.py"
)


# --- agent: query planning -------------------------------------------------
def test_parse_queries_strips_bullets_and_caps():
    text = "1. attention mechanisms\n- transformer architecture\n* long range\nfourth"
    assert agent.parse_queries(text, 3) == [
        "attention mechanisms",
        "transformer architecture",
        "long range",
    ]


def test_plan_queries_falls_back_on_error():
    def boom(messages):
        raise RuntimeError("llm down")

    assert agent.plan_queries("my question", boom) == ["my question"]


def test_plan_queries_falls_back_on_empty():
    assert agent.plan_queries("q", lambda messages: "   ") == ["q"]


# --- agent: helpers --------------------------------------------------------
def test_dedupe_keep_order_case_insensitive():
    papers = [{"title": "A"}, {"title": "a"}, {"title": "B"}]
    assert [p["title"] for p in agent.dedupe_keep_order(papers)] == ["A", "B"]


def test_format_context_numbers_papers():
    ctx = agent.format_context([{"title": "T", "year": "2020", "abstract": "X"}])
    assert "[1] T (2020)" in ctx and "X" in ctx


def test_synthesize_answer_no_papers():
    assert "couldn't find" in agent.synthesize_answer("q", [], lambda m: "x").lower()


# --- agent: end to end with fakes -----------------------------------------
def test_answer_end_to_end_with_fakes():
    calls = {"plan": 0, "synth": 0}

    def fake_chat(messages):
        if "search queries" in messages[0]["content"]:
            calls["plan"] += 1
            return "transformers\nattention"
        calls["synth"] += 1
        assert "[1]" in messages[1]["content"]  # context was passed in
        return "Attention enables long-range modeling [1]."

    def fake_retriever(query, k):
        return [
            {"title": f"paper-{query}", "abstract": "a", "year": "2020", "url": "u"}
        ]

    result = agent.answer(
        "How do transformers work?", fake_retriever, fake_chat, max_queries=2
    )
    assert result["queries"] == ["transformers", "attention"]
    assert [p["title"] for p in result["papers"]] == [
        "paper-transformers",
        "paper-attention",
    ]
    assert "[1]" in result["answer"]
    assert calls == {"plan": 1, "synth": 1}


def test_answer_empty_question():
    res = agent.answer("   ", lambda q, k: [], lambda m: "x")
    assert res["papers"] == [] and "enter a question" in res["answer"].lower()


# --- retrieval -------------------------------------------------------------
def _keyword_encoder(texts):
    vecs = []
    for text in texts:
        low = text.lower()
        vecs.append([1.0, 0.0] if ("cat" in low or "feline" in low) else [0.0, 1.0])
    return np.array(vecs, dtype="float32")


def test_retriever_search_ranks_by_similarity():
    papers = [
        {"title": "cat", "abstract": "feline", "year": "2020", "url": "u"},
        {"title": "dog", "abstract": "canine", "year": "2021", "url": "u"},
    ]
    retriever = retrieval.Retriever(papers, _keyword_encoder)
    top = retriever.search("cat feline", k=2)
    assert top[0]["title"] == "cat"
    assert top[0]["similarity"] >= top[1]["similarity"]


def test_retriever_empty_query_returns_empty():
    retriever = retrieval.Retriever([{"title": "x", "abstract": "y"}], _keyword_encoder)
    assert retriever.search("   ") == []


def test_load_papers_injected_loader_and_failure():
    rows = [
        {
            "title": "T",
            "abstract": "A",
            "authors": ["X", "Y", "Z", "W"],
            "primary_category": "cs.LG",
            "published": "2019",
            "url": "u",
        }
    ]
    papers = retrieval.load_papers(loader=lambda dataset_id, split: rows)
    assert papers and papers[0]["title"] == "T"
    assert papers[0]["authors"].endswith("et al.")

    def boom(dataset_id, split):
        raise RuntimeError("hub down")

    assert retrieval.load_papers(loader=boom) is None
