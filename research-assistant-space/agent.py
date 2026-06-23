"""A small agentic RAG loop: plan search queries -> retrieve -> synthesize.

The loop is genuinely multi-step: an LLM first decomposes the question into
focused search queries, those drive (possibly several) retrieval calls, and a
second LLM step writes a grounded, citation-bearing answer over the retrieved
papers.

``chat_fn(messages) -> str`` and ``retriever(query, k) -> list[paper]`` are
injected, so the orchestration is unit-testable without an LLM or network.
"""

from __future__ import annotations

import re

PLANNER_SYSTEM = (
    "You are a research assistant. Given a user's question, produce up to {n} "
    "short, focused search queries (keywords) that would retrieve relevant "
    "academic papers. Return one query per line, with no numbering or extra text."
)

SYNTH_SYSTEM = (
    "You are a careful research assistant. Answer the question using ONLY the "
    "numbered paper abstracts provided as context. Cite sources inline as [n] "
    "using their numbers. If the context is insufficient, say so explicitly. "
    "Be concise and accurate; never invent papers, numbers, or findings."
)


def parse_queries(text: str, max_queries: int) -> list[str]:
    """Extract clean search queries from an LLM response (one per line)."""
    queries = []
    for line in (text or "").splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\s\"']+", "", line).strip().strip("\"'")
        if cleaned:
            queries.append(cleaned)
    return queries[:max_queries]


def plan_queries(question: str, chat_fn, max_queries: int = 3) -> list[str]:
    """Ask the LLM for focused search queries; fall back to the raw question."""
    try:
        text = chat_fn(
            [
                {"role": "system", "content": PLANNER_SYSTEM.format(n=max_queries)},
                {"role": "user", "content": question},
            ]
        )
        queries = parse_queries(text, max_queries)
    except Exception:
        queries = []
    return queries or [question.strip()]


def dedupe_keep_order(papers: list[dict]) -> list[dict]:
    """Drop duplicate papers (by title) while preserving first-seen order."""
    seen = set()
    out = []
    for paper in papers:
        key = (paper.get("title") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(paper)
    return out


def format_context(papers: list[dict]) -> str:
    """Render retrieved papers as a numbered context block for the LLM."""
    blocks = []
    for i, paper in enumerate(papers, 1):
        blocks.append(
            f"[{i}] {paper.get('title', '')} ({paper.get('year', 'n/a')})\n"
            f"{paper.get('abstract', '')}"
        )
    return "\n\n".join(blocks)


def synthesize_answer(question: str, papers: list[dict], chat_fn) -> str:
    """Ask the LLM to answer the question grounded in the retrieved papers."""
    if not papers:
        return "I couldn't find relevant papers to answer that."
    user = (
        f"Question: {question}\n\n"
        f"Context:\n{format_context(papers)}\n\n"
        "Answer the question using inline [n] citations."
    )
    return chat_fn(
        [
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ]
    ).strip()


def answer(
    question: str,
    retriever,
    chat_fn,
    k_per_query: int = 4,
    max_queries: int = 3,
    max_papers: int = 8,
) -> dict:
    """Run the agent end to end.

    Returns ``{"queries": [...], "papers": [...], "answer": str}``.
    """
    question = (question or "").strip()
    if not question:
        return {"queries": [], "papers": [], "answer": "Please enter a question."}

    queries = plan_queries(question, chat_fn, max_queries)

    collected = []
    for query in queries:
        collected.extend(retriever(query, k_per_query))
    papers = dedupe_keep_order(collected)[:max_papers]

    return {
        "queries": queries,
        "papers": papers,
        "answer": synthesize_answer(question, papers, chat_fn),
    }
