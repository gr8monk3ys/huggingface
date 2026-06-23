"""
Research Assistant - agentic RAG over arXiv papers.

Plans search queries with an LLM, retrieves relevant papers via embedding
similarity, and synthesizes a cited answer grounded in them. Ties together the
portfolio's papers dataset, embedding model, and shared inference helper.
"""

import logging

import gradio as gr
from sentence_transformers import SentenceTransformer

from agent import answer as run_agent
from hf_client import make_client, with_retry
from retrieval import DATASET_ID, FALLBACK_PAPERS, Retriever, load_papers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ---------------------------------------------------------------------------
# Startup: embedding model, corpus, retriever, LLM client
# ---------------------------------------------------------------------------
logger.info("Loading embedding model ...")
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _encode(texts):
    return embedder.encode(texts, convert_to_numpy=True)


_live = load_papers()
if _live:
    PAPERS = _live
    SOURCE_NOTE = f"{len(PAPERS):,} papers from `{DATASET_ID}`"
else:
    PAPERS = FALLBACK_PAPERS
    SOURCE_NOTE = f"{len(PAPERS)} built-in landmark papers (live dataset unavailable)"

retriever = Retriever(PAPERS, _encode)
logger.info("Indexed corpus: %s", SOURCE_NOTE)

client = make_client(LLM_MODEL)


def _chat(messages):
    completion = with_retry(
        client.chat_completion, messages=messages, max_tokens=700, temperature=0.3
    )
    return completion.choices[0].message.content


# ---------------------------------------------------------------------------
# UI handler
# ---------------------------------------------------------------------------
def _format_sources(papers: list[dict]) -> str:
    lines = ["### Sources"]
    for i, paper in enumerate(papers, 1):
        year = f" ({paper['year']})" if paper.get("year") else ""
        url = paper.get("url") or ""
        link = f" — [link]({url})" if url else ""
        sim = paper.get("similarity")
        score = f" · {sim:.0f}% match" if isinstance(sim, (int, float)) else ""
        lines.append(f"{i}. **{paper.get('title', '')}**{year}{link}{score}")
    return "\n".join(lines)


def research(question: str):
    """Run the agent and return (answer markdown, sources markdown)."""
    if not question.strip():
        return "Please enter a research question.", ""
    try:
        result = run_agent(question, retriever.search, _chat)
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.exception("Agent run failed")
        return f"Something went wrong: {exc}", ""

    plan = ", ".join(f"`{q}`" for q in result["queries"])
    answer_md = f"**Search plan:** {plan}\n\n---\n\n{result['answer']}"
    sources_md = _format_sources(result["papers"]) if result["papers"] else ""
    return answer_md, sources_md


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
EXAMPLES = [
    "How do transformers handle long-range dependencies?",
    "What techniques help prevent overfitting in deep neural networks?",
    "Compare approaches to real-time object detection.",
    "Why did attention replace recurrence for sequence modeling?",
]

with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
    # Research Assistant (Agentic RAG)

    Ask a research question. The assistant **plans** focused search queries,
    **retrieves** relevant papers by semantic similarity, and **synthesizes** a
    concise answer with inline `[n]` citations grounded in those papers.

    *Corpus: {SOURCE_NOTE}. Generation via `{LLM_MODEL}` (HuggingFace Inference API).*
    """
    )

    question = gr.Textbox(
        label="Your research question",
        lines=3,
        placeholder="e.g. How does self-attention improve machine translation?",
    )
    ask_btn = gr.Button("Research", variant="primary", size="lg")
    answer_out = gr.Markdown(label="Answer")
    sources_out = gr.Markdown(label="Sources")

    gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[question])

    ask_btn.click(research, inputs=[question], outputs=[answer_out, sources_out])

    gr.Markdown(
        """
    ---

    How it works: query planning + retrieval-augmented generation. Answers are
    grounded in retrieved abstracts; always verify against the linked sources.

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys).
    Requires an `HF_TOKEN` secret (see the README).
    """
    )

if __name__ == "__main__":
    demo.launch()
