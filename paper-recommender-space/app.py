"""
Paper Recommender - Semantic search over academic papers.

Loads the published arXiv dataset from the Hub when available, otherwise falls
back to a built-in set of landmark papers. Embeddings via MiniLM; ranking by
cosine similarity (see core.py).
"""

import logging

import gradio as gr
from sentence_transformers import SentenceTransformer

from core import (
    CATEGORIES,
    DATASET_ID,
    FALLBACK_PAPERS,
    build_index,
    load_papers,
    recommend,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load model + corpus (at startup)
# ---------------------------------------------------------------------------
logger.info("Loading embedding model ...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _encode(texts):
    return model.encode(texts, convert_to_numpy=True)


logger.info("Loading paper corpus ...")
_live = load_papers()
if _live:
    PAPERS = _live
    SOURCE_NOTE = f"{len(PAPERS):,} papers loaded from `{DATASET_ID}`"
else:
    PAPERS = FALLBACK_PAPERS
    SOURCE_NOTE = (
        f"{len(PAPERS)} built-in landmark papers (live dataset unavailable)"
    )
logger.info("Corpus ready: %s", SOURCE_NOTE)

paper_embeddings = build_index(PAPERS, _encode)
logger.info("Indexed %d papers.", len(PAPERS))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def recommend_papers(query: str, category: str, num_results: int) -> str:
    """Find papers similar to the query and render them as Markdown."""
    if not query.strip():
        return "Please enter a search query."
    try:
        results = recommend(
            query,
            PAPERS,
            paper_embeddings,
            _encode,
            CATEGORIES.get(category),
            int(num_results),
        )
    except Exception as exc:  # pragma: no cover - defensive UI guard
        logger.exception("Recommendation failed")
        return f"Something went wrong while searching: {exc}"

    if not results:
        return "No papers found matching your query and filters."

    output = f"## Found {len(results)} Relevant Papers\n\n"
    for i, paper in enumerate(results, 1):
        year = f" ({paper['year']})" if paper.get("year") else ""
        cat = paper.get("category") or "n/a"
        url = paper.get("url") or ""
        link = f" | [View Paper]({url})" if url else ""
        output += f"### {i}. {paper['title']}{year}\n\n"
        output += f"**Similarity Score:** {paper['similarity']:.1f}%\n\n"
        output += f"**Category:** `{cat}`{link}\n\n"
        output += f"**Authors:** {paper['authors']}\n\n"
        output += f"**Abstract:** {paper['abstract']}\n\n"
        output += "---\n\n"
    return output


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    ["transformer attention mechanisms for NLP", "All", 5],
    ["object detection and image recognition", "cs.CV - Computer Vision", 5],
    ["gradient optimization techniques", "cs.LG - Machine Learning", 5],
    ["language model pretraining BERT GPT", "cs.CL - Computation & Language", 5],
]

with gr.Blocks(title="Paper Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
    # Paper Recommender

    Find relevant ML research papers using semantic search. Enter your research
    topic or interests and get recommendations ranked by similarity.

    *Corpus: {SOURCE_NOTE}.*
    """
    )

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Research Query",
                placeholder="Describe your research interests...",
                lines=3,
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(
                choices=list(CATEGORIES.keys()),
                value="All",
                label="Filter by Category",
            )
            num_results_slider = gr.Slider(
                minimum=3, maximum=10, value=5, step=1, label="Number of Results"
            )

    search_btn = gr.Button("Find Papers", variant="primary", size="lg")
    results_output = gr.Markdown(label="Recommendations")

    gr.Examples(
        examples=EXAMPLE_QUERIES,
        inputs=[query_input, category_dropdown, num_results_slider],
        outputs=results_output,
        fn=recommend_papers,
        cache_examples=False,
    )

    search_btn.click(
        fn=recommend_papers,
        inputs=[query_input, category_dropdown, num_results_slider],
        outputs=results_output,
    )

    gr.Markdown(
        """
    ---

    **Model:** sentence-transformers/all-MiniLM-L6-v2 (semantic similarity)

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """
    )

if __name__ == "__main__":
    demo.launch()
