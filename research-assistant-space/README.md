---
title: Research Assistant
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Agentic RAG research assistant over arXiv papers
---

# Research Assistant (Agentic RAG)

Ask a research question and get a concise, **cited** answer grounded in real
papers. This Space ties the portfolio together: it reuses the
[CS/ML Academic Papers dataset](https://huggingface.co/datasets/gr8monk3ys/cs-ml-academic-papers),
the MiniLM embedding model, and the shared HuggingFace Inference helper.

## How it works (the agent loop)

1. **Plan** — an LLM decomposes your question into a few focused search queries.
2. **Retrieve** — each query runs semantic search (MiniLM embeddings + cosine
   similarity) over the paper corpus; results are de-duplicated.
3. **Synthesize** — a second LLM step writes the answer using *only* the
   retrieved abstracts, with inline `[n]` citations back to the sources.

This is retrieval-augmented generation with an explicit planning step, so the
answer is grounded in retrieved evidence rather than the model's memory.

## Architecture

```
Question ──> LLM planner ──> search queries
                                   │
                 MiniLM embeddings + cosine search (per query)
                                   │
                       de-duplicated paper set
                                   │
            LLM synthesizer (answer with [n] citations)
```

| Component | Technology |
|-----------|------------|
| Web framework | Gradio 5.31.0 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Corpus | `gr8monk3ys/cs-ml-academic-papers` (live, with built-in fallback) |
| Generation | `mistralai/Mistral-7B-Instruct-v0.3` via HF Inference API |

The pure logic (query planning, retrieval, synthesis orchestration) lives in
`agent.py` / `retrieval.py` with injectable LLM and embedding functions, so it
is unit-tested without network access.

## Configuration

This Space calls the HuggingFace Inference API for generation, so it needs an
`HF_TOKEN` secret in **Space Settings → Secrets** (a token with inference
access). Without it, the planning/synthesis steps will be rejected. The papers
dataset id can be overridden with the `PAPERS_DATASET` environment variable; if
the dataset can't be reached, the app falls back to a small set of landmark
papers so it still runs.

## Limitations

- Answers are only as good as the retrieved abstracts — always check the linked
  sources.
- Retrieval is over abstracts/titles, not full text.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
