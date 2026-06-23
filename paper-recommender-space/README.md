---
title: Paper Recommender
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Find similar research papers using semantic search
---

# Paper Recommender

An AI-powered tool that recommends relevant research papers based on your research interests or a seed paper abstract. Built with sentence-transformers for semantic similarity search.

## Features

### Semantic Paper Search
Enter your research topic, abstract, or interests in natural language. The system finds the most semantically similar papers. The corpus is loaded **live from the published [CS/ML Academic Papers dataset](https://huggingface.co/datasets/gr8monk3ys/cs-ml-academic-papers)** (up to ~2,500 papers); if that dataset can't be reached, it falls back to a built-in set of ~20 landmark papers so the app always works.

### Category Filtering
Filter results by arXiv category:
- **cs.AI** - Artificial Intelligence
- **cs.CL** - Computation and Language (NLP)
- **cs.CV** - Computer Vision
- **cs.LG** - Machine Learning
- **stat.ML** - Statistics - Machine Learning

### Similarity Scores
Each recommendation includes a similarity score (0-100%) showing how closely the paper matches your query.

### Direct Links
Click through to the original arXiv paper for full text access.

## How It Works

1. **Enter your query** - Describe your research interests or paste an abstract
2. **Select categories** (optional) - Filter to specific arXiv categories
3. **Choose result count** - Get between 3 and 10 recommendations
4. **Browse results** - Review ranked papers with similarity scores

## Technical Architecture

```
HF dataset ──> embed "{title} {abstract}" ──> normalized matrix
   (fallback: ~20 landmark papers)                   │
Query Text ──> Sentence Transformer ──> Query Embedding
                                                      │
                              Cosine similarity (NumPy matmul)
                                                      │
                              Top-K, category-filtered results
```

| Component | Technology |
|-----------|------------|
| Web Framework | Gradio 5.31.0 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Similarity Search | NumPy cosine similarity (corpus is small; no ANN index needed) |
| Dataset | `gr8monk3ys/cs-ml-academic-papers` (live, with built-in fallback) |

## Data Source

Recommendations are drawn live from the [CS/ML Academic Papers Dataset](https://huggingface.co/datasets/gr8monk3ys/cs-ml-academic-papers) (up to ~2,500 recent papers across core arXiv categories). The dataset id can be overridden with the `PAPERS_DATASET` environment variable. If the dataset is unavailable at startup, the app uses a built-in set of ~20 landmark papers.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
