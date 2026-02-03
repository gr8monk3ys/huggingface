---
title: Paper Recommender
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Find similar research papers using semantic search
---

# Paper Recommender

An AI-powered tool that recommends relevant research papers based on your research interests or a seed paper abstract. Built with sentence-transformers and FAISS for fast semantic similarity search.

## Features

### Semantic Paper Search
Enter your research topic, abstract, or interests in natural language. The system finds the most semantically similar papers from our database of 2,500+ CS/ML papers from arXiv.

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
3. **Choose result count** - Get 5, 10, or 20 recommendations
4. **Browse results** - Review ranked papers with similarity scores

## Technical Architecture

```
Query Text ──> Sentence Transformer (all-MiniLM-L6-v2) ──> Query Embedding
                                                              │
Paper Database ──> Pre-computed Embeddings ──> FAISS Index ──┘
                                                              │
                                              Cosine Similarity Search
                                                              │
                                              Top-K Recommendations
```

| Component | Technology |
|-----------|------------|
| Web Framework | Gradio 5.9.1 |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Search | FAISS (Facebook AI Similarity Search) |
| Dataset | gr8monk3ys/cs-ml-academic-papers |

## Data Source

Recommendations are drawn from the [CS/ML Academic Papers Dataset](https://huggingface.co/datasets/gr8monk3ys/cs-ml-academic-papers), containing 2,500+ recent papers across five core arXiv categories.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
