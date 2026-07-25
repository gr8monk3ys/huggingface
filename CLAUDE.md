# HuggingFace Portfolio - Claude Code Guide

## Project Structure

This is a monorepo containing 16 independent HuggingFace projects (datasets, models, and Spaces). Each subfolder is designed to be published independently to the HuggingFace Hub.

```
huggingface/
# --- Academic Paper Intelligence ---
├── academic-papers-dataset/   # Dataset: arXiv paper metadata (cs.AI/CL/CV/LG, stat.ML)
├── paper-classifier-model/    # Model: DistilBERT arXiv paper classifier
├── paper-summarizer-space/    # Space: BART-Large-CNN summarizer (HF Inference API)
├── paper-recommender-space/   # Space: related-paper recommender via embedding similarity
├── research-assistant-space/  # Space: agentic RAG (plan -> retrieve -> cited answer) over the papers dataset
# --- Resume & Career Tools ---
├── resume-section-classifier/ # Model: DistilBERT resume-section classifier (synthetic data)
├── resume-analyzer-space/     # Space: resume <-> job-description semantic matcher
# --- Financial Analysis ---
├── trading-dashboard-space/   # Space: technical-analysis dashboard (yfinance + Plotly), educational only
# --- AI / LLM Utilities ---
├── code-explainer-space/      # Space: code explanations via Mistral-7B (HF Inference API)
├── prompt-enhancer-space/     # Space: prompt rewriter via Mistral-7B (HF Inference API)
├── model-arena-space/         # Space: side-by-side LLM comparison + voting
├── ml-interview-space/        # Space: ML/DS interview question practice
├── model-selector-space/      # Space: find HF models for a task (live Hub query + fallback)
# --- Generative & Data Exploration ---
├── dataset-explorer-space/    # Space: visualize any HF dataset
├── illusion-generator-space/  # Space: optical-illusion image generator (FLUX.1-schnell)
└── style-mixer-space/         # Space: blend two art styles into one image (FLUX.1-schnell)
```

## Conventions

- **Spaces** use Gradio 5.31 (pinned `gradio>=5.31.0,<6.0.0`) with `app.py` as the entry point
- **LLM/image Spaces** call the HuggingFace Inference API via `huggingface_hub.InferenceClient` and need an `HF_TOKEN` secret set in Space Settings
- **Models** use HuggingFace Transformers Trainer API with early stopping
- **Datasets** use HuggingFace Datasets library with Parquet storage
- All projects are MIT licensed
- Each project has its own `requirements.txt`

## Key Patterns

- Models are fine-tuned from pretrained checkpoints (DistilBERT, BART-Large-CNN)
- Spaces handle PDF input via PyMuPDF (fitz)
- Text processing uses intelligent chunking that respects paragraph/sentence boundaries
- Scoring systems use composite metrics (e.g., 60% semantic + 40% keyword overlap)

## Development

```bash
# Run any Space locally
cd <space-folder>
pip install -r requirements.txt
python app.py

# Train a model
cd <model-folder>
pip install -r requirements.txt
python train.py

# Run the test suite (from the repo root)
pip install -r requirements-dev.txt
pytest
```

Tests live in `tests/` and cover the shared, network-free logic: `hf_client`
(the inference helper each LLM/image Space vendors a copy of) and the per-Space
`core.py` modules. They deliberately do not import `app.py` files, which would
require every Space's UI dependencies.

## Publishing to the Hub

`scripts/publish_to_hub.py` is the one-shot publisher. It maps each local
folder to its Hub repo id -- these differ, since folders carry `-space` /
`-model` suffixes that the Hub repos do not -- and uploads cards, scripts,
trained weights, and parquet data to the right repo type.

```bash
hf auth login              # required; --dry-run works without it
python scripts/publish_to_hub.py --dry-run
python scripts/publish_to_hub.py --only spaces|models|datasets
```

Uploads from a project root use an extension allowlist (`.py`, `.txt`, `.md`),
so a stray venv or checkpoint cannot leak to the Hub by being forgotten.

Note: creating a *new* Gradio Space now requires a PRO subscription (the Hub
returns 402); existing Spaces remain writable. The script reports such a repo
as skipped and carries on rather than aborting the run.

## HuggingFace Hub

Projects are published to the `gr8monk3ys` namespace on HuggingFace Hub. Spaces deploy automatically when pushed to their respective HF repos.
