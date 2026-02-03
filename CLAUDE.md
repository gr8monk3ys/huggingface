# HuggingFace Portfolio - Claude Code Guide

## Project Structure

This is a monorepo containing 6 independent HuggingFace projects. Each subfolder is designed to be published independently to the HuggingFace Hub.

```
huggingface/
├── academic-papers-dataset/   # Dataset: arXiv papers collection
│   ├── create_dataset.py      # Fetches papers via arXiv API, publishes to HF Hub
│   └── explore_dataset.py     # EDA visualizations (TF-IDF, timelines)
├── paper-classifier-model/    # Model: DistilBERT paper classifier
│   ├── train.py               # Fine-tuning pipeline (5 epochs, 2e-5 lr)
│   └── inference.py           # Prediction wrapper
├── paper-summarizer-space/    # Space: BART-based paper summarizer
│   └── app.py                 # Gradio app with PDF upload + chunked summarization
├── resume-analyzer-space/     # Space: Resume-job description matcher
│   └── app.py                 # Gradio app with semantic + keyword scoring
├── resume-section-classifier/ # Model: Resume section classifier
│   ├── train.py               # DistilBERT fine-tuning
│   ├── data_generator.py      # Synthetic training data generation
│   └── inference.py           # Section classification API
└── trading-dashboard-space/   # Space: Stock technical analysis dashboard
    └── app.py                 # Gradio app with yfinance + Plotly charts
```

## Conventions

- **Spaces** use Gradio 4.44 with `app.py` as the entry point
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
```

## HuggingFace Hub

Projects are published to the `gr8monk3ys` namespace on HuggingFace Hub. Spaces deploy automatically when pushed to their respective HF repos.
