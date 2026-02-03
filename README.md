# HuggingFace ML Portfolio

A collection of end-to-end machine learning projects published on the HuggingFace Hub, demonstrating NLP, computer vision, and full-stack ML application development.

## Projects

### Academic Paper Intelligence Pipeline

| Project | Type | Description |
|---------|------|-------------|
| [academic-papers-dataset](./academic-papers-dataset) | Dataset | 2,500+ curated arXiv papers across 5 CS/ML categories (cs.AI, cs.CL, cs.CV, cs.LG, stat.ML) |
| [paper-classifier-model](./paper-classifier-model) | Model | Fine-tuned DistilBERT for 8-category arXiv paper classification (~94% F1) |
| [paper-summarizer-space](./paper-summarizer-space) | Space | Gradio app for structured academic paper summarization using BART-Large-CNN |

### Resume & Career Tools

| Project | Type | Description |
|---------|------|-------------|
| [resume-section-classifier](./resume-section-classifier) | Model | DistilBERT fine-tuned on synthetic data to classify resume sections into 8 categories (~95% F1) |
| [resume-analyzer-space](./resume-analyzer-space) | Space | AI-powered resume-to-job-description matcher with semantic similarity scoring and keyword gap analysis |

### Financial Analysis

| Project | Type | Description |
|---------|------|-------------|
| [trading-dashboard-space](./trading-dashboard-space) | Space | Interactive technical analysis dashboard with real-time stock data, indicators (SMA, EMA, RSI, MACD, Bollinger Bands), and backtesting |

## Tech Stack

- **Models**: HuggingFace Transformers (DistilBERT, BART-Large-CNN), sentence-transformers
- **Data**: HuggingFace Datasets, arXiv API, yfinance
- **Applications**: Gradio 4.44, Plotly
- **ML Tools**: PyTorch, scikit-learn, TF-IDF
- **Document Processing**: PyMuPDF (fitz)

## Getting Started

Each project is self-contained. Navigate into any subfolder and follow its specific setup:

```bash
# General pattern for Spaces
cd <project-name>
pip install -r requirements.txt
python app.py

# For models
cd <project-name>
pip install -r requirements.txt
python train.py
```

## License

MIT
