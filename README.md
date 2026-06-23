# HuggingFace ML Portfolio

A collection of end-to-end machine learning projects published on the HuggingFace Hub, spanning NLP, computer vision, generative AI, and full-stack ML application development. The portfolio contains 15 projects: datasets, fine-tuned models, and interactive Gradio Spaces.

## Projects

### Academic Paper Intelligence

| Project | Type | Description |
|---------|------|-------------|
| [academic-papers-dataset](./academic-papers-dataset) | Dataset | arXiv paper metadata across cs.AI, cs.CL, cs.CV, cs.LG, and stat.ML |
| [paper-classifier-model](./paper-classifier-model) | Model | DistilBERT fine-tuned to classify arXiv papers by category |
| [paper-summarizer-space](./paper-summarizer-space) | Space | Summarizes academic papers (PDF or text) using BART-Large-CNN via the HuggingFace Inference API |
| [paper-recommender-space](./paper-recommender-space) | Space | Recommends related papers by embedding similarity |

### Resume & Career Tools

| Project | Type | Description |
|---------|------|-------------|
| [resume-section-classifier](./resume-section-classifier) | Model | DistilBERT resume-section classifier trained on synthetic, self-generated data |
| [resume-analyzer-space](./resume-analyzer-space) | Space | Matches a resume against a job description with semantic similarity and keyword gap analysis |

### Financial Analysis

| Project | Type | Description |
|---------|------|-------------|
| [trading-dashboard-space](./trading-dashboard-space) | Space | Technical-analysis dashboard (yfinance + Plotly) with indicators and backtesting — educational use only |

### AI / LLM Utilities

| Project | Type | Description |
|---------|------|-------------|
| [code-explainer-space](./code-explainer-space) | Space | Explains code snippets using Mistral-7B via the HuggingFace Inference API |
| [prompt-enhancer-space](./prompt-enhancer-space) | Space | Rewrites and enriches prompts using Mistral-7B via the HuggingFace Inference API |
| [model-arena-space](./model-arena-space) | Space | Compares two LLMs side-by-side on the same prompt with response voting |
| [ml-interview-space](./ml-interview-space) | Space | Practice ML/DS interview questions with quiz and browse modes |
| [model-selector-space](./model-selector-space) | Space | Helps find HuggingFace models suited to a given task |

### Generative & Data Exploration

| Project | Type | Description |
|---------|------|-------------|
| [dataset-explorer-space](./dataset-explorer-space) | Space | Visualizes statistics and samples for any HuggingFace dataset |
| [illusion-generator-space](./illusion-generator-space) | Space | Generates optical-illusion images using FLUX.1-schnell via the HuggingFace Inference API |
| [style-mixer-space](./style-mixer-space) | Space | Blends two art styles into a single image using FLUX.1-schnell via the HuggingFace Inference API |

## Tech Stack

- **Models**: HuggingFace Transformers (DistilBERT, BART-Large-CNN), sentence-transformers
- **Data**: HuggingFace Datasets, arXiv API, yfinance
- **Applications**: Gradio 5.31.0, Plotly, Matplotlib
- **Hosted inference**: HuggingFace Inference API (`huggingface_hub.InferenceClient`) for the LLM Spaces (Mistral-7B) and image Spaces (FLUX.1-schnell)
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

The LLM and image Spaces (code-explainer, prompt-enhancer, model-arena, illusion-generator, style-mixer, paper-summarizer) call the **HuggingFace Inference API** and require an `HF_TOKEN` secret (a token with inference access) to be set in Space Settings → Secrets. For the FLUX image Spaces, the token also needs available inference credits/quota.

## License

MIT
