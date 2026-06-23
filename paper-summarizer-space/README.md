---
title: Paper Summarizer
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Summarize academic research papers with AI
---

# Paper Summarizer

An AI-powered tool that transforms lengthy academic research papers into structured, digestible summaries. Built with Facebook's BART-Large-CNN model and deployed as a Gradio web application on HuggingFace Spaces.

## Features

- **PDF Upload** -- Drop a research paper PDF and get an instant structured summary.
- **Text Input** -- Paste raw paper text directly if you prefer.
- **Structured Output** -- Every summary includes:
  - Extracted paper title
  - Concise abstract-length summary
  - Key findings from the results/conclusion sections
  - Methodology overview
  - Word-count statistics with compression ratio
- **Long Document Support** -- Papers of any length are automatically chunked and summarized in multiple passes, then combined into a coherent final summary.
- **Clean PDF Processing** -- Handles hyphenated line breaks, control characters, and other common PDF artifacts.

## How It Works

1. **Text Extraction** -- PDFs are parsed with PyMuPDF (fitz) to extract selectable text from every page.
2. **Cleaning** -- Raw text is normalized: stray control characters are removed, hyphenated line breaks are rejoined, and excessive whitespace is collapsed.
3. **Chunking** -- The cleaned text is split into chunks of approximately 700 words, respecting paragraph and sentence boundaries so context is preserved.
4. **Summarization** -- Each chunk is passed through `facebook/bart-large-cnn` for abstractive summarization. If there are multiple chunks, the individual summaries are combined and summarized again for coherence.
5. **Section Extraction** -- Regex heuristics identify Results, Methodology, and Conclusion sections for targeted summarization of key findings and methods.

## Model

This Space uses [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn), a BART model fine-tuned on the CNN/DailyMail summarization dataset. It runs on the free CPU tier and can process most papers in under a minute.

## Limitations

- **Scanned PDFs** are not supported -- the PDF must contain selectable text (not images of text).
- **Summarization quality** depends on the structure and clarity of the input text.
- **Processing time** may be longer for very large papers due to CPU-only inference.

## Tech Stack

| Component | Library |
|---|---|
| Web framework | Gradio 5.31.0 |
| Summarization model | BART-Large-CNN via the HuggingFace Inference API |
| PDF parsing | PyMuPDF (fitz) |
| Inference backend | HuggingFace Inference API (`huggingface_hub.InferenceClient`) |

## Local Development

```bash
# Clone the repository
git clone https://huggingface.co/spaces/gr8monk3ys/paper-summarizer
cd paper-summarizer

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The app will be available at `http://localhost:7860`.

## Configuration

This Space calls the **HuggingFace Inference API** to run BART-Large-CNN. It requires an `HF_TOKEN` secret — a token with inference access — set in **Space Settings → Secrets**. Without it, summarization will fail.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys).
