"""
Paper Summarizer - A Gradio-based web application for summarizing academic research papers.
Version: 2.0.0 (Gradio 5.x compatible)

This application uses Facebook's BART-Large-CNN model to generate structured summaries
of academic papers. It supports both PDF uploads and pasted text input, handles long
documents through intelligent chunking, and produces summaries with extracted titles,
key findings, methodology notes, and concise abstracts.

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import re
import logging
from typing import Optional

import fitz  # PyMuPDF
import gradio as gr

from hf_client import make_client, with_retry

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "facebook/bart-large-cnn"
# BART-Large-CNN accepts up to 1024 tokens (~750 words). We chunk by words to
# stay safely within that window while leaving room for special tokens.
CHUNK_WORD_LIMIT = 700
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_LENGTH = 180
COMBINE_SUMMARY_MAX_LENGTH = 300

# ---------------------------------------------------------------------------
# Use HuggingFace Inference API (no local model loading - saves memory)
# ---------------------------------------------------------------------------
logger.info("Initializing HuggingFace Inference Client for: %s", MODEL_NAME)
client = make_client(MODEL_NAME)
logger.info("Inference client ready.")


# ===========================================================================
# Text extraction helpers
# ===========================================================================


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text content from a PDF file using PyMuPDF.

    Args:
        pdf_path: Path to the uploaded PDF file.

    Returns:
        The concatenated text of every page, separated by newlines.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(
            f"Could not open the PDF file. It may be corrupted or password-protected. "
            f"Details: {exc}"
        ) from exc

    pages: list[str] = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
        logger.debug("Page %d: extracted %d characters", page_num + 1, len(text))

    doc.close()

    if not pages:
        raise ValueError(
            "The PDF appears to contain no extractable text. "
            "It may be a scanned document or consist only of images."
        )

    return "\n".join(pages)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artefacts.

    Handles excessive newlines, hyphenated line-breaks, and stray control
    characters that often appear in academic PDFs.
    """
    # Remove form-feed and other control characters (keep newlines & tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Re-join hyphenated line breaks (e.g. "summa-\nrization" -> "summarization")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ===========================================================================
# Title extraction heuristic
# ===========================================================================


def extract_title(text: str) -> str:
    """Attempt to extract the paper title from the first few lines.

    Academic papers typically place the title in the first 1-5 lines before the
    author block.  We use a simple heuristic: the longest line among the first
    few non-empty lines that is not all-caps (which would be a header like
    "ABSTRACT") and does not look like an author list.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:12]

    candidates: list[str] = []
    for line in lines:
        # Skip very short lines (page numbers, dates, etc.)
        if len(line) < 10:
            continue
        # Skip lines that are likely author names / affiliations (contain '@')
        if "@" in line:
            continue
        # Skip lines that are section headers (all uppercase, short)
        if line.isupper() and len(line) < 60:
            continue
        # Skip lines that look like emails or URLs
        if re.search(r"https?://|www\.", line):
            continue
        candidates.append(line)

    if not candidates:
        return "Untitled Paper"

    # Return the first substantial candidate (titles usually come first)
    return candidates[0]


# ===========================================================================
# Chunking and summarization
# ===========================================================================


def chunk_text(text: str, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    """Split text into chunks of approximately *max_words* words.

    Splitting is done on paragraph boundaries where possible so that chunks
    remain coherent.  If a single paragraph exceeds the limit it is split on
    sentence boundaries instead.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If adding this paragraph would exceed the limit, finalize the chunk.
        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        # Handle paragraphs that are themselves larger than the limit.
        if para_words > max_words:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sentence in sentences:
                s_words = len(sentence.split())
                if current_word_count + s_words > max_words and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_word_count = 0
                current_chunk.append(sentence)
                current_word_count += s_words
        else:
            current_chunk.append(para)
            current_word_count += para_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def summarize_text(text: str) -> str:
    """Summarize a single chunk of text using the BART model via Inference API.

    Dynamically adjusts min/max summary length based on input length to avoid
    the common transformers warning about min_length exceeding input length.
    """
    word_count = len(text.split())
    # For very short inputs, just return the text as-is.
    if word_count < 50:
        return text

    max_len = min(SUMMARY_MAX_LENGTH, max(50, word_count // 2))
    min_len = min(SUMMARY_MIN_LENGTH, max_len - 10)

    try:
        result = with_retry(
            client.summarization,
            text,
            parameters={
                "max_length": max_len,
                "min_length": min_len,
                "do_sample": False,
            },
        )
        return result.summary_text
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        # Fallback: return truncated text
        return " ".join(text.split()[:100]) + "..."


def generate_full_summary(text: str) -> str:
    """Produce a final summary for arbitrarily long documents.

    Strategy:
    1. Split the document into manageable chunks.
    2. Summarize each chunk individually.
    3. If multiple chunk summaries exist, combine them and run a second-pass
       summarization to produce a coherent final summary.
    """
    chunks = chunk_text(text)
    logger.info("Document split into %d chunk(s) for summarization.", len(chunks))

    chunk_summaries = [summarize_text(chunk) for chunk in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Second pass: combine chunk summaries and re-summarize for coherence.
    combined = " ".join(chunk_summaries)
    combined_words = len(combined.split())

    if combined_words < 50:
        return combined

    max_len = min(COMBINE_SUMMARY_MAX_LENGTH, max(60, combined_words // 2))
    min_len = min(SUMMARY_MIN_LENGTH, max_len - 10)

    try:
        result = with_retry(
            client.summarization,
            combined,
            parameters={
                "max_length": max_len,
                "min_length": min_len,
                "do_sample": False,
            },
        )
        return result.summary_text
    except Exception as e:
        logger.warning("Combined summarization failed: %s", e)
        return combined


# ===========================================================================
# Section extraction helpers
# ===========================================================================


def extract_section(text: str, heading_pattern: str, fallback: str = "") -> str:
    """Extract content under a section heading matched by *heading_pattern*.

    Uses a regex to find the heading and captures everything until the next
    heading of equal or higher level.
    """
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+[\.\)]?\s*)?{heading_pattern}\s*\n(.*?)(?=\n\s*(?:\d+[\.\)]?\s*)?[A-Z][A-Za-z ]+\s*\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        content = match.group(1).strip()
        if len(content) > 30:
            return content
    return fallback


def extract_key_findings(text: str) -> str:
    """Try to extract key findings from Results / Conclusion sections, or
    fall back to summarizing the last portion of the paper."""
    for heading in [
        r"(?:key\s+)?findings",
        r"results?\s*(?:and\s+discussion)?",
        r"conclusions?\s*(?:and\s+future\s+work)?",
        r"discussion",
    ]:
        content = extract_section(text, heading)
        if content:
            return summarize_text(content[:3000])
    # Fallback: summarize the last quarter of the document.
    words = text.split()
    tail = " ".join(words[-(len(words) // 4) :])
    if len(tail.split()) > 50:
        return summarize_text(tail[:3000])
    return "Key findings could not be automatically extracted."


def extract_methodology(text: str) -> str:
    """Try to extract methodology information from the paper."""
    for heading in [
        r"method(?:ology|s)?",
        r"approach",
        r"experimental\s+setup",
        r"materials?\s+and\s+methods",
        r"(?:proposed\s+)?(?:framework|system|model|architecture)",
    ]:
        content = extract_section(text, heading)
        if content:
            return summarize_text(content[:3000])
    return "Methodology section could not be automatically extracted."


# ===========================================================================
# Main processing function
# ===========================================================================


def process_paper(
    pdf_file: Optional[str],
    pasted_text: Optional[str],
) -> str:
    """Process a research paper and return a structured summary.

    Accepts either a PDF file path (from Gradio upload) or raw pasted text.
    Returns a Markdown-formatted structured summary.
    """
    # ------------------------------------------------------------------
    # 1. Obtain raw text
    # ------------------------------------------------------------------
    if pdf_file is not None:
        logger.info("Processing uploaded PDF: %s", pdf_file)
        try:
            raw_text = extract_text_from_pdf(pdf_file)
        except ValueError as exc:
            return f"**Error:** {exc}"
    elif pasted_text and pasted_text.strip():
        raw_text = pasted_text.strip()
    else:
        return (
            "**Error:** Please upload a PDF file or paste the paper text. "
            "Both inputs are currently empty."
        )

    text = clean_text(raw_text)
    original_word_count = len(text.split())

    if original_word_count < 30:
        return (
            "**Error:** The extracted text is too short to summarize. "
            "Please provide a longer document or check that the PDF contains selectable text."
        )

    logger.info("Cleaned text: %d words.", original_word_count)

    # ------------------------------------------------------------------
    # 2. Extract structured components
    # ------------------------------------------------------------------
    title = extract_title(text)
    concise_summary = generate_full_summary(text)
    key_findings = extract_key_findings(text)
    methodology = extract_methodology(text)

    summary_word_count = len(concise_summary.split())

    # ------------------------------------------------------------------
    # 3. Format the output
    # ------------------------------------------------------------------
    output = f"""## {title}

---

### Concise Summary
{concise_summary}

---

### Key Findings
{key_findings}

---

### Methodology
{methodology}

---

### Statistics
| Metric | Value |
|---|---|
| Original length | {original_word_count:,} words |
| Summary length | {summary_word_count:,} words |
| Compression ratio | {original_word_count / max(summary_word_count, 1):.1f}x |
"""
    return output


# ===========================================================================
# Example inputs for the Gradio demo
# ===========================================================================

EXAMPLE_TEXT = """Attention Is All You Need

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

Introduction
Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures. Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

Methods
The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder. The encoder maps an input sequence of symbol representations to a sequence of continuous representations. Given z, the decoder then generates an output sequence of symbols one element at a time. At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next. The Transformer uses multi-head attention to allow the model to jointly attend to information from different representation subspaces at different positions.

Results
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models including ensembles by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers.

Conclusions
In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention. The Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. We achieved new state of the art on both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video."""


# ===========================================================================
# Gradio interface
# ===========================================================================


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="Paper Summarizer",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        ),
        css="""
            .header-text { text-align: center; margin-bottom: 0.5em; }
            .subheader  { text-align: center; color: #6b7280; margin-top: 0; }
            footer { display: none !important; }
        """,
    ) as demo:
        # --- Header ---
        gr.Markdown(
            """
            <h1 class="header-text">Paper Summarizer</h1>
            <p class="subheader">
                Summarize academic research papers into structured, digestible insights.<br>
                Upload a PDF or paste the full text below.
            </p>
            """,
        )

        with gr.Row():
            # --- Input column ---
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                pdf_input = gr.File(
                    label="Upload PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
                text_input = gr.Textbox(
                    label="Or paste paper text",
                    placeholder="Paste the full text of a research paper here...",
                    lines=12,
                    max_lines=30,
                )
                submit_btn = gr.Button("Summarize", variant="primary", size="lg")

            # --- Output column ---
            with gr.Column(scale=1):
                gr.Markdown("### Structured Summary")
                output = gr.Markdown(
                    value="*Your summary will appear here after processing.*",
                    label="Summary",
                )

        # --- Examples ---
        gr.Markdown("---")
        gr.Markdown("### Try an Example")
        gr.Examples(
            examples=[[None, EXAMPLE_TEXT]],
            inputs=[pdf_input, text_input],
            outputs=output,
            fn=process_paper,
            cache_examples=False,
            label="Click to load example paper text",
        )

        # --- About ---
        with gr.Accordion("About this Space", open=False):
            gr.Markdown(
                """
                **Paper Summarizer** uses
                [`facebook/bart-large-cnn`](https://huggingface.co/facebook/bart-large-cnn)
                to generate abstractive summaries of academic papers.

                **Features**
                - PDF upload with automatic text extraction (PyMuPDF)
                - Intelligent chunking for papers of any length
                - Structured output: title, key findings, methodology, and concise summary
                - Word-count statistics and compression ratio

                **Limitations**
                - Scanned PDFs (image-only) are not supported; the PDF must contain selectable text.
                - Summarization quality depends on the input text quality and structure.
                - Running on free CPU tier; very long papers may take a minute to process.

                Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys).
                """
            )

        # --- Event binding ---
        submit_btn.click(
            fn=process_paper,
            inputs=[pdf_input, text_input],
            outputs=output,
        )

    return demo


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app = build_interface()
    app.launch()
