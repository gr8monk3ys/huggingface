"""Paper Summarizer -- a Gradio front end over :mod:`core`.

This module owns the UI and the adapters it needs: it builds the Inference
client, wraps it as the ``summarize_fn`` that :func:`core.summarize_paper`
expects, and renders the returned :class:`core.PaperSummary` as Markdown.

All summarization logic lives in ``core.py`` and is tested there. Nothing in
this file is imported by the test suite -- see
docs/adr/0002-coarse-entry-point-for-space-core-modules.md.

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import logging

import gradio as gr

from core import InputError, PaperSummary, PdfReadError, summarize_paper
from hf_client import InferenceError, make_client, with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = "facebook/bart-large-cnn"

logger.info("Initializing HuggingFace Inference Client for: %s", MODEL_NAME)
client = make_client(MODEL_NAME)
logger.info("Inference client ready.")


# ===========================================================================
# Adapter: the summarizer core.summarize_paper is given
# ===========================================================================


def _summarize(text: str, *, max_length: int, min_length: int) -> str:
    """Call the Inference API, retrying transient failures.

    Raises InferenceError, which process_paper turns into a message. It is
    deliberately not caught here: a caller whose credits are exhausted needs to
    be told so, not handed a truncated document that reads like a summary.
    """
    result = with_retry(
        client.summarization,
        text,
        # `generate_parameters`, not `parameters`: the latter has never been the
        # kwarg on InferenceClient.summarization, so every call raised TypeError.
        # The old bare `except Exception` swallowed it and returned the first
        # 100 words of the input, so the Space appeared to work.
        generate_parameters={
            "max_length": max_length,
            "min_length": min_length,
            "do_sample": False,
        },
    )
    return result.summary_text


# ===========================================================================
# Rendering
# ===========================================================================


def _render(summary: PaperSummary) -> str:
    """Format a PaperSummary as the Markdown shown in the output pane."""
    return f"""## {summary.title}

---

### Concise Summary
{summary.concise_summary}

---

### Key Findings
{summary.key_findings}

---

### Methodology
{summary.methodology}

---

### Statistics
| Metric | Value |
|---|---|
| Original length | {summary.original_word_count:,} words |
| Summary length | {summary.summary_word_count:,} words |
| Compression ratio | {summary.compression_ratio:.1f}x |
"""


# ===========================================================================
# Handler
# ===========================================================================


def process_paper(pdf_file, pasted_text) -> str:
    """Gradio handler: summarize a paper and render it, or report why not."""
    try:
        summary = summarize_paper(
            summarize_fn=_summarize,
            pdf_path=pdf_file,
            pasted_text=pasted_text,
        )
    except (InputError, PdfReadError, InferenceError) as exc:
        return f"**Error:** {exc}"

    return _render(summary)


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
