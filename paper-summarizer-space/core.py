"""Paper summarization logic, independent of Gradio and of the Inference API.

The entry point is :func:`summarize_paper`. It owns the whole sequence -- read
the input, clean it, extract a title, summarize, pull key findings and
methodology -- and returns a :class:`PaperSummary`. Rendering that into Markdown
is ``app.py``'s job; see docs/adr/0002-coarse-entry-point-for-space-core-modules.md.

The summarizer itself is injected as ``summarize_fn`` rather than constructed
here, so tests drive the full sequence with a fake and never touch the network.
``fitz`` is imported lazily inside :func:`read_pdf_text` for the same reason:
importing this module must not require PyMuPDF.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# BART-Large-CNN accepts up to 1024 tokens (~750 words). We chunk by words to
# stay safely within that window while leaving room for special tokens.
CHUNK_WORD_LIMIT = 700
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_LENGTH = 180
COMBINE_SUMMARY_MAX_LENGTH = 300

# Below this, summarizing costs a network round-trip to return roughly the
# input, so the text is passed through unchanged.
PASSTHROUGH_WORD_COUNT = 50

# Below this, there is nothing to summarize and the caller gets an InputError.
MIN_DOCUMENT_WORDS = 30

# How much of a matched section is worth summarizing.
SECTION_CHAR_LIMIT = 3000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PdfReadError(ValueError):
    """A PDF could not be opened, or held no extractable text."""


class InputError(ValueError):
    """The caller supplied no usable document."""


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------
class SummarizeFn(Protocol):
    """Summarize *text*, honouring the requested length bounds.

    Implementations may raise; :func:`summarize_paper` lets those propagate to
    the caller rather than substituting a plausible-looking value.
    """

    def __call__(self, text: str, *, max_length: int, min_length: int) -> str: ...


@dataclass(frozen=True)
class PaperSummary:
    """The structured result of summarizing one paper."""

    title: str
    concise_summary: str
    key_findings: str
    methodology: str
    original_word_count: int

    @property
    def summary_word_count(self) -> int:
        return len(self.concise_summary.split())

    @property
    def compression_ratio(self) -> float:
        """How many times shorter the summary is than the original."""
        return self.original_word_count / max(self.summary_word_count, 1)


# ---------------------------------------------------------------------------
# PDF reading
# ---------------------------------------------------------------------------
def read_pdf_text(pdf_path: str) -> str:
    """Extract every page's text from the PDF at *pdf_path*.

    Raises:
        PdfReadError: if the file cannot be opened, or contains no text.
    """
    import fitz  # PyMuPDF -- imported lazily so this module stays importable

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise PdfReadError(
            "Could not open the PDF file. It may be corrupted or "
            f"password-protected. Details: {exc}"
        ) from exc

    pages = [text for page in doc if (text := page.get_text("text")).strip()]
    doc.close()

    if not pages:
        raise PdfReadError(
            "The PDF appears to contain no extractable text. "
            "It may be a scanned document or consist only of images."
        )

    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Text preparation
# ---------------------------------------------------------------------------
def _clean_text(text: str) -> str:
    """Normalize whitespace and remove common PDF artefacts."""
    # Remove form-feed and other control characters (keep newlines & tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Re-join hyphenated line breaks ("summa-\nrization" -> "summarization")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_title(text: str) -> str:
    """Guess the paper title from the first few lines.

    Academic papers put the title in the first 1-5 lines, before the author
    block. Take the first substantial line that is not a section header, an
    email, or a URL.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()][:12]

    for line in lines:
        if len(line) < 10:  # page numbers, dates
            continue
        if "@" in line:  # author block
            continue
        if line.isupper() and len(line) < 60:  # section header, e.g. "ABSTRACT"
            continue
        if re.search(r"https?://|www\.", line):
            continue
        return line

    return "Untitled Paper"


def _chunk_text(text: str, max_words: int = CHUNK_WORD_LIMIT) -> list[str]:
    """Split *text* into chunks of roughly *max_words* words.

    Splits on paragraph boundaries so chunks stay coherent. A paragraph that is
    itself over the limit is split on sentence boundaries instead.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        if current_word_count + para_words > max_words and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_word_count = 0

        if para_words > max_words:
            for sentence in re.split(r"(?<=[.!?])\s+", para):
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


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def _summarize(text: str, summarize_fn: SummarizeFn, max_cap: int) -> str:
    """Summarize *text*, scaling the length bounds to the input.

    Short inputs are returned unchanged: asking a summarizer to expand 20 words
    to a 40-word minimum produces padding, and costs a round-trip to do it.
    """
    word_count = len(text.split())
    if word_count < PASSTHROUGH_WORD_COUNT:
        return text

    max_len = min(max_cap, max(PASSTHROUGH_WORD_COUNT, word_count // 2))
    min_len = min(SUMMARY_MIN_LENGTH, max_len - 10)
    return summarize_fn(text, max_length=max_len, min_length=min_len)


def _summarize_chunked(text: str, summarize_fn: SummarizeFn) -> str:
    """Summarize a document of any length.

    Chunks the document, summarizes each chunk, then -- if there was more than
    one -- summarizes the concatenated chunk summaries so the result reads as
    one piece rather than a list of fragments.
    """
    chunk_summaries = [
        _summarize(chunk, summarize_fn, SUMMARY_MAX_LENGTH)
        for chunk in _chunk_text(text)
    ]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = " ".join(chunk_summaries)
    return _summarize(combined, summarize_fn, COMBINE_SUMMARY_MAX_LENGTH)


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------
def _extract_section(text: str, heading_pattern: str) -> str:
    """Return the body under a heading matching *heading_pattern*, or ""."""
    pattern = re.compile(
        rf"(?:^|\n)\s*(?:\d+[\.\)]?\s*)?{heading_pattern}\s*\n"
        r"(.*?)(?=\n\s*(?:\d+[\.\)]?\s*)?[A-Z][A-Za-z ]+\s*\n|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        content = match.group(1).strip()
        if len(content) > 30:
            return content
    return ""


def _first_matching_section(text: str, headings: tuple[str, ...]) -> str:
    for heading in headings:
        content = _extract_section(text, heading)
        if content:
            return content
    return ""


_FINDINGS_HEADINGS = (
    r"(?:key\s+)?findings",
    r"results?\s*(?:and\s+discussion)?",
    r"conclusions?\s*(?:and\s+future\s+work)?",
    r"discussion",
)

_METHODOLOGY_HEADINGS = (
    r"method(?:ology|s)?",
    r"approach",
    r"experimental\s+setup",
    r"materials?\s+and\s+methods",
    r"(?:proposed\s+)?(?:framework|system|model|architecture)",
)


def _extract_key_findings(text: str, summarize_fn: SummarizeFn) -> str:
    """Summarize the results/conclusion section, or the tail of the paper."""
    content = _first_matching_section(text, _FINDINGS_HEADINGS)
    if content:
        return _summarize(
            content[:SECTION_CHAR_LIMIT], summarize_fn, SUMMARY_MAX_LENGTH
        )

    # No recognisable heading: papers put their conclusions at the end, so
    # summarize the last quarter.
    words = text.split()
    tail = " ".join(words[-(len(words) // 4) :])
    if len(tail.split()) > PASSTHROUGH_WORD_COUNT:
        return _summarize(tail[:SECTION_CHAR_LIMIT], summarize_fn, SUMMARY_MAX_LENGTH)
    return "Key findings could not be automatically extracted."


def _extract_methodology(text: str, summarize_fn: SummarizeFn) -> str:
    """Summarize the methodology section, if one can be found."""
    content = _first_matching_section(text, _METHODOLOGY_HEADINGS)
    if content:
        return _summarize(
            content[:SECTION_CHAR_LIMIT], summarize_fn, SUMMARY_MAX_LENGTH
        )
    return "Methodology section could not be automatically extracted."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def summarize_paper(
    *,
    summarize_fn: SummarizeFn,
    pdf_path: Optional[str] = None,
    pasted_text: Optional[str] = None,
    read_pdf: Callable[[str], str] = read_pdf_text,
) -> PaperSummary:
    """Summarize a paper supplied as a PDF path or as pasted text.

    Exactly one of *pdf_path* and *pasted_text* is used; *pdf_path* wins if both
    are given.

    Raises:
        InputError: neither input holds a usable document.
        PdfReadError: *pdf_path* could not be read.
        Exception: whatever *summarize_fn* raises. Failures are not swallowed --
            a caller that is out of inference credits must find that out rather
            than receive a truncated document that looks like a summary.
    """
    if pdf_path is not None:
        raw_text = read_pdf(pdf_path)
    elif pasted_text and pasted_text.strip():
        raw_text = pasted_text.strip()
    else:
        raise InputError(
            "Please upload a PDF file or paste the paper text. "
            "Both inputs are currently empty."
        )

    text = _clean_text(raw_text)
    original_word_count = len(text.split())

    if original_word_count < MIN_DOCUMENT_WORDS:
        raise InputError(
            "The extracted text is too short to summarize. Please provide a "
            "longer document, or check that the PDF contains selectable text."
        )

    return PaperSummary(
        title=_extract_title(text),
        concise_summary=_summarize_chunked(text, summarize_fn),
        key_findings=_extract_key_findings(text, summarize_fn),
        methodology=_extract_methodology(text, summarize_fn),
        original_word_count=original_word_count,
    )
