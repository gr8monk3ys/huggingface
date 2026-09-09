"""Tests for the paper summarizer's core logic.

Everything here drives the coarse entry point, ``summarize_paper``, with a fake
summarizer. The private helpers are exercised through it rather than directly:
the recording fake shows exactly what the chunker handed the model, which is a
stronger assertion than calling ``_chunk_text`` in isolation would be.
"""

import pytest

from conftest import load_local_module

core = load_local_module("paper_summarizer_core", "paper-summarizer-space/core.py")


# --- fakes -----------------------------------------------------------------
class RecordingSummarizer:
    """A summarize_fn that records its calls and returns a marker."""

    def __init__(self, reply="SUMMARY"):
        self.calls = []
        self.reply = reply

    def __call__(self, text, *, max_length, min_length):
        self.calls.append({"text": text, "max": max_length, "min": min_length})
        return self.reply


def paper(body_words=400, *, title="A Study of Things", extra=""):
    """Build a document long enough to be summarized."""
    body = " ".join(f"word{i}" for i in range(body_words))
    return f"{title}\n\nIntroduction\n{body}\n{extra}"


# --- input validation ------------------------------------------------------
def test_no_input_at_all_raises_input_error():
    with pytest.raises(core.InputError, match="upload a PDF"):
        core.summarize_paper(summarize_fn=RecordingSummarizer(), pasted_text="")


def test_whitespace_only_input_raises_input_error():
    with pytest.raises(core.InputError):
        core.summarize_paper(summarize_fn=RecordingSummarizer(), pasted_text="   \n  ")


def test_document_below_the_word_floor_raises_input_error():
    with pytest.raises(core.InputError, match="too short"):
        core.summarize_paper(summarize_fn=RecordingSummarizer(), pasted_text="a b c")


def test_validation_happens_before_any_inference_call():
    """A rejected document must not cost a network round-trip."""
    fake = RecordingSummarizer()
    with pytest.raises(core.InputError):
        core.summarize_paper(summarize_fn=fake, pasted_text="too short")
    assert fake.calls == []


# --- the failure that motivated this extraction ----------------------------
def test_summarizer_failure_propagates_rather_than_returning_truncated_text():
    """The old code caught this and returned the first 100 input words as if
    they were a summary, so an out-of-credits user saw a plausible result."""

    def out_of_credits(text, *, max_length, min_length):
        raise RuntimeError("monthly free allowance is used up")

    with pytest.raises(RuntimeError, match="allowance"):
        core.summarize_paper(summarize_fn=out_of_credits, pasted_text=paper())


# --- PDF path --------------------------------------------------------------
def test_pdf_path_is_read_through_the_injected_reader():
    fake = RecordingSummarizer()
    result = core.summarize_paper(
        summarize_fn=fake,
        pdf_path="/nonexistent.pdf",
        read_pdf=lambda path: paper(title="Title From The Pdf File"),
    )
    assert result.title == "Title From The Pdf File"


def test_pdf_read_error_propagates():
    def unreadable(path):
        raise core.PdfReadError("scanned document")

    with pytest.raises(core.PdfReadError, match="scanned"):
        core.summarize_paper(
            summarize_fn=RecordingSummarizer(),
            pdf_path="/scan.pdf",
            read_pdf=unreadable,
        )


def test_pdf_path_wins_when_both_inputs_are_given():
    result = core.summarize_paper(
        summarize_fn=RecordingSummarizer(),
        pdf_path="/x.pdf",
        pasted_text=paper(title="Title From The Pasted Box"),
        read_pdf=lambda path: paper(title="Title From The Pdf File"),
    )
    assert result.title == "Title From The Pdf File"


# --- title extraction, through the entry point -----------------------------
@pytest.mark.parametrize(
    "first_lines, expected",
    [
        ("Deep Learning for Sequences\n\n", "Deep Learning for Sequences"),
        ("ABSTRACT\nDeep Learning for Sequences\n\n", "Deep Learning for Sequences"),
        ("a@b.com\nDeep Learning for Sequences\n\n", "Deep Learning for Sequences"),
        (
            "https://arxiv.org/x\nDeep Learning for Sequences\n\n",
            "Deep Learning for Sequences",
        ),
        ("12\nDeep Learning for Sequences\n\n", "Deep Learning for Sequences"),
    ],
)
def test_title_skips_headers_authors_urls_and_page_numbers(first_lines, expected):
    body = " ".join(f"word{i}" for i in range(200))
    result = core.summarize_paper(
        summarize_fn=RecordingSummarizer(), pasted_text=f"{first_lines}{body}"
    )
    assert result.title == expected


def test_untitled_when_nothing_looks_like_a_title():
    """Every line is too short to be a title (page numbers, stray fragments)."""
    body = "\n".join("ab" for _ in range(200))
    result = core.summarize_paper(summarize_fn=RecordingSummarizer(), pasted_text=body)
    assert result.title == "Untitled Paper"


# --- chunking, observed through what the summarizer was handed -------------
def test_long_document_is_split_into_several_chunks():
    para = " ".join(f"w{i}" for i in range(600))
    text = "Title Line Here\n\n" + "\n\n".join([para] * 3)

    fake = RecordingSummarizer()
    result = core.summarize_paper(summarize_fn=fake, pasted_text=text)

    body_calls = [c for c in fake.calls if c["text"].startswith("w0")]
    assert len(body_calls) >= 2, "expected the body to be split into chunks"
    # Three short chunk summaries join to well under the passthrough floor, so
    # the second pass is skipped and the joined text is the result verbatim.
    assert set(result.concise_summary.split()) == {"SUMMARY"}


def test_second_pass_runs_when_the_joined_summaries_are_long_enough():
    """Several chunks whose summaries add up past the floor get re-summarized,
    so the result reads as one piece rather than a list of fragments."""
    para = " ".join(f"w{i}" for i in range(600))
    text = "Title Line Here\n\n" + "\n\n".join([para] * 3)

    long_reply = " ".join(f"s{i}" for i in range(30))
    fake = RecordingSummarizer(long_reply)
    core.summarize_paper(summarize_fn=fake, pasted_text=text)

    second_pass = [c for c in fake.calls if c["text"].count("s0") > 1]
    assert second_pass, "expected a second pass over the joined chunk summaries"
    assert second_pass[0]["max"] <= core.COMBINE_SUMMARY_MAX_LENGTH


def test_no_chunk_exceeds_the_word_limit():
    para = " ".join(f"w{i}" for i in range(500))
    text = "Title Line Here\n\n" + "\n\n".join([para] * 4)

    fake = RecordingSummarizer()
    core.summarize_paper(summarize_fn=fake, pasted_text=text)

    for call in fake.calls:
        assert len(call["text"].split()) <= core.CHUNK_WORD_LIMIT


def test_short_text_is_passed_through_without_calling_the_summarizer():
    """Under the passthrough floor, summarizing would pad rather than shorten."""
    text = "A Reasonable Title Here\n\n" + " ".join(f"w{i}" for i in range(35))
    fake = RecordingSummarizer()
    result = core.summarize_paper(summarize_fn=fake, pasted_text=text)
    assert result.concise_summary != "SUMMARY"
    assert not any(c["text"].startswith("w0") for c in fake.calls)


def test_length_bounds_scale_to_the_input():
    fake = RecordingSummarizer()
    core.summarize_paper(summarize_fn=fake, pasted_text=paper(400))
    for call in fake.calls:
        assert call["min"] < call["max"] <= core.SUMMARY_MAX_LENGTH


# --- section extraction ----------------------------------------------------
def test_methodology_and_findings_come_from_their_sections():
    body = " ".join(f"w{i}" for i in range(120))
    text = f"A Paper About Things\n\nMethods\n{body}\n\nResults\n{body}\n"

    fake = RecordingSummarizer()
    core.summarize_paper(summarize_fn=fake, pasted_text=text)

    assert len(fake.calls) >= 3, "summary, findings and methodology each summarize"


def test_missing_sections_report_rather_than_inventing():
    text = "A Paper About Things\n\n" + " ".join(f"w{i}" for i in range(40))
    result = core.summarize_paper(summarize_fn=RecordingSummarizer(), pasted_text=text)
    assert "could not be automatically extracted" in result.methodology


# --- the returned value ----------------------------------------------------
def test_summary_reports_compression():
    result = core.summarize_paper(
        summarize_fn=RecordingSummarizer("one two three"), pasted_text=paper(400)
    )
    assert result.summary_word_count == 3
    assert result.compression_ratio == pytest.approx(result.original_word_count / 3)


def test_compression_ratio_survives_an_empty_summary():
    result = core.summarize_paper(
        summarize_fn=RecordingSummarizer(""), pasted_text=paper(400)
    )
    assert result.compression_ratio == result.original_word_count


def test_result_is_immutable():
    result = core.summarize_paper(
        summarize_fn=RecordingSummarizer(), pasted_text=paper(400)
    )
    with pytest.raises(Exception):
        result.title = "changed"


# --- cleaning --------------------------------------------------------------
def test_hyphenated_line_breaks_are_rejoined():
    body = " ".join(f"w{i}" for i in range(200))
    text = f"A Title Goes Here\n\nsumma-\nrization {body}"
    fake = RecordingSummarizer()
    core.summarize_paper(summarize_fn=fake, pasted_text=text)
    assert any("summarization" in c["text"] for c in fake.calls)
