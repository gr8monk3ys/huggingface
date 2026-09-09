"""Tests for the resume-analyzer pure logic (core.py).

Semantic similarity is injected, so the scoring thresholds are exercised
against known values rather than whatever an embedding model returns.
"""

import pytest

from conftest import load_local_module

core = load_local_module("resume_analyzer_core", "resume-analyzer-space/core.py")


# --------------------------------------------------------------------------
# Composite score -- the product's headline claim
# --------------------------------------------------------------------------
def test_weights_sum_to_one():
    assert core.SEMANTIC_WEIGHT + core.KEYWORD_WEIGHT == pytest.approx(1.0)


def test_composite_is_sixty_forty():
    # 60% semantic + 40% keyword overlap
    assert core.composite_score(1.0, 0.0) == pytest.approx(0.6)
    assert core.composite_score(0.0, 1.0) == pytest.approx(0.4)
    assert core.composite_score(0.5, 0.5) == pytest.approx(0.5)


def test_composite_stays_in_range_at_the_extremes():
    assert core.composite_score(0.0, 0.0) == pytest.approx(0.0)
    assert core.composite_score(1.0, 1.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Keyword matching
# --------------------------------------------------------------------------
def test_keyword_match_is_case_and_whitespace_insensitive():
    matched, missing = core.find_matching_and_missing_keywords(
        "Built pipelines with  Apache   Spark and PyTorch.",
        ["apache spark", "PYTORCH", "kubernetes"],
    )
    assert matched == ["apache spark", "PYTORCH"]
    assert missing == ["kubernetes"]


def test_overlap_of_empty_job_is_zero_not_a_crash():
    # The old inline form divided by max(len, 1); this must stay division-safe.
    assert core.keyword_overlap([], []) == 0.0


def test_overlap_is_the_covered_fraction():
    assert core.keyword_overlap(["a", "b"], ["a", "b", "c", "d"]) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Section detection
# --------------------------------------------------------------------------
def test_headers_map_to_canonical_names():
    assert core.match_section_header("PROFESSIONAL SUMMARY") == "summary"
    assert core.match_section_header("Work Experience") == "experience"
    assert core.match_section_header("Core Competencies") == "skills"


def test_long_line_starting_with_a_section_word_is_not_a_header():
    line = "Experience deploying models to production using Docker and Kubernetes"
    assert len(line) > core.MAX_HEADER_CHARS
    assert core.match_section_header(line) is None


def test_decorated_headers_still_match():
    # Real resumes bracket headers with punctuation.
    assert core.match_section_header("--- SKILLS ---") == "skills"
    assert core.match_section_header("Projects:") == "projects"


def test_detect_sections_splits_body_under_each_header():
    resume = "\n".join(
        [
            "Jane Doe",
            "SUMMARY",
            "Backend engineer.",
            "SKILLS",
            "Python, Go",
            "EDUCATION",
            "B.S. Computer Science",
        ]
    )
    got = core.detect_sections(resume)
    assert got["summary"] == "Backend engineer."
    assert got["skills"] == "Python, Go"
    assert got["education"] == "B.S. Computer Science"
    # Text before the first header is not silently dropped.
    assert got["other"] == "Jane Doe"


def test_detect_sections_on_a_resume_with_no_headers():
    got = core.detect_sections("just a paragraph about me")
    assert got == {"other": "just a paragraph about me"}


# --------------------------------------------------------------------------
# Section scoring
# --------------------------------------------------------------------------
def test_missing_section_scores_zero_without_calling_the_model():
    calls = []

    def boom(a, b):  # must never run for an empty section
        calls.append((a, b))
        return 1.0

    out = core.analyze_section("skills", "   ", "job text", boom)
    assert out["score"] == 0.0
    assert "No skills section" in out["comment"]
    assert calls == []


@pytest.mark.parametrize(
    "score,word",
    [
        (0.90, "Strong"),
        (0.70, "Strong"),
        (0.55, "Moderate"),
        (0.45, "Moderate"),
        (0.30, "Weak"),
        (0.0, "Weak"),
    ],
)
def test_alignment_buckets_at_their_boundaries(score, word):
    assert core.describe_alignment(score) == word


def test_analyze_section_reports_the_injected_score():
    out = core.analyze_section("experience", "did things", "job", lambda a, b: 0.82)
    assert out["score"] == pytest.approx(0.82)
    assert "Strong alignment (82.0%)" in out["comment"]


# --------------------------------------------------------------------------
# Suggestions
# --------------------------------------------------------------------------
def _sections(**scores):
    return {name: {"score": s, "comment": ""} for name, s in scores.items()}


def test_weak_sections_each_get_their_own_advice():
    out = core.generate_suggestions(
        [], _sections(experience=0.2, skills=0.1, education=0.1, summary=0.9), 0.3
    )
    joined = " ".join(out)
    assert "Experience section" in joined
    assert "Skills section" in joined
    assert "Education section" in joined


def test_missing_summary_is_called_out():
    out = core.generate_suggestions([], _sections(experience=0.9), 0.9)
    assert any("Professional Summary" in s for s in out)


def test_a_present_summary_is_not_called_out():
    out = core.generate_suggestions(
        [], _sections(summary=0.8, experience=0.9, skills=0.9, education=0.9), 0.9
    )
    assert not any("Professional Summary" in s for s in out)


def test_missing_keywords_are_capped_at_ten():
    kws = [f"kw{i}" for i in range(25)]
    out = core.generate_suggestions(kws, _sections(summary=0.9), 0.9)
    line = next(s for s in out if "high-value keywords" in s)
    assert line.count('"') == 20  # 10 quoted keywords
    assert "kw10" not in line


def test_a_strong_resume_still_gets_one_encouraging_note():
    out = core.generate_suggestions(
        [], _sections(summary=0.9, experience=0.9, skills=0.9, education=0.9), 0.9
    )
    assert len(out) == 1
    assert "well-aligned" in out[0]


# --------------------------------------------------------------------------
# TF-IDF path -- needs scikit-learn, which the Space has but CI does not
# --------------------------------------------------------------------------
def test_extract_keywords_ranks_distinctive_terms_first():
    pytest.importorskip("sklearn")
    docs = [
        "kubernetes docker kubernetes deployment kubernetes orchestration",
        "baking sourdough bread with a starter and steam",
    ]
    got = core.extract_keywords(docs, top_n=5)
    assert "kubernetes" in got[0]
    assert "kubernetes" not in got[1]


# ===========================================================================
# PDF reading and the analyze() entry point
# ===========================================================================
JOB = "We need a Python engineer with experience in machine learning and SQL."
RESUME = """Jane Doe

Summary
Python engineer with five years building machine learning systems.

Experience
Built SQL pipelines and trained models in production.

Skills
Python, SQL, machine learning, pandas
"""


def fixed_similarity(value):
    return lambda a, b: value


def test_analyze_rejects_an_empty_resume():
    with pytest.raises(core.InputError, match="resume text"):
        core.analyze("", JOB, similarity_fn=fixed_similarity(0.8))


def test_analyze_rejects_an_empty_job_description():
    with pytest.raises(core.InputError, match="job description"):
        core.analyze(RESUME, "   ", similarity_fn=fixed_similarity(0.8))


def test_analyze_reads_the_pdf_through_the_injected_reader():
    pytest.importorskip("sklearn")
    result = core.analyze(
        "",
        JOB,
        pdf_path="/x.pdf",
        similarity_fn=fixed_similarity(0.8),
        read_pdf=lambda path: RESUME,
    )
    assert result.overall_pct > 0


def test_pdf_read_errors_propagate_rather_than_becoming_a_ui_type():
    """The old code raised gr.Error from here, welding the reader to gradio."""

    def unreadable(path):
        raise core.PdfReadError("scanned document")

    with pytest.raises(core.PdfReadError, match="scanned"):
        core.analyze(
            "",
            JOB,
            pdf_path="/scan.pdf",
            similarity_fn=fixed_similarity(0.8),
            read_pdf=unreadable,
        )


def test_analysis_reports_the_composite_of_semantic_and_keyword_scores():
    pytest.importorskip("sklearn")
    result = core.analyze(RESUME, JOB, similarity_fn=fixed_similarity(1.0))
    expected = core.composite_score(1.0, result.keyword_overlap_pct / 100) * 100
    assert result.overall_pct == pytest.approx(expected, abs=0.1)
    assert result.semantic_pct == 100.0


def test_analysis_scores_every_scored_section():
    pytest.importorskip("sklearn")
    result = core.analyze(RESUME, JOB, similarity_fn=fixed_similarity(0.7))
    assert set(result.section_scores) == set(core.SCORED_SECTIONS)


@pytest.mark.parametrize(
    "pct, fragment",
    [
        (95, "Excellent"),
        (70, "Excellent"),
        (60, "Good"),
        (50, "Good"),
        (40, "Partial"),
        (30, "Partial"),
        (10, "Low"),
    ],
)
def test_verdict_bands(pct, fragment):
    assert fragment in core.describe_match(pct)


def test_overall_verdict_bands_differ_from_section_alignment_bands():
    """Not an oversight: one judges a whole application, the other one section."""
    assert core.describe_match(60) != core.describe_alignment(0.60)
