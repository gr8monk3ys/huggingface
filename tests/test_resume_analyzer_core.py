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
