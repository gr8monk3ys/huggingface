"""Pure resume/job-description matching logic, independent of Gradio.

Everything here is deterministic and network-free so it can be unit tested
without standing up the Space or downloading an embedding model. The one
function that needs scikit-learn (:func:`extract_keywords`) imports it lazily,
which keeps this module importable — and therefore testable — with nothing but
the standard library.

Semantic similarity is *not* computed here. Callers pass a ``similarity_fn``,
so the scoring thresholds can be exercised against known values instead of
whatever an embedding model happens to return that day.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

# Composite score weighting. Named rather than inlined: the 60/40 split is the
# product's headline claim, so it should be visible and assertable.
SEMANTIC_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# Section header variants -> canonical section name.
RESUME_SECTIONS: dict[str, list[str]] = {
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "career history",
    ],
    "education": [
        "education",
        "academic background",
        "academic history",
        "qualifications",
        "certifications",
        "degrees",
    ],
    "skills": [
        "skills",
        "technical skills",
        "core competencies",
        "competencies",
        "proficiencies",
        "technologies",
        "tools",
    ],
    "projects": [
        "projects",
        "personal projects",
        "portfolio",
        "key projects",
        "selected projects",
    ],
    "summary": [
        "summary",
        "professional summary",
        "profile",
        "objective",
        "career objective",
        "about me",
        "overview",
    ],
}

# Sections scored in the report, in the order they are presented.
SCORED_SECTIONS = ("summary", "experience", "education", "skills", "projects")

# A header line longer than this is prose that happens to start with a section
# word, not a heading.
MAX_HEADER_CHARS = 40


def normalize(text: str) -> str:
    """Lowercase and collapse runs of whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def extract_keywords(texts: list[str], top_n: int = 30) -> list[list[str]]:
    """Return the top TF-IDF keywords for each document in *texts*.

    scikit-learn and numpy are imported here rather than at module scope so the
    rest of this module stays dependency-free.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#.\-]{1,}\b",
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    results: list[list[str]] = []
    for row_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[row_idx].toarray().flatten()
        top_indices = row.argsort()[-top_n:][::-1]
        results.append([feature_names[i] for i in top_indices if row[i] > 0])
    return results


def find_matching_and_missing_keywords(
    resume_text: str, job_keywords: list[str]
) -> tuple[list[str], list[str]]:
    """Split *job_keywords* into those present in the resume and those absent."""
    resume_lower = normalize(resume_text)
    matched, missing = [], []
    for kw in job_keywords:
        (matched if normalize(kw) in resume_lower else missing).append(kw)
    return matched, missing


def keyword_overlap(matched: list[str], job_keywords: list[str]) -> float:
    """Fraction of the job's keywords the resume covers. Empty job -> 0.0."""
    if not job_keywords:
        return 0.0
    return len(matched) / len(job_keywords)


def composite_score(semantic: float, overlap: float) -> float:
    """Blend semantic similarity and keyword overlap into the headline score."""
    return SEMANTIC_WEIGHT * semantic + KEYWORD_WEIGHT * overlap


def match_section_header(line: str) -> Optional[str]:
    """Return the canonical section name if *line* reads as a header, else None."""
    cleaned = re.sub(r"[^a-z\s]", "", line.lower()).strip()
    if not cleaned or len(cleaned) > MAX_HEADER_CHARS:
        return None
    for canonical, variants in RESUME_SECTIONS.items():
        if cleaned in variants:
            return canonical
    return None


def detect_sections(resume_text: str) -> dict[str, str]:
    """Split a resume into named sections by looking for header lines.

    Text appearing before any recognised header is collected under ``other``.
    """
    current_section: Optional[str] = None
    sections: dict[str, list[str]] = {}

    for line in resume_text.split("\n"):
        stripped = line.strip()
        header = match_section_header(stripped)
        if header:
            current_section = header
            sections.setdefault(current_section, [])
        elif current_section:
            sections[current_section].append(stripped)
        else:
            sections.setdefault("other", []).append(stripped)

    return {name: "\n".join(body).strip() for name, body in sections.items()}


def describe_alignment(score: float) -> str:
    """Bucket a 0-1 alignment score into Strong / Moderate / Weak."""
    pct = round(score * 100, 1)
    if pct >= 70:
        return "Strong"
    if pct >= 45:
        return "Moderate"
    return "Weak"


def analyze_section(
    section_name: str,
    section_text: str,
    job_text: str,
    similarity_fn: Callable[[str, str], float],
) -> dict[str, object]:
    """Score one resume section against the job description.

    *similarity_fn* is injected so this stays testable without an embedding
    model, following the same pattern as paper-recommender's ``build_index``.
    """
    if not section_text.strip():
        return {
            "score": 0.0,
            "comment": f"No {section_name} section detected in the resume.",
        }
    score = similarity_fn(section_text, job_text)
    pct = round(score * 100, 1)
    return {
        "score": score,
        "comment": f"{describe_alignment(score)} alignment ({pct}%) with the job description.",
    }


def generate_suggestions(
    missing_keywords: list[str],
    section_scores: dict[str, dict],
    overall_score: float,
) -> list[str]:
    """Return actionable improvements, or a single note when nothing is wrong."""
    suggestions: list[str] = []

    if overall_score < 0.45:
        suggestions.append(
            "Your resume has low overall alignment with this job description. "
            "Consider tailoring it more directly to the role's requirements."
        )

    if missing_keywords:
        suggestions.append(
            "Add these high-value keywords or phrases where truthfully applicable: "
            + ", ".join(f'"{kw}"' for kw in missing_keywords[:10])
            + "."
        )

    for name, data in section_scores.items():
        score = data["score"]
        if name == "experience" and score < 0.5:
            suggestions.append(
                "Strengthen your Experience section by using action verbs and "
                "quantifiable achievements that mirror the job requirements."
            )
        if name == "skills" and score < 0.5:
            suggestions.append(
                "Your Skills section could be improved. List specific tools, "
                "frameworks, and technologies mentioned in the job posting."
            )
        if name == "education" and score < 0.3:
            suggestions.append(
                "Consider highlighting relevant coursework, certifications, or "
                "academic projects in your Education section."
            )

    if not section_scores.get("summary", {}).get("score", 0):
        suggestions.append(
            "Add a Professional Summary at the top of your resume that directly "
            "addresses the key requirements of this role."
        )

    if not suggestions:
        suggestions.append(
            "Your resume is well-aligned with this job description. "
            "Keep refining with specific metrics and results."
        )

    return suggestions
