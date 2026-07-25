"""
Resume Analyzer - AI-Powered Resume Analysis Against Job Descriptions

This Gradio application uses NLP techniques to evaluate how well a resume
matches a given job description. It provides semantic similarity scoring,
keyword extraction, gap analysis, and actionable improvement suggestions.

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import re
import logging
from functools import lru_cache
from typing import Optional

import fitz  # PyMuPDF
import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

RESUME_SECTIONS = {
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

# ---------------------------------------------------------------------------
# Example data shipped with the Space
# ---------------------------------------------------------------------------
EXAMPLE_RESUME = """LORENZO SCATURCHIO
San Francisco, CA | lorenzo@email.com | linkedin.com/in/lorenzo | github.com/gr8monk3ys

PROFESSIONAL SUMMARY
Results-driven Machine Learning Engineer with 4+ years of experience designing,
building, and deploying production ML systems. Skilled in deep learning, NLP,
and scalable data pipelines. Passionate about applying AI to solve real-world
problems and delivering measurable business impact.

EXPERIENCE

Senior Machine Learning Engineer | Acme AI Corp | Jan 2022 - Present
- Designed and deployed a real-time recommendation engine serving 2M+ daily
  active users, improving click-through rate by 18%.
- Built an end-to-end NLP pipeline for document classification using
  Transformers (BERT, RoBERTa) with 94% F1 score.
- Led migration of model training infrastructure to Kubernetes, reducing
  training time by 40% and cloud costs by 25%.
- Mentored a team of 3 junior engineers on MLOps best practices.

Machine Learning Engineer | DataWave Inc | Jun 2020 - Dec 2021
- Developed a customer churn prediction model (XGBoost, LightGBM) that
  reduced churn by 12%, saving $1.2M annually.
- Implemented A/B testing framework for ML model evaluation in production.
- Created automated data quality checks and feature engineering pipelines
  using Apache Spark and Airflow.

Data Science Intern | StartUp Labs | May 2019 - Aug 2019
- Conducted exploratory data analysis on 500K+ records to identify key
  business drivers.
- Built sentiment analysis prototype using spaCy and scikit-learn.

EDUCATION
M.S. Computer Science (Machine Learning Specialization) | Stanford University | 2020
B.S. Computer Science | UC Berkeley | 2018

SKILLS
Languages: Python, SQL, Java, Scala, R
ML/DL Frameworks: PyTorch, TensorFlow, scikit-learn, Hugging Face Transformers
MLOps: Docker, Kubernetes, MLflow, Weights & Biases, Airflow
Cloud: AWS (SageMaker, EC2, S3), GCP (Vertex AI, BigQuery)
Data: Spark, Pandas, NumPy, PostgreSQL, Redis
Other: Git, CI/CD, REST APIs, Agile/Scrum

PROJECTS
- Open-source NLP toolkit for resume analysis (this project!)
- Real-time object detection system using YOLOv5 on edge devices
- Kaggle competition top-5% finish in Tabular Playground Series
"""

EXAMPLE_JOB_DESCRIPTION = """Senior Machine Learning Engineer

About the Role
We are looking for a Senior Machine Learning Engineer to join our AI Platform
team. You will design, build, and maintain production ML systems that power
our core product features. This is a high-impact role where you will work
closely with product, engineering, and data science teams.

Responsibilities
- Design and implement scalable ML pipelines for training and inference.
- Develop and deploy deep learning models for NLP and computer vision tasks.
- Build robust monitoring and alerting for model performance in production.
- Collaborate with cross-functional teams to translate business requirements
  into ML solutions.
- Mentor junior engineers and contribute to engineering best practices.
- Drive adoption of MLOps tools and processes across the organization.

Requirements
- 3+ years of experience in machine learning engineering or a related role.
- Strong proficiency in Python and SQL.
- Hands-on experience with ML frameworks such as PyTorch or TensorFlow.
- Experience deploying models to production using Docker and Kubernetes.
- Familiarity with cloud platforms (AWS or GCP).
- Solid understanding of NLP techniques (transformers, embeddings, etc.).
- Experience with data processing frameworks like Spark or similar.
- Strong communication skills and ability to work in an Agile environment.

Nice to Have
- Experience with recommendation systems or search ranking.
- Contributions to open-source ML projects.
- M.S. or Ph.D. in Computer Science, Machine Learning, or related field.
- Experience with MLflow, Weights & Biases, or similar experiment tracking.
- Familiarity with CI/CD pipelines for ML.
"""


# ---------------------------------------------------------------------------
# Model loading (deferred to first use, then cached)
# ---------------------------------------------------------------------------
# Loading at import cost the Space its UI: a failed download (rate limit, cold
# network) raised during module execution, so the container died and visitors
# got a bare "Runtime error" instead of a page that could explain itself.
@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    logger.info("Loading sentence-transformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Model loaded successfully.")
    return model


# =========================================================================
# Core analysis utilities
# =========================================================================


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract plain text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages = [page.get_text() for page in doc]
        doc.close()
        text = "\n".join(pages).strip()
        if not text:
            raise ValueError("The PDF appears to contain no extractable text.")
        return text
    except Exception as exc:
        logger.error("PDF extraction failed: %s", exc)
        raise gr.Error(f"Could not read the PDF file: {exc}") from exc


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Return cosine similarity (0-1) between two texts using the sentence-transformer."""
    embeddings = _get_model().encode([text_a, text_b], convert_to_numpy=True)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(similarity, 0.0, 1.0))


def extract_keywords(texts: list[str], top_n: int = 30) -> list[list[str]]:
    """
    Use TF-IDF to extract the most important keywords from each document
    in *texts*. Returns a list of keyword-lists, one per input document.
    """
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
        keywords = [feature_names[i] for i in top_indices if row[i] > 0]
        results.append(keywords)
    return results


def _normalize(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def find_matching_and_missing_keywords(
    resume_text: str, job_keywords: list[str]
) -> tuple[list[str], list[str]]:
    """
    Compare job-description keywords against the resume body.
    Returns (matched, missing) keyword lists.
    """
    resume_lower = _normalize(resume_text)
    matched, missing = [], []
    for kw in job_keywords:
        if _normalize(kw) in resume_lower:
            matched.append(kw)
        else:
            missing.append(kw)
    return matched, missing


def detect_sections(resume_text: str) -> dict[str, str]:
    """
    Heuristically split a resume into named sections.
    Returns a dict mapping canonical section names to their content.
    """
    lines = resume_text.split("\n")
    current_section: Optional[str] = None
    sections: dict[str, list[str]] = {}

    for line in lines:
        stripped = line.strip()
        matched_section = _match_section_header(stripped)
        if matched_section:
            current_section = matched_section
            sections.setdefault(current_section, [])
        elif current_section:
            sections[current_section].append(stripped)
        else:
            sections.setdefault("other", []).append(stripped)

    return {name: "\n".join(content).strip() for name, content in sections.items()}


def _match_section_header(line: str) -> Optional[str]:
    """Return a canonical section name if *line* looks like a header, else None."""
    cleaned = re.sub(r"[^a-z\s]", "", line.lower()).strip()
    if not cleaned or len(cleaned) > 40:
        return None
    for canonical, variants in RESUME_SECTIONS.items():
        if cleaned in variants:
            return canonical
    return None


def analyze_section(
    section_name: str, section_text: str, job_text: str
) -> dict[str, object]:
    """Compute a per-section relevance score and short commentary."""
    if not section_text.strip():
        return {
            "score": 0.0,
            "comment": f"No {section_name} section detected in the resume.",
        }
    score = compute_semantic_similarity(section_text, job_text)
    pct = round(score * 100, 1)

    if pct >= 70:
        quality = "Strong"
    elif pct >= 45:
        quality = "Moderate"
    else:
        quality = "Weak"

    comment = f"{quality} alignment ({pct}%) with the job description."
    return {"score": score, "comment": comment}


def generate_suggestions(
    missing_keywords: list[str],
    section_scores: dict[str, dict],
    overall_score: float,
) -> list[str]:
    """Return a list of actionable improvement suggestions."""
    suggestions: list[str] = []

    if overall_score < 0.45:
        suggestions.append(
            "Your resume has low overall alignment with this job description. "
            "Consider tailoring it more directly to the role's requirements."
        )

    # Missing keywords
    if missing_keywords:
        top_missing = missing_keywords[:10]
        suggestions.append(
            "Add these high-value keywords or phrases where truthfully applicable: "
            + ", ".join(f'"{kw}"' for kw in top_missing)
            + "."
        )

    # Section-specific advice
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

    if "summary" not in section_scores or not section_scores.get("summary", {}).get(
        "score", 0
    ):
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


# =========================================================================
# Main analysis orchestrator
# =========================================================================


def run_analysis(
    resume_text: str,
    job_description: str,
    pdf_file: Optional[str] = None,
) -> tuple[str, str, str, str]:
    """
    Run the full resume analysis pipeline.

    Returns a 4-tuple of Markdown strings:
        (overview, keywords_report, section_report, suggestions_report)
    """
    # ------------------------------------------------------------------
    # Input resolution
    # ------------------------------------------------------------------
    if pdf_file is not None:
        resume_text = extract_text_from_pdf(pdf_file)

    if not resume_text or not resume_text.strip():
        raise gr.Error("Please provide resume text or upload a PDF.")
    if not job_description or not job_description.strip():
        raise gr.Error("Please provide a job description.")

    # ------------------------------------------------------------------
    # 1. Semantic similarity
    # ------------------------------------------------------------------
    raw_similarity = compute_semantic_similarity(resume_text, job_description)

    # ------------------------------------------------------------------
    # 2. Keyword analysis
    # ------------------------------------------------------------------
    keyword_lists = extract_keywords([resume_text, job_description], top_n=30)
    resume_keywords, job_keywords = keyword_lists[0], keyword_lists[1]
    matched_kw, missing_kw = find_matching_and_missing_keywords(
        resume_text, job_keywords
    )

    keyword_overlap = len(matched_kw) / max(len(job_keywords), 1)

    # ------------------------------------------------------------------
    # 3. Composite score  (60% semantic + 40% keyword overlap)
    # ------------------------------------------------------------------
    composite = 0.6 * raw_similarity + 0.4 * keyword_overlap
    overall_pct = round(composite * 100, 1)

    # ------------------------------------------------------------------
    # 4. Section-by-section analysis
    # ------------------------------------------------------------------
    sections = detect_sections(resume_text)
    section_scores: dict[str, dict] = {}
    for sec_name in ["summary", "experience", "education", "skills", "projects"]:
        sec_text = sections.get(sec_name, "")
        section_scores[sec_name] = analyze_section(sec_name, sec_text, job_description)

    # ------------------------------------------------------------------
    # 5. Suggestions
    # ------------------------------------------------------------------
    suggestions = generate_suggestions(missing_kw, section_scores, composite)

    # ------------------------------------------------------------------
    # Format outputs as Markdown
    # ------------------------------------------------------------------
    overview_md = _format_overview(
        overall_pct, raw_similarity, keyword_overlap, matched_kw, missing_kw
    )
    keywords_md = _format_keywords(
        resume_keywords, job_keywords, matched_kw, missing_kw
    )
    sections_md = _format_sections(section_scores)
    suggest_md = _format_suggestions(suggestions)

    return overview_md, keywords_md, sections_md, suggest_md


# =========================================================================
# Markdown formatters
# =========================================================================


def _score_bar(pct: float, width: int = 20) -> str:
    """Return a text-based progress bar for Markdown."""
    filled = round(pct / 100 * width)
    empty = width - filled
    return f"`[{'=' * filled}{' ' * empty}]` **{pct}%**"


def _format_overview(
    overall_pct: float,
    semantic_sim: float,
    keyword_overlap: float,
    matched: list[str],
    missing: list[str],
) -> str:
    sem_pct = round(semantic_sim * 100, 1)
    kw_pct = round(keyword_overlap * 100, 1)

    if overall_pct >= 70:
        verdict = "Excellent match - your resume aligns strongly with this role."
    elif overall_pct >= 50:
        verdict = (
            "Good match - some targeted improvements could strengthen your application."
        )
    elif overall_pct >= 30:
        verdict = "Partial match - significant tailoring is recommended."
    else:
        verdict = "Low match - consider whether this role fits your background or rewrite substantially."

    return (
        f"## Overall Match Score\n\n"
        f"# {_score_bar(overall_pct)}\n\n"
        f"**Verdict:** {verdict}\n\n"
        f"---\n\n"
        f"### Score Breakdown\n\n"
        f"| Component | Score |\n"
        f"|---|---|\n"
        f"| Semantic Similarity | {sem_pct}% |\n"
        f"| Keyword Overlap | {kw_pct}% |\n"
        f"| **Composite (60/40)** | **{overall_pct}%** |\n\n"
        f"---\n\n"
        f"**Matched Keywords:** {len(matched)} &nbsp;|&nbsp; "
        f"**Missing Keywords:** {len(missing)}\n"
    )


def _format_keywords(
    resume_kw: list[str],
    job_kw: list[str],
    matched: list[str],
    missing: list[str],
) -> str:
    matched_str = (
        ", ".join(f"**{kw}**" for kw in matched) if matched else "_None detected_"
    )
    missing_str = (
        ", ".join(f"~~{kw}~~" for kw in missing)
        if missing
        else "_None - great coverage!_"
    )
    resume_str = ", ".join(resume_kw[:20]) if resume_kw else "_None detected_"
    job_str = ", ".join(job_kw[:20]) if job_kw else "_None detected_"

    return (
        f"## Keyword Analysis\n\n"
        f"### Matched Keywords (found in your resume)\n{matched_str}\n\n"
        f"---\n\n"
        f"### Missing Keywords (consider adding)\n{missing_str}\n\n"
        f"---\n\n"
        f"### Top Resume Keywords (TF-IDF)\n{resume_str}\n\n"
        f"### Top Job Description Keywords (TF-IDF)\n{job_str}\n"
    )


def _format_sections(section_scores: dict[str, dict]) -> str:
    header = "## Section-by-Section Analysis\n\n"
    rows = "| Section | Score | Assessment |\n|---|---|---|\n"
    details = ""

    for name, data in section_scores.items():
        pct = round(data["score"] * 100, 1)
        comment = data["comment"]
        display_name = name.replace("_", " ").title()
        rows += f"| {display_name} | {pct}% | {comment} |\n"
        details += f"### {display_name}\n{comment}\n\n"

    return header + rows + "\n---\n\n" + details


def _format_suggestions(suggestions: list[str]) -> str:
    items = "\n".join(f"{i}. {s}" for i, s in enumerate(suggestions, 1))
    return (
        f"## Improvement Suggestions\n\n"
        f"{items}\n\n"
        f"---\n\n"
        f"*Tip: Tailor your resume for every application. Mirror the language "
        f"used in the job posting while remaining truthful about your experience.*\n"
    )


# =========================================================================
# Gradio interface
# =========================================================================


def build_interface() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="green",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(theme=theme, title="Resume Analyzer") as demo:
        gr.Markdown(
            "# Resume Analyzer\n"
            "Evaluate how well your resume matches a job description using "
            "**semantic similarity** and **keyword analysis**.\n\n"
            "Paste your resume text (or upload a PDF) and the target job "
            "description, then click **Analyze**."
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                resume_text = gr.Textbox(
                    label="Resume Text",
                    placeholder="Paste your resume here...",
                    lines=18,
                    value=EXAMPLE_RESUME,
                )
                pdf_upload = gr.File(
                    label="Or Upload Resume PDF",
                    file_types=[".pdf"],
                    type="filepath",
                )
            with gr.Column(scale=1):
                job_desc = gr.Textbox(
                    label="Job Description",
                    placeholder="Paste the job description here...",
                    lines=22,
                    value=EXAMPLE_JOB_DESCRIPTION,
                )

        analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Tabs():
            with gr.Tab("Overview"):
                overview_output = gr.Markdown()
            with gr.Tab("Keywords"):
                keywords_output = gr.Markdown()
            with gr.Tab("Sections"):
                sections_output = gr.Markdown()
            with gr.Tab("Suggestions"):
                suggestions_output = gr.Markdown()

        analyze_btn.click(
            fn=run_analysis,
            inputs=[resume_text, job_desc, pdf_upload],
            outputs=[
                overview_output,
                keywords_output,
                sections_output,
                suggestions_output,
            ],
        )

        gr.Markdown(
            "---\n"
            "Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys) | "
            "Model: `sentence-transformers/all-MiniLM-L6-v2` | "
            "[Source Code](https://huggingface.co/spaces/gr8monk3ys/resume-analyzer-space)"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_interface()
    app.launch()
