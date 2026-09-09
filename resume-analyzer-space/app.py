"""
Resume Analyzer - AI-Powered Resume Analysis Against Job Descriptions

This Gradio application uses NLP techniques to evaluate how well a resume
matches a given job description. It provides semantic similarity scoring,
keyword extraction, gap analysis, and actionable improvement suggestions.

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import logging
from functools import lru_cache
from typing import Optional

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core import Analysis, InputError, PdfReadError, analyze

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
# Adapter: the similarity function core.analyze is given
# =========================================================================


def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity (0-1) between two texts, via the sentence-transformer."""
    embeddings = _get_model().encode([text_a, text_b], convert_to_numpy=True)
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(np.clip(similarity, 0.0, 1.0))


# =========================================================================
# Handler
# =========================================================================


def run_analysis(
    resume_text: str,
    job_description: str,
    pdf_file: Optional[str] = None,
) -> tuple[str, str, str, str]:
    """Gradio handler: analyze, then render the four Markdown panes."""
    try:
        analysis = analyze(
            resume_text,
            job_description,
            pdf_file,
            similarity_fn=compute_semantic_similarity,
        )
    except (InputError, PdfReadError) as exc:
        logger.info("Analysis rejected: %s", exc)
        raise gr.Error(str(exc)) from exc

    return (
        _format_overview(analysis),
        _format_keywords(analysis),
        _format_sections(analysis.section_scores),
        _format_suggestions(analysis.suggestions),
    )


# =========================================================================
# Markdown formatters
# =========================================================================


def _score_bar(pct: float, width: int = 20) -> str:
    """Return a text-based progress bar for Markdown."""
    filled = round(pct / 100 * width)
    empty = width - filled
    return f"`[{'=' * filled}{' ' * empty}]` **{pct}%**"


def _format_overview(analysis: Analysis) -> str:
    return (
        f"## Overall Match Score\n\n"
        f"# {_score_bar(analysis.overall_pct)}\n\n"
        f"**Verdict:** {analysis.verdict}\n\n"
        f"---\n\n"
        f"### Score Breakdown\n\n"
        f"| Component | Score |\n"
        f"|---|---|\n"
        f"| Semantic Similarity | {analysis.semantic_pct}% |\n"
        f"| Keyword Overlap | {analysis.keyword_overlap_pct}% |\n"
        f"| **Composite (60/40)** | **{analysis.overall_pct}%** |\n\n"
        f"---\n\n"
        f"**Matched Keywords:** {len(analysis.matched_keywords)} &nbsp;|&nbsp; "
        f"**Missing Keywords:** {len(analysis.missing_keywords)}\n"
    )


def _format_keywords(analysis: Analysis) -> str:
    resume_kw = analysis.resume_keywords
    job_kw = analysis.job_keywords
    matched = analysis.matched_keywords
    missing = analysis.missing_keywords
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
