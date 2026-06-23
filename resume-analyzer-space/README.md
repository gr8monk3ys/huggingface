---
title: Resume Analyzer
emoji: 📋
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: AI-powered resume analysis against job descriptions
---

# Resume Analyzer

An AI-powered tool that analyzes how well your resume matches a target job description. Built with Gradio, sentence-transformers, and scikit-learn.

## Features

### Semantic Similarity Scoring
Uses the `sentence-transformers/all-MiniLM-L6-v2` model to compute deep semantic similarity between your resume and the job description. This goes beyond simple keyword matching to understand the meaning and context of your experience relative to what the role demands.

### TF-IDF Keyword Extraction
Extracts the most important keywords and phrases from both documents using Term Frequency-Inverse Document Frequency (TF-IDF) with unigrams and bigrams. This surfaces the specific terms that carry the most weight in each document.

### Keyword Gap Analysis
Compares the job description's top keywords against your resume content to identify:
- **Matched keywords** -- terms the job requires that your resume already contains.
- **Missing keywords** -- high-value terms you should consider adding (where truthfully applicable).

### Section-by-Section Analysis
Automatically detects standard resume sections (Summary, Experience, Education, Skills, Projects) and scores each one independently against the job description. This pinpoints exactly which parts of your resume need the most attention.

### Composite Match Score
A weighted composite score (60% semantic similarity, 40% keyword overlap) that gives a single 0-100% indicator of overall fit.

### Actionable Suggestions
Generates specific, prioritized recommendations for improving your resume's alignment with the target role.

### PDF Upload Support
Upload your resume as a PDF file instead of pasting text. The app extracts text from the PDF automatically using PyMuPDF.

## How to Use

1. **Paste your resume** into the left text area, or upload a PDF using the file upload widget.
2. **Paste the job description** into the right text area.
3. Click **Analyze**.
4. Browse the results across four tabs:
   - **Overview** -- composite score, breakdown, and verdict.
   - **Keywords** -- matched and missing keywords with TF-IDF rankings.
   - **Sections** -- per-section scores and assessments.
   - **Suggestions** -- numbered, actionable improvement recommendations.

Example data is pre-loaded so you can click Analyze immediately to see the tool in action.

## Technical Architecture

```
Resume Text / PDF  ──┐
                     ├──▶  Semantic Embedding (MiniLM-L6-v2)  ──▶  Cosine Similarity
Job Description  ────┘
                     ├──▶  TF-IDF Vectorization  ──▶  Keyword Extraction & Matching
                     └──▶  Section Detection (regex heuristics)  ──▶  Per-section Scoring
```

| Component | Technology |
|---|---|
| Web framework | Gradio 5.31.0 |
| Semantic model | sentence-transformers/all-MiniLM-L6-v2 |
| Keyword extraction | scikit-learn TfidfVectorizer |
| PDF parsing | PyMuPDF (fitz) |
| Numerical compute | NumPy |

## Running Locally

```bash
# Clone the repository
git clone https://huggingface.co/spaces/gr8monk3ys/resume-analyzer-space
cd resume-analyzer-space

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

The app will be available at `http://localhost:7860`.

## Project Structure

```
resume-analyzer-space/
├── app.py              # Application source (Gradio interface + analysis logic)
├── requirements.txt    # Python dependencies
└── README.md           # This file (includes HF Space metadata)
```

## License

MIT
