---
license: mit
task_categories:
  - text-classification
  - summarization
language:
  - en
tags:
  - arxiv
  - academic-papers
  - research
  - computer-science
  - machine-learning
size_categories:
  - 1K<n<10K
---

# CS/ML Academic Papers Dataset

A curated dataset of academic paper metadata sourced from [arXiv](https://arxiv.org/), spanning key Computer Science and Machine Learning categories. Designed for research in text classification, topic modelling, summarisation, and scientometric analysis.

## Dataset Description

### Overview

The published snapshot (collected **2026-07-12**) contains metadata for **2,019 papers** — 2,500 were fetched and 481 cross-listed duplicates removed — across five core arXiv categories at the intersection of computer science and machine learning:

| Category  | Description                         |
|-----------|-------------------------------------|
| `cs.AI`   | Artificial Intelligence             |
| `cs.CL`   | Computation and Language (NLP)      |
| `cs.CV`   | Computer Vision and Pattern Recog.  |
| `cs.LG`   | Machine Learning                    |
| `stat.ML` | Statistics — Machine Learning       |

Papers are sorted by submission date (most recent first) and deduplicated by arXiv ID, so cross-listed papers appear only once.

### Collection Methodology

1. The [arXiv API](https://info.arxiv.org/help/api/index.html) is queried via the [`arxiv`](https://pypi.org/project/arxiv/) Python package.
2. For each of the five categories, the 500 most recent papers are fetched (configurable).
3. Results are deduplicated on `arxiv_id` — papers cross-listed in multiple categories keep only their first occurrence.
4. Rate limiting (3.5 s between pages, 5 retries on failure) ensures compliance with arXiv's API Terms of Service.

The collection script is fully reproducible: see [`create_dataset.py`](create_dataset.py).

### Schema

| Column             | Type           | Description                                              |
|--------------------|----------------|----------------------------------------------------------|
| `arxiv_id`         | `string`       | arXiv identifier (e.g. `2401.12345v1`)                   |
| `title`            | `string`       | Paper title, whitespace-normalised                       |
| `abstract`         | `string`       | Full abstract text, whitespace-normalised                |
| `authors`          | `list[string]` | Ordered list of author names                             |
| `categories`       | `list[string]` | All arXiv categories assigned to the paper               |
| `primary_category` | `string`       | The primary arXiv category                               |
| `published`        | `string`       | ISO-8601 publication timestamp                           |
| `updated`          | `string`       | ISO-8601 last-updated timestamp                          |
| `doi`              | `string`       | Digital Object Identifier (empty string if unavailable)  |
| `url`              | `string`       | Canonical arXiv abstract URL                             |

### Splits

| Split   | Rows  | Share |
|---------|-------|-------|
| `train` | 1,817 | 90 %  |
| `test`  | 202   | 10 %  |
| **Total** | **2,019** | |

The split is performed with a fixed random seed (`42`) for reproducibility.
Row counts describe the 2026-07-12 snapshot; re-running the collection script
on a different date will produce a different corpus (see Limitations).

## Intended Uses

- **Text classification** — predict the primary arXiv category from the title or abstract.
- **Summarisation** — generate concise summaries from abstracts or use titles as abstractive targets.
- **Topic modelling** — discover latent research themes across categories.
- **Scientometrics** — analyse publication trends, author collaboration patterns, and category overlap.
- **Information retrieval** — build search or recommendation engines over academic literature.

## Example

```python
from datasets import load_dataset

ds = load_dataset("gr8monk3ys/cs-ml-academic-papers")

# Peek at the first training example
print(ds["train"][0])
# {
#     "arxiv_id": "2401.12345v1",
#     "title": "An Example Paper on Large Language Models",
#     "abstract": "We propose a novel method for ...",
#     "authors": ["Alice Researcher", "Bob Scientist"],
#     "categories": ["cs.CL", "cs.AI"],
#     "primary_category": "cs.CL",
#     "published": "2024-01-20T18:00:00+00:00",
#     "updated": "2024-01-22T12:30:00+00:00",
#     "doi": "10.1234/example.2024.001",
#     "url": "http://arxiv.org/abs/2401.12345v1"
#  }
```

## Reproducing the Dataset

```bash
# Install dependencies
pip install -r requirements.txt

# Fetch papers and save locally (Parquet + HF Dataset format)
python create_dataset.py

# Optionally push to the HuggingFace Hub
python create_dataset.py --push --hf-repo gr8monk3ys/cs-ml-academic-papers

# Run exploratory data analysis and generate plots
python explore_dataset.py
```

### CLI Options — `create_dataset.py`

| Flag              | Default                              | Description                            |
|-------------------|--------------------------------------|----------------------------------------|
| `--per-category`  | `500`                                | Papers to fetch per arXiv category     |
| `--push`          | off                                  | Push dataset to HuggingFace Hub        |
| `--hf-repo`       | `gr8monk3ys/cs-ml-academic-papers`   | Hub repository ID                      |
| `--output-dir`    | `./data`                             | Local output directory                 |
| `--verbose`       | off                                  | Enable debug logging                   |

### CLI Options — `explore_dataset.py`

| Flag           | Default                              | Description                            |
|----------------|--------------------------------------|----------------------------------------|
| `--data-dir`   | `./data`                             | Local data directory                   |
| `--from-hub`   | —                                    | Load from HF Hub repo instead          |
| `--plots-dir`  | `./plots`                            | Output directory for PNG plots         |
| `--verbose`    | off                                  | Enable debug logging                   |

## Generated Visualisations

The EDA script (`explore_dataset.py`) produces the following plots:

- **category_distribution.png** — horizontal bar chart of paper counts per primary category.
- **abstract_length_histogram.png** — word-count distribution of abstracts with mean/median markers.
- **abstract_length_by_category.png** — box plots comparing abstract lengths across categories.
- **authors_per_paper.png** — histogram of author counts.
- **publication_timeline.png** — monthly publication volume over time.
- **tfidf_terms_heatmap.png** — heatmap of the top TF-IDF terms per category.

## Limitations and Bias

- The dataset is a **snapshot** — arXiv receives thousands of new submissions daily; re-running the collection script at different times will produce different results.
- Only **five categories** are included. Many relevant papers (e.g., `cs.IR`, `cs.RO`, `math.OC`) are excluded.
- Papers cross-listed under multiple categories are kept only once, so secondary categories are under-represented in the primary-category distribution.
- Metadata only — **full paper text and PDFs are not included**.

## Citation

If you use this dataset in your work, please cite it as:

```bibtex
@misc{scaturchio2024csml_papers,
  author       = {Lorenzo Scaturchio},
  title        = {{CS/ML Academic Papers Dataset}},
  year         = {2024},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/datasets/gr8monk3ys/cs-ml-academic-papers}},
}
```

## License

This dataset is released under the [MIT License](https://opensource.org/licenses/MIT). The underlying paper metadata is sourced from arXiv and is subject to arXiv's [Terms of Use](https://info.arxiv.org/help/api/tou.html).
