#!/usr/bin/env python3
"""
explore_dataset.py

Exploratory Data Analysis (EDA) for the CS/ML Academic Papers Dataset.

Loads the locally-saved dataset (or downloads from HuggingFace Hub), computes
summary statistics, identifies top terms per category via TF-IDF, and saves
publication-ready visualisations as PNG files.

Usage
-----
    # Analyse the local dataset
    python explore_dataset.py

    # Load from the HuggingFace Hub instead
    python explore_dataset.py --from-hub gr8monk3ys/cs-ml-academic-papers

    # Customise the output directory for plots
    python explore_dataset.py --plots-dir ./plots
"""

from __future__ import annotations

import argparse
import logging
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer

# Use non-interactive backend so the script works headlessly.
matplotlib.use("Agg")

LOG = logging.getLogger("explore_dataset")

DATA_DIR: Path = Path(__file__).resolve().parent / "data"
PLOTS_DIR: Path = Path(__file__).resolve().parent / "plots"

# Colour palette (colour-blind friendly, adapted from Tol's muted scheme).
PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
           "#DDCC77", "#CC6677", "#882255", "#AA4499"]

# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_local(data_dir: Path) -> pd.DataFrame:
    """Load the dataset from a local ``save_to_disk`` directory."""
    ds_path = data_dir / "hf_dataset"
    if not ds_path.exists():
        raise FileNotFoundError(
            f"No saved dataset found at {ds_path}. "
            "Run create_dataset.py first or use --from-hub."
        )
    dd = load_from_disk(str(ds_path))
    frames = [dd[split].to_pandas() for split in dd]
    return pd.concat(frames, ignore_index=True)


def load_hub(repo_id: str) -> pd.DataFrame:
    """Download the dataset from the HuggingFace Hub."""
    from datasets import load_dataset

    dd = load_dataset(repo_id)
    frames = [dd[split].to_pandas() for split in dd]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print high-level summary statistics to stdout."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("  CS/ML Academic Papers Dataset — Summary Statistics")
    print(separator)

    print(f"\n  Total papers           : {len(df):,}")
    print(f"  Unique arXiv IDs       : {df['arxiv_id'].nunique():,}")
    print(f"  Unique primary cats    : {df['primary_category'].nunique()}")

    # Date range
    if "published" in df.columns:
        dates = pd.to_datetime(df["published"], errors="coerce")
        valid = dates.dropna()
        if len(valid) > 0:
            print(f"  Published date range   : {valid.min():%Y-%m-%d} to {valid.max():%Y-%m-%d}")

    # Authors
    author_counts = df["authors"].apply(len)
    print(f"\n  Authors per paper (mean): {author_counts.mean():.1f}")
    print(f"  Authors per paper (med) : {author_counts.median():.0f}")

    # Abstract lengths
    abs_len = df["abstract"].str.split().str.len()
    print(f"\n  Abstract length (words):")
    print(f"    mean   : {abs_len.mean():.0f}")
    print(f"    median : {abs_len.median():.0f}")
    print(f"    min    : {abs_len.min():.0f}")
    print(f"    max    : {abs_len.max():.0f}")
    print(f"    std    : {abs_len.std():.1f}")

    # Category distribution
    print(f"\n  Primary category distribution:")
    for cat, count in df["primary_category"].value_counts().items():
        pct = 100.0 * count / len(df)
        print(f"    {cat:<12s}  {count:>5,}  ({pct:5.1f}%)")

    # DOI availability
    has_doi = (df["doi"].str.len() > 0).sum()
    print(f"\n  Papers with DOI        : {has_doi:,} ({100*has_doi/len(df):.1f}%)")

    print(f"\n{separator}\n")


# ---------------------------------------------------------------------------
# TF-IDF keyword extraction
# ---------------------------------------------------------------------------


def top_tfidf_terms(
    df: pd.DataFrame,
    text_col: str = "abstract",
    group_col: str = "primary_category",
    top_n: int = 15,
) -> dict[str, list[tuple[str, float]]]:
    """
    For each group in *group_col*, fit a TF-IDF vectoriser on the documents
    belonging to that group and return the top-*n* terms by mean TF-IDF score.
    """
    results: dict[str, list[tuple[str, float]]] = {}

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        min_df=5,
        max_df=0.85,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z+#\-]{2,}\b",
    )

    for group, sub_df in df.groupby(group_col):
        texts = sub_df[text_col].tolist()
        if len(texts) < 10:
            LOG.warning("Skipping group %s — too few documents (%d).", group, len(texts))
            continue

        tfidf_matrix = vectorizer.fit_transform(texts)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        top_indices = mean_scores.argsort()[::-1][:top_n]
        results[group] = [
            (feature_names[i], float(mean_scores[i])) for i in top_indices
        ]

    return results


def print_top_terms(terms_by_cat: dict[str, list[tuple[str, float]]]) -> None:
    """Pretty-print TF-IDF top terms per category."""
    print("=" * 60)
    print("  Top TF-IDF Terms per Category")
    print("=" * 60)
    for cat in sorted(terms_by_cat):
        print(f"\n  [{cat}]")
        for rank, (term, score) in enumerate(terms_by_cat[cat], 1):
            print(f"    {rank:>2}. {term:<30s}  (score: {score:.4f})")
    print()


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------


def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    LOG.info("Saved plot -> %s", path)


def plot_category_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of primary-category counts."""
    counts = df["primary_category"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[: len(counts)])
    ax.bar_label(bars, padding=4, fontsize=9)
    ax.set_xlabel("Number of Papers")
    ax.set_title("Papers by Primary arXiv Category")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_dir / "category_distribution.png")


def plot_abstract_length_histogram(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of abstract word counts."""
    lengths = df["abstract"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lengths, bins=50, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.axvline(lengths.median(), color=PALETTE[6], linestyle="--", linewidth=1.5,
               label=f"Median ({lengths.median():.0f} words)")
    ax.axvline(lengths.mean(), color=PALETTE[4], linestyle=":", linewidth=1.5,
               label=f"Mean ({lengths.mean():.0f} words)")
    ax.set_xlabel("Abstract Length (words)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Abstract Lengths")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_dir / "abstract_length_histogram.png")


def plot_abstract_length_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """Box plot of abstract lengths grouped by primary category."""
    df = df.copy()
    df["abstract_words"] = df["abstract"].str.split().str.len()

    cats = df["primary_category"].value_counts().index.tolist()
    data = [df.loc[df["primary_category"] == c, "abstract_words"].values for c in cats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=cats, patch_artist=True, showfliers=False)
    for patch, colour in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)
    ax.set_ylabel("Abstract Length (words)")
    ax.set_title("Abstract Length by Primary Category")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_dir / "abstract_length_by_category.png")


def plot_authors_per_paper(df: pd.DataFrame, output_dir: Path) -> None:
    """Histogram of author counts per paper."""
    author_counts = df["authors"].apply(len)

    fig, ax = plt.subplots(figsize=(8, 5))
    max_display = int(author_counts.quantile(0.99)) + 1
    ax.hist(
        author_counts.clip(upper=max_display),
        bins=range(1, max_display + 2),
        color=PALETTE[2],
        edgecolor="white",
        alpha=0.85,
        align="left",
    )
    ax.set_xlabel("Number of Authors")
    ax.set_ylabel("Frequency")
    ax.set_title("Authors per Paper")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_dir / "authors_per_paper.png")


def plot_publication_timeline(df: pd.DataFrame, output_dir: Path) -> None:
    """Monthly publication counts over time."""
    dates = pd.to_datetime(df["published"], errors="coerce").dropna()
    monthly = dates.dt.to_period("M").value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(monthly)),
        monthly.values,
        color=PALETTE[1],
        edgecolor="white",
        width=1.0,
    )
    # Show a subset of tick labels to avoid crowding.
    step = max(1, len(monthly) // 12)
    tick_indices = list(range(0, len(monthly), step))
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [str(monthly.index[i]) for i in tick_indices],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Publication Timeline (Monthly)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _savefig(fig, output_dir / "publication_timeline.png")


def plot_top_terms_heatmap(
    terms_by_cat: dict[str, list[tuple[str, float]]],
    output_dir: Path,
    top_n: int = 10,
) -> None:
    """Heatmap-style visualisation of top TF-IDF terms across categories."""
    # Gather the union of top terms across all categories.
    all_terms: list[str] = []
    for cat in sorted(terms_by_cat):
        for term, _ in terms_by_cat[cat][:top_n]:
            if term not in all_terms:
                all_terms.append(term)

    cats = sorted(terms_by_cat.keys())
    matrix = np.zeros((len(all_terms), len(cats)))
    for j, cat in enumerate(cats):
        term_map = dict(terms_by_cat[cat])
        for i, term in enumerate(all_terms):
            matrix[i, j] = term_map.get(term, 0.0)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(all_terms))))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=9)
    ax.set_yticks(range(len(all_terms)))
    ax.set_yticklabels(all_terms, fontsize=8)
    ax.set_title("Top TF-IDF Terms by Category")
    fig.colorbar(im, ax=ax, label="Mean TF-IDF Score", shrink=0.6)
    fig.tight_layout()
    _savefig(fig, output_dir / "tfidf_terms_heatmap.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for the CS/ML Academic Papers Dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"Local data directory (default: {DATA_DIR}).",
    )
    parser.add_argument(
        "--from-hub",
        type=str,
        default=None,
        help="Load the dataset from a HuggingFace Hub repo instead of locally.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=str(PLOTS_DIR),
        help=f"Directory for saved plots (default: {PLOTS_DIR}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if args.from_hub:
        LOG.info("Loading dataset from HuggingFace Hub: %s", args.from_hub)
        df = load_hub(args.from_hub)
    else:
        LOG.info("Loading dataset from local directory: %s", args.data_dir)
        df = load_local(Path(args.data_dir))

    LOG.info("Loaded %d papers.", len(df))

    # ------------------------------------------------------------------
    # 2. Summary statistics
    # ------------------------------------------------------------------
    print_summary(df)

    # ------------------------------------------------------------------
    # 3. TF-IDF keyword extraction
    # ------------------------------------------------------------------
    LOG.info("Computing TF-IDF top terms per category ...")
    terms_by_cat = top_tfidf_terms(df)
    print_top_terms(terms_by_cat)

    # ------------------------------------------------------------------
    # 4. Visualisations
    # ------------------------------------------------------------------
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Generating visualisations -> %s", plots_dir)

    plot_category_distribution(df, plots_dir)
    plot_abstract_length_histogram(df, plots_dir)
    plot_abstract_length_by_category(df, plots_dir)
    plot_authors_per_paper(df, plots_dir)
    plot_publication_timeline(df, plots_dir)

    if terms_by_cat:
        plot_top_terms_heatmap(terms_by_cat, plots_dir)

    LOG.info("All plots saved to %s", plots_dir)
    print(f"Visualisations saved to: {plots_dir}")


if __name__ == "__main__":
    main()
