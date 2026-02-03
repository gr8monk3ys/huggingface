#!/usr/bin/env python3
"""
create_dataset.py

Fetches academic paper metadata from the arXiv API across multiple CS/ML
categories, deduplicates the results, and publishes a clean HuggingFace
Dataset in Parquet format.

Usage
-----
    # Fetch papers and save locally
    python create_dataset.py

    # Fetch papers and push to the HuggingFace Hub
    python create_dataset.py --push --hf-repo gr8monk3ys/cs-ml-academic-papers

    # Customise the number of papers per category
    python create_dataset.py --per-category 1000
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from huggingface_hub import HfApi
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CATEGORIES: list[str] = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "stat.ML"]

DEFAULT_PER_CATEGORY: int = 500

# arXiv API courtesy: wait between successive queries to avoid rate-limiting.
# The arXiv API Terms of Service recommend no more than one request every
# three seconds.  We are conservative and wait a bit longer between pages.
REQUEST_DELAY_SECONDS: float = 3.5

# Each arXiv search page can return at most this many results.
PAGE_SIZE: int = 100

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "data"

LOG = logging.getLogger("create_dataset")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Collapse whitespace and strip leading/trailing blanks."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_doi(entry: arxiv.Result) -> str:
    """Return the DOI if present, otherwise an empty string."""
    return entry.doi or ""


def _extract_authors(entry: arxiv.Result) -> list[str]:
    """Return a sorted list of author names."""
    return [str(a) for a in entry.authors]


def _entry_to_record(entry: arxiv.Result) -> dict[str, Any]:
    """Convert an arxiv.Result into a flat dictionary."""
    return {
        "arxiv_id": entry.entry_id.split("/abs/")[-1],
        "title": _clean_text(entry.title),
        "abstract": _clean_text(entry.summary),
        "authors": _extract_authors(entry),
        "categories": list(entry.categories),
        "primary_category": entry.primary_category,
        "published": entry.published.isoformat() if entry.published else "",
        "updated": entry.updated.isoformat() if entry.updated else "",
        "doi": _extract_doi(entry),
        "url": entry.entry_id,
    }


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_papers_for_category(
    category: str,
    max_results: int = DEFAULT_PER_CATEGORY,
) -> list[dict[str, Any]]:
    """
    Query the arXiv API for papers in *category*, respecting rate limits.

    Parameters
    ----------
    category:
        An arXiv category string such as ``"cs.AI"`` or ``"stat.ML"``.
    max_results:
        Maximum number of papers to retrieve for this category.

    Returns
    -------
    list[dict]
        A list of paper-metadata dictionaries.
    """
    LOG.info("Fetching up to %d papers for category: %s", max_results, category)

    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(
        page_size=PAGE_SIZE,
        delay_seconds=REQUEST_DELAY_SECONDS,
        num_retries=5,
    )

    records: list[dict[str, Any]] = []
    try:
        for entry in tqdm(
            client.results(search),
            total=max_results,
            desc=f"  {category}",
            unit="paper",
        ):
            records.append(_entry_to_record(entry))
    except arxiv.UnexpectedEmptyPageError:
        LOG.warning(
            "Received an empty page from arXiv for %s after %d results. "
            "Continuing with what we have.",
            category,
            len(records),
        )
    except arxiv.HTTPError as exc:
        LOG.error(
            "HTTP error while fetching %s (collected %d so far): %s",
            category,
            len(records),
            exc,
        )

    LOG.info("Collected %d papers for %s", len(records), category)
    return records


def fetch_all_papers(
    categories: list[str] | None = None,
    per_category: int = DEFAULT_PER_CATEGORY,
) -> pd.DataFrame:
    """
    Fetch papers across all requested categories and return a deduplicated
    :class:`pandas.DataFrame`.
    """
    categories = categories or CATEGORIES
    all_records: list[dict[str, Any]] = []

    for cat in categories:
        records = fetch_papers_for_category(cat, max_results=per_category)
        all_records.extend(records)
        # Extra courtesy pause between categories
        LOG.info("Pausing between categories ...")
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(all_records)
    before = len(df)
    df = df.drop_duplicates(subset=["arxiv_id"], keep="first").reset_index(drop=True)
    after = len(df)
    LOG.info(
        "Deduplicated %d -> %d records (%d duplicates removed)",
        before,
        after,
        before - after,
    )
    return df


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

FEATURES = Features(
    {
        "arxiv_id": Value("string"),
        "title": Value("string"),
        "abstract": Value("string"),
        "authors": Sequence(Value("string")),
        "categories": Sequence(Value("string")),
        "primary_category": Value("string"),
        "published": Value("string"),
        "updated": Value("string"),
        "doi": Value("string"),
        "url": Value("string"),
    }
)


def build_dataset(df: pd.DataFrame) -> DatasetDict:
    """
    Convert a :class:`pandas.DataFrame` of paper records into a
    :class:`datasets.DatasetDict` with ``train`` / ``test`` splits
    (90 / 10).
    """
    dataset = Dataset.from_pandas(df, features=FEATURES, preserve_index=False)
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    return DatasetDict({"train": splits["train"], "test": splits["test"]})


def save_dataset(dataset_dict: DatasetDict, output_dir: Path) -> None:
    """Save the dataset to disk in Parquet format."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir / "hf_dataset"))

    # Also save individual Parquet files for easy inspection.
    for split_name, split_ds in dataset_dict.items():
        parquet_path = output_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(parquet_path))
        LOG.info("Saved %s split (%d rows) -> %s", split_name, len(split_ds), parquet_path)


def push_to_hub(dataset_dict: DatasetDict, repo_id: str) -> None:
    """Push the dataset to the HuggingFace Hub."""
    LOG.info("Pushing dataset to HuggingFace Hub: %s", repo_id)
    dataset_dict.push_to_hub(repo_id, private=False)
    LOG.info("Successfully pushed to %s", repo_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch arXiv paper metadata and create a HuggingFace Dataset.",
    )
    parser.add_argument(
        "--per-category",
        type=int,
        default=DEFAULT_PER_CATEGORY,
        help=f"Number of papers to fetch per category (default: {DEFAULT_PER_CATEGORY}).",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the dataset to the HuggingFace Hub after creation.",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="gr8monk3ys/cs-ml-academic-papers",
        help="HuggingFace Hub repo ID (default: gr8monk3ys/cs-ml-academic-papers).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Local output directory (default: {OUTPUT_DIR}).",
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

    output_dir = Path(args.output_dir)
    LOG.info("=" * 60)
    LOG.info("arXiv Academic Papers Dataset Builder")
    LOG.info("=" * 60)
    LOG.info("Categories      : %s", ", ".join(CATEGORIES))
    LOG.info("Per category    : %d", args.per_category)
    LOG.info("Output directory: %s", output_dir)
    LOG.info("")

    start = time.time()

    # 1. Fetch ----------------------------------------------------------
    df = fetch_all_papers(per_category=args.per_category)

    # 2. Build dataset --------------------------------------------------
    dataset_dict = build_dataset(df)
    LOG.info(
        "Dataset built — train: %d, test: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["test"]),
    )

    # 3. Save locally ---------------------------------------------------
    save_dataset(dataset_dict, output_dir)

    # 4. (Optional) push to Hub -----------------------------------------
    if args.push:
        push_to_hub(dataset_dict, args.hf_repo)

    elapsed = time.time() - start
    LOG.info("Done in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
