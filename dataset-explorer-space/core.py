"""Dataset exploration logic, independent of Gradio.

The entry point is :func:`explore`. It owns the sequence -- load samples,
describe the columns, draw the charts -- and returns an :class:`Exploration`.

Per ADR-0002 the rendered image crosses the seam (a caller that is not a UI can
still save it) while the statistics come back structured, for ``app.py`` to turn
into Markdown. ``datasets`` and ``matplotlib`` are imported lazily so this
module stays importable without them.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POPULAR_DATASETS = [
    "imdb",
    "squad",
    "glue",
    "wikitext",
    "ag_news",
    "yelp_review_full",
    "amazon_polarity",
    "dbpedia_14",
    "emotion",
    "financial_phrasebank",
]

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


# Charts get unreadable past a handful of panels, and a high-cardinality column
# makes a bar chart of nothing but hairlines.
MAX_PANELS = 4
MAX_CATEGORICAL_CARDINALITY = 20
TOP_VALUES_SHOWN = 5
UNIQUE_VALUES_BEFORE_LISTING = 10


class DatasetLoadError(RuntimeError):
    """The dataset could not be loaded."""


@dataclass(frozen=True)
class ColumnStats:
    """What is known about one column of the sample."""

    name: str
    dtype: str
    non_null: int
    non_null_pct: float
    unique: Optional[int] = None
    top_values: Optional[dict] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    @property
    def is_numeric(self) -> bool:
        return self.mean is not None


@dataclass(frozen=True)
class DatasetStats:
    """Shape and per-column summary of the loaded sample."""

    rows: int
    columns: list[ColumnStats] = field(default_factory=list)


@dataclass(frozen=True)
class Exploration:
    """Everything one exploration produced."""

    stats: DatasetStats
    configs: list[str]
    chart: Any
    sample: pd.DataFrame


def load_samples(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    num_samples: int = 100,
    *,
    loader: Optional[Callable] = None,
    config_lister: Optional[Callable] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Stream *num_samples* rows of a Hub dataset into a DataFrame.

    Raises:
        DatasetLoadError: the dataset, config or split could not be read.
    """
    if loader is None or config_lister is None:
        from datasets import get_dataset_config_names, load_dataset

        loader = loader or load_dataset
        config_lister = config_lister or get_dataset_config_names

    try:
        configs = list(config_lister(dataset_id))
        chosen = config or (configs[0] if configs else None)

        if chosen:
            stream = loader(dataset_id, chosen, split=split, streaming=True)
        else:
            stream = loader(dataset_id, split=split, streaming=True)

        rows = []
        for index, item in enumerate(stream):
            if index >= num_samples:
                break
            rows.append(item)
    except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
        raise DatasetLoadError(str(exc)) from exc

    return pd.DataFrame(rows), configs


def describe(df: pd.DataFrame) -> DatasetStats:
    """Summarize every column of *df*."""
    if df is None or df.empty:
        return DatasetStats(rows=0)

    columns = []
    for name in df.columns:
        series = df[name]
        dtype = series.dtype
        non_null = int(series.count())
        non_null_pct = 100 - (series.isnull().sum() / len(df)) * 100

        common = {
            "name": str(name),
            "dtype": str(dtype),
            "non_null": non_null,
            "non_null_pct": float(non_null_pct),
        }

        if dtype == "object" or dtype.name == "string":
            unique = int(series.nunique())
            top = None
            if unique <= UNIQUE_VALUES_BEFORE_LISTING:
                top = series.value_counts().head(TOP_VALUES_SHOWN).to_dict()
            columns.append(ColumnStats(**common, unique=unique, top_values=top))
        elif np.issubdtype(dtype, np.number):
            columns.append(
                ColumnStats(
                    **common,
                    minimum=float(series.min()),
                    maximum=float(series.max()),
                    mean=float(series.mean()),
                    std=float(series.std()),
                )
            )
        else:
            columns.append(ColumnStats(**common))

    return DatasetStats(rows=len(df), columns=columns)


def visualize(df: pd.DataFrame):
    """Draw distribution charts for *df*, or return None if nothing plots."""
    if df is None or df.empty:
        return None

    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    # Find columns to visualize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Limit columns to visualize
    numeric_cols = numeric_cols[:4]
    categorical_cols = [c for c in categorical_cols if df[c].nunique() <= 20][:4]

    if not numeric_cols and not categorical_cols:
        return None

    # Create figure
    n_plots = len(numeric_cols) + len(categorical_cols)
    if n_plots == 0:
        return None

    fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(4 * min(n_plots, 4), 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot numeric columns as histograms
    for col in numeric_cols[:2]:
        if plot_idx < len(axes):
            axes[plot_idx].hist(df[col].dropna(), bins=20, color="#3498db", alpha=0.7)
            axes[plot_idx].set_title(f"{col} Distribution")
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel("Count")
            plot_idx += 1

    # Plot categorical columns as bar charts
    for col in categorical_cols[:2]:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts().head(10)
            axes[plot_idx].barh(
                range(len(value_counts)),
                value_counts.values,
                color="#2ecc71",
                alpha=0.7,
            )
            axes[plot_idx].set_yticks(range(len(value_counts)))
            axes[plot_idx].set_yticklabels([str(v)[:20] for v in value_counts.index])
            axes[plot_idx].set_title(f"{col} Distribution")
            axes[plot_idx].set_xlabel("Count")
            plot_idx += 1

    plt.tight_layout()

    # Render the figure to an in-memory PNG and return a PIL image
    # (gr.Image cannot reliably render a raw BytesIO buffer).
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    image = Image.open(buf).copy()
    buf.close()
    return image


def explore(
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    num_samples: int = 100,
    *,
    loader: Optional[Callable] = None,
    config_lister: Optional[Callable] = None,
    chart_fn: Optional[Callable] = None,
) -> Exploration:
    """Load a dataset sample and describe it.

    *chart_fn* defaults to :func:`visualize`; it is injectable so a caller
    without matplotlib installed can still reach the rest of the sequence.

    Raises:
        ValueError: *dataset_id* is empty, or the load returned no rows.
        DatasetLoadError: the dataset could not be read.
    """
    if not dataset_id.strip():
        raise ValueError("Please enter a dataset ID")

    df, configs = load_samples(
        dataset_id.strip(),
        config.strip() if config and config.strip() else None,
        split,
        int(num_samples),
        loader=loader,
        config_lister=config_lister,
    )

    if df is None or df.empty:
        raise ValueError("No data found")

    return Exploration(
        stats=describe(df),
        configs=configs,
        chart=(chart_fn or visualize)(df),
        sample=df.head(10),
    )
