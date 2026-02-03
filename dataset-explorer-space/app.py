"""
Dataset Explorer - Explore any HuggingFace dataset visually.

Enter a dataset ID to see statistics, sample data, and distributions.
"""

import gradio as gr
import pandas as pd
import numpy as np
from datasets import load_dataset, get_dataset_config_names
import matplotlib.pyplot as plt
import io
import base64

# ---------------------------------------------------------------------------
# Popular Datasets for Quick Access
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

def get_dataset_info(dataset_id: str, config: str = None, split: str = "train", num_samples: int = 100):
    """Load dataset and extract information."""
    try:
        # Get available configs
        configs = get_dataset_config_names(dataset_id)
        if not config and configs:
            config = configs[0]

        # Load dataset
        if config:
            ds = load_dataset(dataset_id, config, split=split, streaming=True)
        else:
            ds = load_dataset(dataset_id, split=split, streaming=True)

        # Take samples
        samples = []
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            samples.append(item)

        df = pd.DataFrame(samples)

        return df, configs, None

    except Exception as e:
        return None, [], str(e)


def generate_stats(df: pd.DataFrame) -> str:
    """Generate statistics for the dataset."""
    if df is None or df.empty:
        return "No data available"

    stats = []
    stats.append(f"## Dataset Statistics\n")
    stats.append(f"**Rows loaded:** {len(df)}")
    stats.append(f"**Columns:** {len(df.columns)}")
    stats.append(f"\n### Column Information\n")

    for col in df.columns:
        dtype = df[col].dtype
        non_null = df[col].count()
        null_pct = (df[col].isnull().sum() / len(df)) * 100

        col_info = f"**{col}** ({dtype})"
        col_info += f"\n- Non-null: {non_null} ({100-null_pct:.1f}%)"

        if dtype == 'object' or dtype.name == 'string':
            unique = df[col].nunique()
            col_info += f"\n- Unique values: {unique}"
            if unique <= 10:
                top_vals = df[col].value_counts().head(5).to_dict()
                col_info += f"\n- Top values: {top_vals}"
        elif np.issubdtype(dtype, np.number):
            col_info += f"\n- Range: [{df[col].min():.2f}, {df[col].max():.2f}]"
            col_info += f"\n- Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f}"

        stats.append(col_info + "\n")

    return "\n".join(stats)


def generate_visualization(df: pd.DataFrame) -> str:
    """Generate visualizations for the dataset."""
    if df is None or df.empty:
        return None

    # Find columns to visualize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

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
            axes[plot_idx].hist(df[col].dropna(), bins=20, color='#3498db', alpha=0.7)
            axes[plot_idx].set_title(f'{col} Distribution')
            axes[plot_idx].set_xlabel(col)
            axes[plot_idx].set_ylabel('Count')
            plot_idx += 1

    # Plot categorical columns as bar charts
    for col in categorical_cols[:2]:
        if plot_idx < len(axes):
            value_counts = df[col].value_counts().head(10)
            axes[plot_idx].barh(range(len(value_counts)), value_counts.values, color='#2ecc71', alpha=0.7)
            axes[plot_idx].set_yticks(range(len(value_counts)))
            axes[plot_idx].set_yticklabels([str(v)[:20] for v in value_counts.index])
            axes[plot_idx].set_title(f'{col} Distribution')
            axes[plot_idx].set_xlabel('Count')
            plot_idx += 1

    plt.tight_layout()

    # Convert to base64 for display
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    return buf


def explore_dataset(dataset_id: str, config: str, split: str, num_samples: int):
    """Main function to explore a dataset."""
    if not dataset_id.strip():
        return "Please enter a dataset ID", "", None, None

    # Clean input
    dataset_id = dataset_id.strip()
    config = config.strip() if config and config.strip() else None

    # Load dataset
    df, configs, error = get_dataset_info(dataset_id, config, split, int(num_samples))

    if error:
        return f"Error loading dataset: {error}", "", None, None

    if df is None or df.empty:
        return "No data found", "", None, None

    # Generate outputs
    stats = generate_stats(df)
    config_info = f"**Available configs:** {', '.join(configs) if configs else 'None'}"

    # Generate visualization
    viz_buf = generate_visualization(df)

    # Format sample data
    sample_df = df.head(10)

    return stats, config_info, viz_buf, sample_df


def load_popular_dataset(dataset_name: str):
    """Load a popular dataset quickly."""
    return dataset_name, "", "train", 100


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Dataset Explorer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Dataset Explorer

    Explore any HuggingFace dataset instantly. Enter a dataset ID to see statistics,
    visualizations, and sample data.

    **Examples:** `imdb`, `squad`, `wikitext`, `emotion`, `ag_news`
    """)

    with gr.Row():
        with gr.Column(scale=2):
            dataset_id = gr.Textbox(
                label="Dataset ID",
                placeholder="e.g., imdb, squad, username/dataset-name",
                value="imdb"
            )
        with gr.Column(scale=1):
            config_input = gr.Textbox(
                label="Config (optional)",
                placeholder="Leave empty for default"
            )
        with gr.Column(scale=1):
            split_input = gr.Dropdown(
                choices=["train", "test", "validation"],
                value="train",
                label="Split"
            )
        with gr.Column(scale=1):
            num_samples = gr.Slider(
                minimum=10,
                maximum=500,
                value=100,
                step=10,
                label="Samples to load"
            )

    with gr.Row():
        explore_btn = gr.Button("Explore Dataset", variant="primary", size="lg")

    # Quick access buttons
    gr.Markdown("### Quick Access - Popular Datasets")
    with gr.Row():
        for ds in POPULAR_DATASETS[:5]:
            btn = gr.Button(ds, size="sm")
            btn.click(fn=lambda x=ds: x, outputs=[dataset_id])
    with gr.Row():
        for ds in POPULAR_DATASETS[5:]:
            btn = gr.Button(ds, size="sm")
            btn.click(fn=lambda x=ds: x, outputs=[dataset_id])

    # Outputs
    with gr.Row():
        with gr.Column(scale=1):
            stats_output = gr.Markdown(label="Statistics")
            config_output = gr.Markdown(label="Configs")
        with gr.Column(scale=1):
            viz_output = gr.Image(label="Visualizations", type="pil")

    gr.Markdown("### Sample Data")
    sample_output = gr.Dataframe(label="First 10 rows", wrap=True)

    # Event handlers
    explore_btn.click(
        fn=explore_dataset,
        inputs=[dataset_id, config_input, split_input, num_samples],
        outputs=[stats_output, config_output, viz_output, sample_output]
    )

    gr.Markdown("""
    ---

    ### Tips
    - For datasets with multiple configs (like `glue`), specify the config name
    - Use streaming to handle large datasets efficiently
    - Check the [HuggingFace Datasets Hub](https://huggingface.co/datasets) for available datasets

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)


if __name__ == "__main__":
    demo.launch()
