"""Dataset Explorer -- a Gradio front end over :mod:`core`.

Owns the UI and the rendering. All loading, describing and charting lives in
core.py; see docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import gradio as gr

from core import (
    POPULAR_DATASETS,
    DatasetLoadError,
    DatasetStats,
    explore,
)


def _render_column(column) -> str:
    lines = [
        f"**{column.name}** ({column.dtype})",
        f"- Non-null: {column.non_null} ({column.non_null_pct:.1f}%)",
    ]
    if column.unique is not None:
        lines.append(f"- Unique values: {column.unique}")
        if column.top_values:
            lines.append(f"- Top values: {column.top_values}")
    if column.is_numeric:
        lines.append(f"- Range: [{column.minimum:.2f}, {column.maximum:.2f}]")
        lines.append(f"- Mean: {column.mean:.2f}, Std: {column.std:.2f}")
    return "\n".join(lines)


def _render_stats(stats: DatasetStats) -> str:
    """Format DatasetStats as the Markdown shown in the statistics pane."""
    if not stats.rows:
        return "No data available"

    parts = [
        "## Dataset Statistics",
        "",
        f"**Rows loaded:** {stats.rows}",
        f"**Columns:** {len(stats.columns)}",
        "",
        "### Column Information",
        "",
    ]
    parts.extend(_render_column(column) + "\n" for column in stats.columns)
    return "\n".join(parts)


def handle_explore(dataset_id: str, config: str, split: str, num_samples: int):
    """Gradio handler: explore a dataset, or report why not."""
    try:
        result = explore(dataset_id, config, split, num_samples)
    except DatasetLoadError as exc:
        return f"Error loading dataset: {exc}", "", None, None
    except ValueError as exc:
        return str(exc), "", None, None

    configs = ", ".join(result.configs) if result.configs else "None"
    return (
        _render_stats(result.stats),
        f"**Available configs:** {configs}",
        result.chart,
        result.sample,
    )


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
                value="imdb",
            )
        with gr.Column(scale=1):
            config_input = gr.Textbox(
                label="Config (optional)", placeholder="Leave empty for default"
            )
        with gr.Column(scale=1):
            split_input = gr.Dropdown(
                choices=["train", "test", "validation"], value="train", label="Split"
            )
        with gr.Column(scale=1):
            num_samples = gr.Slider(
                minimum=10, maximum=500, value=100, step=10, label="Samples to load"
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
        fn=handle_explore,
        inputs=[dataset_id, config_input, split_input, num_samples],
        outputs=[stats_output, config_output, viz_output, sample_output],
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
