"""Model Selector -- a Gradio front end over :mod:`core`.

Owns the UI and the rendering. Task data, ranking, the live Hub query and the
curated fallback all live in core.py; see
docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import gradio as gr

from core import (
    SIZE_PREFERENCES,
    TASKS,
    NoMatchError,
    Recommendation,
    UnknownTaskError,
    recommend,
)


def _render_live(model, position: int) -> list[str]:
    return [
        f"### {position}. {model.name}",
        f"- **Downloads:** {model.downloads:,} | **Likes:** {model.likes:,}",
        f"- **Link:** [View on HuggingFace]({model.url})",
        "",
    ]


def _render_curated(model, position: int) -> list[str]:
    return [
        f"### {position}. {model.name}",
        f"- **Size:** {model.size} parameters",
        f"- **License:** {model.license}",
        f"- **Link:** [View on HuggingFace]({model.url})",
        "",
    ]


def _render(result: Recommendation, use_case: str) -> str:
    """Format a Recommendation as the Markdown shown in the results pane."""
    parts = [
        f"## Recommendations for: {result.task}\n",
        f"*{result.description}*\n",
    ]
    if use_case:
        parts.append(f"**Your use case:** {use_case}\n")

    if result.source == "live":
        parts.append("_Live from the HuggingFace Hub, sorted by downloads._\n")
        if result.size_filter_ignored:
            parts.append(
                "> Size filtering applies to the curated fallback; live results "
                "are ranked by popularity.\n"
            )
        render_one = _render_live
    else:
        parts.append("_Curated picks (live Hub query unavailable right now)._\n")
        render_one = _render_curated

    parts.append("---\n")
    for position, model in enumerate(result.models, 1):
        parts.extend(render_one(model, position))

    return "\n".join(parts)


def get_recommendations(
    task: str, size_pref: str, priority: str, use_case: str
) -> tuple[str, str]:
    """Gradio handler: recommend models, or report why not."""
    try:
        result = recommend(task, size_pref, priority)
    except (UnknownTaskError, NoMatchError) as exc:
        return str(exc), ""

    return _render(result, use_case), result.code_example


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Model Selector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Model Selector

    Find the perfect HuggingFace model for your task. Answer a few questions
    and get personalized recommendations with code examples.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            task_select = gr.Dropdown(
                choices=list(TASKS.keys()),
                label="What do you want to do?",
                value="Text Generation",
            )

            task_description = gr.Markdown(
                value=f"*{TASKS['Text Generation']['description']}*"
            )

            size_select = gr.Dropdown(
                choices=list(SIZE_PREFERENCES.keys()),
                label="Model size preference?",
                value="Any size",
                info="Smaller = faster, larger = higher quality",
            )

            priority_select = gr.Radio(
                choices=["Most Popular", "Smallest/Fastest", "Best Quality"],
                label="What matters most?",
                value="Most Popular",
            )

            use_case = gr.Textbox(
                label="Describe your use case (optional)",
                placeholder="e.g., Customer support chatbot for e-commerce",
            )

            recommend_btn = gr.Button(
                "Get Recommendations", variant="primary", size="lg"
            )

        with gr.Column(scale=1):
            recommendations = gr.Markdown(label="Recommendations")
            code_example = gr.Markdown(label="Code Example")

    # Use cases display
    use_cases_display = gr.Markdown(
        value=f"**Common use cases:** {', '.join(TASKS['Text Generation']['use_cases'])}"
    )

    # Event handlers
    def update_task_info(task):
        desc = f"*{TASKS[task]['description']}*"
        uses = f"**Common use cases:** {', '.join(TASKS[task]['use_cases'])}"
        return desc, uses

    task_select.change(
        fn=update_task_info,
        inputs=[task_select],
        outputs=[task_description, use_cases_display],
    )

    recommend_btn.click(
        fn=get_recommendations,
        inputs=[task_select, size_select, priority_select, use_case],
        outputs=[recommendations, code_example],
    )

    gr.Markdown("""
    ---

    ### Quick Reference

    | Task | Best For | Typical Size |
    |------|----------|--------------|
    | Text Generation | Chatbots, content | 3B - 70B |
    | Text Classification | Sentiment, topics | 50M - 300M |
    | Embeddings | Search, RAG | 20M - 100M |
    | Speech Recognition | Transcription | 200M - 1.5B |
    | Image Generation | Art, visualization | 1B - 12B |

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)


if __name__ == "__main__":
    demo.launch()
