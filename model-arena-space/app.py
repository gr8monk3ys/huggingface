"""AI Model Arena -- a Gradio front end over :mod:`core`.

Owns the UI, the adapters, and the vote tally. The tally is mutable
per-process state, which is why it lives here rather than in core: see
docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import gradio as gr

from core import (
    CATEGORIES,
    MODELS,
    Battle,
    InputError,
    battle,
    example_prompt,
    new_tally,
    random_battle,
    rank,
    record_vote,
)
from hf_client import friendly_error, make_client, with_retry

# Resets when the Space restarts, as the leaderboard footer says.
vote_counts = new_tally()


def _respond(model_id: str, prompt: str, *, max_tokens: int) -> str:
    """Call one model, retrying transient failures.

    Raises with a user-ready message: core records str(exc) verbatim, so the
    classification hf_client does must survive to this point.
    """
    try:
        response = with_retry(
            client_for(model_id).chat_completion,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with a classified message
        raise RuntimeError(friendly_error(exc)) from exc
    return response.choices[0].message.content


def client_for(model_id: str):
    """A client bound to one model. The arena switches models per battle."""
    return make_client(model_id)


def _render_response(response) -> str:
    if not response.ok:
        return f"**Error:** {response.error}"
    return f"{response.text}\n\n---\n*Response time: {response.seconds:.2f}s*"


def _render_info(model_name: str) -> str:
    model = MODELS[model_name]
    return (
        f"**{model_name}**\n{model['description']}\n*Strengths: {model['strengths']}*"
    )


def _render_battle(result: Battle) -> tuple:
    """Unpack a Battle into the six outputs the UI binds."""
    return (
        _render_response(result.first),
        _render_response(result.second),
        _render_info(result.first.model_name),
        _render_info(result.second.model_name),
        result.prompt,
        # Return the models that actually produced these responses, so a vote
        # attributes to them rather than to whatever the dropdowns show later.
        (result.first.model_name, result.second.model_name),
    )


def handle_battle(prompt: str, model1_name: str, model2_name: str) -> tuple:
    """Gradio handler: run a battle, or report why not."""
    try:
        result = battle(prompt, model1_name, model2_name, respond_fn=_respond)
    except InputError as exc:
        return str(exc), "", "", "", "", ("", "")

    return _render_battle(result)


def handle_vote(winner: str, loser: str, prompt: str) -> str:
    """Gradio handler: record a vote against the module-level tally."""
    global vote_counts

    if not prompt or not winner:
        return "Run a battle first before voting!"

    try:
        vote_counts = record_vote(vote_counts, winner, loser)
    except KeyError:
        return "Run a battle first before voting!"

    stats = vote_counts[winner]
    return f"Voted for **{winner}**! Total wins: {stats['wins']}/{stats['battles']}"


def handle_leaderboard() -> str:
    """Gradio handler: render the current standings as a Markdown table."""
    rows = [
        "## Leaderboard",
        "",
        "| Rank | Model | Wins | Battles | Win Rate |",
        "|------|-------|------|---------|----------|",
    ]
    for position, standing in enumerate(rank(vote_counts), 1):
        rate = f"{standing.win_rate:.1f}%" if standing.battles else "-"
        rows.append(
            f"| {position} | {standing.model} | {standing.wins} "
            f"| {standing.battles} | {rate} |"
        )
    rows.append("")
    rows.append("*Leaderboard resets when the Space restarts*")
    return "\n".join(rows)


# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="AI Model Arena", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI Model Arena

    **Compare AI models head-to-head!**

    Test the same prompt across different models and vote for the best response.
    See which models excel at different tasks.

    *All models run via HuggingFace Inference API*
    """)

    # Hidden state for the current battle: prompt + the two models that ran it
    current_prompt = gr.State("")
    current_models = gr.State(("", ""))

    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Your Prompt",
                placeholder="Enter a prompt to test both models...",
                lines=3,
            )

            with gr.Row():
                category_dropdown = gr.Dropdown(
                    choices=list(CATEGORIES.keys()),
                    label="Example Category",
                    value="Creative Writing",
                )
                example_btn = gr.Button("📝 Get Example", size="sm")

        with gr.Column(scale=1):
            model1_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Mistral-7B",
                label="Model A",
            )
            model2_dropdown = gr.Dropdown(
                choices=list(MODELS.keys()),
                value="Llama-3.1-8B",
                label="Model B",
            )

    with gr.Row():
        random_btn = gr.Button("🎲 Random Battle", variant="secondary")
        battle_btn = gr.Button("⚔️ Start Battle!", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            model1_info = gr.Markdown("**Model A**")
            model1_output = gr.Markdown(label="Model A Response")
            vote1_btn = gr.Button("👍 Vote for Model A", variant="secondary")

        with gr.Column():
            model2_info = gr.Markdown("**Model B**")
            model2_output = gr.Markdown(label="Model B Response")
            vote2_btn = gr.Button("👍 Vote for Model B", variant="secondary")

    vote_result = gr.Markdown("")

    with gr.Accordion("📊 Leaderboard", open=False):
        leaderboard_output = gr.Markdown(handle_leaderboard())
        refresh_btn = gr.Button("🔄 Refresh Leaderboard")

    gr.Markdown("""
    ---

    ## Available Models

    | Model | Size | Strengths |
    |-------|------|-----------|
    | Mistral-7B | 7B | Speed, reasoning, code |
    | Llama-3.1-8B | 8B | General knowledge, instructions |
    | Qwen2.5-7B | 7B | Multilingual, math, coding |
    | Phi-3-mini | 3.8B | Efficiency, reasoning |
    | Gemma-2-9B | 9B | Quality, safety |
    | Zephyr-7B | 7B | Helpfulness, alignment |

    ---

    ## Test Categories

    - **Creative Writing** - Poetry, stories, creative tasks
    - **Coding** - Programming challenges
    - **Reasoning** - Logic puzzles, math
    - **Knowledge** - Explanations, facts
    - **Summarization** - Condensing information

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)

    # Event handlers
    battle_btn.click(
        fn=handle_battle,
        inputs=[prompt_input, model1_dropdown, model2_dropdown],
        outputs=[
            model1_output,
            model2_output,
            model1_info,
            model2_info,
            current_prompt,
            current_models,
        ],
    )

    example_btn.click(
        fn=example_prompt,
        inputs=[category_dropdown],
        outputs=[prompt_input],
    )

    random_btn.click(
        fn=random_battle,
        outputs=[model1_dropdown, model2_dropdown, prompt_input],
    )

    vote1_btn.click(
        fn=lambda models, p: handle_vote(models[0], models[1], p),
        inputs=[current_models, current_prompt],
        outputs=[vote_result],
    ).then(
        fn=handle_leaderboard,
        outputs=[leaderboard_output],
    )

    vote2_btn.click(
        fn=lambda models, p: handle_vote(models[1], models[0], p),
        inputs=[current_models, current_prompt],
        outputs=[vote_result],
    ).then(
        fn=handle_leaderboard,
        outputs=[leaderboard_output],
    )

    refresh_btn.click(
        fn=handle_leaderboard,
        outputs=[leaderboard_output],
    )

if __name__ == "__main__":
    demo.launch()
