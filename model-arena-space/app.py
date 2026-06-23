"""
AI Model Arena - Compare AI model outputs side by side.
Test prompts across multiple models and vote for the best response.
"""

import random
import time

import gradio as gr

from hf_client import InferenceError, friendly_error, make_client, with_retry

# ---------------------------------------------------------------------------
# Model Configurations
# ---------------------------------------------------------------------------

MODELS = {
    "Mistral-7B": {
        "id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Fast, efficient 7B parameter model from Mistral AI",
        "strengths": "Speed, reasoning, code",
    },
    "Llama-3.1-8B": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta's latest open LLM with strong capabilities",
        "strengths": "General knowledge, instruction following",
    },
    "Qwen2.5-7B": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Alibaba's powerful multilingual model",
        "strengths": "Multilingual, math, coding",
    },
    "Phi-3-mini": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft's compact but capable model",
        "strengths": "Efficiency, reasoning, small size",
    },
    "Gemma-2-9B": {
        "id": "google/gemma-2-9b-it",
        "description": "Google's instruction-tuned Gemma model",
        "strengths": "Quality, safety, general tasks",
    },
    "Zephyr-7B": {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "description": "Fine-tuned Mistral with DPO alignment",
        "strengths": "Helpfulness, alignment, chat",
    },
}

CATEGORIES = {
    "Creative Writing": [
        "Write a haiku about artificial intelligence",
        "Create a short story opening about a robot discovering emotions",
        "Write a limerick about machine learning",
        "Compose a brief poem about the future of technology",
    ],
    "Coding": [
        "Write a Python function to check if a number is prime",
        "Create a JavaScript function to reverse a string",
        "Write a SQL query to find duplicate emails in a users table",
        "Implement a simple stack data structure in Python",
    ],
    "Reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    ],
    "Knowledge": [
        "Explain quantum entanglement in simple terms",
        "What are the main differences between TCP and UDP?",
        "Briefly explain how transformers work in machine learning",
        "What is the difference between machine learning and deep learning?",
    ],
    "Summarization": [
        "Summarize the concept of blockchain technology in 2-3 sentences",
        "Explain the main idea behind reinforcement learning briefly",
        "Summarize what makes Python popular for data science",
        "Briefly explain the concept of transfer learning",
    ],
}

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------

vote_counts = {model: {"wins": 0, "battles": 0} for model in MODELS}

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def get_model_response(model_id: str, prompt: str, max_tokens: int = 500) -> tuple:
    """Get response from a model with timing."""
    client = make_client(model_id)

    start_time = time.time()
    try:
        response = with_retry(
            client.chat_completion,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        elapsed = time.time() - start_time
        return response.choices[0].message.content, elapsed, None
    except InferenceError as e:
        return None, time.time() - start_time, str(e)
    except Exception as e:  # noqa: BLE001 - surfaced to the UI
        return None, time.time() - start_time, friendly_error(e)


def battle(prompt: str, model1_name: str, model2_name: str) -> tuple:
    """Run a battle between two models."""
    if not prompt.strip():
        return "Please enter a prompt.", "", "", "", "", ("", "")

    model1_id = MODELS[model1_name]["id"]
    model2_id = MODELS[model2_name]["id"]

    # Get responses
    resp1, time1, err1 = get_model_response(model1_id, prompt)
    resp2, time2, err2 = get_model_response(model2_id, prompt)

    # Format responses
    if err1:
        output1 = f"**Error:** {err1}"
    else:
        output1 = f"{resp1}\n\n---\n*Response time: {time1:.2f}s*"

    if err2:
        output2 = f"**Error:** {err2}"
    else:
        output2 = f"{resp2}\n\n---\n*Response time: {time2:.2f}s*"

    # Model info
    info1 = f"**{model1_name}**\n{MODELS[model1_name]['description']}\n*Strengths: {MODELS[model1_name]['strengths']}*"
    info2 = f"**{model2_name}**\n{MODELS[model2_name]['description']}\n*Strengths: {MODELS[model2_name]['strengths']}*"

    # Return the battled model names so votes attribute to the models that
    # actually produced these responses, not whatever the dropdowns show later.
    return output1, output2, info1, info2, prompt, (model1_name, model2_name)


def vote_model(winner: str, loser: str, prompt: str) -> str:
    """Record a vote for the winning model."""
    if not prompt or not winner or winner not in vote_counts:
        return "Run a battle first before voting!"

    vote_counts[winner]["wins"] += 1
    vote_counts[winner]["battles"] += 1
    if loser in vote_counts:
        vote_counts[loser]["battles"] += 1

    return f"Voted for **{winner}**! Total wins: {vote_counts[winner]['wins']}/{vote_counts[winner]['battles']}"


def get_leaderboard() -> str:
    """Generate leaderboard markdown."""
    # Calculate win rates
    rankings = []
    for model, stats in vote_counts.items():
        if stats["battles"] > 0:
            win_rate = stats["wins"] / stats["battles"] * 100
        else:
            win_rate = 0
        rankings.append((model, stats["wins"], stats["battles"], win_rate))

    # Sort by win rate, then by total wins
    rankings.sort(key=lambda x: (x[3], x[1]), reverse=True)

    # Generate markdown table
    md = "## Leaderboard\n\n"
    md += "| Rank | Model | Wins | Battles | Win Rate |\n"
    md += "|------|-------|------|---------|----------|\n"

    for i, (model, wins, battles, rate) in enumerate(rankings, 1):
        if battles > 0:
            md += f"| {i} | {model} | {wins} | {battles} | {rate:.1f}% |\n"
        else:
            md += f"| {i} | {model} | 0 | 0 | - |\n"

    md += "\n*Leaderboard resets when the Space restarts*"
    return md


def random_battle() -> tuple:
    """Set up a random battle."""
    models = list(MODELS.keys())
    model1 = random.choice(models)
    model2 = random.choice([m for m in models if m != model1])
    category = random.choice(list(CATEGORIES.keys()))
    prompt = random.choice(CATEGORIES[category])
    return model1, model2, prompt


def get_example_prompt(category: str) -> str:
    """Get a random prompt from a category."""
    if category in CATEGORIES:
        return random.choice(CATEGORIES[category])
    return ""


# ---------------------------------------------------------------------------
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
        leaderboard_output = gr.Markdown(get_leaderboard())
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
        fn=battle,
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
        fn=get_example_prompt,
        inputs=[category_dropdown],
        outputs=[prompt_input],
    )

    random_btn.click(
        fn=random_battle,
        outputs=[model1_dropdown, model2_dropdown, prompt_input],
    )

    vote1_btn.click(
        fn=lambda models, p: vote_model(models[0], models[1], p),
        inputs=[current_models, current_prompt],
        outputs=[vote_result],
    ).then(
        fn=get_leaderboard,
        outputs=[leaderboard_output],
    )

    vote2_btn.click(
        fn=lambda models, p: vote_model(models[1], models[0], p),
        inputs=[current_models, current_prompt],
        outputs=[vote_result],
    ).then(
        fn=get_leaderboard,
        outputs=[leaderboard_output],
    )

    refresh_btn.click(
        fn=get_leaderboard,
        outputs=[leaderboard_output],
    )

if __name__ == "__main__":
    demo.launch()
