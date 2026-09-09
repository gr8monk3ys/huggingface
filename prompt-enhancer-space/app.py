"""Prompt Enhancer -- a Gradio front end over :mod:`core`.

Owns the UI and the adapters: builds the Inference client, wraps it as the
``chat_fn`` core expects, and unpacks the returned Enhancement.
"""

import gradio as gr

from core import PROMPT_TYPES, InputError, enhance_prompt, example_for
from hf_client import InferenceError, make_client, with_retry

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = make_client(MODEL_ID)


def _chat(messages, *, max_tokens, temperature, top_p) -> str:
    """Call the Inference API, retrying transient failures."""
    completion = with_retry(
        client.chat_completion,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return completion.choices[0].message.content


def handle_enhance(
    basic_prompt: str,
    prompt_type: str,
    creativity: float = 0.7,
    enhancement_level: str = "Balanced",
) -> tuple[str, str]:
    """Gradio handler: enhance the prompt, or report why not."""
    try:
        result = enhance_prompt(
            basic_prompt,
            prompt_type,
            creativity,
            enhancement_level,
            chat_fn=_chat,
        )
    except (InputError, InferenceError) as exc:
        return str(exc), ""

    return result.enhanced, result.tips


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.enhanced-output {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 2px;
}
.enhanced-output > div {
    background: white;
    border-radius: 8px;
}
"""

with gr.Blocks(title="Prompt Enhancer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("""
    # Prompt Enhancer

    Transform basic prompts into powerful, detailed prompts that get better AI results.
    Works for image generation, text/chat, code, and creative writing.

    *Inspired by the success of [MagicPrompt](https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion) - now supporting all AI types!*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_type = gr.Dropdown(
                choices=list(PROMPT_TYPES.keys()),
                value="Image Generation",
                label="Prompt Type",
                info="Select the type of AI you're prompting",
            )

            type_description = gr.Markdown(
                value=f"*{PROMPT_TYPES['Image Generation']['description']}*"
            )

            basic_prompt = gr.Textbox(
                label="Your Basic Prompt",
                placeholder="Enter your simple prompt here...",
                lines=3,
            )

            with gr.Row():
                enhancement_level = gr.Radio(
                    choices=["Minimal", "Balanced", "Maximum"],
                    value="Balanced",
                    label="Enhancement Level",
                )

            with gr.Row():
                creativity = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Creativity",
                    info="Higher = more creative variations",
                )

            with gr.Row():
                enhance_btn = gr.Button("Enhance Prompt", variant="primary", size="lg")
                example_btn = gr.Button("Load Example", variant="secondary")

        with gr.Column(scale=1):
            enhanced_prompt = gr.Textbox(
                label="Enhanced Prompt",
                lines=12,
                show_copy_button=True,
                elem_classes=["enhanced-output"],
            )

            tips_output = gr.Markdown(label="Tips")

    # Examples section
    gr.Markdown("### Quick Examples")
    with gr.Row():
        gr.Examples(
            examples=[
                ["a sunset over mountains", "Image Generation"],
                ["explain quantum computing", "Text/Chat"],
                ["function to validate email", "Code Generation"],
                ["story about time travel", "Creative Writing"],
            ],
            inputs=[basic_prompt, prompt_type],
            label="",
        )

    # Stats section
    gr.Markdown("""
    ---
    ### Why Enhanced Prompts Work Better

    | Basic Prompt | Enhanced Prompt |
    |-------------|-----------------|
    | Vague, open to interpretation | Specific, guided direction |
    | Missing context | Rich context and constraints |
    | Generic output | Tailored, high-quality output |
    | Trial and error needed | First-try success rate higher |

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys) |
    [GitHub](https://github.com/gr8monk3ys)
    """)

    # Event handlers
    def update_description(prompt_type):
        return f"*{PROMPT_TYPES[prompt_type]['description']}*"

    prompt_type.change(
        fn=update_description, inputs=[prompt_type], outputs=[type_description]
    )

    enhance_btn.click(
        fn=handle_enhance,
        inputs=[basic_prompt, prompt_type, creativity, enhancement_level],
        outputs=[enhanced_prompt, tips_output],
    )

    example_btn.click(
        fn=example_for, inputs=[prompt_type], outputs=[basic_prompt, enhanced_prompt]
    )


if __name__ == "__main__":
    demo.launch()
