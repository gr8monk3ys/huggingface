"""Code Explainer -- a Gradio front end over :mod:`core`.

Owns the UI and the adapters: builds the Inference client, wraps it as the
``chat_fn`` core expects, and renders the returned Explanation.
"""

import gradio as gr

from core import (
    AUTO_DETECT,
    EXPLANATION_LEVELS,
    LANGUAGES,
    Explanation,
    InputError,
    detect_language,
    explain_code,
    highlight_code,
)
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


def _render(explanation: Explanation) -> str:
    """Format an Explanation as the Markdown shown in the output pane."""
    return (
        f"**Detected Language:** `{explanation.detected_language}`"
        f"\n\n---\n\n{explanation.body}"
    )


def handle_explain(code: str, language: str, level: str) -> tuple[str, str]:
    """Gradio handler: explain the code, or report why not.

    The code pane is filled in either way -- a failed explanation should not
    also cost the user the highlighted view of what they pasted.
    """
    try:
        explanation = explain_code(code, language, level, chat_fn=_chat)
    except InputError as exc:
        return str(exc), ""
    except InferenceError as exc:
        resolved = detect_language(code) if language == AUTO_DETECT else language
        return f"**{exc}**", highlight_code(code, resolved)

    return _render(explanation), explanation.highlighted_code


# ---------------------------------------------------------------------------

EXAMPLE_CODE = '''def fibonacci(n):
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]

    sequence = [0, 1]
    while len(sequence) < n:
        next_num = sequence[-1] + sequence[-2]
        sequence.append(next_num)

    return sequence

# Example usage
result = fibonacci(10)
print(result)'''

with gr.Blocks(title="Code Explainer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Code Explainer

    Paste any code snippet and get a clear, educational explanation.
    Choose your experience level for tailored explanations.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            code_input = gr.Code(
                label="Paste Your Code",
                language="python",
                lines=15,
                value=EXAMPLE_CODE,
            )

            with gr.Row():
                language_dropdown = gr.Dropdown(
                    choices=LANGUAGES,
                    value="Auto-detect",
                    label="Language",
                    scale=1,
                )
                level_dropdown = gr.Dropdown(
                    choices=list(EXPLANATION_LEVELS.keys()),
                    value="Intermediate",
                    label="Explanation Level",
                    scale=1,
                )

            explain_btn = gr.Button("Explain Code", variant="primary")

        with gr.Column(scale=1):
            formatted_code_output = gr.HTML(label="Formatted Code")

    explanation_output = gr.Markdown(label="Explanation")

    explain_btn.click(
        fn=handle_explain,
        inputs=[code_input, language_dropdown, level_dropdown],
        outputs=[explanation_output, formatted_code_output],
    )

    gr.Markdown("""
    ---

    **Model:** Mistral-7B-Instruct via HuggingFace Inference API

    **Tip:** For best results, include complete functions or logical code blocks.

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)

if __name__ == "__main__":
    demo.launch()
