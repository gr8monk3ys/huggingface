"""
Code Explainer - AI-powered code explanation using HuggingFace Inference API.
"""

import html

import gradio as gr
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter

from hf_client import InferenceError, make_client, with_retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

LANGUAGES = [
    "Auto-detect",
    "Python",
    "JavaScript",
    "TypeScript",
    "Java",
    "C++",
    "C",
    "Go",
    "Rust",
    "Ruby",
    "PHP",
    "Swift",
    "Kotlin",
    "SQL",
    "Bash",
    "HTML/CSS",
]

EXPLANATION_LEVELS = {
    "Beginner": "Explain this code in simple terms that a programming beginner would understand. Use analogies and avoid jargon. Focus on the 'what' rather than technical details.",
    "Intermediate": "Explain this code for someone with basic programming knowledge. Include technical details but explain any advanced concepts. Discuss both what the code does and how it works.",
    "Advanced": "Provide a thorough technical analysis of this code. Discuss implementation details, time/space complexity, potential edge cases, and possible improvements or alternatives.",
}

# ---------------------------------------------------------------------------
# Initialize client
# ---------------------------------------------------------------------------

client = make_client(MODEL_ID)

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def detect_language(code: str) -> str:
    """Attempt to detect the programming language."""
    try:
        lexer = guess_lexer(code)
        return lexer.name
    except Exception:
        return "Unknown"


def format_code_html(code: str, language: str) -> str:
    """Apply syntax highlighting to code."""
    try:
        if language == "Auto-detect" or language == "Unknown":
            lexer = guess_lexer(code)
        else:
            lang_map = {
                "JavaScript": "javascript",
                "TypeScript": "typescript",
                "C++": "cpp",
                "HTML/CSS": "html",
                "Bash": "bash",
            }
            lang_key = lang_map.get(language, language.lower())
            lexer = get_lexer_by_name(lang_key)

        formatter = HtmlFormatter(style="monokai", noclasses=True)
        return highlight(code, lexer, formatter)
    except Exception:
        return f"<pre><code>{html.escape(code)}</code></pre>"


# ---------------------------------------------------------------------------
# Main explanation function
# ---------------------------------------------------------------------------


def explain_code(code: str, language: str, level: str) -> tuple[str, str]:
    """Generate an explanation for the provided code."""
    if not code.strip():
        return "Please paste some code to explain.", ""

    # Detect language if auto
    detected_lang = language
    if language == "Auto-detect":
        detected_lang = detect_language(code)

    # Build prompt
    level_instruction = EXPLANATION_LEVELS.get(
        level, EXPLANATION_LEVELS["Intermediate"]
    )
    lexer_hint = detected_lang.split()[0].lower() if detected_lang.strip() else ""

    system_prompt = f"You are an expert programming tutor. {level_instruction}"
    user_prompt = f"""Explain the following code.

```{lexer_hint}
{code}
```

Provide a structured explanation with the following sections:

## Overview
A brief summary of what this code does (2-3 sentences).

## Step-by-Step Breakdown
Explain the code section by section, describing what each part does.

## Key Concepts
List and briefly explain the important programming concepts used in this code.

## Potential Improvements
Suggest any improvements, best practices, or potential issues to be aware of.

Keep your explanation clear, accurate, and educational."""

    try:
        completion = with_retry(
            client.chat_completion,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1500,
            temperature=0.7,
            top_p=0.95,
        )
        explanation = completion.choices[0].message.content.strip()

        # Add language badge
        explanation = (
            f"**Detected Language:** `{detected_lang}`\n\n---\n\n{explanation}"
        )

        # Format the code with syntax highlighting
        formatted_code = format_code_html(code, detected_lang)

        return explanation, formatted_code

    except InferenceError as e:
        return f"**{e}**", format_code_html(code, detected_lang)


# ---------------------------------------------------------------------------
# Gradio Interface
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
        fn=explain_code,
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
