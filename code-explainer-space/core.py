"""Code explanation logic, independent of Gradio and of the Inference API.

The entry point is :func:`explain_code`, which owns the sequence: validate,
resolve the language, build the prompt, call the model, and return an
:class:`Explanation`. Rendering is ``app.py``'s job -- see
docs/adr/0002-coarse-entry-point-for-space-core-modules.md.

:func:`detect_language` and :func:`highlight_code` are public because the UI
needs them on their own: when explanation fails, the code pane is still shown
highlighted.
"""

from dataclasses import dataclass
from typing import Protocol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
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


# Pygments' name for a language, where it differs from the label we show.
_LEXER_NAMES = {
    "JavaScript": "javascript",
    "TypeScript": "typescript",
    "C++": "cpp",
    "HTML/CSS": "html",
    "Bash": "bash",
}

AUTO_DETECT = "Auto-detect"
UNKNOWN = "Unknown"

MAX_TOKENS = 1500
TOP_P = 0.95
TEMPERATURE = 0.7


class InputError(ValueError):
    """The caller supplied no code to explain."""


class ChatFn(Protocol):
    """Send *messages* to a chat model and return its reply.

    Implementations may raise; :func:`explain_code` lets those propagate rather
    than substituting a plausible-looking explanation.
    """

    def __call__(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str: ...


@dataclass(frozen=True)
class Explanation:
    """The result of explaining one snippet of code."""

    detected_language: str
    body: str
    highlighted_code: str


def detect_language(code: str) -> str:
    """Guess the language of *code*, or return ``"Unknown"``."""
    from pygments.lexers import guess_lexer

    try:
        return guess_lexer(code).name
    except Exception:  # noqa: BLE001 - pygments raises freely; Unknown is the fallback
        return UNKNOWN


def highlight_code(code: str, language: str) -> str:
    """Return *code* as syntax-highlighted HTML.

    Falls back to escaped plain text: highlighting is decoration, and failing to
    decorate must not cost the user sight of their own code.
    """
    import html

    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import get_lexer_by_name, guess_lexer

    try:
        if language in (AUTO_DETECT, UNKNOWN):
            lexer = guess_lexer(code)
        else:
            lexer = get_lexer_by_name(_LEXER_NAMES.get(language, language.lower()))
        return highlight(code, lexer, HtmlFormatter(style="monokai", noclasses=True))
    except Exception:  # noqa: BLE001 - decoration only; show the code regardless
        return f"<pre><code>{html.escape(code)}</code></pre>"


def _build_user_prompt(code: str, language: str) -> str:
    lexer_hint = language.split()[0].lower() if language.strip() else ""
    return f"""Explain the following code.

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


def explain_code(
    code: str,
    language: str,
    level: str,
    *,
    chat_fn: ChatFn,
) -> Explanation:
    """Explain *code* at the requested *level*.

    An unrecognised *level* falls back to Intermediate rather than failing: the
    value comes from a dropdown, so a bad one means a UI change, not bad input.

    Raises:
        InputError: *code* is empty or whitespace.
        Exception: whatever *chat_fn* raises.
    """
    if not code.strip():
        raise InputError("Please paste some code to explain.")

    resolved = detect_language(code) if language == AUTO_DETECT else language
    instruction = EXPLANATION_LEVELS.get(level, EXPLANATION_LEVELS["Intermediate"])

    body = chat_fn(
        [
            {
                "role": "system",
                "content": f"You are an expert programming tutor. {instruction}",
            },
            {"role": "user", "content": _build_user_prompt(code, resolved)},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    return Explanation(
        detected_language=resolved,
        body=body.strip(),
        highlighted_code=highlight_code(code, resolved),
    )
