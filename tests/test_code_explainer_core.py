"""Tests for the code explainer's core logic.

Drives the coarse entry point with a recording fake, so the prompt actually sent
to the model is assertable -- which is where this Space's behaviour lives.
"""

import pytest

from conftest import load_local_module

core = load_local_module("code_explainer_core", "code-explainer-space/core.py")

PY_SNIPPET = "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"


class RecordingChat:
    def __init__(self, reply="an explanation"):
        self.calls = []
        self.reply = reply

    def __call__(self, messages, *, max_tokens, temperature, top_p):
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        return self.reply

    @property
    def system(self):
        return self.calls[-1]["messages"][0]["content"]

    @property
    def user(self):
        return self.calls[-1]["messages"][1]["content"]


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize("code", ["", "   ", "\n\t "])
def test_empty_code_raises_input_error(code):
    with pytest.raises(core.InputError, match="paste some code"):
        core.explain_code(code, "Python", "Beginner", chat_fn=RecordingChat())


def test_validation_happens_before_any_inference_call():
    fake = RecordingChat()
    with pytest.raises(core.InputError):
        core.explain_code("  ", "Python", "Beginner", chat_fn=fake)
    assert fake.calls == []


# --- level selection -------------------------------------------------------
@pytest.mark.parametrize("level", ["Beginner", "Intermediate", "Advanced"])
def test_each_level_sends_its_own_instruction(level):
    fake = RecordingChat()
    core.explain_code(PY_SNIPPET, "Python", level, chat_fn=fake)
    assert core.EXPLANATION_LEVELS[level] in fake.system


def test_unknown_level_falls_back_to_intermediate():
    """The value comes from a dropdown; a bad one is a UI bug, not bad input."""
    fake = RecordingChat()
    core.explain_code(PY_SNIPPET, "Python", "Wizard", chat_fn=fake)
    assert core.EXPLANATION_LEVELS["Intermediate"] in fake.system


# --- language resolution ---------------------------------------------------
def test_explicit_language_is_used_verbatim():
    fake = RecordingChat()
    result = core.explain_code(PY_SNIPPET, "Rust", "Beginner", chat_fn=fake)
    assert result.detected_language == "Rust"
    assert "```rust" in fake.user


def test_auto_detect_resolves_a_language():
    fake = RecordingChat()
    result = core.explain_code(PY_SNIPPET, core.AUTO_DETECT, "Beginner", chat_fn=fake)
    assert result.detected_language != core.AUTO_DETECT


def test_multiword_language_uses_only_its_first_word_as_the_fence_hint():
    fake = RecordingChat()
    core.explain_code(PY_SNIPPET, "HTML/CSS", "Beginner", chat_fn=fake)
    assert "```html/css" in fake.user


# --- the prompt ------------------------------------------------------------
def test_user_prompt_carries_the_code_and_the_required_sections():
    fake = RecordingChat()
    core.explain_code(PY_SNIPPET, "Python", "Beginner", chat_fn=fake)
    assert PY_SNIPPET in fake.user
    for heading in ("## Overview", "## Step-by-Step Breakdown", "## Key Concepts"):
        assert heading in fake.user


def test_generation_settings_are_passed_through():
    fake = RecordingChat()
    core.explain_code(PY_SNIPPET, "Python", "Beginner", chat_fn=fake)
    assert fake.calls[-1]["max_tokens"] == core.MAX_TOKENS
    assert fake.calls[-1]["top_p"] == core.TOP_P


# --- the result ------------------------------------------------------------
def test_reply_is_stripped_and_returned_unadorned():
    """core returns the model's text; app.py adds the language badge."""
    result = core.explain_code(
        PY_SNIPPET, "Python", "Beginner", chat_fn=RecordingChat("  spaced  ")
    )
    assert result.body == "spaced"
    assert "Detected Language" not in result.body


def test_result_carries_highlighted_code():
    result = core.explain_code(
        PY_SNIPPET, "Python", "Beginner", chat_fn=RecordingChat()
    )
    assert "<pre" in result.highlighted_code or "<div" in result.highlighted_code


def test_chat_failures_propagate():
    def boom(messages, *, max_tokens, temperature, top_p):
        raise RuntimeError("credits exhausted")

    with pytest.raises(RuntimeError, match="credits"):
        core.explain_code(PY_SNIPPET, "Python", "Beginner", chat_fn=boom)


# --- highlighting, which the UI also calls on its own ----------------------
def test_highlight_falls_back_to_escaped_text_for_an_unknown_language():
    out = core.highlight_code("<script>x</script>", "NotALanguage")
    assert "&lt;script&gt;" in out


def test_highlight_escapes_rather_than_emitting_raw_markup_on_fallback():
    out = core.highlight_code("<img onerror=1>", "NotALanguage")
    assert "<img" not in out


def test_detect_language_returns_unknown_rather_than_raising():
    assert isinstance(core.detect_language(""), str)
