"""Tests for the prompt enhancer's core logic."""

import pytest

from conftest import load_local_module

core = load_local_module("prompt_enhancer_core", "prompt-enhancer-space/core.py")


class RecordingChat:
    def __init__(self, reply="an enhanced prompt"):
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
@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_prompt_raises_input_error(text):
    with pytest.raises(core.InputError, match="enter a prompt"):
        core.enhance_prompt(text, "Text/Chat", chat_fn=RecordingChat())


def test_validation_happens_before_any_inference_call():
    fake = RecordingChat()
    with pytest.raises(core.InputError):
        core.enhance_prompt("", "Text/Chat", chat_fn=fake)
    assert fake.calls == []


# --- type and level selection ---------------------------------------------
@pytest.mark.parametrize("prompt_type", list(core.PROMPT_TYPES))
def test_each_type_sends_its_own_system_prompt(prompt_type):
    fake = RecordingChat()
    core.enhance_prompt("a cat", prompt_type, chat_fn=fake)
    assert core.PROMPT_TYPES[prompt_type]["system_prompt"] in fake.system


def test_unknown_type_falls_back_to_the_default():
    fake = RecordingChat()
    core.enhance_prompt("a cat", "Interpretive Dance", chat_fn=fake)
    assert core.PROMPT_TYPES[core.DEFAULT_TYPE]["system_prompt"] in fake.system


@pytest.mark.parametrize("level", list(core.LEVEL_INSTRUCTIONS))
def test_level_instruction_is_appended_to_the_system_prompt(level):
    fake = RecordingChat()
    core.enhance_prompt("a cat", "Text/Chat", 0.7, level, chat_fn=fake)
    assert fake.system.endswith(core.LEVEL_INSTRUCTIONS[level])


def test_unknown_level_appends_nothing_rather_than_failing():
    fake = RecordingChat()
    core.enhance_prompt("a cat", "Text/Chat", 0.7, "Extreme", chat_fn=fake)
    assert fake.system == core.PROMPT_TYPES["Text/Chat"]["system_prompt"]


# --- generation settings ---------------------------------------------------
def test_creativity_is_the_temperature():
    """The UI slider is labelled Creativity; it is the sampling temperature."""
    fake = RecordingChat()
    core.enhance_prompt("a cat", "Text/Chat", 0.25, chat_fn=fake)
    assert fake.calls[-1]["temperature"] == 0.25


def test_fixed_generation_settings_are_passed_through():
    fake = RecordingChat()
    core.enhance_prompt("a cat", "Text/Chat", chat_fn=fake)
    assert fake.calls[-1]["max_tokens"] == core.MAX_TOKENS
    assert fake.calls[-1]["top_p"] == core.TOP_P


def test_user_message_carries_the_original_prompt():
    fake = RecordingChat()
    core.enhance_prompt("a cat on a windowsill", "Text/Chat", chat_fn=fake)
    assert "a cat on a windowsill" in fake.user


# --- the result ------------------------------------------------------------
def test_result_is_stripped_and_carries_matching_tips():
    result = core.enhance_prompt(
        "a cat", "Image Generation", chat_fn=RecordingChat("  spaced  ")
    )
    assert result.enhanced == "spaced"
    assert result.tips == core.tips_for("Image Generation")
    assert result.tips


def test_chat_failures_propagate():
    def boom(messages, *, max_tokens, temperature, top_p):
        raise RuntimeError("credits exhausted")

    with pytest.raises(RuntimeError, match="credits"):
        core.enhance_prompt("a cat", "Text/Chat", chat_fn=boom)


# --- lookups the UI calls directly ----------------------------------------
@pytest.mark.parametrize("prompt_type", list(core.PROMPT_TYPES))
def test_every_type_has_tips_and_a_runnable_example(prompt_type):
    assert core.tips_for(prompt_type).strip()
    example_in, example_out = core.example_for(prompt_type)
    assert example_in.strip() and example_out.strip()


def test_unknown_type_yields_empty_tips_rather_than_raising():
    assert core.tips_for("Interpretive Dance") == ""
