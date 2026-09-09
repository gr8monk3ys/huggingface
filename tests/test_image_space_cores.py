"""Tests for the two image Spaces' core logic.

style-mixer and illusion-generator are structural twins: resolve a subject,
look up config, build a prompt, generate. They share a test module so the
divergences between them stay visible.
"""

import random

import pytest

from conftest import load_local_module

mixer = load_local_module("style_mixer_core", "style-mixer-space/core.py")
illusion = load_local_module("illusion_core", "illusion-generator-space/core.py")


class RecordingGenerator:
    """A generate_image that records prompts and returns a sentinel image."""

    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return f"<image {len(self.prompts)}>"


def boom(prompt):
    raise RuntimeError("credits exhausted")


# ===========================================================================
# Shared behaviour
# ===========================================================================
def mix(**kw):
    args = {
        "style1": "Impressionism",
        "style2": "Cubism",
        "blend_ratio": 0.5,
        "subject": "a cat",
        "custom_subject": "",
        "generate_image": RecordingGenerator(),
    }
    args.update(kw)
    return mixer.mix_styles(**args)


def illus(**kw):
    args = {
        "illusion_type": next(iter(illusion.ILLUSION_TYPES)),
        "subject": "a cat",
        "custom_subject": "",
        "intensity": "Medium",
        "color_scheme": "Neon",
        "generate_image": RecordingGenerator(),
    }
    args.update(kw)
    return illusion.generate_illusion(**args)


@pytest.mark.parametrize("call", [mix, illus])
def test_no_subject_raises_input_error(call):
    with pytest.raises((mixer.InputError, illusion.InputError), match="subject"):
        call(subject="", custom_subject="")


@pytest.mark.parametrize("call", [mix, illus])
def test_whitespace_custom_subject_falls_back_to_the_dropdown(call):
    assert call(subject="a fox", custom_subject="   ").subject == "a fox"


@pytest.mark.parametrize("call", [mix, illus])
def test_custom_subject_wins_and_is_stripped(call):
    assert call(subject="a fox", custom_subject="  a dragon  ").subject == "a dragon"


@pytest.mark.parametrize("call", [mix, illus])
def test_validation_happens_before_any_generation_call(call):
    gen = RecordingGenerator()
    with pytest.raises(ValueError):
        call(subject="", custom_subject="", generate_image=gen)
    assert gen.prompts == []


@pytest.mark.parametrize("call", [mix, illus])
def test_generation_failures_propagate(call):
    with pytest.raises(RuntimeError, match="credits"):
        call(generate_image=boom)


@pytest.mark.parametrize("call", [mix, illus])
def test_subject_reaches_the_prompt(call):
    gen = RecordingGenerator()
    call(subject="", custom_subject="a brass telescope", generate_image=gen)
    assert "a brass telescope" in gen.prompts[0]


# ===========================================================================
# style-mixer: the blend bands
# ===========================================================================
@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.0, (70, 30)),
        (0.3, (70, 30)),  # inclusive lower edge
        (0.5, (50, 50)),
        (0.7, (30, 70)),  # inclusive upper edge
        (1.0, (30, 70)),
    ],
)
def test_blend_ratio_maps_to_three_discrete_bands(ratio, expected):
    result = mix(blend_ratio=ratio)
    assert (result.percent1, result.percent2) == expected


def test_reported_percentages_match_the_weights_actually_sent():
    """The displayed split must describe the prompt that was really used."""
    gen = RecordingGenerator()
    result = mix(blend_ratio=0.1, generate_image=gen)
    assert (result.percent1, result.percent2) == (70, 30)
    assert ":1.3)" in gen.prompts[0] and ":0.5)" in gen.prompts[0]


def test_unknown_style_falls_back_to_its_own_name():
    """Unlike illusion types, a style with no description is still usable --
    the name itself goes into the prompt."""
    assert "Zdzislaw Beksinski" not in mixer.ART_STYLES
    result = mix(style1="Zdzislaw Beksinski")
    assert result.style1_description == "Zdzislaw Beksinski"


def test_mix_carries_the_prompt_it_used():
    gen = RecordingGenerator()
    result = mix(generate_image=gen)
    assert result.prompt == gen.prompts[0]


def test_random_mix_picks_two_distinct_styles():
    rng = random.Random(0)
    for _ in range(25):
        style1, style2, ratio, subject, custom = mixer.random_mix(rng)
        assert style1 != style2
        assert mixer.LOW_BAND <= ratio <= mixer.HIGH_BAND
        assert custom == ""


# ===========================================================================
# illusion-generator: config lookup
# ===========================================================================
def test_unknown_illusion_type_raises_rather_than_substituting():
    """There is no sensible illusion to fall back to, unlike intensity/colour."""
    with pytest.raises(illusion.UnknownIllusionError, match="Unknown illusion"):
        illus(illusion_type="Escher Staircase")


@pytest.mark.parametrize("field", ["intensity", "color_scheme"])
def test_unknown_dropdown_values_fall_back_to_defaults(field):
    """These come from dropdowns, so a bad value is a UI bug, not bad input."""
    result = illus(**{field: "Nonexistent"})
    assert result.image is not None


def test_every_illusion_type_produces_a_prompt_and_a_tip():
    for name in illusion.ILLUSION_TYPES:
        result = illus(illusion_type=name)
        assert result.technique
        assert result.tip
        assert result.prompt.strip()


def test_tip_for_unknown_type_is_the_default():
    assert illusion.tip_for("Escher Staircase") == illusion.DEFAULT_TIP


def test_intensity_and_colour_reach_the_prompt():
    gen = RecordingGenerator()
    illus(intensity="Intense", color_scheme="Monochrome", generate_image=gen)
    assert illusion.INTENSITY_DESCRIPTIONS["Intense"] in gen.prompts[0]
    assert illusion.COLOR_SCHEMES["Monochrome"] in gen.prompts[0]


def test_random_illusion_stays_within_the_configured_choices():
    rng = random.Random(0)
    for _ in range(25):
        kind, subject, custom, intensity, colour = illusion.random_illusion(rng)
        assert kind in illusion.ILLUSION_TYPES
        assert subject in illusion.SUBJECTS
        assert intensity in illusion.INTENSITY_LEVELS
        assert colour in illusion.COLOR_SCHEMES
        assert custom == ""
