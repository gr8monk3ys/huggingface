"""Optical-illusion prompt logic, independent of Gradio and of the Inference API.

The entry point is :func:`generate_illusion`. It owns the sequence -- resolve the
subject, look up the illusion config, build the prompt, generate -- and returns
an :class:`Illusion` carrying the image and the viewing guidance the UI shows.
See docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import random
from dataclasses import dataclass
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ILLUSION_TYPES = {
    "Spiral Illusion": {
        "pattern": "mesmerizing spiral pattern, hypnotic swirl, concentric circles morphing into spiral",
        "technique": "The image appears to rotate when you stare at the center",
        "prompt_prefix": "optical illusion spiral pattern containing",
    },
    "Hidden Image": {
        "pattern": "stereogram style, hidden 3D image within pattern, magic eye illusion",
        "technique": "Relax your eyes and look 'through' the image to reveal the hidden object",
        "prompt_prefix": "magic eye stereogram pattern hiding",
    },
    "Motion Illusion": {
        "pattern": "peripheral drift illusion, asymmetric patterns that appear to move, pulsating design",
        "technique": "The pattern appears to move in your peripheral vision",
        "prompt_prefix": "motion illusion pattern featuring",
    },
    "Impossible Object": {
        "pattern": "M.C. Escher style, impossible geometry, paradoxical architecture, infinite stairs",
        "technique": "Objects that could not exist in 3D reality",
        "prompt_prefix": "impossible object in the style of M.C. Escher showing",
    },
    "Color Afterimage": {
        "pattern": "high contrast complementary colors, stare and look away effect, negative afterimage",
        "technique": "Stare at the center for 30 seconds, then look at a white surface",
        "prompt_prefix": "high contrast optical illusion for afterimage effect featuring",
    },
    "Size Illusion": {
        "pattern": "forced perspective, Ebbinghaus illusion, relative size comparison",
        "technique": "Objects appear different sizes due to their surroundings",
        "prompt_prefix": "size perception illusion demonstrating",
    },
    "Ambiguous Figure": {
        "pattern": "Rubin vase style, duck-rabbit illusion, two images in one, reversible figure",
        "technique": "You can see two different images depending on your focus",
        "prompt_prefix": "ambiguous figure optical illusion combining",
    },
    "Geometric Pattern": {
        "pattern": "repeating geometric shapes, tessellation, sacred geometry, fractal-like pattern",
        "technique": "Patterns create depth and movement through repetition",
        "prompt_prefix": "geometric optical illusion pattern with",
    },
}

SUBJECTS = [
    "a majestic dragon",
    "a peaceful Buddha",
    "a cosmic galaxy",
    "a blooming flower",
    "a mysterious eye",
    "a sacred mandala",
    "a dancing figure",
    "a mythical phoenix",
    "ocean waves",
    "a mystical tree",
    "geometric shapes",
    "a hidden face",
]

INTENSITY_LEVELS = ["Subtle", "Medium", "Intense"]


INTENSITY_DESCRIPTIONS = {
    "Subtle": "subtle, gentle",
    "Medium": "clear, pronounced",
    "Intense": "extreme, overwhelming, very strong",
}

COLOR_SCHEMES = {
    "Psychedelic": "vibrant rainbow colors, neon, psychedelic",
    "Monochrome": "black and white, grayscale, high contrast",
    "Warm": "warm colors, reds, oranges, yellows, sunset tones",
    "Cool": "cool colors, blues, greens, purples, ocean tones",
    "Gold & Black": "gold and black, luxurious, elegant",
    "Neon": "neon pink and cyan, vaporwave, glowing",
}

VIEWING_TIPS = {
    "Spiral Illusion": "Focus on the exact center and let your peripheral vision do the work",
    "Hidden Image": "Try crossing your eyes slightly or looking 'through' the screen",
    "Motion Illusion": "Don't focus on one spot - let your eyes wander naturally",
    "Impossible Object": "Trace the edges with your eyes to see the paradox",
    "Color Afterimage": "After staring, look at a white wall for the afterimage effect",
    "Size Illusion": "Compare similar objects in different parts of the image",
    "Ambiguous Figure": "Blink or shift your focus to switch between interpretations",
    "Geometric Pattern": "Look at the center, then slowly move your gaze outward",
}

DEFAULT_TIP = "Experiment with different viewing distances"
DEFAULT_INTENSITY = "Medium"
DEFAULT_COLOR_SCHEME = "Psychedelic"

RANDOM_COLOR_SCHEMES = ("Psychedelic", "Monochrome", "Neon", "Gold & Black")


class InputError(ValueError):
    """The caller supplied no subject."""


class UnknownIllusionError(ValueError):
    """The requested illusion type has no configuration."""


class GenerateImageFn(Protocol):
    """Generate an image from *prompt*. May raise; failures propagate."""

    def __call__(self, prompt: str) -> Any: ...


@dataclass(frozen=True)
class Illusion:
    """A generated illusion plus how to look at it."""

    image: Any
    illusion_type: str
    subject: str
    intensity: str
    color_scheme: str
    technique: str
    tip: str
    prompt: str


def tip_for(illusion_type: str) -> str:
    """Viewing advice specific to *illusion_type*."""
    return VIEWING_TIPS.get(illusion_type, DEFAULT_TIP)


def generate_illusion(
    illusion_type: str,
    subject: str,
    custom_subject: str = "",
    intensity: str = DEFAULT_INTENSITY,
    color_scheme: str = DEFAULT_COLOR_SCHEME,
    *,
    generate_image: GenerateImageFn,
) -> Illusion:
    """Generate an optical illusion of *subject*.

    *custom_subject* wins over *subject* when it holds anything. Unrecognised
    *intensity* and *color_scheme* fall back to defaults, since both come from
    dropdowns -- but an unrecognised *illusion_type* raises, because there is no
    sensible illusion to substitute.

    Raises:
        InputError: neither subject field holds anything.
        UnknownIllusionError: *illusion_type* is not configured.
        Exception: whatever *generate_image* raises.
    """
    final_subject = custom_subject.strip() if custom_subject.strip() else subject
    if not final_subject:
        raise InputError("Please select or enter a subject.")

    config = ILLUSION_TYPES.get(illusion_type)
    if config is None:
        raise UnknownIllusionError(f"Unknown illusion type: {illusion_type}")

    intensity_desc = INTENSITY_DESCRIPTIONS.get(
        intensity, INTENSITY_DESCRIPTIONS[DEFAULT_INTENSITY]
    )
    color_desc = COLOR_SCHEMES.get(color_scheme, COLOR_SCHEMES[DEFAULT_COLOR_SCHEME])

    prompt = f"""{config["prompt_prefix"]} {final_subject}.

Style: {config["pattern"]}
Effect intensity: {intensity_desc}
Color scheme: {color_desc}

This is a {intensity.lower()} optical illusion that creates the visual effect of {config["technique"].lower()}.
Highly detailed, mesmerizing, hypnotic, professional quality optical illusion art.
The illusion effect should be clearly visible and striking."""

    return Illusion(
        image=generate_image(prompt),
        illusion_type=illusion_type,
        subject=final_subject,
        intensity=intensity,
        color_scheme=color_scheme,
        technique=config["technique"],
        tip=tip_for(illusion_type),
        prompt=prompt,
    )


def random_illusion(rng: random.Random = random) -> tuple[str, str, str, str, str]:
    """Pick random illusion settings for the "surprise me" button."""
    return (
        rng.choice(list(ILLUSION_TYPES)),
        rng.choice(SUBJECTS),
        "",
        rng.choice(INTENSITY_LEVELS),
        rng.choice(list(RANDOM_COLOR_SCHEMES)),
    )
