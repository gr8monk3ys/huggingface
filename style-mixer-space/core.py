"""Style-mixing logic, independent of Gradio and of the Inference API.

The entry point is :func:`mix_styles`. It owns the sequence -- resolve the
subject, pick a blend band, build the prompt, generate -- and returns a
:class:`StyleMix` carrying the image and everything the UI needs to describe it.
See docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import random
from dataclasses import dataclass
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ART_STYLES = {
    "Van Gogh": "swirling brushstrokes, vibrant colors, post-impressionist, starry night style, thick impasto paint",
    "Picasso Cubism": "geometric shapes, fragmented forms, multiple perspectives, cubist, angular",
    "Monet Impressionism": "soft brushstrokes, light effects, water lilies style, dreamy, pastel colors",
    "Japanese Ukiyo-e": "flat colors, bold outlines, wave patterns, woodblock print style, Mount Fuji",
    "Art Deco": "geometric patterns, gold accents, 1920s glamour, symmetrical, elegant lines",
    "Cyberpunk": "neon lights, futuristic city, rain-slicked streets, holographic, dystopian",
    "Studio Ghibli": "anime style, whimsical, nature spirits, soft colors, Miyazaki inspired",
    "Baroque": "dramatic lighting, rich colors, ornate details, chiaroscuro, Caravaggio style",
    "Pop Art": "bold colors, comic book style, Ben-Day dots, Warhol inspired, high contrast",
    "Watercolor": "soft edges, transparent layers, flowing pigments, wet-on-wet technique",
    "Pixel Art": "8-bit style, retro gaming, blocky pixels, limited color palette, nostalgic",
    "Steampunk": "Victorian era, brass gears, clockwork, industrial, sepia tones",
    "Vaporwave": "80s aesthetic, pink and cyan, greek statues, sunset gradients, retro tech",
    "Gothic": "dark atmosphere, cathedral architecture, ravens, moonlight, dramatic shadows",
    "Minimalist": "clean lines, simple shapes, negative space, monochromatic, zen-like",
}

SUBJECTS = [
    "a majestic lion",
    "a serene mountain landscape",
    "a bustling city street",
    "a mysterious forest",
    "a peaceful ocean sunset",
    "a cozy coffee shop",
    "an ancient temple",
    "a futuristic spacecraft",
    "a beautiful garden",
    "a magical castle",
]


# The slider is continuous but the prompt weights are not: three discrete bands.
# The reported percentages describe the band actually used, so what the UI shows
# matches the weights that were really sent.
LOW_BAND = 0.3
HIGH_BAND = 0.7


class InputError(ValueError):
    """The caller supplied no subject."""


class GenerateImageFn(Protocol):
    """Generate an image from *prompt*. May raise; failures propagate."""

    def __call__(self, prompt: str) -> Any: ...


@dataclass(frozen=True)
class StyleMix:
    """A generated image plus the blend that produced it."""

    image: Any
    subject: str
    style1: str
    style2: str
    style1_description: str
    style2_description: str
    percent1: int
    percent2: int
    prompt: str


def _blend(style1: str, style2: str, ratio: float) -> tuple[str, str, int, int]:
    """Return (description, weight expression, percent1, percent2)."""
    desc1 = ART_STYLES.get(style1, style1)
    desc2 = ART_STYLES.get(style2, style2)

    if ratio <= LOW_BAND:
        return (
            f"primarily in {style1} style with subtle hints of {style2}",
            f"({desc1}:1.3), ({desc2}:0.5)",
            70,
            30,
        )
    if ratio >= HIGH_BAND:
        return (
            f"primarily in {style2} style with subtle hints of {style1}",
            f"({desc1}:0.5), ({desc2}:1.3)",
            30,
            70,
        )
    return (
        f"harmoniously blending {style1} and {style2} styles",
        f"({desc1}:1.0), ({desc2}:1.0)",
        50,
        50,
    )


def mix_styles(
    style1: str,
    style2: str,
    blend_ratio: float,
    subject: str,
    custom_subject: str = "",
    *,
    generate_image: GenerateImageFn,
) -> StyleMix:
    """Blend two art styles into one image of *subject*.

    *custom_subject* wins over *subject* when it holds anything.

    Raises:
        InputError: neither subject field holds anything.
        Exception: whatever *generate_image* raises.
    """
    final_subject = custom_subject.strip() if custom_subject.strip() else subject
    if not final_subject:
        raise InputError("Please select or enter a subject.")

    blend_desc, style_weight, pct1, pct2 = _blend(style1, style2, blend_ratio)

    prompt = f"""A stunning artistic rendering of {final_subject}, {blend_desc}.

Style fusion: {style_weight}

The artwork masterfully combines elements from both styles, creating a unique and visually striking piece.
High quality, detailed, professional artwork, museum quality."""

    return StyleMix(
        image=generate_image(prompt),
        subject=final_subject,
        style1=style1,
        style2=style2,
        style1_description=ART_STYLES.get(style1, style1),
        style2_description=ART_STYLES.get(style2, style2),
        percent1=pct1,
        percent2=pct2,
        prompt=prompt,
    )


def random_mix(rng: random.Random = random) -> tuple[str, str, float, str, str]:
    """Pick a random pair of distinct styles, a subject, and a mid-range ratio."""
    styles = list(ART_STYLES)
    style1 = rng.choice(styles)
    style2 = rng.choice([s for s in styles if s != style1])
    return style1, style2, rng.uniform(LOW_BAND, HIGH_BAND), rng.choice(SUBJECTS), ""
