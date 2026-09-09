"""Optical Illusion Generator -- a Gradio front end over :mod:`core`.

Owns the UI and the adapters: builds the Inference client, wraps it as the
``generate_image`` core expects, and renders the returned Illusion.
"""

import gradio as gr

from core import (
    ILLUSION_TYPES,
    INTENSITY_LEVELS,
    SUBJECTS,
    Illusion,
    InputError,
    UnknownIllusionError,
    generate_illusion,
    random_illusion,
)
from hf_client import InferenceError, make_client, with_retry

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
client = make_client()


def _generate_image(prompt: str):
    """Call the Inference API, retrying transient failures."""
    return with_retry(client.text_to_image, prompt, model=IMAGE_MODEL)


def _render(illusion: Illusion) -> str:
    """Format an Illusion as the Markdown shown beside the image."""
    return f"""## {illusion.illusion_type} Generated!

**Subject:** {illusion.subject}
**Intensity:** {illusion.intensity}
**Color Scheme:** {illusion.color_scheme}

### How to Experience This Illusion

{illusion.technique}

### Tips for Best Effect

- View on a large screen if possible
- Adjust your distance from the screen
- Try looking at different parts of the image
- {illusion.tip}

---

*Generated with FLUX.1-schnell*
"""


def handle_generate(
    illusion_type, subject, custom_subject, intensity, color_scheme
) -> tuple:
    """Gradio handler: generate the illusion, or report why not."""
    try:
        illusion = generate_illusion(
            illusion_type,
            subject,
            custom_subject,
            intensity,
            color_scheme,
            generate_image=_generate_image,
        )
    except (InputError, UnknownIllusionError, InferenceError) as exc:
        return None, str(exc)

    return illusion.image, _render(illusion)


EXAMPLES = [
    ["Spiral Illusion", "a mystical tree", "", "Intense", "Psychedelic"],
    ["Impossible Object", "infinite stairs", "", "Medium", "Monochrome"],
    ["Hidden Image", "a hidden face", "", "Intense", "Gold & Black"],
    ["Motion Illusion", "ocean waves", "", "Intense", "Cool"],
    ["Geometric Pattern", "sacred mandala", "", "Medium", "Neon"],
]

with gr.Blocks(title="Optical Illusion Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Optical Illusion Generator

    **Create mesmerizing visual illusions with AI!**

    Generate spiral illusions, hidden images, impossible objects, and more.
    Each illusion type creates a different visual experience.

    *Warning: Some illusions may cause mild visual effects. View responsibly.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            illusion_dropdown = gr.Dropdown(
                choices=list(ILLUSION_TYPES.keys()),
                value="Spiral Illusion",
                label="Illusion Type",
            )

            subject_dropdown = gr.Dropdown(
                choices=SUBJECTS,
                value="a mystical tree",
                label="Subject",
            )

            custom_subject = gr.Textbox(
                label="Custom Subject (optional)",
                placeholder="Or describe your own subject...",
            )

            intensity_dropdown = gr.Dropdown(
                choices=INTENSITY_LEVELS,
                value="Medium",
                label="Effect Intensity",
            )

            color_dropdown = gr.Dropdown(
                choices=[
                    "Psychedelic",
                    "Monochrome",
                    "Warm",
                    "Cool",
                    "Gold & Black",
                    "Neon",
                ],
                value="Psychedelic",
                label="Color Scheme",
            )

            with gr.Row():
                random_btn = gr.Button("🎲 Random", variant="secondary")
                generate_btn = gr.Button(
                    "✨ Generate Illusion", variant="primary", size="lg"
                )

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Illusion", type="pil")
            output_description = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            illusion_dropdown,
            subject_dropdown,
            custom_subject,
            intensity_dropdown,
            color_dropdown,
        ],
        outputs=[output_image, output_description],
        fn=handle_generate,
        cache_examples=False,
    )

    # Event handlers
    generate_btn.click(
        fn=handle_generate,
        inputs=[
            illusion_dropdown,
            subject_dropdown,
            custom_subject,
            intensity_dropdown,
            color_dropdown,
        ],
        outputs=[output_image, output_description],
    )

    random_btn.click(
        fn=random_illusion,
        outputs=[
            illusion_dropdown,
            subject_dropdown,
            custom_subject,
            intensity_dropdown,
            color_dropdown,
        ],
    )

    gr.Markdown("""
    ---

    ## Illusion Types Explained

    | Type | Effect | Best For |
    |------|--------|----------|
    | **Spiral Illusion** | Appears to rotate | Hypnotic, meditative |
    | **Hidden Image** | 3D image within pattern | Puzzles, discoveries |
    | **Motion Illusion** | Appears to move | Dynamic displays |
    | **Impossible Object** | Paradoxical geometry | Mind-bending art |
    | **Color Afterimage** | Ghost image after staring | Science demonstrations |
    | **Size Illusion** | Distorted size perception | Perspective tricks |
    | **Ambiguous Figure** | Two images in one | Duality, hidden meanings |
    | **Geometric Pattern** | Depth through repetition | Decorative, hypnotic |

    ---

    ## Color Schemes

    - **Psychedelic** - Vibrant rainbow neon colors
    - **Monochrome** - Classic black and white
    - **Warm** - Sunset reds, oranges, yellows
    - **Cool** - Ocean blues and greens
    - **Gold & Black** - Elegant and luxurious
    - **Neon** - Vaporwave pink and cyan

    ---

    **Pro Tips:**
    - Intense illusions work best on larger screens
    - Monochrome often creates the strongest optical effects
    - Spiral + Psychedelic = Maximum trippy effect

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)

if __name__ == "__main__":
    demo.launch()
