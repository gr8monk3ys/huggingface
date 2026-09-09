"""Style Mixer -- a Gradio front end over :mod:`core`.

Owns the UI and the adapters: builds the Inference client, wraps it as the
``generate_image`` core expects, and renders the returned StyleMix.
"""

import gradio as gr

from core import (
    ART_STYLES,
    SUBJECTS,
    InputError,
    StyleMix,
    mix_styles,
    random_mix,
)
from hf_client import InferenceError, make_client, with_retry

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
client = make_client()


def _generate_image(prompt: str):
    """Call the Inference API, retrying transient failures."""
    return with_retry(client.text_to_image, prompt, model=IMAGE_MODEL)


def _render(mix: StyleMix) -> str:
    """Format a StyleMix as the Markdown shown beside the image."""
    return f"""## Style Mix Complete!

**Subject:** {mix.subject}

**Style Blend:**
- **{mix.style1}** ({mix.percent1}%): {mix.style1_description}
- **{mix.style2}** ({mix.percent2}%): {mix.style2_description}

**Prompt Used:**
```
{mix.prompt}
```

*Try adjusting the blend ratio or mixing different styles!*
"""


def handle_mix(style1, style2, blend_ratio, subject, custom_subject) -> tuple:
    """Gradio handler: generate the blend, or report why not."""
    try:
        mix = mix_styles(
            style1,
            style2,
            blend_ratio,
            subject,
            custom_subject,
            generate_image=_generate_image,
        )
    except (InputError, InferenceError) as exc:
        return None, str(exc)

    return mix.image, _render(mix)


EXAMPLES = [
    ["Van Gogh", "Cyberpunk", 0.5, "a bustling city street", ""],
    ["Japanese Ukiyo-e", "Vaporwave", 0.6, "a majestic lion", ""],
    ["Studio Ghibli", "Watercolor", 0.4, "a magical castle", ""],
    ["Art Deco", "Steampunk", 0.5, "a futuristic spacecraft", ""],
    ["Monet Impressionism", "Pixel Art", 0.3, "a beautiful garden", ""],
]

with gr.Blocks(title="Style Mixer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Style Mixer

    **Blend two art styles into one unique masterpiece!**

    Select two artistic styles, adjust the blend ratio, and watch as AI creates
    a fusion of both aesthetics. Perfect for creating unique art or exploring
    how different artistic movements might combine.

    *Powered by FLUX.1-schnell*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            style1_dropdown = gr.Dropdown(
                choices=list(ART_STYLES.keys()),
                value="Van Gogh",
                label="Style 1",
            )

            style2_dropdown = gr.Dropdown(
                choices=list(ART_STYLES.keys()),
                value="Cyberpunk",
                label="Style 2",
            )

            blend_slider = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.5,
                step=0.1,
                label="Blend Ratio (← Style 1 | Style 2 →)",
            )

            subject_dropdown = gr.Dropdown(
                choices=SUBJECTS,
                value="a bustling city street",
                label="Subject",
            )

            custom_subject = gr.Textbox(
                label="Custom Subject (optional)",
                placeholder="Or describe your own subject...",
            )

            with gr.Row():
                random_btn = gr.Button("🎲 Random Mix", variant="secondary")
                generate_btn = gr.Button(
                    "🎨 Generate Art", variant="primary", size="lg"
                )

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Artwork", type="pil")
            output_description = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            style1_dropdown,
            style2_dropdown,
            blend_slider,
            subject_dropdown,
            custom_subject,
        ],
        outputs=[output_image, output_description],
        fn=handle_mix,
        cache_examples=False,
    )

    # Event handlers
    generate_btn.click(
        fn=handle_mix,
        inputs=[
            style1_dropdown,
            style2_dropdown,
            blend_slider,
            subject_dropdown,
            custom_subject,
        ],
        outputs=[output_image, output_description],
    )

    random_btn.click(
        fn=random_mix,
        outputs=[
            style1_dropdown,
            style2_dropdown,
            blend_slider,
            subject_dropdown,
            custom_subject,
        ],
    )

    gr.Markdown("""
    ---

    ## Available Styles

    | Style | Description |
    |-------|-------------|
    | Van Gogh | Swirling brushstrokes, vibrant post-impressionism |
    | Picasso Cubism | Geometric, fragmented, multiple perspectives |
    | Monet Impressionism | Soft, dreamy, light effects |
    | Japanese Ukiyo-e | Flat colors, bold outlines, woodblock style |
    | Cyberpunk | Neon, futuristic, dystopian |
    | Studio Ghibli | Whimsical anime, nature spirits |
    | Vaporwave | 80s aesthetic, pink/cyan, retro |
    | And more... | Gothic, Steampunk, Pixel Art, etc. |

    ---

    **Tips:**
    - Try contrasting styles (e.g., Classical + Cyberpunk) for dramatic results
    - Adjust the blend ratio to emphasize one style over another
    - Use custom subjects for personalized artwork

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)

if __name__ == "__main__":
    demo.launch()
