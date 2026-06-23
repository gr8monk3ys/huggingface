"""
Style Mixer - Blend artistic styles using AI.
Create unique art by mixing different visual styles.
"""

import random

import gradio as gr

from hf_client import InferenceError, make_client, with_retry

# ---------------------------------------------------------------------------
# Style Definitions
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

# ---------------------------------------------------------------------------
# Initialize Client
# ---------------------------------------------------------------------------

IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
client = make_client()

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def mix_styles(style1: str, style2: str, blend_ratio: float, subject: str, custom_subject: str) -> tuple:
    """Generate an image blending two art styles."""

    # Use custom subject if provided
    final_subject = custom_subject.strip() if custom_subject.strip() else subject

    if not final_subject:
        return None, "Please select or enter a subject."

    # Get style descriptions
    style1_desc = ART_STYLES.get(style1, style1)
    style2_desc = ART_STYLES.get(style2, style2)

    # Create the blended prompt. Weighting is applied in three discrete bands,
    # so the displayed percentages reflect the band actually used rather than
    # the raw slider value (which would diverge from the real prompt weights).
    if blend_ratio <= 0.3:
        blend_desc = f"primarily in {style1} style with subtle hints of {style2}"
        style_weight = f"({style1_desc}:1.3), ({style2_desc}:0.5)"
        pct1, pct2 = 70, 30
    elif blend_ratio >= 0.7:
        blend_desc = f"primarily in {style2} style with subtle hints of {style1}"
        style_weight = f"({style1_desc}:0.5), ({style2_desc}:1.3)"
        pct1, pct2 = 30, 70
    else:
        blend_desc = f"harmoniously blending {style1} and {style2} styles"
        style_weight = f"({style1_desc}:1.0), ({style2_desc}:1.0)"
        pct1, pct2 = 50, 50

    # Construct the prompt
    prompt = f"""A stunning artistic rendering of {final_subject}, {blend_desc}.

Style fusion: {style_weight}

The artwork masterfully combines elements from both styles, creating a unique and visually striking piece.
High quality, detailed, professional artwork, museum quality."""

    try:
        image = with_retry(client.text_to_image, prompt, model=IMAGE_MODEL)
    except InferenceError as e:
        return None, str(e)

    description = f"""## Style Mix Complete!

**Subject:** {final_subject}

**Style Blend:**
- **{style1}** ({pct1}%): {style1_desc}
- **{style2}** ({pct2}%): {style2_desc}

**Prompt Used:**
```
{prompt}
```

*Try adjusting the blend ratio or mixing different styles!*
"""
    return image, description


def random_mix():
    """Generate random style combination."""
    styles = list(ART_STYLES.keys())
    style1 = random.choice(styles)
    style2 = random.choice([s for s in styles if s != style1])
    subject = random.choice(SUBJECTS)
    ratio = random.uniform(0.3, 0.7)
    return style1, style2, ratio, subject, ""


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

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
                generate_btn = gr.Button("🎨 Generate Art", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Artwork", type="pil")
            output_description = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[style1_dropdown, style2_dropdown, blend_slider, subject_dropdown, custom_subject],
        outputs=[output_image, output_description],
        fn=mix_styles,
        cache_examples=False,
    )

    # Event handlers
    generate_btn.click(
        fn=mix_styles,
        inputs=[style1_dropdown, style2_dropdown, blend_slider, subject_dropdown, custom_subject],
        outputs=[output_image, output_description],
    )

    random_btn.click(
        fn=random_mix,
        outputs=[style1_dropdown, style2_dropdown, blend_slider, subject_dropdown, custom_subject],
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
