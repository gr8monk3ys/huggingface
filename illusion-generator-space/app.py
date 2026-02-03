"""
Optical Illusion Generator - Create mesmerizing visual illusions with AI.
Generate spiral illusions, hidden images, and mind-bending patterns.
"""

import gradio as gr
from huggingface_hub import InferenceClient
import random

# ---------------------------------------------------------------------------
# Illusion Types and Patterns
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

INTENSITY_LEVELS = {
    "Subtle": 0.3,
    "Medium": 0.6,
    "Intense": 0.9,
}

# ---------------------------------------------------------------------------
# Initialize Client
# ---------------------------------------------------------------------------

client = InferenceClient()

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def generate_illusion(
    illusion_type: str,
    subject: str,
    custom_subject: str,
    intensity: str,
    color_scheme: str,
) -> tuple:
    """Generate an optical illusion image."""

    # Use custom subject if provided
    final_subject = custom_subject.strip() if custom_subject.strip() else subject

    if not final_subject:
        return None, "Please select or enter a subject."

    illusion_config = ILLUSION_TYPES[illusion_type]

    # Build the prompt
    intensity_desc = {
        "Subtle": "subtle, gentle",
        "Medium": "clear, pronounced",
        "Intense": "extreme, overwhelming, very strong",
    }

    color_desc = {
        "Psychedelic": "vibrant rainbow colors, neon, psychedelic",
        "Monochrome": "black and white, grayscale, high contrast",
        "Warm": "warm colors, reds, oranges, yellows, sunset tones",
        "Cool": "cool colors, blues, greens, purples, ocean tones",
        "Gold & Black": "gold and black, luxurious, elegant",
        "Neon": "neon pink and cyan, vaporwave, glowing",
    }

    prompt = f"""{illusion_config['prompt_prefix']} {final_subject}.

Style: {illusion_config['pattern']}
Effect intensity: {intensity_desc[intensity]}
Color scheme: {color_desc[color_scheme]}

This is a {intensity.lower()} optical illusion that creates the visual effect of {illusion_config['technique'].lower()}.
Highly detailed, mesmerizing, hypnotic, professional quality optical illusion art.
The illusion effect should be clearly visible and striking."""

    try:
        # Generate image using FLUX
        image = client.text_to_image(
            prompt,
            model="black-forest-labs/FLUX.1-schnell",
        )

        # Create description
        description = f"""## {illusion_type} Generated!

**Subject:** {final_subject}
**Intensity:** {intensity}
**Color Scheme:** {color_scheme}

### How to Experience This Illusion

{illusion_config['technique']}

### Tips for Best Effect

- View on a large screen if possible
- Adjust your distance from the screen
- Try looking at different parts of the image
- {get_specific_tip(illusion_type)}

---

*Generated with FLUX.1-schnell*
"""
        return image, description

    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return None, "Rate limited. Please wait a moment and try again."
        return None, f"Generation failed: {error_msg}"


def get_specific_tip(illusion_type: str) -> str:
    """Get illusion-specific viewing tips."""
    tips = {
        "Spiral Illusion": "Focus on the exact center and let your peripheral vision do the work",
        "Hidden Image": "Try crossing your eyes slightly or looking 'through' the screen",
        "Motion Illusion": "Don't focus on one spot - let your eyes wander naturally",
        "Impossible Object": "Trace the edges with your eyes to see the paradox",
        "Color Afterimage": "After staring, look at a white wall for the afterimage effect",
        "Size Illusion": "Compare similar objects in different parts of the image",
        "Ambiguous Figure": "Blink or shift your focus to switch between interpretations",
        "Geometric Pattern": "Look at the center, then slowly move your gaze outward",
    }
    return tips.get(illusion_type, "Experiment with different viewing distances")


def random_illusion():
    """Generate random illusion settings."""
    illusion_type = random.choice(list(ILLUSION_TYPES.keys()))
    subject = random.choice(SUBJECTS)
    intensity = random.choice(list(INTENSITY_LEVELS.keys()))
    color = random.choice(["Psychedelic", "Monochrome", "Neon", "Gold & Black"])
    return illusion_type, subject, "", intensity, color


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

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
                choices=list(INTENSITY_LEVELS.keys()),
                value="Medium",
                label="Effect Intensity",
            )

            color_dropdown = gr.Dropdown(
                choices=["Psychedelic", "Monochrome", "Warm", "Cool", "Gold & Black", "Neon"],
                value="Psychedelic",
                label="Color Scheme",
            )

            with gr.Row():
                random_btn = gr.Button("🎲 Random", variant="secondary")
                generate_btn = gr.Button("✨ Generate Illusion", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Illusion", type="pil")
            output_description = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=[illusion_dropdown, subject_dropdown, custom_subject, intensity_dropdown, color_dropdown],
        outputs=[output_image, output_description],
        fn=generate_illusion,
        cache_examples=False,
    )

    # Event handlers
    generate_btn.click(
        fn=generate_illusion,
        inputs=[illusion_dropdown, subject_dropdown, custom_subject, intensity_dropdown, color_dropdown],
        outputs=[output_image, output_description],
    )

    random_btn.click(
        fn=random_illusion,
        outputs=[illusion_dropdown, subject_dropdown, custom_subject, intensity_dropdown, color_dropdown],
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
