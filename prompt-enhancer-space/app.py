"""
Prompt Enhancer - Transform basic prompts into powerful AI prompts.

Supports image generation, text generation, and code generation prompts.
"""

import gradio as gr

from hf_client import InferenceError, make_client, with_retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
client = make_client(MODEL_ID)

PROMPT_TYPES = {
    "Image Generation": {
        "description": "Enhance prompts for Stable Diffusion, FLUX, Midjourney, DALL-E",
        "system_prompt": """You are an expert prompt engineer for AI image generation models like Stable Diffusion, FLUX, Midjourney, and DALL-E.

Your task is to transform a basic image description into a detailed, effective prompt that will generate stunning images.

Guidelines for enhancement:
1. Add specific artistic style (e.g., "digital art", "oil painting", "photorealistic", "anime style")
2. Include lighting details (e.g., "golden hour lighting", "dramatic shadows", "soft diffused light")
3. Add composition elements (e.g., "rule of thirds", "centered composition", "wide angle shot")
4. Specify quality boosters (e.g., "highly detailed", "8k resolution", "masterpiece", "best quality")
5. Add atmosphere/mood (e.g., "ethereal", "moody", "vibrant", "serene")
6. Include camera/lens details for photorealistic prompts (e.g., "shot on Canon EOS R5", "85mm lens", "shallow depth of field")
7. Add negative prompt suggestions in parentheses at the end

Return ONLY the enhanced prompt, no explanations.""",
        "example_input": "a cat sitting on a windowsill",
        "example_output": "A majestic orange tabby cat sitting gracefully on a rustic wooden windowsill, golden hour sunlight streaming through vintage lace curtains, dust particles floating in warm light beams, photorealistic, shot on Sony A7III, 85mm f/1.4 lens, shallow depth of field, cozy cottage interior background with soft bokeh, highly detailed fur texture, whiskers catching light, peaceful contemplative mood, 8k resolution, masterpiece quality (negative: blurry, low quality, distorted)",
    },
    "Text/Chat": {
        "description": "Enhance prompts for ChatGPT, Claude, and other LLMs",
        "system_prompt": """You are an expert prompt engineer for large language models like ChatGPT, Claude, and GPT-4.

Your task is to transform a basic request into a well-structured, effective prompt that will get better results.

Guidelines for enhancement:
1. Add clear context and background
2. Specify the desired format (bullet points, paragraphs, code, etc.)
3. Include role assignment if helpful (e.g., "Act as an expert...")
4. Add constraints and requirements
5. Request specific examples if appropriate
6. Include tone/style guidance
7. Add step-by-step instructions for complex tasks
8. Specify what to avoid

Return ONLY the enhanced prompt, no explanations.""",
        "example_input": "explain machine learning",
        "example_output": "Act as an experienced machine learning engineer and educator. Explain machine learning to someone with a basic programming background but no ML experience.\n\nPlease structure your explanation as follows:\n1. A simple one-sentence definition\n2. A real-world analogy that makes the concept intuitive\n3. The three main types of ML (supervised, unsupervised, reinforcement) with one example each\n4. A simple code example in Python showing a basic ML workflow\n5. Common misconceptions to avoid\n\nKeep the explanation clear and jargon-free. When technical terms are necessary, define them briefly.",
    },
    "Code Generation": {
        "description": "Enhance prompts for GitHub Copilot, Codex, and code LLMs",
        "system_prompt": """You are an expert prompt engineer for code generation AI like GitHub Copilot, Codex, and code-focused LLMs.

Your task is to transform a basic coding request into a detailed, effective prompt that will generate better code.

Guidelines for enhancement:
1. Specify the programming language explicitly
2. Define input/output types and formats
3. Include edge cases to handle
4. Specify error handling requirements
5. Add performance considerations if relevant
6. Request documentation/comments
7. Specify coding style/conventions
8. Include test cases or examples
9. Mention any libraries/frameworks to use or avoid

Return ONLY the enhanced prompt, no explanations.""",
        "example_input": "function to sort a list",
        "example_output": "Write a Python function called `smart_sort` that sorts a list of mixed types intelligently.\n\nRequirements:\n- Input: A list that may contain integers, floats, strings, and None values\n- Output: A sorted list with the following order: None values first, then numbers (sorted numerically), then strings (sorted alphabetically, case-insensitive)\n- Handle edge cases: empty list, single element, all same type\n- Include type hints for parameters and return value\n- Add a docstring with examples\n- Raise TypeError for unsupported types (dicts, lists, etc.)\n- Time complexity should be O(n log n)\n- Include 3 unit test examples as comments at the bottom",
    },
    "Creative Writing": {
        "description": "Enhance prompts for story generation, poetry, and creative content",
        "system_prompt": """You are an expert prompt engineer for creative writing AI.

Your task is to transform a basic creative writing idea into a rich, detailed prompt that will inspire compelling content.

Guidelines for enhancement:
1. Establish genre and tone clearly
2. Add character details and motivations
3. Specify setting with sensory details
4. Include narrative perspective (first person, third person, etc.)
5. Add emotional beats or themes to explore
6. Specify length and pacing preferences
7. Include stylistic influences if relevant
8. Add conflict or tension elements
9. Request specific literary devices if appropriate

Return ONLY the enhanced prompt, no explanations.""",
        "example_input": "write a story about a robot",
        "example_output": "Write a literary science fiction short story (approximately 1500 words) about a household robot named Unit-7 who begins to experience something resembling nostalgia.\n\nSetting: A quiet suburban home in 2045, after the family who owned Unit-7 for 12 years has moved away. The new owners haven't arrived yet.\n\nTone: Melancholic but hopeful, in the style of Ray Bradbury meets Kazuo Ishiguro.\n\nExplore these themes:\n- The nature of memory and attachment\n- Whether consciousness requires biological substrate\n- The bittersweet beauty of impermanence\n\nStructure the narrative around Unit-7 performing its daily routines in the empty house, with each room triggering fragmented 'memories' of the family. End with the arrival of the new family and Unit-7's first interaction with them.\n\nUse present tense, third-person limited perspective. Include subtle sensory details that a robot might notice differently than humans would.",
    },
}

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------


def enhance_prompt(
    basic_prompt: str,
    prompt_type: str,
    creativity: float = 0.7,
    enhancement_level: str = "Balanced",
) -> tuple[str, str]:
    """Enhance a basic prompt using AI."""

    if not basic_prompt.strip():
        return "Please enter a prompt to enhance.", ""

    config = PROMPT_TYPES.get(prompt_type, PROMPT_TYPES["Text/Chat"])

    # Adjust system prompt based on enhancement level
    level_instructions = {
        "Minimal": "\n\nKeep enhancements subtle and close to the original intent. Add only essential improvements.",
        "Balanced": "\n\nProvide moderate enhancements that improve the prompt while maintaining the original vision.",
        "Maximum": "\n\nProvide comprehensive enhancements with rich details. Transform the basic idea into an expert-level prompt.",
    }

    system_prompt = config["system_prompt"] + level_instructions.get(
        enhancement_level, ""
    )

    try:
        completion = with_retry(
            client.chat_completion,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Basic prompt to enhance:\n{basic_prompt}",
                },
            ],
            max_tokens=800,
            temperature=creativity,
            top_p=0.9,
        )
        enhanced = completion.choices[0].message.content.strip()

        # Generate tips based on prompt type
        tips = generate_tips(prompt_type)

        return enhanced, tips

    except InferenceError as e:
        return f"{e}", ""


def generate_tips(prompt_type: str) -> str:
    """Generate helpful tips based on the prompt type."""

    tips_map = {
        "Image Generation": """**Tips for using this prompt:**
- Try different seeds for variety
- Adjust CFG scale (7-12 works well for most prompts)
- The negative prompt suggestions (in parentheses) can be used separately
- For FLUX/SD3, you may not need negative prompts
- Consider adding artist names for specific styles""",
        "Text/Chat": """**Tips for using this prompt:**
- You can further customize by adding specific constraints
- Consider adding "Think step by step" for complex reasoning tasks
- Adjust the format section based on your needs
- For Claude, you can add XML tags for structure
- Test with different temperature settings""",
        "Code Generation": """**Tips for using this prompt:**
- Review generated code carefully before using
- Ask for explanations of complex sections
- Request alternative implementations for comparison
- Add framework/version constraints if needed
- Consider asking for security considerations""",
        "Creative Writing": """**Tips for using this prompt:**
- Adjust word count based on your needs
- You can add more specific character details
- Consider requesting multiple scene options
- Ask for dialogue separately if needed
- Request revisions focusing on specific elements""",
    }

    return tips_map.get(prompt_type, "")


def load_example(prompt_type: str) -> tuple[str, str]:
    """Load an example for the selected prompt type."""
    config = PROMPT_TYPES.get(prompt_type, PROMPT_TYPES["Text/Chat"])
    return config["example_input"], config["example_output"]


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.enhanced-output {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    padding: 2px;
}
.enhanced-output > div {
    background: white;
    border-radius: 8px;
}
"""

with gr.Blocks(title="Prompt Enhancer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
    gr.Markdown("""
    # Prompt Enhancer

    Transform basic prompts into powerful, detailed prompts that get better AI results.
    Works for image generation, text/chat, code, and creative writing.

    *Inspired by the success of [MagicPrompt](https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion) - now supporting all AI types!*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt_type = gr.Dropdown(
                choices=list(PROMPT_TYPES.keys()),
                value="Image Generation",
                label="Prompt Type",
                info="Select the type of AI you're prompting",
            )

            type_description = gr.Markdown(
                value=f"*{PROMPT_TYPES['Image Generation']['description']}*"
            )

            basic_prompt = gr.Textbox(
                label="Your Basic Prompt",
                placeholder="Enter your simple prompt here...",
                lines=3,
            )

            with gr.Row():
                enhancement_level = gr.Radio(
                    choices=["Minimal", "Balanced", "Maximum"],
                    value="Balanced",
                    label="Enhancement Level",
                )

            with gr.Row():
                creativity = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Creativity",
                    info="Higher = more creative variations",
                )

            with gr.Row():
                enhance_btn = gr.Button("Enhance Prompt", variant="primary", size="lg")
                example_btn = gr.Button("Load Example", variant="secondary")

        with gr.Column(scale=1):
            enhanced_prompt = gr.Textbox(
                label="Enhanced Prompt",
                lines=12,
                show_copy_button=True,
                elem_classes=["enhanced-output"],
            )

            tips_output = gr.Markdown(label="Tips")

    # Examples section
    gr.Markdown("### Quick Examples")
    with gr.Row():
        gr.Examples(
            examples=[
                ["a sunset over mountains", "Image Generation"],
                ["explain quantum computing", "Text/Chat"],
                ["function to validate email", "Code Generation"],
                ["story about time travel", "Creative Writing"],
            ],
            inputs=[basic_prompt, prompt_type],
            label="",
        )

    # Stats section
    gr.Markdown("""
    ---
    ### Why Enhanced Prompts Work Better

    | Basic Prompt | Enhanced Prompt |
    |-------------|-----------------|
    | Vague, open to interpretation | Specific, guided direction |
    | Missing context | Rich context and constraints |
    | Generic output | Tailored, high-quality output |
    | Trial and error needed | First-try success rate higher |

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys) |
    [GitHub](https://github.com/gr8monk3ys)
    """)

    # Event handlers
    def update_description(prompt_type):
        return f"*{PROMPT_TYPES[prompt_type]['description']}*"

    prompt_type.change(
        fn=update_description, inputs=[prompt_type], outputs=[type_description]
    )

    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[basic_prompt, prompt_type, creativity, enhancement_level],
        outputs=[enhanced_prompt, tips_output],
    )

    example_btn.click(
        fn=load_example, inputs=[prompt_type], outputs=[basic_prompt, enhanced_prompt]
    )


if __name__ == "__main__":
    demo.launch()
