---
title: Prompt Enhancer
emoji: ✨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Transform basic prompts into powerful AI prompts
---

# Prompt Enhancer

Transform your basic prompts into powerful, detailed prompts that get better AI results. Works for image generation, text/chat, code, and creative writing.

## Features

### Multi-Model Support
- **Image Generation** - Stable Diffusion, FLUX, Midjourney, DALL-E
- **Text/Chat** - ChatGPT, Claude, GPT-4, Llama
- **Code Generation** - GitHub Copilot, Codex, Code Llama
- **Creative Writing** - Story generation, poetry, scripts

### Smart Enhancement
- Adds specific details and context
- Includes style and quality modifiers
- Suggests constraints and requirements
- Provides format guidance

### Customizable
- **Enhancement Level** - Minimal, Balanced, or Maximum
- **Creativity Slider** - Control variation in outputs
- **Quick Examples** - Learn from effective prompts

## How It Works

1. **Select your prompt type** (Image, Text, Code, or Creative)
2. **Enter your basic prompt** (e.g., "a cat on a windowsill")
3. **Click Enhance** to transform it
4. **Copy the enhanced prompt** to use with your AI tool

## Example Transformations

### Image Generation
**Basic:** "a cat sitting on a windowsill"

**Enhanced:** "A majestic orange tabby cat sitting gracefully on a rustic wooden windowsill, golden hour sunlight streaming through vintage lace curtains, dust particles floating in warm light beams, photorealistic, shot on Sony A7III, 85mm f/1.4 lens, shallow depth of field, cozy cottage interior background with soft bokeh, highly detailed fur texture, 8k resolution, masterpiece quality"

### Code Generation
**Basic:** "function to sort a list"

**Enhanced:** "Write a Python function called `smart_sort` that sorts a list of mixed types intelligently. Requirements: Input accepts integers, floats, strings, and None values. Output ordering: None first, then numbers (numerically), then strings (alphabetically, case-insensitive). Include type hints, docstring with examples, and handle edge cases. Raise TypeError for unsupported types."

## Why Enhanced Prompts Work Better

| Aspect | Basic Prompt | Enhanced Prompt |
|--------|-------------|-----------------|
| Clarity | Vague | Specific |
| Context | Missing | Rich |
| Output | Generic | Tailored |
| Success Rate | Trial & error | First-try success |

## Technical Details

| Component | Technology |
|-----------|------------|
| Web Framework | Gradio 5.9.1 |
| AI Model | Mistral-7B via HuggingFace Inference API |
| Enhancement | Custom prompt engineering |

## Inspiration

Inspired by the success of [MagicPrompt-Stable-Diffusion](https://huggingface.co/spaces/Gustavosta/MagicPrompt-Stable-Diffusion) (2,000+ likes), extended to support all types of AI prompts.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
