---
title: Optical Illusion Generator
emoji: 🌀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Create mesmerizing visual illusions with AI
---

# Optical Illusion Generator

Create mesmerizing visual illusions with AI! Generate spiral illusions, hidden images, impossible objects, and mind-bending patterns.

## Features

### 8 Illusion Types

| Type | Effect |
|------|--------|
| **Spiral Illusion** | Appears to rotate when you stare at the center |
| **Hidden Image** | Contains a 3D image revealed by relaxing your eyes |
| **Motion Illusion** | Patterns appear to move in peripheral vision |
| **Impossible Object** | M.C. Escher-style paradoxical geometry |
| **Color Afterimage** | Creates ghost image after staring |
| **Size Illusion** | Objects appear different sizes |
| **Ambiguous Figure** | Two images in one, reversible perception |
| **Geometric Pattern** | Hypnotic repeating patterns |

### 12 Subjects
- Mystical creatures (dragon, phoenix)
- Spiritual symbols (Buddha, mandala)
- Nature (tree, flowers, waves)
- Abstract (eye, geometric shapes, galaxy)

### 6 Color Schemes
- **Psychedelic** - Maximum trippy effect
- **Monochrome** - Classic high-contrast
- **Warm** - Sunset tones
- **Cool** - Ocean blues
- **Gold & Black** - Elegant luxury
- **Neon** - Vaporwave aesthetic

### 3 Intensity Levels
- **Subtle** - Gentle, easy on the eyes
- **Medium** - Clear optical effect
- **Intense** - Maximum mind-bending

## How to Use

1. **Choose Illusion Type** - Select the visual effect you want
2. **Pick a Subject** - What appears in the illusion
3. **Set Intensity** - How strong the effect should be
4. **Select Colors** - The color palette
5. **Generate** - Create your unique illusion!

## Viewing Tips

Each illusion type has specific viewing instructions:

- **Spiral**: Focus on center, let peripheral vision work
- **Hidden Image**: Cross eyes slightly or look "through" screen
- **Motion**: Don't focus on one spot, let eyes wander
- **Impossible**: Trace edges to see the paradox

## Examples

| Combination | Effect |
|-------------|--------|
| Spiral + Intense + Psychedelic | Maximum hypnotic rotation |
| Hidden Image + Gold & Black | Elegant reveal experience |
| Motion + Cool + Ocean Waves | Flowing water effect |
| Impossible + Monochrome + Stairs | Classic Escher tribute |

## Technical Details

- **Model**: FLUX.1-schnell
- **Output**: High-quality illusion art
- **Generation**: Real-time via HuggingFace Inference API

## Safety Note

Some optical illusions may cause mild visual effects like:
- Brief afterimages
- Perceived motion
- Slight disorientation

Take breaks if you experience discomfort.

## Configuration

This Space calls the **HuggingFace Inference API** for image generation (FLUX.1-schnell). It requires an `HF_TOKEN` secret — a token with inference access — set in **Space Settings → Secrets**. Because FLUX runs on a serverless image model, the token also needs available inference credits/quota; without it, generation will fail.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)

*Inspired by the viral success of IllusionDiffusion*
