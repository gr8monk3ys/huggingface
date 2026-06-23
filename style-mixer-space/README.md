---
title: Style Mixer
emoji: 🎨
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Blend two art styles into unique masterpieces
---

# Style Mixer

Blend two artistic styles into one unique masterpiece using AI!

## Features

### 15 Art Styles
- **Van Gogh** - Swirling post-impressionist brushstrokes
- **Picasso Cubism** - Geometric, fragmented forms
- **Monet Impressionism** - Soft, dreamy light effects
- **Japanese Ukiyo-e** - Bold woodblock print style
- **Cyberpunk** - Neon futuristic dystopia
- **Studio Ghibli** - Whimsical Miyazaki anime
- **Vaporwave** - 80s retro aesthetic
- **Gothic** - Dark, dramatic atmosphere
- **Steampunk** - Victorian clockwork fantasy
- **And more...**

### Smart Blending
- Adjust the blend ratio between styles
- Emphasize one style over another
- Create unique fusion aesthetics

### 10 Pre-set Subjects
- Majestic lion, mountain landscape, city street
- Mysterious forest, ocean sunset, coffee shop
- Ancient temple, spacecraft, garden, castle
- Or enter your own custom subject!

## How It Works

1. **Select Style 1** - Your primary artistic influence
2. **Select Style 2** - The style to blend with
3. **Adjust Blend Ratio** - Control the mix (50/50 or weighted)
4. **Choose Subject** - What to depict
5. **Generate** - Watch the AI create your unique artwork!

## Example Combinations

| Style 1 | Style 2 | Result |
|---------|---------|--------|
| Van Gogh | Cyberpunk | Swirling neon cityscapes |
| Ukiyo-e | Vaporwave | Retro-futuristic waves |
| Ghibli | Watercolor | Dreamy animated scenes |
| Art Deco | Steampunk | Elegant clockwork glamour |

## Technical Details

- **Model**: FLUX.1-schnell (fast, high-quality generation) via the HuggingFace Inference API
- **Style Weighting**: Dynamic prompt weighting based on blend ratio
- **Resolution**: High-quality (~1024px) output (the model's default; no custom width/height is set)

## Tips

- Contrasting styles often produce the most interesting results
- Try pairing old (Baroque) with new (Cyberpunk)
- Use the random button to discover unexpected combinations
- Custom subjects allow for personalized artwork

## Configuration

This Space calls the **HuggingFace Inference API** for image generation (FLUX.1-schnell). It requires an `HF_TOKEN` secret — a token with inference access — set in **Space Settings → Secrets**. Because FLUX runs on a serverless image model, the token also needs available inference credits/quota; without it, generation will fail.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
