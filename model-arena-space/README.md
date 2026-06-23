---
title: AI Model Arena
emoji: ⚔️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Compare AI models head-to-head and vote for the best
---

# AI Model Arena

Compare AI models head-to-head! Test the same prompt across different models and vote for the best response.

## Features

### 6 Top Open Models
- **Mistral-7B** - Fast and efficient, great at reasoning and code
- **Llama-3.1-8B** - Meta's latest with strong general capabilities
- **Qwen2.5-7B** - Excellent at multilingual tasks, math, and coding
- **Phi-3-mini** - Microsoft's compact powerhouse
- **Gemma-2-9B** - Google's quality-focused instruction model
- **Zephyr-7B** - Aligned for helpfulness and safety

### Battle System
- Run any two models against each other
- See response times for each model
- Vote for the better response
- Track wins on the leaderboard

### 5 Test Categories
- **Creative Writing** - Poetry, stories, creative prompts
- **Coding** - Programming challenges and algorithms
- **Reasoning** - Logic puzzles and math problems
- **Knowledge** - Explanations and factual queries
- **Summarization** - Condensing complex topics

## How to Use

1. **Enter a prompt** or use an example from a category
2. **Select two models** to compare
3. **Click "Start Battle"** to generate responses
4. **Read both responses** and compare quality, accuracy, and style
5. **Vote** for the better response
6. **Check the leaderboard** to see which models are winning!

## Example Battles

| Category | Sample Prompt |
|----------|--------------|
| Creative | Write a haiku about AI |
| Coding | Implement a prime number checker |
| Reasoning | Solve: Bat + Ball = $1.10, Bat costs $1 more... |
| Knowledge | Explain quantum entanglement simply |

## Why This Matters

Different models have different strengths:
- Some are faster, some more accurate
- Some excel at code, others at creative tasks
- Testing helps you choose the right model for your needs

## Technical Details

- All models accessed via HuggingFace Inference API
- Response times measured for comparison
- Leaderboard persists during Space session

## Configuration

This Space calls the **HuggingFace Inference API** to run every model. It requires an `HF_TOKEN` secret — a token with inference access — set in **Space Settings → Secrets**. Without it, battles will fail.

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
