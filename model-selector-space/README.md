---
title: Model Selector
emoji: 🎯
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.1
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Find the perfect HuggingFace model for your task
---

# Model Selector

Find the perfect HuggingFace model for your task. Answer a few simple questions and get personalized recommendations with ready-to-use code examples.

## Features

### 10 Task Categories
- **Text Generation** - Chatbots, content writing, code
- **Text Classification** - Sentiment, spam, topics
- **Question Answering** - Document QA, FAQs
- **Translation** - 200+ languages
- **Summarization** - Articles, documents
- **Image Classification** - Photos, medical images
- **Object Detection** - Detect objects in images
- **Image Generation** - Create images from text
- **Speech Recognition** - Audio to text
- **Embeddings** - Semantic search, RAG

### Smart Filtering
- Filter by model size (tiny to large)
- Prioritize by speed, quality, or popularity
- Get recommendations tailored to your use case

### Ready-to-Use Code
Every recommendation includes:
- Working Python code example
- Direct link to the model
- License information
- Size/speed tradeoffs

## How to Use

1. **Select your task** (e.g., Text Generation)
2. **Choose size preference** based on your hardware
3. **Set priority** (speed, quality, or popularity)
4. **Describe your use case** (optional)
5. **Get recommendations** with code examples!

## Example Output

For "Text Generation" with "Small" size preference:

| Rank | Model | Size | License |
|------|-------|------|---------|
| 1 | microsoft/phi-3-mini | 3.8B | MIT |
| 2 | Qwen/Qwen2.5-3B-Instruct | 3B | Apache |
| 3 | mistralai/Mistral-7B | 7B | Apache |

## Quick Reference

| Task | Typical Size | Best For |
|------|--------------|----------|
| Text Generation | 3B - 70B | Chatbots, content |
| Classification | 50M - 300M | Sentiment, spam |
| Embeddings | 20M - 100M | Search, RAG |
| Speech | 200M - 1.5B | Transcription |

## Why Use This Tool?

- **Save time** - Don't search through thousands of models
- **Avoid mistakes** - Get proven, popular models
- **Quick start** - Copy-paste code examples
- **Right-sized** - Match models to your hardware

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
