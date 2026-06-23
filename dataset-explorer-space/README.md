---
title: Dataset Explorer
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.31.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
short_description: Explore any HuggingFace dataset visually
---

# Dataset Explorer

Explore any HuggingFace dataset instantly with statistics, visualizations, and sample data.

## Features

### Instant Exploration
- Enter any dataset ID from the HuggingFace Hub
- See statistics, distributions, and samples immediately
- Works with streaming for large datasets

### Visual Statistics
- Column types and null percentages
- Value distributions for numeric columns
- Category counts for text columns
- Automatic visualization generation

### Quick Access
Popular datasets available with one click:
- `imdb` - Movie reviews sentiment
- `squad` - Question answering
- `ag_news` - News classification
- `emotion` - Emotion detection
- And more!

## How to Use

1. **Enter a dataset ID** (e.g., `imdb`, `squad`, `username/dataset-name`)
2. **Optionally specify config** for multi-config datasets
3. **Select split** (train/test/validation)
4. **Click Explore** to analyze

## Example Datasets

| Dataset | Description | Configs |
|---------|-------------|---------|
| `imdb` | Movie reviews | None |
| `squad` | Question answering | None |
| `glue` | NLU benchmark | mrpc, sst2, cola, etc. |
| `wikitext` | Language modeling | wikitext-2-raw-v1, etc. |
| `emotion` | Emotion classification | None |

## Technical Details

| Component | Technology |
|-----------|------------|
| Web Framework | Gradio 5.31.0 |
| Data Loading | HuggingFace Datasets |
| Visualization | Matplotlib |
| Data Processing | Pandas |

## Use Cases

- **Quick dataset preview** before downloading
- **Understanding data structure** for new projects
- **Comparing datasets** for your use case
- **Teaching** data science concepts

## License

MIT

## Author

Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
