"""Pure logic for the model selector (no Gradio import) so it is unit-testable.

The live Hub query is isolated in :func:`fetch_live_models` with a lazy
``huggingface_hub`` import and an injectable ``lister`` for testing, and it
fails soft (returns ``None``) so the UI can fall back to a curated list.
"""

from __future__ import annotations

import re
from typing import Optional

SIZE_PREFERENCES = {
    "Tiny (< 100M)": {"min": 0, "max": 100},
    "Small (100M - 500M)": {"min": 100, "max": 500},
    "Medium (500M - 2B)": {"min": 500, "max": 2000},
    "Large (2B - 10B)": {"min": 2000, "max": 10000},
    "Any size": {"min": 0, "max": 100000},
}


def parse_size(size_str: str) -> float:
    """Parse a parameter-count string like '7B', '67M', '1.5B' to millions.

    Returns 0.0 for unparseable input rather than guessing.
    """
    if not size_str:
        return 0.0
    s = str(size_str).strip().upper().replace(",", "")
    match = re.match(r"([0-9]*\.?[0-9]+)\s*([BMK]?)", s)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2)
    if unit == "B":
        return val * 1000.0
    if unit == "K":
        return val / 1000.0
    return val  # 'M' or unspecified -> already millions


def rank_curated(models: list[dict], size_pref: str, priority: str) -> list[dict]:
    """Filter curated models by size preference and order them by priority."""
    size_range = SIZE_PREFERENCES.get(size_pref, SIZE_PREFERENCES["Any size"])
    if size_pref != "Any size":
        models = [
            m
            for m in models
            if size_range["min"] <= parse_size(m["size"]) <= size_range["max"]
        ]
    if priority == "Smallest/Fastest":
        models = sorted(models, key=lambda x: parse_size(x["size"]))
    elif priority == "Best Quality":
        models = sorted(models, key=lambda x: parse_size(x["size"]), reverse=True)
    # "Most Popular" keeps the curated order.
    return models


def fetch_live_models(
    task_id: str, limit: int = 8, lister=None
) -> Optional[list[dict]]:
    """Return live top models for a pipeline task from the Hub.

    Sorted by downloads (desc). Returns ``None`` on any failure so callers can
    fall back to a curated list. ``lister`` is injectable for testing.
    """
    try:
        if lister is None:
            from huggingface_hub import list_models as lister
        results = lister(filter=task_id, sort="downloads", direction=-1, limit=limit)
        out = []
        for m in results:
            name = getattr(m, "id", None) or getattr(m, "modelId", None)
            if not name:
                continue
            out.append(
                {
                    "name": name,
                    "downloads": int(getattr(m, "downloads", 0) or 0),
                    "likes": int(getattr(m, "likes", 0) or 0),
                }
            )
        return out or None
    except Exception:
        return None


def generate_code_example(
    task_label: str, task_id: str, model_name: Optional[str]
) -> str:
    """Generate a code snippet for using the recommended model."""
    if not model_name:
        return ""

    code_templates = {
        "Text Generation": f'''```python
from transformers import pipeline

generator = pipeline("text-generation", model="{model_name}")

result = generator(
    "Write a story about a robot:",
    max_length=100,
    num_return_sequences=1
)
print(result[0]["generated_text"])
```''',
        "Text Classification": f'''```python
from transformers import pipeline

classifier = pipeline("text-classification", model="{model_name}")

result = classifier("I love this product! It's amazing!")
print(result)  # [{{'label': 'POSITIVE', 'score': 0.99}}]
```''',
        "Question Answering": f'''```python
from transformers import pipeline

qa = pipeline("question-answering", model="{model_name}")

result = qa(
    question="What is the capital of France?",
    context="France is a country in Europe. Paris is its capital city."
)
print(result["answer"])  # Paris
```''',
        "Translation": f'''```python
from transformers import pipeline

translator = pipeline("translation", model="{model_name}")

result = translator("Hello, how are you?")
print(result[0]["translation_text"])
```''',
        "Summarization": f'''```python
from transformers import pipeline

summarizer = pipeline("summarization", model="{model_name}")

long_text = """Your long article text here..."""
result = summarizer(long_text, max_length=130, min_length=30)
print(result[0]["summary_text"])
```''',
        "Image Classification": f'''```python
from transformers import pipeline

classifier = pipeline("image-classification", model="{model_name}")

result = classifier("path/to/image.jpg")
print(result)  # [{{'label': 'cat', 'score': 0.95}}]
```''',
        "Speech Recognition": f'''```python
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", model="{model_name}")

result = transcriber("audio.mp3")
print(result["text"])
```''',
        "Embeddings": f'''```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{model_name}")

sentences = ["This is a sentence", "This is another sentence"]
embeddings = model.encode(sentences)
print(embeddings.shape)  # (2, 384)
```''',
    }

    return code_templates.get(
        task_label,
        f'''```python
from transformers import pipeline

pipe = pipeline("{task_id}", model="{model_name}")
result = pipe("Your input here")
print(result)
```''',
    )
