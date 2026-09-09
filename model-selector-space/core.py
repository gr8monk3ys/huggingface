"""Model-recommendation logic, independent of Gradio.

The entry point is :func:`recommend`. It owns the sequence -- resolve the task,
try the Hub live, fall back to the curated list -- and returns a
:class:`Recommendation` for ``app.py`` to render.

The TASKS table lives here, beside the code that ranks it. It used to sit in
app.py while core.py ranked data it did not own, which is the shape
docs/adr/0002-coarse-entry-point-for-space-core-modules.md records as the
reason for this change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from hf_client import with_retry

TASKS = {
    "Text Generation": {
        "id": "text-generation",
        "description": "Generate text, stories, code, or continue prompts",
        "use_cases": [
            "Chatbots",
            "Content writing",
            "Code completion",
            "Story generation",
        ],
        "top_models": [
            {
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "size": "8B",
                "license": "llama3.1",
            },
            {
                "name": "mistralai/Mistral-7B-Instruct-v0.3",
                "size": "7B",
                "license": "apache-2.0",
            },
            {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "license": "apache-2.0"},
            {"name": "google/gemma-2-9b-it", "size": "9B", "license": "gemma"},
            {
                "name": "microsoft/phi-3-mini-4k-instruct",
                "size": "3.8B",
                "license": "mit",
            },
        ],
    },
    "Text Classification": {
        "id": "text-classification",
        "description": "Classify text into categories (sentiment, topic, intent)",
        "use_cases": [
            "Sentiment analysis",
            "Spam detection",
            "Topic classification",
            "Intent detection",
        ],
        "top_models": [
            {
                "name": "distilbert-base-uncased-finetuned-sst-2-english",
                "size": "67M",
                "license": "apache-2.0",
            },
            {
                "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "size": "125M",
                "license": "mit",
            },
            {"name": "facebook/bart-large-mnli", "size": "400M", "license": "mit"},
        ],
    },
    "Question Answering": {
        "id": "question-answering",
        "description": "Answer questions based on context or knowledge",
        "use_cases": [
            "Customer support",
            "Document QA",
            "Knowledge retrieval",
            "FAQ bots",
        ],
        "top_models": [
            {
                "name": "deepset/roberta-base-squad2",
                "size": "125M",
                "license": "cc-by-4.0",
            },
            {
                "name": "distilbert-base-cased-distilled-squad",
                "size": "67M",
                "license": "apache-2.0",
            },
            {"name": "google/flan-t5-base", "size": "250M", "license": "apache-2.0"},
            {"name": "Intel/dynamic_tinybert", "size": "15M", "license": "apache-2.0"},
        ],
    },
    "Translation": {
        "id": "translation",
        "description": "Translate text between languages",
        "use_cases": [
            "Multilingual apps",
            "Document translation",
            "Real-time translation",
        ],
        "top_models": [
            {
                "name": "facebook/nllb-200-distilled-600M",
                "size": "600M",
                "license": "cc-by-nc-4.0",
            },
            {
                "name": "Helsinki-NLP/opus-mt-en-de",
                "size": "74M",
                "license": "apache-2.0",
            },
            {"name": "google/madlad400-3b-mt", "size": "3B", "license": "apache-2.0"},
            {
                "name": "facebook/mbart-large-50-many-to-many-mmt",
                "size": "611M",
                "license": "mit",
            },
        ],
    },
    "Summarization": {
        "id": "summarization",
        "description": "Summarize long documents or articles",
        "use_cases": [
            "News summarization",
            "Document condensing",
            "Meeting notes",
            "Research papers",
        ],
        "top_models": [
            {"name": "facebook/bart-large-cnn", "size": "400M", "license": "mit"},
            {"name": "google/pegasus-xsum", "size": "568M", "license": "apache-2.0"},
            {
                "name": "philschmid/bart-large-cnn-samsum",
                "size": "400M",
                "license": "mit",
            },
            {"name": "google/flan-t5-large", "size": "780M", "license": "apache-2.0"},
        ],
    },
    "Image Classification": {
        "id": "image-classification",
        "description": "Classify images into categories",
        "use_cases": [
            "Product categorization",
            "Medical imaging",
            "Quality control",
            "Content moderation",
        ],
        "top_models": [
            {
                "name": "google/vit-base-patch16-224",
                "size": "86M",
                "license": "apache-2.0",
            },
            {"name": "microsoft/resnet-50", "size": "25M", "license": "apache-2.0"},
            {
                "name": "facebook/convnext-base-224",
                "size": "88M",
                "license": "apache-2.0",
            },
            {
                "name": "timm/efficientnet_b0.ra_in1k",
                "size": "5M",
                "license": "apache-2.0",
            },
        ],
    },
    "Object Detection": {
        "id": "object-detection",
        "description": "Detect and locate objects in images",
        "use_cases": [
            "Autonomous vehicles",
            "Security cameras",
            "Inventory management",
            "Sports analytics",
        ],
        "top_models": [
            {"name": "facebook/detr-resnet-50", "size": "41M", "license": "apache-2.0"},
            {"name": "hustvl/yolos-tiny", "size": "6M", "license": "apache-2.0"},
            {
                "name": "microsoft/table-transformer-detection",
                "size": "42M",
                "license": "mit",
            },
            {
                "name": "facebook/detr-resnet-101",
                "size": "60M",
                "license": "apache-2.0",
            },
        ],
    },
    "Image Generation": {
        "id": "text-to-image",
        "description": "Generate images from text descriptions",
        "use_cases": [
            "Art creation",
            "Product visualization",
            "Marketing content",
            "Game assets",
        ],
        "top_models": [
            {
                "name": "stabilityai/stable-diffusion-xl-base-1.0",
                "size": "6.9B",
                "license": "openrail++",
            },
            {
                "name": "black-forest-labs/FLUX.1-schnell",
                "size": "12B",
                "license": "apache-2.0",
            },
            {
                "name": "runwayml/stable-diffusion-v1-5",
                "size": "1B",
                "license": "creativeml-openrail-m",
            },
            {"name": "stabilityai/sdxl-turbo", "size": "6.9B", "license": "openrail++"},
        ],
    },
    "Speech Recognition": {
        "id": "automatic-speech-recognition",
        "description": "Convert speech to text",
        "use_cases": [
            "Transcription",
            "Voice commands",
            "Meeting notes",
            "Accessibility",
        ],
        "top_models": [
            {
                "name": "openai/whisper-large-v3",
                "size": "1.5B",
                "license": "apache-2.0",
            },
            {"name": "openai/whisper-medium", "size": "769M", "license": "apache-2.0"},
            {"name": "openai/whisper-small", "size": "244M", "license": "apache-2.0"},
            {
                "name": "facebook/wav2vec2-base-960h",
                "size": "95M",
                "license": "apache-2.0",
            },
        ],
    },
    "Embeddings": {
        "id": "feature-extraction",
        "description": "Generate embeddings for semantic search and similarity",
        "use_cases": [
            "Semantic search",
            "Recommendation systems",
            "Clustering",
            "RAG systems",
        ],
        "top_models": [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "size": "22M",
                "license": "apache-2.0",
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "size": "109M",
                "license": "apache-2.0",
            },
            {"name": "BAAI/bge-small-en-v1.5", "size": "33M", "license": "mit"},
            {"name": "intfloat/e5-small-v2", "size": "33M", "license": "mit"},
        ],
    },
}


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
        # A cold start or a rate limit is exactly what with_retry exists for;
        # without it a transient blip silently demoted every user to the
        # curated list.
        results = with_retry(
            lister, filter=task_id, sort="downloads", direction=-1, limit=limit
        )
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
    except Exception:  # noqa: BLE001 - a Hub outage degrades to the curated list
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
LIVE_QUERY_LIMIT = 8
LIVE_RESULTS_SHOWN = 5
CURATED_RESULTS_SHOWN = 4

ANY_SIZE = "Any size"
BEST_QUALITY = "Best Quality"


class UnknownTaskError(ValueError):
    """The requested task is not configured."""


class NoMatchError(ValueError):
    """No curated model matches the requested size."""


@dataclass(frozen=True)
class ModelRec:
    """One recommended model. Live and curated results carry different fields."""

    name: str
    downloads: Optional[int] = None
    likes: Optional[int] = None
    size: Optional[str] = None
    license: Optional[str] = None

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.name}"


@dataclass(frozen=True)
class Recommendation:
    """Models for a task, plus where they came from."""

    task: str
    description: str
    source: str  # "live" or "curated"
    models: list[ModelRec] = field(default_factory=list)
    code_example: str = ""
    size_filter_ignored: bool = False


def recommend(
    task: str,
    size_pref: str = ANY_SIZE,
    priority: str = "Most Popular",
    *,
    lister=None,
) -> Recommendation:
    """Recommend models for *task*, querying the Hub with a curated fallback.

    The fallback is deliberate, and the result says which path was taken via
    ``source`` -- a Hub outage should show curated picks rather than an error,
    but the user is told that is what happened.

    Raises:
        UnknownTaskError: *task* is not configured.
        NoMatchError: the curated list has nothing matching *size_pref*.
    """
    if task not in TASKS:
        raise UnknownTaskError("Please select a task.")

    info = TASKS[task]
    task_id = info["id"]

    live = fetch_live_models(task_id, limit=LIVE_QUERY_LIMIT, lister=lister)
    if live:
        if priority == BEST_QUALITY:
            live = sorted(live, key=lambda m: m["likes"], reverse=True)
        models = [
            ModelRec(name=m["name"], downloads=m["downloads"], likes=m["likes"])
            for m in live[:LIVE_RESULTS_SHOWN]
        ]
        return Recommendation(
            task=task,
            description=info["description"],
            source="live",
            models=models,
            code_example=generate_code_example(task, task_id, models[0].name),
            # Size is a curated-list concept; live results rank by popularity.
            size_filter_ignored=size_pref != ANY_SIZE,
        )

    curated = rank_curated(info["top_models"], size_pref, priority)
    if not curated:
        raise NoMatchError("No models match your size preference. Try 'Any size'.")

    models = [
        ModelRec(name=m["name"], size=m["size"], license=m["license"])
        for m in curated[:CURATED_RESULTS_SHOWN]
    ]
    return Recommendation(
        task=task,
        description=info["description"],
        source="curated",
        models=models,
        code_example=generate_code_example(task, task_id, models[0].name),
    )
