"""
Model Selector - Find the right HuggingFace model for your task.

Answer a few questions and get personalized model recommendations.
"""

import gradio as gr

from core import (
    SIZE_PREFERENCES,
    fetch_live_models,
    generate_code_example,
    rank_curated,
)

# ---------------------------------------------------------------------------
# Task Categories and Model Recommendations
# ---------------------------------------------------------------------------

TASKS = {
    "Text Generation": {
        "id": "text-generation",
        "description": "Generate text, stories, code, or continue prompts",
        "use_cases": ["Chatbots", "Content writing", "Code completion", "Story generation"],
        "top_models": [
            {"name": "meta-llama/Llama-3.1-8B-Instruct", "size": "8B", "license": "llama3.1"},
            {"name": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B", "license": "apache-2.0"},
            {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B", "license": "apache-2.0"},
            {"name": "google/gemma-2-9b-it", "size": "9B", "license": "gemma"},
            {"name": "microsoft/phi-3-mini-4k-instruct", "size": "3.8B", "license": "mit"},
        ]
    },
    "Text Classification": {
        "id": "text-classification",
        "description": "Classify text into categories (sentiment, topic, intent)",
        "use_cases": ["Sentiment analysis", "Spam detection", "Topic classification", "Intent detection"],
        "top_models": [
            {"name": "distilbert-base-uncased-finetuned-sst-2-english", "size": "67M", "license": "apache-2.0"},
            {"name": "cardiffnlp/twitter-roberta-base-sentiment-latest", "size": "125M", "license": "mit"},
            {"name": "facebook/bart-large-mnli", "size": "400M", "license": "mit"},
        ]
    },
    "Question Answering": {
        "id": "question-answering",
        "description": "Answer questions based on context or knowledge",
        "use_cases": ["Customer support", "Document QA", "Knowledge retrieval", "FAQ bots"],
        "top_models": [
            {"name": "deepset/roberta-base-squad2", "size": "125M", "license": "cc-by-4.0"},
            {"name": "distilbert-base-cased-distilled-squad", "size": "67M", "license": "apache-2.0"},
            {"name": "google/flan-t5-base", "size": "250M", "license": "apache-2.0"},
            {"name": "Intel/dynamic_tinybert", "size": "15M", "license": "apache-2.0"},
        ]
    },
    "Translation": {
        "id": "translation",
        "description": "Translate text between languages",
        "use_cases": ["Multilingual apps", "Document translation", "Real-time translation"],
        "top_models": [
            {"name": "facebook/nllb-200-distilled-600M", "size": "600M", "license": "cc-by-nc-4.0"},
            {"name": "Helsinki-NLP/opus-mt-en-de", "size": "74M", "license": "apache-2.0"},
            {"name": "google/madlad400-3b-mt", "size": "3B", "license": "apache-2.0"},
            {"name": "facebook/mbart-large-50-many-to-many-mmt", "size": "611M", "license": "mit"},
        ]
    },
    "Summarization": {
        "id": "summarization",
        "description": "Summarize long documents or articles",
        "use_cases": ["News summarization", "Document condensing", "Meeting notes", "Research papers"],
        "top_models": [
            {"name": "facebook/bart-large-cnn", "size": "400M", "license": "mit"},
            {"name": "google/pegasus-xsum", "size": "568M", "license": "apache-2.0"},
            {"name": "philschmid/bart-large-cnn-samsum", "size": "400M", "license": "mit"},
            {"name": "google/flan-t5-large", "size": "780M", "license": "apache-2.0"},
        ]
    },
    "Image Classification": {
        "id": "image-classification",
        "description": "Classify images into categories",
        "use_cases": ["Product categorization", "Medical imaging", "Quality control", "Content moderation"],
        "top_models": [
            {"name": "google/vit-base-patch16-224", "size": "86M", "license": "apache-2.0"},
            {"name": "microsoft/resnet-50", "size": "25M", "license": "apache-2.0"},
            {"name": "facebook/convnext-base-224", "size": "88M", "license": "apache-2.0"},
            {"name": "timm/efficientnet_b0.ra_in1k", "size": "5M", "license": "apache-2.0"},
        ]
    },
    "Object Detection": {
        "id": "object-detection",
        "description": "Detect and locate objects in images",
        "use_cases": ["Autonomous vehicles", "Security cameras", "Inventory management", "Sports analytics"],
        "top_models": [
            {"name": "facebook/detr-resnet-50", "size": "41M", "license": "apache-2.0"},
            {"name": "hustvl/yolos-tiny", "size": "6M", "license": "apache-2.0"},
            {"name": "microsoft/table-transformer-detection", "size": "42M", "license": "mit"},
            {"name": "facebook/detr-resnet-101", "size": "60M", "license": "apache-2.0"},
        ]
    },
    "Image Generation": {
        "id": "text-to-image",
        "description": "Generate images from text descriptions",
        "use_cases": ["Art creation", "Product visualization", "Marketing content", "Game assets"],
        "top_models": [
            {"name": "stabilityai/stable-diffusion-xl-base-1.0", "size": "6.9B", "license": "openrail++"},
            {"name": "black-forest-labs/FLUX.1-schnell", "size": "12B", "license": "apache-2.0"},
            {"name": "runwayml/stable-diffusion-v1-5", "size": "1B", "license": "creativeml-openrail-m"},
            {"name": "stabilityai/sdxl-turbo", "size": "6.9B", "license": "openrail++"},
        ]
    },
    "Speech Recognition": {
        "id": "automatic-speech-recognition",
        "description": "Convert speech to text",
        "use_cases": ["Transcription", "Voice commands", "Meeting notes", "Accessibility"],
        "top_models": [
            {"name": "openai/whisper-large-v3", "size": "1.5B", "license": "apache-2.0"},
            {"name": "openai/whisper-medium", "size": "769M", "license": "apache-2.0"},
            {"name": "openai/whisper-small", "size": "244M", "license": "apache-2.0"},
            {"name": "facebook/wav2vec2-base-960h", "size": "95M", "license": "apache-2.0"},
        ]
    },
    "Embeddings": {
        "id": "feature-extraction",
        "description": "Generate embeddings for semantic search and similarity",
        "use_cases": ["Semantic search", "Recommendation systems", "Clustering", "RAG systems"],
        "top_models": [
            {"name": "sentence-transformers/all-MiniLM-L6-v2", "size": "22M", "license": "apache-2.0"},
            {"name": "sentence-transformers/all-mpnet-base-v2", "size": "109M", "license": "apache-2.0"},
            {"name": "BAAI/bge-small-en-v1.5", "size": "33M", "license": "mit"},
            {"name": "intfloat/e5-small-v2", "size": "33M", "license": "mit"},
        ]
    },
}

# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def get_recommendations(task: str, size_pref: str, priority: str, use_case: str) -> tuple[str, str]:
    """Recommend models for a task, querying the Hub live with a curated fallback."""
    if task not in TASKS:
        return "Please select a task.", ""

    task_info = TASKS[task]
    task_id = task_info["id"]

    header = [
        f"## Recommendations for: {task}\n",
        f"*{task_info['description']}*\n",
    ]
    if use_case:
        header.append(f"**Your use case:** {use_case}\n")

    # 1) Try a live query against the HuggingFace Hub.
    live = fetch_live_models(task_id, limit=8)
    if live:
        if priority == "Best Quality":
            live = sorted(live, key=lambda m: m["likes"], reverse=True)
        recs = header + ["_Live from the HuggingFace Hub, sorted by downloads._\n"]
        if size_pref != "Any size":
            recs.append(
                "> Size filtering applies to the curated fallback; live results "
                "are ranked by popularity.\n"
            )
        recs.append("---\n")
        for i, m in enumerate(live[:5], 1):
            recs.append(f"### {i}. {m['name']}")
            recs.append(f"- **Downloads:** {m['downloads']:,} | **Likes:** {m['likes']:,}")
            recs.append(f"- **Link:** [View on HuggingFace](https://huggingface.co/{m['name']})")
            recs.append("")
        return "\n".join(recs), generate_code_example(task, task_id, live[0]["name"])

    # 2) Fall back to the curated list (e.g. when offline or rate-limited).
    models = rank_curated(task_info["top_models"], size_pref, priority)
    if not models:
        return "No models match your size preference. Try 'Any size'.", ""
    recs = header + ["_Curated picks (live Hub query unavailable right now)._\n", "---\n"]
    for i, model in enumerate(models[:4], 1):
        recs.append(f"### {i}. {model['name']}")
        recs.append(f"- **Size:** {model['size']} parameters")
        recs.append(f"- **License:** {model['license']}")
        recs.append(f"- **Link:** [View on HuggingFace](https://huggingface.co/{model['name']})")
        recs.append("")
    return "\n".join(recs), generate_code_example(task, task_id, models[0]["name"])


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

with gr.Blocks(title="Model Selector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Model Selector

    Find the perfect HuggingFace model for your task. Answer a few questions
    and get personalized recommendations with code examples.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            task_select = gr.Dropdown(
                choices=list(TASKS.keys()),
                label="What do you want to do?",
                value="Text Generation"
            )

            task_description = gr.Markdown(
                value=f"*{TASKS['Text Generation']['description']}*"
            )

            size_select = gr.Dropdown(
                choices=list(SIZE_PREFERENCES.keys()),
                label="Model size preference?",
                value="Any size",
                info="Smaller = faster, larger = higher quality"
            )

            priority_select = gr.Radio(
                choices=["Most Popular", "Smallest/Fastest", "Best Quality"],
                label="What matters most?",
                value="Most Popular"
            )

            use_case = gr.Textbox(
                label="Describe your use case (optional)",
                placeholder="e.g., Customer support chatbot for e-commerce"
            )

            recommend_btn = gr.Button("Get Recommendations", variant="primary", size="lg")

        with gr.Column(scale=1):
            recommendations = gr.Markdown(label="Recommendations")
            code_example = gr.Markdown(label="Code Example")

    # Use cases display
    use_cases_display = gr.Markdown(
        value=f"**Common use cases:** {', '.join(TASKS['Text Generation']['use_cases'])}"
    )

    # Event handlers
    def update_task_info(task):
        desc = f"*{TASKS[task]['description']}*"
        uses = f"**Common use cases:** {', '.join(TASKS[task]['use_cases'])}"
        return desc, uses

    task_select.change(
        fn=update_task_info,
        inputs=[task_select],
        outputs=[task_description, use_cases_display]
    )

    recommend_btn.click(
        fn=get_recommendations,
        inputs=[task_select, size_select, priority_select, use_case],
        outputs=[recommendations, code_example]
    )

    gr.Markdown("""
    ---

    ### Quick Reference

    | Task | Best For | Typical Size |
    |------|----------|--------------|
    | Text Generation | Chatbots, content | 3B - 70B |
    | Text Classification | Sentiment, topics | 50M - 300M |
    | Embeddings | Search, RAG | 20M - 100M |
    | Speech Recognition | Transcription | 200M - 1.5B |
    | Image Generation | Art, visualization | 1B - 12B |

    ---

    Built by [Lorenzo Scaturchio](https://huggingface.co/gr8monk3ys)
    """)


if __name__ == "__main__":
    demo.launch()
