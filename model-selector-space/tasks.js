// Generated from the former core.py -- the single source of task data.
// tests/test_model_selector_data.py parses this file and validates it, so a
// static Space does not mean unvalidated data.
const DATA = {
  "tasks": {
    "Text Generation": {
      "id": "text-generation",
      "description": "Generate text, stories, code, or continue prompts",
      "use_cases": [
        "Chatbots",
        "Content writing",
        "Code completion",
        "Story generation"
      ],
      "top_models": [
        {
          "name": "meta-llama/Llama-3.1-8B-Instruct",
          "size": "8B",
          "license": "llama3.1"
        },
        {
          "name": "mistralai/Mistral-7B-Instruct-v0.2",
          "size": "7B",
          "license": "apache-2.0"
        },
        {
          "name": "Qwen/Qwen2.5-7B-Instruct",
          "size": "7B",
          "license": "apache-2.0"
        },
        {
          "name": "google/gemma-2-9b-it",
          "size": "9B",
          "license": "gemma"
        },
        {
          "name": "microsoft/phi-3-mini-4k-instruct",
          "size": "3.8B",
          "license": "mit"
        }
      ]
    },
    "Text Classification": {
      "id": "text-classification",
      "description": "Classify text into categories (sentiment, topic, intent)",
      "use_cases": [
        "Sentiment analysis",
        "Spam detection",
        "Topic classification",
        "Intent detection"
      ],
      "top_models": [
        {
          "name": "distilbert-base-uncased-finetuned-sst-2-english",
          "size": "67M",
          "license": "apache-2.0"
        },
        {
          "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
          "size": "125M",
          "license": "mit"
        },
        {
          "name": "facebook/bart-large-mnli",
          "size": "400M",
          "license": "mit"
        }
      ]
    },
    "Question Answering": {
      "id": "question-answering",
      "description": "Answer questions based on context or knowledge",
      "use_cases": [
        "Customer support",
        "Document QA",
        "Knowledge retrieval",
        "FAQ bots"
      ],
      "top_models": [
        {
          "name": "deepset/roberta-base-squad2",
          "size": "125M",
          "license": "cc-by-4.0"
        },
        {
          "name": "distilbert-base-cased-distilled-squad",
          "size": "67M",
          "license": "apache-2.0"
        },
        {
          "name": "google/flan-t5-base",
          "size": "250M",
          "license": "apache-2.0"
        },
        {
          "name": "Intel/dynamic_tinybert",
          "size": "15M",
          "license": "apache-2.0"
        }
      ]
    },
    "Translation": {
      "id": "translation",
      "description": "Translate text between languages",
      "use_cases": [
        "Multilingual apps",
        "Document translation",
        "Real-time translation"
      ],
      "top_models": [
        {
          "name": "facebook/nllb-200-distilled-600M",
          "size": "600M",
          "license": "cc-by-nc-4.0"
        },
        {
          "name": "Helsinki-NLP/opus-mt-en-de",
          "size": "74M",
          "license": "apache-2.0"
        },
        {
          "name": "google/madlad400-3b-mt",
          "size": "3B",
          "license": "apache-2.0"
        },
        {
          "name": "facebook/mbart-large-50-many-to-many-mmt",
          "size": "611M",
          "license": "mit"
        }
      ]
    },
    "Summarization": {
      "id": "summarization",
      "description": "Summarize long documents or articles",
      "use_cases": [
        "News summarization",
        "Document condensing",
        "Meeting notes",
        "Research papers"
      ],
      "top_models": [
        {
          "name": "facebook/bart-large-cnn",
          "size": "400M",
          "license": "mit"
        },
        {
          "name": "google/pegasus-xsum",
          "size": "568M",
          "license": "apache-2.0"
        },
        {
          "name": "philschmid/bart-large-cnn-samsum",
          "size": "400M",
          "license": "mit"
        },
        {
          "name": "google/flan-t5-large",
          "size": "780M",
          "license": "apache-2.0"
        }
      ]
    },
    "Image Classification": {
      "id": "image-classification",
      "description": "Classify images into categories",
      "use_cases": [
        "Product categorization",
        "Medical imaging",
        "Quality control",
        "Content moderation"
      ],
      "top_models": [
        {
          "name": "google/vit-base-patch16-224",
          "size": "86M",
          "license": "apache-2.0"
        },
        {
          "name": "microsoft/resnet-50",
          "size": "25M",
          "license": "apache-2.0"
        },
        {
          "name": "facebook/convnext-base-224",
          "size": "88M",
          "license": "apache-2.0"
        },
        {
          "name": "timm/efficientnet_b0.ra_in1k",
          "size": "5M",
          "license": "apache-2.0"
        }
      ]
    },
    "Object Detection": {
      "id": "object-detection",
      "description": "Detect and locate objects in images",
      "use_cases": [
        "Autonomous vehicles",
        "Security cameras",
        "Inventory management",
        "Sports analytics"
      ],
      "top_models": [
        {
          "name": "facebook/detr-resnet-50",
          "size": "41M",
          "license": "apache-2.0"
        },
        {
          "name": "hustvl/yolos-tiny",
          "size": "6M",
          "license": "apache-2.0"
        },
        {
          "name": "microsoft/table-transformer-detection",
          "size": "42M",
          "license": "mit"
        },
        {
          "name": "facebook/detr-resnet-101",
          "size": "60M",
          "license": "apache-2.0"
        }
      ]
    },
    "Image Generation": {
      "id": "text-to-image",
      "description": "Generate images from text descriptions",
      "use_cases": [
        "Art creation",
        "Product visualization",
        "Marketing content",
        "Game assets"
      ],
      "top_models": [
        {
          "name": "stabilityai/stable-diffusion-xl-base-1.0",
          "size": "6.9B",
          "license": "openrail++"
        },
        {
          "name": "black-forest-labs/FLUX.1-schnell",
          "size": "12B",
          "license": "apache-2.0"
        },
        {
          "name": "runwayml/stable-diffusion-v1-5",
          "size": "1B",
          "license": "creativeml-openrail-m"
        },
        {
          "name": "stabilityai/sdxl-turbo",
          "size": "6.9B",
          "license": "openrail++"
        }
      ]
    },
    "Speech Recognition": {
      "id": "automatic-speech-recognition",
      "description": "Convert speech to text",
      "use_cases": [
        "Transcription",
        "Voice commands",
        "Meeting notes",
        "Accessibility"
      ],
      "top_models": [
        {
          "name": "openai/whisper-large-v3",
          "size": "1.5B",
          "license": "apache-2.0"
        },
        {
          "name": "openai/whisper-medium",
          "size": "769M",
          "license": "apache-2.0"
        },
        {
          "name": "openai/whisper-small",
          "size": "244M",
          "license": "apache-2.0"
        },
        {
          "name": "facebook/wav2vec2-base-960h",
          "size": "95M",
          "license": "apache-2.0"
        }
      ]
    },
    "Embeddings": {
      "id": "feature-extraction",
      "description": "Generate embeddings for semantic search and similarity",
      "use_cases": [
        "Semantic search",
        "Recommendation systems",
        "Clustering",
        "RAG systems"
      ],
      "top_models": [
        {
          "name": "sentence-transformers/all-MiniLM-L6-v2",
          "size": "22M",
          "license": "apache-2.0"
        },
        {
          "name": "sentence-transformers/all-mpnet-base-v2",
          "size": "109M",
          "license": "apache-2.0"
        },
        {
          "name": "BAAI/bge-small-en-v1.5",
          "size": "33M",
          "license": "mit"
        },
        {
          "name": "intfloat/e5-small-v2",
          "size": "33M",
          "license": "mit"
        }
      ]
    }
  },
  "sizePreferences": {
    "Tiny (< 100M)": {
      "min": 0,
      "max": 100
    },
    "Small (100M - 500M)": {
      "min": 100,
      "max": 500
    },
    "Medium (500M - 2B)": {
      "min": 500,
      "max": 2000
    },
    "Large (2B - 10B)": {
      "min": 2000,
      "max": 10000
    },
    "Any size": {
      "min": 0,
      "max": 100000
    }
  },
  "codeTemplates": {
    "Text Generation": "```python\nfrom transformers import pipeline\n\ngenerator = pipeline(\"text-generation\", model=\"__MODEL__\")\n\nresult = generator(\n    \"Write a story about a robot:\",\n    max_length=100,\n    num_return_sequences=1\n)\nprint(result[0][\"generated_text\"])\n```",
    "Text Classification": "```python\nfrom transformers import pipeline\n\nclassifier = pipeline(\"text-classification\", model=\"__MODEL__\")\n\nresult = classifier(\"I love this product! It's amazing!\")\nprint(result)  # [{'label': 'POSITIVE', 'score': 0.99}]\n```",
    "Question Answering": "```python\nfrom transformers import pipeline\n\nqa = pipeline(\"question-answering\", model=\"__MODEL__\")\n\nresult = qa(\n    question=\"What is the capital of France?\",\n    context=\"France is a country in Europe. Paris is its capital city.\"\n)\nprint(result[\"answer\"])  # Paris\n```",
    "Translation": "```python\nfrom transformers import pipeline\n\ntranslator = pipeline(\"translation\", model=\"__MODEL__\")\n\nresult = translator(\"Hello, how are you?\")\nprint(result[0][\"translation_text\"])\n```",
    "Summarization": "```python\nfrom transformers import pipeline\n\nsummarizer = pipeline(\"summarization\", model=\"__MODEL__\")\n\nlong_text = \"\"\"Your long article text here...\"\"\"\nresult = summarizer(long_text, max_length=130, min_length=30)\nprint(result[0][\"summary_text\"])\n```",
    "Image Classification": "```python\nfrom transformers import pipeline\n\nclassifier = pipeline(\"image-classification\", model=\"__MODEL__\")\n\nresult = classifier(\"path/to/image.jpg\")\nprint(result)  # [{'label': 'cat', 'score': 0.95}]\n```",
    "Object Detection": "```python\nfrom transformers import pipeline\n\npipe = pipeline(\"object-detection\", model=\"__MODEL__\")\nresult = pipe(\"Your input here\")\nprint(result)\n```",
    "Image Generation": "```python\nfrom transformers import pipeline\n\npipe = pipeline(\"text-to-image\", model=\"__MODEL__\")\nresult = pipe(\"Your input here\")\nprint(result)\n```",
    "Speech Recognition": "```python\nfrom transformers import pipeline\n\ntranscriber = pipeline(\"automatic-speech-recognition\", model=\"__MODEL__\")\n\nresult = transcriber(\"audio.mp3\")\nprint(result[\"text\"])\n```",
    "Embeddings": "```python\nfrom sentence_transformers import SentenceTransformer\n\nmodel = SentenceTransformer(\"__MODEL__\")\n\nsentences = [\"This is a sentence\", \"This is another sentence\"]\nembeddings = model.encode(sentences)\nprint(embeddings.shape)  # (2, 384)\n```"
  },
  "genericTemplate": "```python\nfrom transformers import pipeline\n\npipe = pipeline(\"some-task\", model=\"__MODEL__\")\nresult = pipe(\"Your input here\")\nprint(result)\n```"
};
