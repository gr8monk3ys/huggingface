"""Pure logic for the paper recommender (no Gradio import) so it is testable.

Papers are loaded from the published arXiv dataset on the Hub when available
(:data:`DATASET_ID`), falling back to a built-in set of landmark papers when the
dataset can't be reached. Ranking is cosine similarity over the embedding model;
the corpus is small enough that a plain NumPy matmul is fast and an ANN index
(e.g. FAISS) isn't needed.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

DATASET_ID = os.environ.get("PAPERS_DATASET", "gr8monk3ys/cs-ml-academic-papers")
MAX_PAPERS = int(os.environ.get("PAPERS_MAX", "2500"))

CATEGORIES = {
    "All": None,
    "cs.AI - Artificial Intelligence": "cs.AI",
    "cs.CL - Computation & Language": "cs.CL",
    "cs.CV - Computer Vision": "cs.CV",
    "cs.LG - Machine Learning": "cs.LG",
    "stat.ML - Machine Learning (Stats)": "stat.ML",
}

FALLBACK_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We show that the Transformer generalizes well to other tasks.",
        "authors": "Vaswani, Shazeer, Parmar, et al.",
        "category": "cs.CL",
        "year": "2017",
        "url": "https://arxiv.org/abs/1706.03762"
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "abstract": "We introduce BERT, a new language representation model that is designed to pre-train deep bidirectional representations from unlabeled text. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context. It obtains state-of-the-art results on eleven NLP tasks.",
        "authors": "Devlin, Chang, Lee, Toutanova",
        "category": "cs.CL",
        "year": "2018",
        "url": "https://arxiv.org/abs/1810.04805"
    },
    {
        "title": "GPT-4 Technical Report",
        "abstract": "We report the development of GPT-4, a large-scale multimodal model which can accept image and text inputs and produce text outputs. GPT-4 exhibits human-level performance on various professional and academic benchmarks. We describe the predictable scaling of model capabilities and safety challenges.",
        "authors": "OpenAI",
        "category": "cs.CL",
        "year": "2023",
        "url": "https://arxiv.org/abs/2303.08774"
    },
    {
        "title": "ResNet: Deep Residual Learning for Image Recognition",
        "abstract": "We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs. Residual networks are easier to optimize and can gain accuracy from considerably increased depth.",
        "authors": "He, Zhang, Ren, Sun",
        "category": "cs.CV",
        "year": "2015",
        "url": "https://arxiv.org/abs/1512.03385"
    },
    {
        "title": "YOLO: Real-Time Object Detection",
        "abstract": "We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem. YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes. It is extremely fast.",
        "authors": "Redmon, Divvala, Girshick, Farhadi",
        "category": "cs.CV",
        "year": "2016",
        "url": "https://arxiv.org/abs/1506.02640"
    },
    {
        "title": "Vision Transformer (ViT)",
        "abstract": "We show that the reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks, Vision Transformer attains excellent results.",
        "authors": "Dosovitskiy, Beyer, Kolesnikov, et al.",
        "category": "cs.CV",
        "year": "2020",
        "url": "https://arxiv.org/abs/2010.11929"
    },
    {
        "title": "Proximal Policy Optimization (PPO)",
        "abstract": "We propose a new family of policy gradient methods for reinforcement learning, which alternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent. PPO strikes a balance between ease of implementation, sample complexity, and wall-clock time.",
        "authors": "Schulman, Wolski, Dhariwal, et al.",
        "category": "cs.LG",
        "year": "2017",
        "url": "https://arxiv.org/abs/1707.06347"
    },
    {
        "title": "XGBoost: A Scalable Tree Boosting System",
        "abstract": "We describe XGBoost, a scalable machine learning system for tree boosting. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. XGBoost scales beyond billions of examples and is the state-of-the-art method for many ML challenges.",
        "authors": "Chen, Guestrin",
        "category": "cs.LG",
        "year": "2016",
        "url": "https://arxiv.org/abs/1603.02754"
    },
    {
        "title": "Dropout: Preventing Neural Network Overfitting",
        "abstract": "Deep neural nets with many parameters are powerful but prone to overfitting. Dropout is a technique for addressing this problem. The key idea is to randomly drop units from the neural network during training. This prevents units from co-adapting too much and significantly reduces overfitting.",
        "authors": "Srivastava, Hinton, Krizhevsky, et al.",
        "category": "cs.LG",
        "year": "2014",
        "url": "https://jmlr.org/papers/v15/srivastava14a.html"
    },
    {
        "title": "Stable Diffusion: High-Resolution Image Synthesis",
        "abstract": "We apply diffusion models in the latent space of powerful pretrained autoencoders. Training diffusion models on such a representation enables high-resolution image synthesis on limited computational resources. Latent Diffusion Models achieve state-of-the-art synthesis results for image inpainting, class-conditional synthesis, and text-to-image synthesis.",
        "authors": "Rombach, Blattmann, Lorenz, et al.",
        "category": "cs.CV",
        "year": "2022",
        "url": "https://arxiv.org/abs/2112.10752"
    },
    {
        "title": "LLaMA: Open Foundation Language Models",
        "abstract": "We introduce LLaMA, a collection of foundation language models ranging from 7B to 65B parameters. We train our models on trillions of tokens using publicly available datasets exclusively. LLaMA-13B outperforms GPT-3 (175B) on most benchmarks and LLaMA-65B is competitive with the best models.",
        "authors": "Touvron, Lavril, Izacard, et al.",
        "category": "cs.CL",
        "year": "2023",
        "url": "https://arxiv.org/abs/2302.13971"
    },
    {
        "title": "Segment Anything Model (SAM)",
        "abstract": "We introduce the Segment Anything project: a new task, model, and dataset for image segmentation. Using our efficient model, SAM, in a data collection loop, we collected the largest segmentation dataset to date with over 1 billion masks. SAM is designed to be promptable and can transfer zero-shot to new image distributions.",
        "authors": "Kirillov, Mintun, Ravi, et al.",
        "category": "cs.CV",
        "year": "2023",
        "url": "https://arxiv.org/abs/2304.02643"
    },
    {
        "title": "Adam: A Method for Stochastic Optimization",
        "abstract": "We introduce Adam, an algorithm for first-order gradient-based optimization of stochastic objective functions. Adam combines the advantages of AdaGrad and RMSProp. It is computationally efficient, has little memory requirement, and is well suited for problems with large data or parameters.",
        "authors": "Kingma, Ba",
        "category": "cs.LG",
        "year": "2014",
        "url": "https://arxiv.org/abs/1412.6980"
    },
    {
        "title": "Batch Normalization: Accelerating Deep Network Training",
        "abstract": "Training deep neural networks is complicated by the distribution of layer inputs changing during training. We propose batch normalization, which allows us to use much higher learning rates and be less careful about initialization. It also acts as a regularizer and reduces the need for Dropout.",
        "authors": "Ioffe, Szegedy",
        "category": "cs.LG",
        "year": "2015",
        "url": "https://arxiv.org/abs/1502.03167"
    },
    {
        "title": "Word2Vec: Distributed Representations of Words",
        "abstract": "We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task. We observe large improvements in accuracy at much lower computational cost compared to previous techniques.",
        "authors": "Mikolov, Chen, Corrado, Dean",
        "category": "cs.CL",
        "year": "2013",
        "url": "https://arxiv.org/abs/1301.3781"
    },
    {
        "title": "GAN: Generative Adversarial Networks",
        "abstract": "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G.",
        "authors": "Goodfellow, Pouget-Abadie, Mirza, et al.",
        "category": "cs.LG",
        "year": "2014",
        "url": "https://arxiv.org/abs/1406.2661"
    },
    {
        "title": "Neural Machine Translation by Jointly Learning to Align and Translate",
        "abstract": "We conjecture that a fixed-length vector is a bottleneck in improving performance of encoder-decoder architecture. We propose to extend this by allowing a model to automatically search for parts of a source sentence that are relevant to predicting a target word, through an attention mechanism.",
        "authors": "Bahdanau, Cho, Bengio",
        "category": "cs.CL",
        "year": "2014",
        "url": "https://arxiv.org/abs/1409.0473"
    },
    {
        "title": "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)",
        "abstract": "We trained a large, deep convolutional neural network to classify the 1.2 million images in ImageNet into 1000 classes. Our network won the ImageNet Large Scale Visual Recognition Challenge in 2012. The neural network has 60 million parameters and 650,000 neurons.",
        "authors": "Krizhevsky, Sutskever, Hinton",
        "category": "cs.CV",
        "year": "2012",
        "url": "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks"
    },
    {
        "title": "U-Net: Convolutional Networks for Biomedical Image Segmentation",
        "abstract": "We present a network and training strategy that relies on data augmentation to use available annotated samples more efficiently. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. U-Net works with very few training images.",
        "authors": "Ronneberger, Fischer, Brox",
        "category": "cs.CV",
        "year": "2015",
        "url": "https://arxiv.org/abs/1505.04597"
    },
    {
        "title": "CLIP: Learning Transferable Visual Models From Natural Language Supervision",
        "abstract": "We demonstrate that a simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn image representations from scratch on a dataset of 400 million image-text pairs. CLIP learns to perform a wide variety of tasks during pre-training.",
        "authors": "Radford, Kim, Hallacy, et al.",
        "category": "cs.CV",
        "year": "2021",
        "url": "https://arxiv.org/abs/2103.00020"
    },
]


def _first(row: dict, *keys: str, default: str = ""):
    """Return the first present, non-empty value among *keys*."""
    for key in keys:
        if key in row and row[key] not in (None, "", []):
            return row[key]
    return default


def normalize_row(row: dict) -> dict:
    """Map a raw dataset row (varied schemas) to our paper dict."""
    category = _first(row, "primary_category", "category", "categories")
    if isinstance(category, list):
        category = category[0] if category else ""

    year = str(_first(row, "year", "published", "updated", "date"))[:4]

    authors = _first(row, "authors", "author")
    if isinstance(authors, list):
        shown = ", ".join(authors[:4])
        authors = shown + (", et al." if len(authors) > 4 else "")

    return {
        "title": str(_first(row, "title")).strip(),
        "abstract": str(_first(row, "abstract", "summary")).strip(),
        "authors": authors or "Unknown",
        "category": category or "",
        "year": year,
        "url": _first(row, "url", "arxiv_url", "pdf_url", "entry_id", "arxiv_id"),
    }


def load_papers(
    dataset_id: str = DATASET_ID,
    split: str = "train",
    max_papers: int = MAX_PAPERS,
    loader=None,
) -> Optional[list[dict]]:
    """Load papers from the Hub dataset, or return ``None`` on any failure.

    ``loader`` is injectable for testing; by default ``datasets.load_dataset``
    is imported lazily so this module imports without the dependency.
    """
    try:
        if loader is None:
            from datasets import load_dataset as loader
        dataset = loader(dataset_id, split=split)
        papers = []
        for row in dataset:
            paper = normalize_row(row)
            if paper["title"] and paper["abstract"]:
                papers.append(paper)
            if len(papers) >= max_papers:
                break
        return papers or None
    except Exception:
        return None


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows, guarding against zero vectors."""
    matrix = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def build_index(papers: list[dict], encode_fn) -> np.ndarray:
    """Embed ``"{title} {abstract}"`` for each paper and L2-normalize."""
    texts = [f"{p['title']} {p['abstract']}" for p in papers]
    return normalize(encode_fn(texts))


def recommend(
    query: str,
    papers: list[dict],
    paper_matrix: np.ndarray,
    encode_fn,
    category_value: Optional[str],
    num_results: int,
) -> list[dict]:
    """Return ranked, category-filtered results for *query* (most similar first)."""
    if not query.strip():
        return []
    query_vec = np.asarray(encode_fn([query]), dtype="float32").reshape(-1)
    query_vec = query_vec / (np.linalg.norm(query_vec) or 1.0)
    sims = paper_matrix @ query_vec
    order = np.argsort(sims)[::-1]

    results = []
    for idx in order:
        if len(results) >= num_results:
            break
        paper = papers[int(idx)]
        if category_value and paper["category"] != category_value:
            continue
        results.append({**paper, "similarity": float(sims[idx]) * 100})
    return results
