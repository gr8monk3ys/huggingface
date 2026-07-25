---
license: mit
base_model: distilbert-base-uncased
tags:
  - text-classification
  - arxiv
  - academic-papers
  - distilbert
datasets:
  - ccdv/arxiv-classification
metrics:
  - accuracy
  - f1
pipeline_tag: text-classification
---

# Academic Paper Classifier

A DistilBERT model fine-tuned to classify academic paper abstracts into arxiv
subject categories. Given the abstract of a research paper, the model predicts
which area of computer science or statistics the paper belongs to.

## Intended Use

This model is designed for:

- **Automated paper triage** -- quickly routing new submissions to the
  appropriate reviewers or reading lists.
- **Literature search** -- filtering large collections of papers by
  predicted subject area.
- **Research tooling** -- as a building block in larger academic-paper
  analysis pipelines.

The model is **not** intended for high-stakes decisions such as publication
acceptance or funding allocation.

## Labels

| Id | Label    | Description                       |
|----|----------|-----------------------------------|
| 0  | cs.AI    | Artificial Intelligence           |
| 1  | cs.CL    | Computation and Language (NLP)    |
| 2  | cs.CV    | Computer Vision                   |
| 3  | cs.LG    | Machine Learning                  |
| 4  | cs.NE    | Neural and Evolutionary Computing |
| 5  | cs.RO    | Robotics                          |
| 6  | math.ST  | Statistics Theory                 |
| 7  | stat.ML  | Machine Learning (Statistics)     |

## Training Procedure

### Base Model

[`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) --
a distilled version of BERT that is 60% faster while retaining 97% of BERT's
language-understanding performance.

### Dataset

[`ccdv/arxiv-classification`](https://huggingface.co/datasets/ccdv/arxiv-classification)
-- a curated collection of arxiv paper abstracts with subject category labels.

### Hyperparameters

| Parameter              | Value  |
|------------------------|--------|
| Learning rate          | 2e-5   |
| LR scheduler           | Linear with warmup |
| Warmup ratio           | 0.1    |
| Weight decay           | 0.01   |
| Epochs                 | 5      |
| Batch size (train)     | 16     |
| Batch size (eval)      | 32     |
| Max sequence length    | 512    |
| Early stopping patience| 3      |
| Seed                   | 42     |

### Metrics

The model is evaluated on accuracy, weighted F1, weighted precision, and
weighted recall. The best checkpoint is selected by weighted F1.

### Measured results

The hyperparameter table above lists the script's defaults. The weights
published to the Hub were trained with a smaller budget than those defaults:
a subsample of **8,000 training / 1,500 evaluation** examples, **3 epochs**,
`--max_length 384`, on Apple MPS. Scores below are on the 1,500 held-out
papers from that run:

| Metric               | Score |
|----------------------|-------|
| Accuracy             | 0.829 |
| F1 (weighted)        | 0.827 |
| Precision (weighted) | 0.828 |
| Recall (weighted)    | 0.829 |
| Eval loss            | 0.576 |

Training on the full corpus for the default 5 epochs at 512 tokens would
likely score higher; these numbers describe the checkpoint you actually
download, not the best achievable configuration.

## How to Use

### With the `transformers` pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="gr8monk3ys/paper-classifier",
)

abstract = (
    "We introduce a new method for neural machine translation that uses "
    "attention mechanisms to align source and target sentences, achieving "
    "state-of-the-art results on WMT benchmarks."
)

result = classifier(abstract)
print(result)
# [{'label': 'cs.NE', 'score': 0.61}]
```

### With the included inference script

```bash
python inference.py \
    --model_path gr8monk3ys/paper-classifier \
    --abstract "We propose a convolutional neural network for image recognition..."
```

### Training from scratch

```bash
pip install -r requirements.txt

python train.py \
    --num_train_epochs 5 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 16 \
    --push_to_hub
```

## Limitations

- The model covers exactly the 11 categories present in
  `ccdv/arxiv-classification`: `cs.AI`, `cs.CE`, `cs.CV`, `cs.DS`, `cs.IT`,
  `cs.NE`, `cs.PL`, `cs.SY`, `math.AC`, `math.GR`, `math.ST`. Papers from any
  other field -- including common ones like `cs.CL` and `cs.LG`, which are
  **not** in the label space -- will be forced into one of these buckets.
- Performance may degrade on abstracts that are unusually short, written in a
  language other than English, or that span multiple subject areas.
- The model inherits any biases present in the DistilBERT base weights and in
  the training dataset.

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{scaturchio2025paperclassifier,
    title  = {Academic Paper Classifier},
    author = {Lorenzo Scaturchio},
    year   = {2025},
    url    = {https://huggingface.co/gr8monk3ys/paper-classifier}
}
```
