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

## How to Use

### With the `transformers` pipeline

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="gr8monk3ys/paper-classifier-model",
)

abstract = (
    "We introduce a new method for neural machine translation that uses "
    "attention mechanisms to align source and target sentences, achieving "
    "state-of-the-art results on WMT benchmarks."
)

result = classifier(abstract)
print(result)
# [{'label': 'cs.CL', 'score': 0.95}]
```

### With the included inference script

```bash
python inference.py \
    --model_path gr8monk3ys/paper-classifier-model \
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

- The model only covers a fixed set of 8 arxiv categories. Papers from other
  fields will be forced into one of these buckets.
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
    url    = {https://huggingface.co/gr8monk3ys/paper-classifier-model}
}
```
