#!/usr/bin/env python3
"""
Fine-tune DistilBERT for academic paper abstract classification.

This script downloads arxiv paper abstracts, preprocesses them, and fine-tunes
a DistilBERT model for multi-class sequence classification. Supports pushing
the trained model to the HuggingFace Hub.

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import ClassLabel, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DATASET = "ccdv/arxiv-classification"
DEFAULT_OUTPUT_DIR = "./results"
DEFAULT_MODEL_DIR = "./model"

# Canonical label order so the id<->label mapping is deterministic.
LABEL_NAMES = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.LG",
    "cs.NE",
    "cs.RO",
    "math.ST",
    "stat.ML",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training hyperparameters."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on arxiv paper classification."
    )

    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET,
        help="HuggingFace dataset identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for the tokenizer (default: %(default)s).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap the number of training samples (useful for debugging).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Cap the number of evaluation samples (useful for debugging).",
    )

    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for training checkpoints (default: %(default)s).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Directory where the final model is saved (default: %(default)s).",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Total number of training epochs (default: %(default)s).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per device during training (default: %(default)s).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size per device during evaluation (default: %(default)s).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate (default: %(default)s).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient (default: %(default)s).",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total steps used for linear warmup (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: %(default)s).",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement before stopping (default: %(default)s).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use mixed-precision (FP16) training.",
    )

    # Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push the trained model to the HuggingFace Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="gr8monk3ys/paper-classifier-model",
        help="Repository id on the HuggingFace Hub (default: %(default)s).",
    )

    return parser.parse_args()


def build_label_mappings(label_names: list[str]) -> tuple[dict, dict]:
    """Return (label2id, id2label) dicts for the given label names."""
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for idx, label in enumerate(label_names)}
    return label2id, id2label


def load_and_prepare_dataset(
    dataset_name: str,
    label2id: dict[str, int],
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    """Load the dataset and normalise the label column.

    The function handles two common dataset layouts:
      1. The dataset already has train / validation / test splits and a
         numeric ``label`` column whose values match our ``label2id``.
      2. The dataset has a string ``label`` column that needs mapping.

    Returns a ``DatasetDict`` with ``train`` and ``validation`` splits.
    """
    logger.info("Loading dataset: %s", dataset_name)
    raw = load_dataset(dataset_name, trust_remote_code=True)

    # Determine the text and label column names --------------------------
    sample_columns = list(next(iter(raw.values())).column_names)
    text_col = None
    for candidate in ("text", "abstract", "input", "sentence"):
        if candidate in sample_columns:
            text_col = candidate
            break
    if text_col is None:
        # Fall back to the first string-typed column
        text_col = sample_columns[0]
    logger.info("Using text column: '%s'", text_col)

    label_col = None
    for candidate in ("label", "labels", "category", "class"):
        if candidate in sample_columns:
            label_col = candidate
            break
    if label_col is None:
        label_col = sample_columns[-1]
    logger.info("Using label column: '%s'", label_col)

    # Rename columns so downstream code can rely on 'text' and 'label' ---
    def _rename(example):
        return {"text": str(example[text_col]), "label": example[label_col]}

    raw = raw.map(_rename, remove_columns=sample_columns)

    # If labels are strings, map them to ints using label2id -------------
    sample_label = raw[list(raw.keys())[0]][0]["label"]
    if isinstance(sample_label, str):
        logger.info("Mapping string labels to integer ids.")

        def _map_label(example):
            lbl = example["label"]
            if lbl in label2id:
                example["label"] = label2id[lbl]
            else:
                example["label"] = -1  # will be filtered out
            return example

        raw = raw.map(_map_label)
        raw = raw.filter(lambda ex: ex["label"] != -1)

    # Ensure we have a ClassLabel feature --------------------------------
    label_feature = ClassLabel(
        num_classes=len(label2id), names=list(label2id.keys())
    )
    raw = raw.cast_column("label", label_feature)

    # Build train / validation splits ------------------------------------
    if "validation" not in raw and "test" in raw:
        raw["validation"] = raw.pop("test")
    elif "validation" not in raw:
        split = raw["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
        raw = DatasetDict({"train": split["train"], "validation": split["test"]})

    # Subsample if requested ---------------------------------------------
    if max_train_samples is not None:
        raw["train"] = raw["train"].select(range(min(max_train_samples, len(raw["train"]))))
    if max_eval_samples is not None:
        raw["validation"] = raw["validation"].select(
            range(min(max_eval_samples, len(raw["validation"])))
        )

    logger.info(
        "Dataset sizes -> train: %d, validation: %d",
        len(raw["train"]),
        len(raw["validation"]),
    )
    return raw


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int,
) -> DatasetDict:
    """Tokenize the ``text`` column using the supplied tokenizer."""

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    logger.info("Tokenizing dataset (max_length=%d) ...", max_length)
    tokenized = dataset.map(_tokenize, batched=True, desc="Tokenizing")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def build_compute_metrics_fn():
    """Return a ``compute_metrics`` callable for the HF Trainer.

    Loads the ``accuracy``, ``f1``, ``precision`` and ``recall`` evaluate
    metrics once at creation time to avoid repeated disk access.
    """
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    prec_metric = evaluate.load("precision")
    rec_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        results = {}
        results.update(acc_metric.compute(predictions=predictions, references=labels))
        results.update(
            f1_metric.compute(
                predictions=predictions, references=labels, average="weighted"
            )
        )
        results.update(
            prec_metric.compute(
                predictions=predictions, references=labels, average="weighted"
            )
        )
        results.update(
            rec_metric.compute(
                predictions=predictions, references=labels, average="weighted"
            )
        )
        return results

    return compute_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)
    logger.info("Seed set to %d", args.seed)

    # Device info
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Label mappings
    label2id, id2label = build_label_mappings(LABEL_NAMES)
    num_labels = len(LABEL_NAMES)
    logger.info("Number of labels: %d", num_labels)

    # Dataset
    dataset = load_and_prepare_dataset(
        dataset_name=args.dataset_name,
        label2id=label2id,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    # Tokenizer
    logger.info("Loading tokenizer: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, args.max_length)

    # Model
    logger.info("Loading model: %s", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
        push_to_hub=False,  # we push manually after training
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics_fn(),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    # Train
    logger.info("Starting training ...")
    train_result = trainer.train()
    logger.info("Training complete.")

    # Log final training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    logger.info("Running final evaluation ...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    # Save model + tokenizer
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", model_dir)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Push to Hub
    if args.push_to_hub:
        logger.info("Pushing model to HuggingFace Hub: %s", args.hub_model_id)
        try:
            model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
            logger.info("Model pushed successfully.")
        except Exception:
            logger.exception("Failed to push model to Hub.")
            sys.exit(1)

    logger.info("All done.")


if __name__ == "__main__":
    main()
