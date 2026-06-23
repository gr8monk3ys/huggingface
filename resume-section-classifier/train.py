"""
Resume Section Classifier – Training Script

Fine-tunes distilbert-base-uncased for classifying resume text sections
into 8 categories: education, experience, skills, projects, summary,
certifications, contact, awards.

Author: Lorenzo Scaturchio (gr8monk3ys)

Usage:
    python train.py                           # Train with defaults
    python train.py --epochs 5 --batch-size 32
    python train.py --push-to-hub             # Push to HuggingFace Hub
    python train.py --output-dir ./my_model
"""

import json
import logging
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_generator import generate_dataset, get_label_mapping, load_as_hf_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = "./model_output"
DEFAULT_LOGGING_DIR = "./logs"
HUB_MODEL_ID = "gr8monk3ys/resume-section-classifier"
MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def build_compute_metrics(id2label: dict):
    """Build a compute_metrics function with access to label mappings."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
        f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
        precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
        recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")

        return {
            "accuracy": acc["accuracy"],
            "f1_macro": f1_macro["f1"],
            "f1_weighted": f1_weighted["f1"],
            "precision": precision["precision"],
            "recall": recall["recall"],
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def tokenize_dataset(dataset_dict: DatasetDict, tokenizer, label2id: dict, max_length: int = MAX_LENGTH):
    """Tokenize all splits and encode labels as integers."""

    def preprocess(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding via DataCollator
        )
        tokenized["labels"] = [label2id[label] for label in examples["label"]]
        return tokenized

    tokenized = dataset_dict.map(
        preprocess,
        batched=True,
        remove_columns=["text", "label"],
        desc="Tokenizing",
    )

    return tokenized


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_length: int = MAX_LENGTH,
    examples_per_category: int = 80,
    augmented_copies: int = 2,
    seed: int = 42,
    push_to_hub: bool = False,
    hub_model_id: str = HUB_MODEL_ID,
    fp16: bool = None,
    gradient_accumulation_steps: int = 1,
    early_stopping_patience: int = 3,
):
    """
    Full training pipeline.

    Args:
        output_dir: Directory to save model and artifacts.
        model_name: Pretrained model identifier.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_ratio: Fraction of steps for warmup.
        max_length: Maximum token sequence length.
        examples_per_category: Base synthetic examples per category.
        augmented_copies: Augmented copies per base example.
        seed: Random seed.
        push_to_hub: Whether to push to HuggingFace Hub.
        hub_model_id: Hub model repository ID.
        fp16: Use mixed precision (auto-detected if None).
        gradient_accumulation_steps: Gradient accumulation steps.
        early_stopping_patience: Early stopping patience (epochs).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Auto-detect fp16
    if fp16 is None:
        fp16 = torch.cuda.is_available()

    logger.info("=" * 60)
    logger.info("Resume Section Classifier – Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'}")
    logger.info(f"FP16: {fp16}")

    # ------------------------------------------------------------------
    # 1. Generate synthetic data
    # ------------------------------------------------------------------
    logger.info("\n[1/5] Generating synthetic training data...")
    raw_dataset = generate_dataset(
        examples_per_category=examples_per_category,
        augmented_copies=augmented_copies,
        seed=seed,
    )
    label2id, id2label = get_label_mapping(raw_dataset)
    num_labels = len(label2id)

    logger.info(f"  Total examples: {len(raw_dataset)}")
    logger.info(f"  Labels ({num_labels}): {list(label2id.keys())}")

    # Create HF DatasetDict with train/val/test splits
    dataset_dict = load_as_hf_dataset(raw_dataset)
    logger.info(f"  Train: {len(dataset_dict['train'])}")
    logger.info(f"  Validation: {len(dataset_dict['validation'])}")
    logger.info(f"  Test: {len(dataset_dict['test'])}")

    # ------------------------------------------------------------------
    # 2. Tokenize
    # ------------------------------------------------------------------
    logger.info("\n[2/5] Loading tokenizer and tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_dataset = tokenize_dataset(dataset_dict, tokenizer, label2id, max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    logger.info("\n[3/5] Loading pretrained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    logger.info("\n[4/5] Training...")

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Training hyperparameters
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        # Evaluation
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # Logging
        logging_dir=DEFAULT_LOGGING_DIR,
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        # Efficiency
        fp16=fp16,
        dataloader_num_workers=0,
        # Reproducibility
        seed=seed,
        data_seed=seed,
        # Hub
        push_to_hub=False,  # We'll push manually after evaluation
        # Misc
        save_total_limit=3,
        disable_tqdm=False,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
        callbacks=callbacks,
    )

    train_result = trainer.train()

    # Log training metrics
    logger.info("\nTraining Results:")
    for key, value in train_result.metrics.items():
        logger.info(f"  {key}: {value}")

    # ------------------------------------------------------------------
    # 5. Evaluation
    # ------------------------------------------------------------------
    logger.info("\n[5/5] Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])

    logger.info("\nTest Results:")
    for key, value in test_results.items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    logger.info("\nSaving model and artifacts...")

    # Save model + tokenizer
    final_path = output_path / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    # Save label mapping
    label_mapping = {
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "labels": list(label2id.keys()),
    }
    with open(final_path / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Save training config
    train_config = {
        "model_name": model_name,
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "examples_per_category": examples_per_category,
        "augmented_copies": augmented_copies,
        "seed": seed,
        "num_labels": num_labels,
        "train_size": len(dataset_dict["train"]),
        "val_size": len(dataset_dict["validation"]),
        "test_size": len(dataset_dict["test"]),
    }
    with open(final_path / "training_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    # Save metrics
    all_metrics = {
        "train": train_result.metrics,
        "test": test_results,
    }
    with open(final_path / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nAll artifacts saved to: {final_path}")

    # ------------------------------------------------------------------
    # Optional: Push to Hub
    # ------------------------------------------------------------------
    if push_to_hub:
        logger.info(f"\nPushing to HuggingFace Hub: {hub_model_id}")
        try:
            trainer.push_to_hub(
                repo_id=hub_model_id,
                commit_message="Upload fine-tuned resume section classifier",
            )
            tokenizer.push_to_hub(hub_model_id)
            logger.info("Successfully pushed to Hub!")
        except Exception as e:
            logger.error(f"Failed to push to Hub: {e}")
            logger.info("You can push manually later with:")
            logger.info(f"  huggingface-cli upload {hub_model_id} {final_path}")

    logger.info("\nTraining complete!")
    return test_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for resume section classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & output
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Pretrained model name or path")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for model and artifacts")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Training batch size per device")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for AdamW")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Fraction of total steps for linear warmup")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                        help="Maximum token sequence length")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Number of gradient accumulation steps")

    # Data
    parser.add_argument("--examples-per-category", type=int, default=80,
                        help="Base synthetic examples per category")
    parser.add_argument("--augmented-copies", type=int, default=2,
                        help="Augmented copies per base example")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Training config
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Force FP16 training")
    parser.add_argument("--no-fp16", action="store_true",
                        help="Disable FP16 training")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Early stopping patience (0 to disable)")

    # Hub
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push trained model to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str, default=HUB_MODEL_ID,
                        help="HuggingFace Hub model ID")

    args = parser.parse_args()

    # Handle fp16 flags
    fp16 = args.fp16
    if args.no_fp16:
        fp16 = False

    results = train(
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_length=args.max_length,
        examples_per_category=args.examples_per_category,
        augmented_copies=args.augmented_copies,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        fp16=fp16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
