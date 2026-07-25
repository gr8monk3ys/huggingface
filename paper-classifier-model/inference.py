#!/usr/bin/env python3
"""
Inference script for the Academic Paper Classifier.

Loads a fine-tuned DistilBERT model and predicts the arxiv category for a
given paper abstract. Returns the predicted category along with per-class
confidence scores.

Usage examples:
    # Use a local model directory
    python inference.py --model_path ./model --abstract "We propose a novel ..."

    # Use a HuggingFace Hub model
    python inference.py --model_path gr8monk3ys/paper-classifier \
                        --abstract "We propose a novel ..."

    # Interactive mode (reads from stdin)
    python inference.py --model_path ./model

Author: Lorenzo Scaturchio (gr8monk3ys)
License: MIT
"""

import argparse
import json
import logging
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
# Classifier wrapper
# ---------------------------------------------------------------------------
class PaperClassifier:
    """Thin wrapper around a fine-tuned sequence-classification model.

    Parameters
    ----------
    model_path : str
        Path to a local model directory **or** a HuggingFace Hub model id.
    device : str | None
        Target device (``"cpu"``, ``"cuda"``, ``"mps"``).  If *None* the best
        available device is selected automatically.
    """

    def __init__(self, model_path: str, device: str | None = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        logger.info("Loading tokenizer from: %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info("Loading model from: %s", model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)

        # Read label mapping stored in the model config
        self.id2label: dict[int, str] = self.model.config.id2label
        logger.info("Labels: %s", list(self.id2label.values()))

    @torch.no_grad()
    def predict(self, abstract: str, top_k: int | None = None) -> dict:
        """Classify a single paper abstract.

        Parameters
        ----------
        abstract : str
            The paper abstract to classify.
        top_k : int | None
            If given, only the *top_k* categories (by confidence) are returned
            in ``scores``.  Pass *None* to return all categories.

        Returns
        -------
        dict
            ``{"label": str, "confidence": float, "scores": {label: prob}}``
        """
        self.model.eval()

        inputs = self.tokenizer(
            abstract,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        sorted_indices = probs.argsort()[::-1]
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]

        scores = {
            self.id2label[int(idx)]: float(probs[idx]) for idx in sorted_indices
        }

        best_idx = int(probs.argmax())
        return {
            "label": self.id2label[best_idx],
            "confidence": float(probs[best_idx]),
            "scores": scores,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify an academic paper abstract into an arxiv category."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./model",
        help="Path to the fine-tuned model directory or HF Hub id (default: %(default)s).",
    )
    parser.add_argument(
        "--abstract",
        type=str,
        default=None,
        help="Paper abstract text.  If omitted, the script enters interactive mode.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Only show the top-k predictions (default: show all).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to run inference on (default: auto-detect).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        dest="output_json",
        help="Output raw JSON instead of human-readable text.",
    )
    return parser.parse_args()


def _print_result(result: dict, output_json: bool) -> None:
    """Pretty-print or JSON-dump a prediction result."""
    if output_json:
        print(json.dumps(result, indent=2))
        return

    print(f"\n  Predicted category : {result['label']}")
    print(f"  Confidence         : {result['confidence']:.4f}")
    print("  ---------------------------------")
    for label, score in result["scores"].items():
        bar = "#" * int(score * 40)
        print(f"  {label:<10s} {score:6.4f}  {bar}")
    print()


def main() -> None:
    args = parse_args()
    classifier = PaperClassifier(model_path=args.model_path, device=args.device)

    if args.abstract is not None:
        result = classifier.predict(args.abstract, top_k=args.top_k)
        _print_result(result, args.output_json)
        return

    # Interactive mode
    print("Academic Paper Classifier - Interactive Mode")
    print("Enter a paper abstract (or 'quit' to exit).")
    print("For multi-line input, end with an empty line.\n")

    while True:
        try:
            lines: list[str] = []
            prompt = "abstract> " if sys.stdin.isatty() else ""
            while True:
                line = input(prompt)
                if line.strip().lower() == "quit":
                    logger.info("Exiting.")
                    return
                if line == "" and lines:
                    break
                lines.append(line)
                prompt = "... " if sys.stdin.isatty() else ""

            abstract = " ".join(lines).strip()
            if not abstract:
                continue

            result = classifier.predict(abstract, top_k=args.top_k)
            _print_result(result, args.output_json)

        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Exiting.")
            return


if __name__ == "__main__":
    main()
