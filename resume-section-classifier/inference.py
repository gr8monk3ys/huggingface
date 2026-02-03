"""
Resume Section Classifier – Inference Script

Takes raw resume text, splits it into sections, and classifies each section
into one of 8 categories with confidence scores.

Author: Lorenzo Scaturchio (gr8monk3ys)

Usage:
    # Classify a resume file
    python inference.py --file resume.txt

    # Classify inline text
    python inference.py --text "Bachelor of Science in Computer Science, MIT, 2023"

    # Use a custom model path
    python inference.py --model ./model_output/final_model --file resume.txt

    # Output as JSON
    python inference.py --file resume.txt --format json

    # Python API
    from inference import ResumeSectionClassifier
    classifier = ResumeSectionClassifier("./model_output/final_model")
    results = classifier.classify_resume(resume_text)
"""

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SectionPrediction:
    """A single section classification result."""
    text: str
    label: str
    confidence: float
    all_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
        }


@dataclass
class ResumeAnalysis:
    """Complete resume analysis output."""
    sections: list
    section_count: int = 0
    label_distribution: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "section_count": self.section_count,
            "label_distribution": self.label_distribution,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Resume Analysis: {self.section_count} sections detected",
            "=" * 50,
        ]
        for i, sec in enumerate(self.sections, 1):
            text_preview = sec.text[:80].replace("\n", " ")
            if len(sec.text) > 80:
                text_preview += "..."
            lines.append(
                f"\n[{i}] {sec.label.upper()} (confidence: {sec.confidence:.1%})"
            )
            lines.append(f"    {text_preview}")

        lines.append("\n" + "-" * 50)
        lines.append("Label Distribution:")
        for label, count in sorted(self.label_distribution.items()):
            lines.append(f"  {label}: {count}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section splitting heuristics
# ---------------------------------------------------------------------------

# Common resume section headers (case-insensitive patterns)
SECTION_HEADER_PATTERNS = [
    r"^#{1,3}\s+.+$",  # Markdown headers
    r"^[A-Z][A-Z\s&/,]{2,}$",  # ALL CAPS headers
    r"^(?:EDUCATION|EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|"
    r"SKILLS|TECHNICAL SKILLS|PROJECTS|PERSONAL PROJECTS|"
    r"SUMMARY|PROFESSIONAL SUMMARY|OBJECTIVE|PROFILE|ABOUT|"
    r"CERTIFICATIONS|CERTIFICATES|LICENSES|"
    r"CONTACT|CONTACT INFORMATION|PERSONAL INFORMATION|"
    r"AWARDS|HONORS|ACHIEVEMENTS|RECOGNITION|"
    r"PUBLICATIONS|REFERENCES|VOLUNTEER|LANGUAGES|INTERESTS|"
    r"ACTIVITIES|LEADERSHIP|RESEARCH)\s*:?\s*$",
]

COMPILED_HEADERS = [re.compile(p, re.MULTILINE | re.IGNORECASE) for p in SECTION_HEADER_PATTERNS]


def is_section_header(line: str) -> bool:
    """Check if a line looks like a section header."""
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False

    for pattern in COMPILED_HEADERS:
        if pattern.match(stripped):
            return True

    # Heuristic: short all-caps line
    if stripped.isupper() and len(stripped.split()) <= 5 and len(stripped) < 50:
        return True

    return False


def split_resume_into_sections(text: str, min_section_length: int = 20) -> list:
    """
    Split raw resume text into logical sections.

    Strategy:
    1. First try to split on detected section headers.
    2. Fall back to splitting on double newlines (paragraph breaks).
    3. Filter out very short fragments.

    Args:
        text: Raw resume text.
        min_section_length: Minimum character length for a section.

    Returns:
        List of text sections.
    """
    lines = text.split("\n")
    sections = []
    current_section_lines = []

    # Pass 1: Try header-based splitting
    header_found = False
    for line in lines:
        if is_section_header(line):
            header_found = True
            # Save previous section
            if current_section_lines:
                section_text = "\n".join(current_section_lines).strip()
                if len(section_text) >= min_section_length:
                    sections.append(section_text)
            current_section_lines = [line]
        else:
            current_section_lines.append(line)

    # Don't forget the last section
    if current_section_lines:
        section_text = "\n".join(current_section_lines).strip()
        if len(section_text) >= min_section_length:
            sections.append(section_text)

    # If no headers found, fall back to paragraph splitting
    if not header_found or len(sections) <= 1:
        sections = []
        paragraphs = re.split(r"\n\s*\n", text)
        for para in paragraphs:
            stripped = para.strip()
            if len(stripped) >= min_section_length:
                sections.append(stripped)

    # If still just one big block, return it as-is
    if not sections:
        stripped = text.strip()
        if stripped:
            sections = [stripped]

    return sections


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ResumeSectionClassifier:
    """
    Classifies resume text sections into categories.

    Supports both single-section and full-resume classification.
    """

    def __init__(
        self,
        model_path: str = "./model_output/final_model",
        device: str = None,
        max_length: int = 256,
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to fine-tuned model directory.
            device: Device string ('cpu', 'cuda', 'mps'). Auto-detected if None.
            max_length: Maximum token sequence length.
        """
        self.model_path = Path(model_path)
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path)
        ).to(self.device)
        self.model.eval()

        # Load label mapping
        label_mapping_path = self.model_path / "label_mapping.json"
        if label_mapping_path.exists():
            with open(label_mapping_path) as f:
                mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]
        else:
            # Fall back to model config
            self.id2label = self.model.config.id2label
            self.label2id = self.model.config.label2id

        self.labels = sorted(self.label2id.keys())

    def classify_section(self, text: str) -> SectionPrediction:
        """
        Classify a single text section.

        Args:
            text: Section text to classify.

        Returns:
            SectionPrediction with label, confidence, and all scores.
        """
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        scores = {self.id2label[i]: probs[i].item() for i in range(len(probs))}
        predicted_id = probs.argmax().item()
        predicted_label = self.id2label[predicted_id]
        confidence = probs[predicted_id].item()

        return SectionPrediction(
            text=text,
            label=predicted_label,
            confidence=confidence,
            all_scores=scores,
        )

    def classify_sections(self, texts: list) -> list:
        """
        Classify multiple text sections (batched).

        Args:
            texts: List of section texts.

        Returns:
            List of SectionPrediction objects.
        """
        if not texts:
            return []

        inputs = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

        results = []
        for i, text in enumerate(texts):
            scores = {self.id2label[j]: probs[i][j].item() for j in range(probs.shape[1])}
            predicted_id = probs[i].argmax().item()
            predicted_label = self.id2label[predicted_id]
            confidence = probs[i][predicted_id].item()

            results.append(SectionPrediction(
                text=text,
                label=predicted_label,
                confidence=confidence,
                all_scores=scores,
            ))

        return results

    def classify_resume(
        self,
        resume_text: str,
        min_section_length: int = 20,
    ) -> ResumeAnalysis:
        """
        Classify a full resume by splitting into sections and classifying each.

        Args:
            resume_text: Full resume text.
            min_section_length: Minimum section length in characters.

        Returns:
            ResumeAnalysis with all section predictions.
        """
        sections = split_resume_into_sections(resume_text, min_section_length)
        predictions = self.classify_sections(sections)

        # Compute label distribution
        label_dist = {}
        for pred in predictions:
            label_dist[pred.label] = label_dist.get(pred.label, 0) + 1

        return ResumeAnalysis(
            sections=predictions,
            section_count=len(predictions),
            label_distribution=label_dist,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify resume sections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --file resume.txt
  python inference.py --text "BS in Computer Science, MIT, 2023"
  python inference.py --file resume.txt --format json
  python inference.py --model ./model_output/final_model --file resume.txt
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--file", type=str, help="Path to resume text file")
    input_group.add_argument("--text", type=str, help="Direct text to classify")

    parser.add_argument("--model", type=str, default="./model_output/final_model",
                        help="Path to fine-tuned model (default: ./model_output/final_model)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, mps (auto-detected if omitted)")
    parser.add_argument("--max-length", type=int, default=256,
                        help="Maximum token sequence length (default: 256)")
    parser.add_argument("--min-section-length", type=int, default=20,
                        help="Minimum section length in characters (default: 20)")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text",
                        help="Output format (default: text)")
    parser.add_argument("--single", action="store_true",
                        help="Classify as single section (no splitting)")

    args = parser.parse_args()

    # Load classifier
    try:
        classifier = ResumeSectionClassifier(
            model_path=args.model,
            device=args.device,
            max_length=args.max_length,
        )
    except Exception as e:
        print(f"Error loading model from '{args.model}': {e}", file=sys.stderr)
        print("Have you trained the model yet? Run: python train.py", file=sys.stderr)
        sys.exit(1)

    # Get input text
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        text = file_path.read_text(encoding="utf-8")
    else:
        text = args.text

    # Classify
    if args.single:
        result = classifier.classify_section(text)
        if args.format == "json":
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.1%}")
            print("\nAll scores:")
            for label, score in sorted(result.all_scores.items(), key=lambda x: -x[1]):
                bar = "#" * int(score * 40)
                print(f"  {label:20s} {score:.4f} {bar}")
    else:
        analysis = classifier.classify_resume(text, min_section_length=args.min_section_length)
        if args.format == "json":
            print(analysis.to_json())
        else:
            print(analysis.summary())


if __name__ == "__main__":
    main()
