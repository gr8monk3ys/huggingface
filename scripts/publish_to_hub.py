#!/usr/bin/env python3
"""
Publish every project in this monorepo to the HuggingFace Hub.

Idempotent: the Hub dedupes by content hash, so re-running only transfers
what actually changed. Requires `hf auth login` (or HF_TOKEN) beforehand.

Usage:
    python scripts/publish_to_hub.py                 # publish everything
    python scripts/publish_to_hub.py --dry-run       # show the plan only
    python scripts/publish_to_hub.py --only spaces   # spaces|models|datasets
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMESPACE = "gr8monk3ys"

# local folder -> Hub repo name (the two differ; the folder carries a
# -space/-model suffix that the Hub repo does not).
SPACES = {
    "code-explainer-space": "code-explainer",
    "dataset-explorer-space": "dataset-explorer",
    "illusion-generator-space": "illusion-generator",
    "ml-interview-space": "ml-interview-prep",
    "model-arena-space": "model-arena",
    "model-selector-space": "model-selector",
    "paper-recommender-space": "paper-recommender",
    "paper-summarizer-space": "paper-summarizer",
    "prompt-enhancer-space": "prompt-enhancer",
    "research-assistant-space": "research-assistant",
    "resume-analyzer-space": "resume-analyzer",
    "style-mixer-space": "style-mixer",
    "trading-dashboard-space": "trading-dashboard",
}

# local folder -> (Hub repo name, dir holding the trained weights)
MODELS = {
    "paper-classifier-model": ("paper-classifier", "model"),
    "resume-section-classifier": ("resume-section-classifier", "model_output/final_model"),
}

# local folder -> (Hub repo name, dir holding the parquet)
DATASETS = {
    "academic-papers-dataset": ("academic-papers-dataset", "data"),
}

# Only these extensions are ever uploaded from a project root. An allowlist
# rather than an ignore-list, so a stray venv or checkpoint cannot leak to
# the Hub just because someone forgot to add it to a skip list.
SCRIPT_EXTS = {".py", ".txt", ".md"}

# Weight/data dirs are uploaded wholesale; these never belong in them.
ARTIFACT_IGNORE = ["__pycache__/*", "*.pyc", ".DS_Store", "optimizer.pt", "scheduler.pt", "rng_state.pth"]


def project_files(folder: Path):
    """Card + scripts + requirements at the project root only (never recursive)."""
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in SCRIPT_EXTS]


def ensure_repo(api, repo_id, repo_type, dry_run):
    """Create the repo if it does not exist yet (no-op when it does)."""
    if dry_run:
        return
    kwargs = {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": True}
    if repo_type == "space":
        kwargs["space_sdk"] = "gradio"
    api.create_repo(**kwargs)


def push_project(api, repo_id, repo_type, folder, dry_run):
    """Upload a project's root-level card/scripts as one atomic commit."""
    from huggingface_hub import CommitOperationAdd

    files = project_files(folder)
    for p in files:
        print(f"  {repo_type}:{repo_id}  <-  {p.relative_to(ROOT)}")
    if dry_run or not files:
        return
    api.create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=[
            CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p)) for p in files
        ],
        commit_message="Sync card and scripts from the monorepo",
    )


def push_artifacts(api, repo_id, repo_type, src: Path, dry_run, path_in_repo=""):
    """Upload a weights or data directory."""
    if not src.is_dir():
        print(f"  !! missing: {src.relative_to(ROOT)} -- skipped")
        return
    dest = path_in_repo or "<root>"
    print(f"  {repo_type}:{repo_id}  <-  {src.relative_to(ROOT)}/ -> {dest}")
    if dry_run:
        return
    api.upload_folder(
        folder_path=str(src),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ARTIFACT_IGNORE,
        commit_message="Upload artifacts from the monorepo",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["spaces", "models", "datasets"])
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    if not args.dry_run:
        try:
            user = api.whoami()["name"]
        except Exception as exc:
            sys.exit(f"Not authenticated ({exc}). Run `hf auth login` first.")
        if user != NAMESPACE:
            sys.exit(f"Logged in as {user!r}, expected {NAMESPACE!r} -- aborting.")

    if args.only in (None, "spaces"):
        print("\n== Spaces ==")
        for folder, repo in SPACES.items():
            repo_id = f"{NAMESPACE}/{repo}"
            ensure_repo(api, repo_id, "space", args.dry_run)
            push_project(api, repo_id, "space", ROOT / folder, args.dry_run)

    if args.only in (None, "models"):
        print("\n== Models ==")
        for folder, (repo, weights) in MODELS.items():
            repo_id = f"{NAMESPACE}/{repo}"
            ensure_repo(api, repo_id, "model", args.dry_run)
            push_project(api, repo_id, "model", ROOT / folder, args.dry_run)
            push_artifacts(api, repo_id, "model", ROOT / folder / weights, args.dry_run)

    if args.only in (None, "datasets"):
        print("\n== Datasets ==")
        for folder, (repo, data_dir) in DATASETS.items():
            repo_id = f"{NAMESPACE}/{repo}"
            ensure_repo(api, repo_id, "dataset", args.dry_run)
            push_project(api, repo_id, "dataset", ROOT / folder, args.dry_run)
            push_artifacts(api, repo_id, "dataset", ROOT / folder / data_dir,
                           args.dry_run, path_in_repo="data")

    print("\nDone." + (" (dry run -- nothing uploaded)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
