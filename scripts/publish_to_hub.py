#!/usr/bin/env python3
"""
Publish every project in this monorepo to the HuggingFace Hub.

Idempotent: the Hub dedupes by content hash, so re-running only transfers
what actually changed. Publishing requires `hf auth login` (or HF_TOKEN).

Usage:
    python scripts/publish_to_hub.py                 # publish everything
    python scripts/publish_to_hub.py --dry-run       # show the plan only
    python scripts/publish_to_hub.py --only spaces   # spaces|models|datasets

Deciding what to upload and uploading it are separate: :func:`plan` is pure and
imports nothing from ``huggingface_hub``, so ``--dry-run`` works with the
library absent, and the plan can be asserted on without a network.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
    "resume-section-classifier": (
        "resume-section-classifier",
        "model_output/final_model",
    ),
}

# local folder -> (Hub repo name, dir holding the parquet)
DATASETS = {
    "academic-papers-dataset": ("academic-papers-dataset", "data"),
}

# Top-level directories that are deliberately not published. Anything else
# unmapped is an error rather than a silent skip: nothing used to reconcile
# these tables against the filesystem, so a new or renamed project folder was
# simply never uploaded, with no warning printed and no failure raised.
NOT_PUBLISHED = {
    "docs",
    "scripts",
    "tests",
}

# Only these extensions are ever uploaded from a project root. An allowlist
# rather than an ignore-list, so a stray venv or checkpoint cannot leak to
# the Hub just because someone forgot to add it to a skip list.
# .html/.css/.js are here for static Spaces (sdk: static), which ship a page
# instead of an app.py.
SCRIPT_EXTS = {".py", ".txt", ".md", ".html", ".css", ".js"}

# Weight/data dirs are uploaded wholesale; these never belong in them.
#
# The .arrow entries are not housekeeping. A dataset repo containing Arrow IPC
# files is refused by the Hub's dataset viewer -- "Datasets with Arrow IPC files
# are temporarily unavailable in the dataset viewer" -- so uploading a
# save_to_disk() dump alongside the parquet silently kills the preview on the
# dataset page. The parquet is the published format; the arrow dump is a local
# build artifact that happens to live in the same directory.
ARTIFACT_IGNORE = [
    "__pycache__/*",
    "*.pyc",
    ".DS_Store",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "*.arrow",
    "*hf_dataset*",
]


class UnmappedProjectError(RuntimeError):
    """A project folder is neither mapped to a Hub repo nor opted out."""


@dataclass(frozen=True)
class Upload:
    """One repo's worth of work: what goes where, decided but not yet done."""

    repo_id: str
    repo_type: str  # "space" | "model" | "dataset"
    folder: Path
    files: tuple = ()
    artifacts: Optional[Path] = None
    artifacts_path_in_repo: str = ""

    @property
    def artifacts_missing(self) -> bool:
        return self.artifacts is not None and not self.artifacts.is_dir()


@dataclass
class Plan:
    """Everything a run would do, and what it noticed on the way."""

    uploads: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def project_files(folder: Path):
    """Card + scripts + requirements at the project root only (never recursive).

    Non-recursive on purpose, and the reason a Space cannot import from outside
    its own folder -- see docs/adr/0001-vendoring-is-the-only-sharing-mechanism.md.
    """
    if not folder.is_dir():
        return []
    return [
        p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in SCRIPT_EXTS
    ]


def published_folders() -> set:
    """Every folder name the three tables claim."""
    return set(SPACES) | set(MODELS) | set(DATASETS)


def check_reconciled(root: Path = ROOT) -> None:
    """Fail if a project folder is neither mapped nor explicitly opted out.

    Raises:
        UnmappedProjectError: listing what is unmapped, or mapped but missing.
    """
    on_disk = {
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    }
    mapped = published_folders()

    unmapped = sorted(on_disk - mapped - NOT_PUBLISHED)
    missing = sorted(mapped - on_disk)

    problems = []
    if unmapped:
        problems.append(
            "not mapped to a Hub repo and not in NOT_PUBLISHED: " + ", ".join(unmapped)
        )
    if missing:
        problems.append("mapped but absent from disk: " + ", ".join(missing))

    if problems:
        raise UnmappedProjectError("; ".join(problems))


def plan(root: Path = ROOT, only: Optional[str] = None) -> Plan:
    """Decide what a run would upload. Pure: no network, no huggingface_hub.

    Raises:
        UnmappedProjectError: the tables and the filesystem disagree.
    """
    check_reconciled(root)
    result = Plan()

    def add(folder, repo, repo_type, artifacts=None, path_in_repo=""):
        upload = Upload(
            repo_id=f"{NAMESPACE}/{repo}",
            repo_type=repo_type,
            folder=root / folder,
            files=tuple(project_files(root / folder)),
            artifacts=(root / folder / artifacts) if artifacts else None,
            artifacts_path_in_repo=path_in_repo,
        )
        if not upload.files:
            result.warnings.append(f"{upload.repo_id}: no uploadable files at root")
        if upload.artifacts_missing:
            result.warnings.append(
                f"{upload.repo_id}: missing {upload.artifacts.relative_to(root)}/"
            )
        result.uploads.append(upload)

    if only in (None, "spaces"):
        for folder, repo in SPACES.items():
            add(folder, repo, "space")

    if only in (None, "models"):
        for folder, (repo, weights) in MODELS.items():
            add(folder, repo, "model", artifacts=weights)

    if only in (None, "datasets"):
        for folder, (repo, data_dir) in DATASETS.items():
            add(folder, repo, "dataset", artifacts=data_dir, path_in_repo="data")

    return result


def describe(plan_result: Plan, root: Path = ROOT) -> str:
    """Render a plan for a human. No side effects, so --dry-run is just this."""
    lines = []
    current_type = None
    for upload in plan_result.uploads:
        if upload.repo_type != current_type:
            current_type = upload.repo_type
            lines.append(f"\n== {current_type.capitalize()}s ==")
        for path in upload.files:
            lines.append(
                f"  {upload.repo_type}:{upload.repo_id}  <-  {path.relative_to(root)}"
            )
        if upload.artifacts is not None:
            if upload.artifacts_missing:
                lines.append(
                    f"  !! missing: {upload.artifacts.relative_to(root)} -- skipped"
                )
            else:
                dest = upload.artifacts_path_in_repo or "<root>"
                lines.append(
                    f"  {upload.repo_type}:{upload.repo_id}  <-  "
                    f"{upload.artifacts.relative_to(root)}/ -> {dest}"
                )
    for warning in plan_result.warnings:
        lines.append(f"  !! {warning}")
    return "\n".join(lines)


def ensure_repo(api, repo_id, repo_type) -> bool:
    """Make sure the repo exists. Returns False if it can't be used.

    Checks existence before attempting creation: ``create_repo`` still POSTs to
    /api/repos/create even with ``exist_ok=True``, and the Hub now answers that
    with 402 for Gradio Spaces on the free tier -- so calling it unconditionally
    fails on repos that are already there and perfectly writable.

    A repo that genuinely cannot be created is reported and skipped rather than
    aborting the run, so one paywalled Space doesn't block the other projects.
    """
    from huggingface_hub.errors import HfHubHTTPError

    if api.repo_exists(repo_id=repo_id, repo_type=repo_type):
        return True

    kwargs = {"repo_id": repo_id, "repo_type": repo_type}
    if repo_type == "space":
        kwargs["space_sdk"] = "gradio"
    try:
        api.create_repo(**kwargs)
        print(f"  {repo_type}:{repo_id}  created")
        return True
    except HfHubHTTPError as exc:
        first_line = str(exc).split("\n")[0]
        print(f"  !! cannot create {repo_type}:{repo_id} -- skipped ({first_line})")
        return False


def execute(api, upload: Upload) -> None:
    """Perform one planned Upload."""
    from huggingface_hub import CommitOperationAdd

    if not ensure_repo(api, upload.repo_id, upload.repo_type):
        return

    if upload.files:
        api.create_commit(
            repo_id=upload.repo_id,
            repo_type=upload.repo_type,
            operations=[
                CommitOperationAdd(path_in_repo=p.name, path_or_fileobj=str(p))
                for p in upload.files
            ],
            commit_message="Sync card and scripts from the monorepo",
        )

    if upload.artifacts is not None and not upload.artifacts_missing:
        api.upload_folder(
            folder_path=str(upload.artifacts),
            path_in_repo=upload.artifacts_path_in_repo,
            repo_id=upload.repo_id,
            repo_type=upload.repo_type,
            ignore_patterns=ARTIFACT_IGNORE,
            commit_message="Upload artifacts from the monorepo",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["spaces", "models", "datasets"])
    args = ap.parse_args()

    try:
        result = plan(ROOT, args.only)
    except UnmappedProjectError as exc:
        sys.exit(f"Publish plan is out of sync with the repo: {exc}")

    print(describe(result, ROOT))

    if args.dry_run:
        print("\nDone. (dry run -- nothing uploaded)")
        return

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, LocalTokenNotFoundError

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except (HfHubHTTPError, LocalTokenNotFoundError) as exc:
        sys.exit(f"Not authenticated ({exc}). Run `hf auth login --force` first.")
    if user != NAMESPACE:
        sys.exit(f"Logged in as {user!r}, expected {NAMESPACE!r} -- aborting.")

    for upload in result.uploads:
        execute(api, upload)

    print("\nDone.")


if __name__ == "__main__":
    main()
