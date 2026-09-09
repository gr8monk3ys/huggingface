# 1. Vendoring is the only way Spaces share code

Date: 2026-09-08

## Status

Accepted

## Context

The 16 projects in this repo are published independently to the HuggingFace
Hub. A Space is uploaded as a flat folder, and `scripts/publish_to_hub.py`
decides what goes in it:

```python
def project_files(folder: Path):
    """Card + scripts + requirements at the project root only (never recursive)."""
    return [
        p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in SCRIPT_EXTS
    ]
```

`iterdir()` is not recursive and `p.is_file()` excludes directories outright, so
only files sitting directly beside `app.py` are uploaded. The one code path that
uploads a tree — `push_artifacts`, which calls `upload_folder` — runs for
`MODELS` and `DATASETS` only, never for Spaces.

Three further facts close off the alternatives:

- `pyproject.toml` has no `[project]` table, so there is no installable package.
- No Space manipulates `sys.path`. The only such code in the repo is
  `tests/conftest.py`, which exists precisely to work around this.
- `hf_client.py` resolves at Space runtime *because* it sits flat beside
  `app.py`. That is not an accident of style; it is the only arrangement that
  imports.

So a root-level `shared/` package cannot reach the Hub, and a Space importing
from one would `ImportError` on startup. The Space would look fine locally,
where the repo root is on the path, and fail only once deployed.

## Decision

Code shared between two or more Spaces is **vendored**: a byte-identical copy
in each Space folder, guarded by a test that discovers the copies by glob and
asserts they hash to a single digest.

We do not create a root-level shared package for Space code, and we do not add
`sys.path` manipulation to a Space to simulate one.

The threshold for vendoring is **two or more real callers**. One caller is a
hypothetical seam; copying a module for it buys duplication and a guard in
exchange for nothing.

## Consequences

Vendoring is duplication, and duplication drifts. The guard is what makes it
survivable, and the guard has already earned its place: the comment in
`tests/test_hf_client.py` records that a hardcoded list of copies silently
excluded a new Space, which is how an unguarded copy shipped. Copy discovery
must stay glob-driven for exactly that reason.

A change to a vendored module has to touch every copy in the same commit, or
the guard fails. That is the intended cost — it is cheaper than the failure it
replaces, which is two copies quietly disagreeing about behaviour while both
are covered by their own passing tests.

Where this rule does *not* apply: `tests/` and `scripts/` are never uploaded,
so anything there can import freely across the repo.
