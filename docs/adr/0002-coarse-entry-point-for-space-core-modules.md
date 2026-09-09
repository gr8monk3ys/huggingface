# 2. A Space's `core.py` exposes a coarse entry point

Date: 2026-09-08

## Status

Accepted

## Context

Four Spaces had logic extracted from `app.py` into a `core.py` before this
decision, and all four landed on the same shape: a block of fine-grained public
functions, with `app.py` importing them by name and doing the sequencing.

That shape produced two very different outcomes.

It worked for `trading-dashboard`. Its four stages thread one DataFrame, the
handler that calls them is six lines, and deleting `core.py` would force
re-deriving an entire backtest state machine. The module is deep.

It failed for `model-selector`. The four helpers moved; the 253-line `TASKS`
dict they operate on stayed in `app.py`, along with the live-then-curated
fallback that decides whether a user sees Hub results or the curated list. The
interface is nearly as wide as the implementation, 85 of its 180 lines are
string templates, and `app.py` re-sorts on `likes` for "Best Quality" —
duplicating the priority concept `rank_curated` owns, with different semantics.
Deleting that `core.py` would move complexity, not concentrate it.

`resume-analyzer` sits in between: a genuinely good module, but a 30-line
orchestration block and four verdict thresholds stayed behind an `import
gradio`, where no test can reach them.

The pattern in the failures is not granularity itself. It is that **sequencing
left outside the interface never gets tested**, because the test suite
deliberately does not import `app.py`.

## Decision

When a Space's logic involves non-trivial sequencing — validate, build, call,
post-process — `core.py` exposes **one coarse entry point** that owns that
sequencing. The steps become underscore-private helpers, tested through the
entry point rather than around it.

Fine-grained public functions remain acceptable where the handler is a short
pipeline that reads as one expression of intent.

Two rules follow from this and apply to every `core.py`:

- **`core.py` returns data; `app.py` owns every string a user reads.** No
  Markdown assembly, no `**bold**`, no score bars in `core.py`. The test is
  whether the value would still make sense to a caller that is not a UI. A
  `matplotlib` `Figure` passes that test and may cross the seam; a Markdown
  blob does not.
- **`core.py` raises domain errors; `app.py` translates them.** `gr.Error` and
  user-facing formatting belong on the `app.py` side of the seam. Domain
  exceptions are defined locally in the `core.py` that raises them, subclassing
  a builtin where one fits, so a caller that does not care can still catch
  generically.

## Consequences

This departs from all four prior extractions, and the repo will carry both
shapes. That is deliberate: `trading-dashboard` and `paper-recommender` already
pass the deletion test, and rewriting a module that works to match a convention
is churn. Uniformity is not the goal — the deletion test is.

The interface is the test surface. Putting sequencing behind it means tests
exercise the thing that actually breaks, at the cost of not being able to reach
individual steps directly. That cost is intended: a step that needs its own
test through the public interface is usually a sign it should be its own module.

Injecting the expensive dependency — the inference client, the encoder, the
similarity function — is what makes the coarse entry point testable at all. A
`core.py` that constructs its own client at import time has no seam, whatever
its function signatures look like.
