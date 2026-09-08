# 3. The two "resume section" concepts stay separate

Date: 2026-09-08

## Status

Accepted

## Context

Two projects in this repo split a resume into sections, and an architecture
review flagged them as one domain concept implemented twice. On inspection they
are not.

`resume-analyzer-space` has `detect_sections`, which returns a
`dict[str, str]` keyed by **five** canonical names, driven by a synonym lookup
table with a `MAX_HEADER_CHARS` guard so prose is not mistaken for a heading.
Its purpose is to score each section against a job description; only the five
scored sections matter, and a section it cannot classify is simply not scored.

`resume-section-classifier` has `split_resume_into_sections`, which returns a
`list[str]` of raw chunks fed to a DistilBERT classifier over **eight** labels —
the analyzer's five plus `certifications`, `contact` and `awards`. Its splitter
is deliberately permissive, using a broader regex set and a three-tier fallback,
because a chunk it fails to split is a chunk the model never sees.

The two are entirely uncoupled, verified rather than assumed:
`resume-analyzer-space` contains no reference to the classifier, its
`requirements.txt` has no `transformers` and no model id, and
`scripts/publish_to_hub.py` maps them to different Hub repo *types* — one a
Space, one a Model.

So the label sets differ, the return types differ, the splitting heuristics
differ, and each is correct for its own job. They share a word.

## Decision

The two stay separate. No shared module, no unified label set.

`CONTEXT.md` names them distinctly — **scored section** for the analyzer's five,
**section label** for the classifier's eight — so the collision stops being
invisible in prose and code review.

## Consequences

A future architecture review will notice the shared word again and propose
merging them; that is exactly how this ADR came to be written. This document is
the answer, and the distinct names in `CONTEXT.md` are the cheaper first line of
defence.

Unifying them remains a legitimate *product* decision. Adopting the classifier's
eight labels would change what the analyzer scores and what a user sees, so it
belongs in a discussion about the product, not in a refactor justified by
removing duplication. Nothing here forecloses it.

The cost of separation is that a genuine improvement to section splitting has to
be made twice, or deliberately made in one place only. Given that the two
heuristics are tuned for opposite failure modes — precision for scoring,
recall for classification — an improvement to one is not usually an improvement
to the other.
