# Domain glossary

Terms this repo uses with a specific meaning, where the ordinary reading would
be wrong or ambiguous.

This file is grown lazily. A term is added when a decision actually turns on it,
not in advance — so its absence from this list means a term has not yet caused
a misunderstanding, never that it is unimportant. Architecture vocabulary
(module, interface, seam, depth) lives in `docs/adr/`, not here.

---

**Space** — one of the 13 published HuggingFace Spaces, each a folder uploaded
to the Hub as a flat set of root-level files. A Space folder is a publishing
unit, not a package: see [ADR-0001](docs/adr/0001-vendoring-is-the-only-sharing-mechanism.md)
for what that forbids.

**Static Space** — a Space with no server (`sdk: static`, an `index.html`, no
`requirements.txt`). `ml-interview-space` is the only one. Being serverless
makes it free and always awake, so it is the preferred shape for any Space that
does not call a model — a sleeping Gradio Space serves nobody.

**Core module** — the `core.py` beside a Space's `app.py`, holding the logic
that is not UI. Its shape is fixed by
[ADR-0002](docs/adr/0002-coarse-entry-point-for-space-core-modules.md): it
returns data, raises domain errors, and accepts its expensive dependencies
rather than constructing them.

**Vendored module** — a module deliberately duplicated byte-for-byte across
several Space folders, because a Space cannot import from outside its own
folder. `hf_client.py` is the canonical example. Copies are discovered by glob
and asserted to hash identically; a vendored module is never edited in one
place. See [ADR-0001](docs/adr/0001-vendoring-is-the-only-sharing-mechanism.md).

**Classified error** — an `InferenceError` raised by `hf_client`, carrying a
message already sorted into permanent (credits exhausted, bad token) or
transient (cold start, rate limit). The distinction is the module's whole
purpose, so catching it and substituting a generic message — or worse, a
plausible-looking fallback value — throws away the thing that was paid for.

**Fallback corpus** — the small hardcoded set of papers a Space falls back to
when the live dataset cannot be loaded. It is a degraded mode, not a fixture:
users see it without being told, so its contents are a product surface.

**Scored section** — one of the **five** resume sections `resume-analyzer-space`
recognises and scores against a job description (summary, experience, education,
skills, projects). Precision-tuned: a heading it cannot place is left unscored.

**Section label** — one of the **eight** classes
`resume-section-classifier` predicts, the five scored sections plus
certifications, contact and awards. Recall-tuned: its splitter is deliberately
permissive, because a chunk it fails to split never reaches the model.

> Scored sections and section labels are *different concepts that share a word*.
> The two projects are uncoupled and stay that way; see
> [ADR-0003](docs/adr/0003-resume-section-concepts-stay-separate.md).
