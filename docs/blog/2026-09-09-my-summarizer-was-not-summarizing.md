# My paper summarizer wasn't summarizing anything

For some unknown number of weeks, my Paper Summarizer Space took a research
paper, ignored it completely, and returned the first hundred words of whatever
you pasted in. With a heading that said "Concise Summary" above it.

It looked fine. That's the part I keep thinking about.

## What broke

The Space calls `bart-large-cnn` through the HuggingFace Inference API. The
call looked like this:

```python
result = with_retry(
    client.summarization,
    text,
    parameters={"max_length": max_len, "min_length": min_len, "do_sample": False},
)
```

`InferenceClient.summarization` has never accepted a kwarg called `parameters`.
In `huggingface_hub` 0.25 it was there. In 0.30 it got renamed to
`generate_parameters`. My `requirements.txt` said:

```
huggingface_hub>=0.25.0,<1.0.0
```

So when the Space rebuilt at some point, it resolved to something past 0.30, and
every summarization call started raising `TypeError`. Not sometimes. Every call,
every time.

## Why nobody noticed

Because of this, sitting directly underneath:

```python
except Exception as e:
    logger.warning("Summarization failed: %s", e)
    # Fallback: return truncated text
    return " ".join(text.split()[:100]) + "..."
```

I wrote that. I remember the reasoning: a summarizer that returns *something*
beats a summarizer that throws a stack trace at a user. Degrade gracefully.

The problem is that the graceful degradation produced output indistinguishable
from success. The first hundred words of a paper's abstract read like a summary
of the paper. They're in the right register, roughly the right length, on the
right topic. If you weren't holding the original next to it, you would not
catch it. I didn't, and it's my Space.

A crash would have told me in about four seconds.

## Why the tests didn't catch it either

My test suite passes. It passed the whole time this was broken.

The suite deliberately doesn't import any `app.py`, because doing so would pull
in Gradio, torch and PyMuPDF for every test run. Reasonable tradeoff, and I'd
probably make it again. But it means the tests could never reach the one line
that mattered, and I had mistaken "the tests are green" for "the thing works."

Those are different claims. I knew that abstractly. I still shipped this.

## Finding it

It turned up during a refactor. I was moving the summarization logic out of
`app.py` into a `core.py` so it could actually be tested, and I had a rule for
myself on that piece of work: every change gets run for real, not just
type-checked and unit-tested. Boot the Space. Put a paper through it. Look at
what comes out.

First real call:

```
**Error:** Inference failed: InferenceClient.summarization()
           got an unexpected keyword argument 'parameters'
```

There it was. The refactor didn't cause the bug. It removed the thing that was
hiding it, because I hadn't carried the bare `except Exception` into the new
file.

## Then it got worse

Renaming `parameters` to `generate_parameters` gets you past the client. The
server then rejects it:

```
The following `model_kwargs` are not used by the model: ['generate_parameters']
```

I tested the endpoint directly, with and without. The serverless provider for
`bart-large-cnn` accepts no length controls at all right now. So the fix wasn't
"use the current kwarg." It was "stop sending length parameters, and write down
why, so the next person doesn't put them back."

Three layers. Wrong kwarg, then renamed kwarg, then a kwarg the backend won't
take. Only the first is findable by reading code.

## It wasn't just the one Space

Once I started actually running things instead of reading them, the same session
turned up more:

Both of my model training scripts had `requirements.txt` files that could not be
installed. `transformers>=5.5.0` requires `huggingface_hub>=1.0`, and the pin
said `<1.0.0`. Straight resolver failure. I had bumped transformers to v5 months
earlier to clear a security advisory and never reinstalled from a clean
environment.

Both scripts also passed `warmup_ratio` and `logging_dir` to `TrainingArguments`,
which transformers 5 removed. So even with the dependencies fixed, both died with
`TypeError` before a single training step.

Neither script had been run since that version bump. Nothing anywhere told me.
Training scripts don't have uptime monitoring. They just sit there looking like
code.

And a lint rule: eight files in the repo carried `# noqa: BLE001` comments with
carefully written justifications for each blind `except`. There was no ruff
configuration in the project at all, so `BLE001` had never once been enabled.
Someone, meaning me, had written eight annotations suppressing a rule that
wasn't running.

## What I actually changed

The summarizer's logic moved into a `core.py` that takes the model call as an
argument, so a test can drive the whole sequence with a fake and assert on what
the chunker handed the model. The failure no longer gets swallowed. If your
inference credits are gone, you now see:

> This account is out of HuggingFace Inference credits, so the request was
> declined. The HF_TOKEN is valid, the monthly free allowance is used up.

I know that message renders correctly because I burned through the month's
allowance testing all this, and then watched the Space tell me so.

I also turned `BLE001` on, made it run on pull requests, and annotated the
twelve blind excepts it found. Five of them were tagged as real defects rather
than justified, and got fixed in later changes.

## The thing I'd tell past me

A fallback that returns plausible output is worse than no fallback. It converts
a loud failure into a quiet wrong answer, and quiet wrong answers can run for
months.

If you catch an exception and substitute a value, ask what the user sees when
the substitution happens. If the answer is "something that looks like a normal
result," you have built a machine for hiding your own bugs. Either let it fail,
or make the degraded state obvious in the output. My paper recommender does the
second thing, falling back to a built-in corpus when the dataset won't load, and
it says so on screen. That's fine. Returning truncated input under a heading
that says "Summary" is not.

The other one is duller and I'll probably need to learn it again: green tests
mean the code you tested works. If your tests structurally cannot reach a code
path, that path has no coverage, no matter what the number at the bottom of the
run says. Mine said 70 passing. It says 289 now. The bug lived in the gap
between those two numbers the entire time.
