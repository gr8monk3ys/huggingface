"""Model-arena logic, independent of Gradio and of the Inference API.

Two entry points, because the arena does two separable things: :func:`battle`
runs a prompt past two models, and :func:`record_vote` / :func:`rank` maintain
the tally.

The tally is passed in and a new one returned rather than kept here. A module
global would relocate the untestability rather than remove it -- tests would
need reset fixtures between cases. ``app.py`` owns the mutable state, which is
per-process and ephemeral anyway: a sleeping Space loses it on restart.

See docs/adr/0002-coarse-entry-point-for-space-core-modules.md.
"""

import random
import time
from dataclasses import dataclass
from typing import Optional, Protocol

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "Mistral-7B": {
        "id": "mistralai/Mistral-7B-Instruct-v0.2",
        "description": "Fast, efficient 7B parameter model from Mistral AI",
        "strengths": "Speed, reasoning, code",
    },
    "Llama-3.1-8B": {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Meta's latest open LLM with strong capabilities",
        "strengths": "General knowledge, instruction following",
    },
    "Qwen2.5-7B": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Alibaba's powerful multilingual model",
        "strengths": "Multilingual, math, coding",
    },
    "Phi-3-mini": {
        "id": "microsoft/Phi-3-mini-4k-instruct",
        "description": "Microsoft's compact but capable model",
        "strengths": "Efficiency, reasoning, small size",
    },
    "Gemma-2-9B": {
        "id": "google/gemma-2-9b-it",
        "description": "Google's instruction-tuned Gemma model",
        "strengths": "Quality, safety, general tasks",
    },
    "Zephyr-7B": {
        "id": "HuggingFaceH4/zephyr-7b-beta",
        "description": "Fine-tuned Mistral with DPO alignment",
        "strengths": "Helpfulness, alignment, chat",
    },
}

CATEGORIES = {
    "Creative Writing": [
        "Write a haiku about artificial intelligence",
        "Create a short story opening about a robot discovering emotions",
        "Write a limerick about machine learning",
        "Compose a brief poem about the future of technology",
    ],
    "Coding": [
        "Write a Python function to check if a number is prime",
        "Create a JavaScript function to reverse a string",
        "Write a SQL query to find duplicate emails in a users table",
        "Implement a simple stack data structure in Python",
    ],
    "Reasoning": [
        "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "What comes next in the sequence: 2, 6, 12, 20, 30, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    ],
    "Knowledge": [
        "Explain quantum entanglement in simple terms",
        "What are the main differences between TCP and UDP?",
        "Briefly explain how transformers work in machine learning",
        "What is the difference between machine learning and deep learning?",
    ],
    "Summarization": [
        "Summarize the concept of blockchain technology in 2-3 sentences",
        "Explain the main idea behind reinforcement learning briefly",
        "Summarize what makes Python popular for data science",
        "Briefly explain the concept of transfer learning",
    ],
}

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------


MAX_TOKENS = 500
TEMPERATURE = 0.7

Tally = dict[str, dict[str, int]]


class InputError(ValueError):
    """The caller supplied no prompt."""


class RespondFn(Protocol):
    """Send *prompt* to the model at *model_id* and return its reply.

    Whatever this raises must already carry a message fit to show a user;
    :func:`battle` records ``str(exc)`` verbatim.
    """

    def __call__(self, model_id: str, prompt: str, *, max_tokens: int) -> str: ...


@dataclass(frozen=True)
class Response:
    """One model's answer, or its failure.

    A failure is data rather than an exception here: when one model is down the
    battle is still worth showing, with the other model's answer beside the
    error.
    """

    model_name: str
    text: Optional[str]
    seconds: float
    error: Optional[str]

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Battle:
    """Both sides of one head-to-head."""

    prompt: str
    first: Response
    second: Response


@dataclass(frozen=True)
class Standing:
    """One model's position on the leaderboard."""

    model: str
    wins: int
    battles: int

    @property
    def win_rate(self) -> float:
        """Percentage, or 0.0 for a model that has not been voted on."""
        return self.wins / self.battles * 100 if self.battles else 0.0


def new_tally() -> Tally:
    """A fresh vote tally with every configured model at zero."""
    return {model: {"wins": 0, "battles": 0} for model in MODELS}


def _ask(model_name: str, prompt: str, respond_fn: RespondFn) -> Response:
    model_id = MODELS[model_name]["id"]
    start = time.perf_counter()
    try:
        text = respond_fn(model_id, prompt, max_tokens=MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - recorded per model so the other side still shows
        return Response(model_name, None, time.perf_counter() - start, str(exc))
    return Response(model_name, text, time.perf_counter() - start, None)


def battle(
    prompt: str,
    model1_name: str,
    model2_name: str,
    *,
    respond_fn: RespondFn,
) -> Battle:
    """Run *prompt* past both models and return both answers.

    Raises:
        InputError: *prompt* is empty or whitespace.
        KeyError: either name is not a configured model.
    """
    if not prompt.strip():
        raise InputError("Please enter a prompt.")

    return Battle(
        prompt=prompt,
        first=_ask(model1_name, prompt, respond_fn),
        second=_ask(model2_name, prompt, respond_fn),
    )


def record_vote(tally: Tally, winner: str, loser: str) -> Tally:
    """Return a new tally with a win for *winner* and a battle for both.

    Raises:
        KeyError: *winner* is not in *tally*. An unknown *loser* is ignored --
            the winner's vote still counts.
    """
    if winner not in tally:
        raise KeyError(winner)

    updated = {model: dict(stats) for model, stats in tally.items()}
    updated[winner]["wins"] += 1
    updated[winner]["battles"] += 1
    if loser in updated:
        updated[loser]["battles"] += 1
    return updated


def rank(tally: Tally) -> list[Standing]:
    """Standings, best first: by win rate, then by total wins."""
    standings = [
        Standing(model, stats["wins"], stats["battles"])
        for model, stats in tally.items()
    ]
    standings.sort(key=lambda s: (s.win_rate, s.wins), reverse=True)
    return standings


def random_battle(rng: random.Random = random) -> tuple[str, str, str]:
    """Two distinct models and a prompt, for the "surprise me" button."""
    models = list(MODELS)
    model1 = rng.choice(models)
    model2 = rng.choice([m for m in models if m != model1])
    category = rng.choice(list(CATEGORIES))
    return model1, model2, rng.choice(CATEGORIES[category])


def example_prompt(category: str, rng: random.Random = random) -> str:
    """A random prompt from *category*, or "" if it is not configured."""
    return rng.choice(CATEGORIES[category]) if category in CATEGORIES else ""
