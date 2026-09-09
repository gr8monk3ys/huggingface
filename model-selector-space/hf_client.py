"""Helpers for talking to the HuggingFace Inference API.

Centralises the three things every Space here needs and previously got wrong:

* **Authentication** -- reads ``HF_TOKEN`` from the environment. The serverless
  Inference Providers backend rejects most anonymous calls (401/402), so a token
  is effectively required. Add an ``HF_TOKEN`` secret in the Space settings.
* **Timeouts** -- a hung provider request no longer blocks the worker forever.
* **Retries** -- transient errors (model cold-start 503s, rate limits) are
  retried with exponential back-off instead of failing on the first hit.

The ``huggingface_hub`` import is deferred into :func:`make_client` so the pure
helpers (:func:`friendly_error`, :func:`with_retry`) can be imported and unit
tested without the dependency installed.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120  # seconds
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds

T = TypeVar("T")

_TRANSIENT_MARKERS = (
    "loading",
    "currently loading",
    "starting",
    "503",
    "rate limit",
    "429",
    "too many requests",
    "quota",
    "timeout",
    "timed out",
)

# Billing failures are permanent until the account changes -- retrying just
# delays the error. These win over _TRANSIENT_MARKERS, which would otherwise
# match a credits message that happens to mention "quota".
_PERMANENT_MARKERS = (
    "402",
    "payment required",
    "included credits",
    "subscribe to pro",
    "exceeded your monthly",
)


class InferenceError(RuntimeError):
    """Raised when an inference call fails after exhausting retries."""


def get_token() -> Optional[str]:
    """Return the HF token from the environment, if configured."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def make_client(model: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT):
    """Create an :class:`InferenceClient` authenticated with the Space token."""
    from huggingface_hub import InferenceClient  # deferred: keeps helpers dep-free

    token = get_token()
    if token is None:
        logger.warning(
            "No HF_TOKEN found in the environment; inference calls will likely be "
            "rejected. Add an HF_TOKEN secret in the Space settings."
        )
    return InferenceClient(model=model, token=token, timeout=timeout)


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    if any(marker in msg for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


def friendly_error(exc: Exception) -> str:
    """Map a raw inference exception to an actionable, user-facing message."""
    msg = str(exc).lower()
    # Checked before rate limiting: a credits message often mentions "quota"
    # too, and before auth: the token is valid, so telling the user to replace
    # it sends them to fix something that is not broken.
    if any(k in msg for k in _PERMANENT_MARKERS):
        return (
            "This account is out of HuggingFace Inference credits, so the request "
            "was declined. The HF_TOKEN is valid -- the monthly free allowance is "
            "used up. Wait for the monthly reset or subscribe to PRO for more."
        )
    if any(k in msg for k in ("rate limit", "429", "too many requests", "quota")):
        return (
            "The model is rate-limited right now. Wait a moment and try again, or "
            "use an HF_TOKEN that has inference credits."
        )
    if any(k in msg for k in ("currently loading", "loading", "503", "starting")):
        return "The model is warming up (cold start). Please try again in ~20 seconds."
    if any(k in msg for k in ("401", "unauthorized", "authentication")):
        return (
            "Inference was rejected for authentication. Set a valid HF_TOKEN secret "
            "in the Space settings."
        )
    if "timeout" in msg or "timed out" in msg:
        return "The request timed out -- the model may be busy. Please try again."
    return f"Inference failed: {exc}"


def with_retry(
    fn: Callable[..., T],
    *args,
    retries: int = MAX_RETRIES,
    sleep: Callable[[float], None] = time.sleep,
    **kwargs,
) -> T:
    """Call *fn* with exponential back-off on transient errors.

    Raises :class:`InferenceError` with a friendly message if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised via InferenceError
            last_exc = exc
            if not _is_transient(exc) or attempt == retries - 1:
                break
            wait = BACKOFF_BASE * (2**attempt)
            logger.info(
                "Transient inference error (attempt %d/%d); retrying in %.1fs: %s",
                attempt + 1,
                retries,
                wait,
                exc,
            )
            sleep(wait)
    raise InferenceError(friendly_error(last_exc)) from last_exc
