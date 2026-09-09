"""Tests for the shared HuggingFace inference helper (hf_client.py).

These exercise the pure logic; no network. The drift guard over the per-Space
copies lives in test_vendored.py, which covers every vendored module rather
than this one alone.
"""

import pytest

import hf_client


def test_friendly_error_rate_limit():
    assert (
        "rate-limited"
        in hf_client.friendly_error(Exception("429 Too Many Requests")).lower()
    )


def test_friendly_error_loading():
    assert (
        "warming up"
        in hf_client.friendly_error(Exception("Model is currently loading")).lower()
    )


def test_friendly_error_auth():
    assert "hf_token" in hf_client.friendly_error(Exception("401 Unauthorized")).lower()


# The message HuggingFace actually returns once the free monthly allowance is
# spent. It must not be reported as a bad token: the token is fine.
REAL_402 = (
    "402 Client Error: Payment Required. You have exceeded your monthly "
    "included credits for Inference Providers. Subscribe to PRO for more."
)


def test_friendly_error_out_of_credits_is_not_an_auth_problem():
    msg = hf_client.friendly_error(Exception(REAL_402)).lower()
    assert "credits" in msg
    assert "set a valid hf_token" not in msg


def test_credits_error_mentioning_quota_beats_the_rate_limit_branch():
    exc = Exception("402 Payment Required: quota exceeded")
    msg = hf_client.friendly_error(exc).lower()
    assert "credits" in msg
    assert "rate-limited" not in msg


def test_billing_errors_are_never_retried():
    attempts = {"n": 0}

    def broke():
        attempts["n"] += 1
        raise RuntimeError(REAL_402)

    with pytest.raises(hf_client.InferenceError):
        hf_client.with_retry(broke, retries=5, sleep=lambda _s: None)
    assert attempts["n"] == 1


def test_friendly_error_generic():
    assert "splines" in hf_client.friendly_error(Exception("reticulating splines"))


def test_with_retry_returns_value_without_sleeping():
    slept = []
    assert hf_client.with_retry(lambda: 7, sleep=slept.append) == 7
    assert slept == []


def test_with_retry_retries_transient_then_raises():
    attempts = {"n": 0}

    def boom():
        attempts["n"] += 1
        raise RuntimeError("503 loading")

    with pytest.raises(hf_client.InferenceError):
        hf_client.with_retry(boom, retries=3, sleep=lambda _s: None)
    assert attempts["n"] == 3


def test_with_retry_fast_fails_on_permanent_error():
    attempts = {"n": 0}

    def bad():
        attempts["n"] += 1
        raise ValueError("bad request")

    with pytest.raises(hf_client.InferenceError):
        hf_client.with_retry(bad, retries=5, sleep=lambda _s: None)
    assert attempts["n"] == 1  # not retried


def test_get_token_reads_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    assert hf_client.get_token() is None
    monkeypatch.setenv("HF_TOKEN", "secret-abc")
    assert hf_client.get_token() == "secret-abc"
