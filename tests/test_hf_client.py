"""Tests for the shared HuggingFace inference helper (hf_client.py).

These exercise the pure logic (no network) and guard against the six per-Space
copies drifting apart.
"""

import hashlib
from pathlib import Path

import pytest

import hf_client

ROOT = Path(__file__).resolve().parent.parent
SPACES_WITH_HELPER = [
    "code-explainer-space",
    "prompt-enhancer-space",
    "model-arena-space",
    "style-mixer-space",
    "illusion-generator-space",
    "paper-summarizer-space",
]


def test_friendly_error_rate_limit():
    assert "rate-limited" in hf_client.friendly_error(Exception("429 Too Many Requests")).lower()


def test_friendly_error_loading():
    assert "warming up" in hf_client.friendly_error(Exception("Model is currently loading")).lower()


def test_friendly_error_auth():
    assert "hf_token" in hf_client.friendly_error(Exception("401 Unauthorized")).lower()


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


def test_all_helper_copies_are_identical():
    """Every Space must ship the exact same helper (guards against drift)."""
    digests = {}
    for space in SPACES_WITH_HELPER:
        path = ROOT / space / "hf_client.py"
        assert path.exists(), f"hf_client.py missing in {space}"
        digests[space] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(set(digests.values())) == 1, f"hf_client.py copies diverged: {digests}"
