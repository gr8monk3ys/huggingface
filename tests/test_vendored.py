"""Drift guard for every vendored module.

A Space is uploaded as a flat folder and cannot import from outside itself, so
code shared between Spaces is vendored: a byte-identical copy per folder. That
only stays safe if something checks the copies still match --
docs/adr/0001-vendoring-is-the-only-sharing-mechanism.md.

Copies are discovered by glob rather than listed. A hardcoded list silently
excludes any new Space from the check, which is exactly how
research-assistant-space once shipped an unguarded hf_client.py.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Every module vendored across Space folders. Add a name here when a new one is
# introduced; the copies themselves are found by glob.
VENDORED_MODULES = ("hf_client.py", "papers.py")


def copies_of(module: str) -> list[Path]:
    return sorted(ROOT.glob(f"*/{module}"))


@pytest.mark.parametrize("module", VENDORED_MODULES)
def test_every_copy_is_byte_identical(module):
    copies = copies_of(module)
    assert copies, f"no copies of {module} found -- is it still vendored?"

    digests = {
        path.parent.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in copies
    }
    assert len(set(digests.values())) == 1, f"{module} copies diverged: {digests}"


@pytest.mark.parametrize("module", VENDORED_MODULES)
def test_more_than_one_copy_exists(module):
    """One copy is not a vendored module, it is just a file.

    ADR-0001 puts the threshold at two or more real callers: below that,
    vendoring buys duplication and a guard in exchange for nothing.
    """
    assert len(copies_of(module)) >= 2, (
        f"{module} has a single copy; either a Space lost it or it should not "
        f"be vendored"
    )


@pytest.mark.parametrize("module", VENDORED_MODULES)
def test_copies_declare_that_they_are_vendored(module):
    """A copy that does not say so invites being edited in one place only."""
    for path in copies_of(module):
        head = path.read_text()[:1200].upper()
        assert "VENDORED" in head, (
            f"{path.parent.name}/{module} does not announce that it is vendored"
        )
