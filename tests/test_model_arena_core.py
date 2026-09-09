"""Tests for model-arena's core logic.

dataset-explorer used to be tested here too. It became a static Space, so its
Python is gone and tests/test_dataset_explorer_data.py covers what remains.
"""

import random

import pytest

from conftest import load_local_module

arena = load_local_module("model_arena_core", "model-arena-space/core.py")

NAMES = list(arena.MODELS)


# ===========================================================================
# model-arena: battles
# ===========================================================================
class RecordingResponder:
    def __init__(self, fail_for=()):
        self.calls = []
        self.fail_for = set(fail_for)

    def __call__(self, model_id, prompt, *, max_tokens):
        self.calls.append(model_id)
        if model_id in self.fail_for:
            raise RuntimeError("monthly free allowance is used up")
        return f"answer from {model_id}"


@pytest.mark.parametrize("prompt", ["", "   ", "\n"])
def test_empty_prompt_raises_input_error(prompt):
    with pytest.raises(arena.InputError, match="enter a prompt"):
        arena.battle(prompt, NAMES[0], NAMES[1], respond_fn=RecordingResponder())


def test_validation_happens_before_any_inference_call():
    fake = RecordingResponder()
    with pytest.raises(arena.InputError):
        arena.battle("  ", NAMES[0], NAMES[1], respond_fn=fake)
    assert fake.calls == []


def test_both_models_are_asked_and_named_in_the_result():
    fake = RecordingResponder()
    result = arena.battle("why is the sky blue", NAMES[0], NAMES[1], respond_fn=fake)
    assert len(fake.calls) == 2
    assert result.first.model_name == NAMES[0]
    assert result.second.model_name == NAMES[1]
    assert result.first.ok and result.second.ok


def test_one_model_failing_still_returns_the_other_side():
    """A half-broken battle is worth showing, so a failure is data, not a raise."""
    failing_id = arena.MODELS[NAMES[0]]["id"]
    fake = RecordingResponder(fail_for=[failing_id])

    result = arena.battle("a prompt", NAMES[0], NAMES[1], respond_fn=fake)

    assert not result.first.ok
    assert "allowance" in result.first.error
    assert result.first.text is None
    assert result.second.ok and result.second.text


def test_every_response_is_timed():
    result = arena.battle(
        "a prompt", NAMES[0], NAMES[1], respond_fn=RecordingResponder()
    )
    assert result.first.seconds >= 0
    assert result.second.seconds >= 0


def test_a_failed_response_is_still_timed():
    fake = RecordingResponder(fail_for=[arena.MODELS[NAMES[0]]["id"]])
    result = arena.battle("a prompt", NAMES[0], NAMES[1], respond_fn=fake)
    assert result.first.seconds >= 0


# ===========================================================================
# model-arena: the tally
# ===========================================================================
def test_new_tally_starts_every_model_at_zero():
    tally = arena.new_tally()
    assert set(tally) == set(arena.MODELS)
    assert all(s == {"wins": 0, "battles": 0} for s in tally.values())


def test_record_vote_does_not_mutate_the_tally_it_was_given():
    """core returns a new tally; app.py owns the mutable state."""
    before = arena.new_tally()
    after = arena.record_vote(before, NAMES[0], NAMES[1])
    assert before[NAMES[0]]["wins"] == 0
    assert after[NAMES[0]]["wins"] == 1


def test_a_vote_counts_a_battle_for_both_sides():
    tally = arena.record_vote(arena.new_tally(), NAMES[0], NAMES[1])
    assert tally[NAMES[0]] == {"wins": 1, "battles": 1}
    assert tally[NAMES[1]] == {"wins": 0, "battles": 1}


def test_unknown_winner_raises():
    with pytest.raises(KeyError):
        arena.record_vote(arena.new_tally(), "Nonexistent", NAMES[0])


def test_unknown_loser_is_ignored_so_the_winners_vote_still_counts():
    tally = arena.record_vote(arena.new_tally(), NAMES[0], "Nonexistent")
    assert tally[NAMES[0]] == {"wins": 1, "battles": 1}


def test_win_rate_is_zero_for_a_model_with_no_battles():
    standing = arena.Standing(model="x", wins=0, battles=0)
    assert standing.win_rate == 0.0


def test_rank_orders_by_win_rate_then_by_wins():
    tally = {
        "half": {"wins": 5, "battles": 10},  # 50%, more wins
        "perfect": {"wins": 1, "battles": 1},  # 100%
        "unplayed": {"wins": 0, "battles": 0},  # 0%
        "also_half": {"wins": 2, "battles": 4},  # 50%, fewer wins
    }
    assert [s.model for s in arena.rank(tally)] == [
        "perfect",
        "half",
        "also_half",
        "unplayed",
    ]


def test_random_battle_picks_two_distinct_models_and_a_real_prompt():
    rng = random.Random(0)
    all_prompts = {p for prompts in arena.CATEGORIES.values() for p in prompts}
    for _ in range(25):
        model1, model2, prompt = arena.random_battle(rng)
        assert model1 != model2
        assert prompt in all_prompts


def test_example_prompt_for_unknown_category_is_empty():
    assert arena.example_prompt("Nonexistent") == ""
