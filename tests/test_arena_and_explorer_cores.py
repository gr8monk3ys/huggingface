"""Tests for model-arena and dataset-explorer core logic.

These are the two Spaces that do not fit the LLM shape: one keeps a vote tally,
the other returns a chart. Both are covered here.
"""

import random

import pandas as pd
import pytest

from conftest import load_local_module

arena = load_local_module("model_arena_core", "model-arena-space/core.py")
explorer = load_local_module("dataset_explorer_core", "dataset-explorer-space/core.py")

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


# ===========================================================================
# dataset-explorer
# ===========================================================================
def frame():
    return pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0],
            "label": ["a", "b", "a", "c"],
            "missing": [1.0, None, 3.0, None],
        }
    )


def fake_loader(rows):
    def loader(dataset_id, *args, **kwargs):
        return iter(rows)

    return loader


def test_describe_reports_shape():
    stats = explorer.describe(frame())
    assert stats.rows == 4
    assert len(stats.columns) == 3


def test_describe_of_an_empty_frame_reports_no_rows():
    assert explorer.describe(pd.DataFrame()).rows == 0


def test_numeric_columns_carry_range_and_moments():
    score = next(c for c in explorer.describe(frame()).columns if c.name == "score")
    assert score.is_numeric
    assert (score.minimum, score.maximum) == (1.0, 4.0)
    assert score.mean == 2.5


def test_categorical_columns_carry_cardinality_and_top_values():
    label = next(c for c in explorer.describe(frame()).columns if c.name == "label")
    assert not label.is_numeric
    assert label.unique == 3
    assert label.top_values["a"] == 2


def test_top_values_are_omitted_for_high_cardinality_columns():
    many = pd.DataFrame({"id": [f"v{i}" for i in range(50)]})
    column = explorer.describe(many).columns[0]
    assert column.unique == 50
    assert column.top_values is None


def test_null_percentage_is_reported():
    missing = next(c for c in explorer.describe(frame()).columns if c.name == "missing")
    assert missing.non_null == 2
    assert missing.non_null_pct == 50.0


def test_load_samples_stops_at_the_requested_count():
    rows = [{"n": i} for i in range(100)]
    df, configs = explorer.load_samples(
        "some/dataset",
        num_samples=7,
        loader=fake_loader(rows),
        config_lister=lambda _: ["default"],
    )
    assert len(df) == 7
    assert configs == ["default"]


def test_loader_failures_become_a_domain_error():
    def broken(*args, **kwargs):
        raise OSError("connection reset")

    with pytest.raises(explorer.DatasetLoadError, match="connection reset"):
        explorer.load_samples(
            "some/dataset", loader=broken, config_lister=lambda _: ["default"]
        )


@pytest.mark.parametrize("dataset_id", ["", "   "])
def test_explore_rejects_an_empty_dataset_id(dataset_id):
    with pytest.raises(ValueError, match="enter a dataset ID"):
        explorer.explore(dataset_id, loader=fake_loader([]), config_lister=lambda _: [])


def test_explore_rejects_a_dataset_that_yields_no_rows():
    with pytest.raises(ValueError, match="No data found"):
        explorer.explore(
            "some/dataset", loader=fake_loader([]), config_lister=lambda _: ["default"]
        )


def test_explore_returns_stats_configs_and_a_ten_row_sample():
    rows = [{"n": i, "t": "x"} for i in range(40)]
    result = explorer.explore(
        "some/dataset",
        num_samples=30,
        loader=fake_loader(rows),
        config_lister=lambda _: ["default", "other"],
        chart_fn=lambda df: "<chart>",
    )
    assert result.stats.rows == 30
    assert result.configs == ["default", "other"]
    assert len(result.sample) == 10
    assert result.chart == "<chart>"


def test_visualize_returns_nothing_for_an_empty_frame():
    assert explorer.visualize(pd.DataFrame()) is None
