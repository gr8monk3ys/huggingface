"""Tests for the model-selector pure logic (core.py)."""

import pytest

from conftest import load_local_module

core = load_local_module("model_selector_core", "model-selector-space/core.py")


def test_parse_size_units():
    assert core.parse_size("7B") == 7000.0
    assert core.parse_size("67M") == 67.0
    assert core.parse_size("1.5B") == 1500.0
    assert core.parse_size("500K") == 0.5
    assert core.parse_size("") == 0.0
    assert core.parse_size("garbage") == 0.0


def test_rank_curated_size_filter():
    models = [
        {"name": "a", "size": "67M"},
        {"name": "b", "size": "7B"},
        {"name": "c", "size": "400M"},
    ]
    small = core.rank_curated(models, "Small (100M - 500M)", "Most Popular")
    assert [m["name"] for m in small] == ["c"]


def test_rank_curated_priority_ordering():
    models = [{"name": "big", "size": "7B"}, {"name": "small", "size": "67M"}]
    smallest = core.rank_curated(models, "Any size", "Smallest/Fastest")
    assert [m["name"] for m in smallest] == ["small", "big"]
    best = core.rank_curated(models, "Any size", "Best Quality")
    assert [m["name"] for m in best] == ["big", "small"]


class _FakeModel:
    def __init__(self, id_, downloads, likes):
        self.id = id_
        self.downloads = downloads
        self.likes = likes


def test_fetch_live_models_with_injected_lister():
    def fake_lister(**kwargs):
        assert kwargs["filter"] == "text-generation"
        return [_FakeModel("org/m1", 1000, 50), _FakeModel("org/m2", 500, 10)]

    out = core.fetch_live_models("text-generation", limit=8, lister=fake_lister)
    assert out == [
        {"name": "org/m1", "downloads": 1000, "likes": 50},
        {"name": "org/m2", "downloads": 500, "likes": 10},
    ]


def test_fetch_live_models_returns_none_on_failure():
    def boom(**kwargs):
        raise RuntimeError("hub down")

    assert core.fetch_live_models("text-generation", lister=boom) is None


def test_fetch_live_models_none_on_empty():
    assert core.fetch_live_models("x", lister=lambda **k: []) is None


def test_generate_code_example_known_and_fallback():
    code = core.generate_code_example("Text Generation", "text-generation", "org/m")
    assert "text-generation" in code and "org/m" in code
    generic = core.generate_code_example("Totally New Task", "some-task", "org/m")
    assert "some-task" in generic and "pipeline(" in generic
    assert core.generate_code_example("Text Generation", "text-generation", None) == ""


# ===========================================================================
# recommend: the coarse entry point
# ===========================================================================
TASK = "Text Generation"


def stub_lister(models):
    """A list_models stand-in returning objects shaped like the Hub's."""

    class Model:
        def __init__(self, name, downloads, likes):
            self.id = name
            self.downloads = downloads
            self.likes = likes

    def lister(**kwargs):
        return [Model(*m) for m in models]

    return lister


def dead_lister(**kwargs):
    raise RuntimeError("hub down")


def test_unknown_task_raises():
    with pytest.raises(core.UnknownTaskError, match="select a task"):
        core.recommend("Interpretive Dance", lister=dead_lister)


def test_live_results_are_used_when_the_hub_answers():
    lister = stub_lister([("org/a", 100, 5), ("org/b", 50, 9)])
    result = core.recommend(TASK, lister=lister)

    assert result.source == "live"
    assert [m.name for m in result.models] == ["org/a", "org/b"]
    assert result.models[0].downloads == 100


def test_best_quality_reorders_live_results_by_likes():
    lister = stub_lister([("org/a", 100, 5), ("org/b", 50, 9)])
    result = core.recommend(TASK, priority="Best Quality", lister=lister)
    assert [m.name for m in result.models] == ["org/b", "org/a"]


def test_size_filter_is_flagged_as_ignored_for_live_results():
    """Size ranking is a curated-list concept; the UI has to say so."""
    lister = stub_lister([("org/a", 100, 5)])
    assert core.recommend(TASK, "Tiny (< 100M)", lister=lister).size_filter_ignored
    assert not core.recommend(TASK, "Any size", lister=lister).size_filter_ignored


def test_hub_failure_falls_back_to_curated_and_says_so():
    """A Hub outage shows curated picks rather than an error -- but labelled."""
    result = core.recommend(TASK, lister=dead_lister)
    assert result.source == "curated"
    assert result.models
    assert result.models[0].size is not None


def test_curated_fallback_with_no_size_match_raises():
    with pytest.raises(core.NoMatchError, match="size preference"):
        core.recommend(TASK, "Tiny (< 100M)", lister=dead_lister)


def test_every_configured_task_can_be_recommended_from_the_curated_list():
    for task in core.TASKS:
        result = core.recommend(task, lister=dead_lister)
        assert result.source == "curated"
        assert result.models
        assert result.description


def test_result_carries_a_code_example_for_its_top_model():
    result = core.recommend(TASK, lister=stub_lister([("org/a", 100, 5)]))
    assert "org/a" in result.code_example


def test_model_url_is_built_from_the_name():
    result = core.recommend(TASK, lister=stub_lister([("org/a", 1, 1)]))
    assert result.models[0].url == "https://huggingface.co/org/a"
