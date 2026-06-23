"""Tests for the model-selector pure logic (core.py)."""

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
