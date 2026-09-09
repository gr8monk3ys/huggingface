"""Validation for the static dataset-explorer.

The Space became static, so its describing and charting logic is JS and this
repo has no JS runner. That is a real loss of coverage. What is checked here is
the part that silently rotted before and would rot again: the preset list.

Every one of the ten original presets ("imdb", "squad", "glue", ...) had been
renamed on the Hub, so every quick-start button in the Space returned "The
dataset has been renamed" and nothing worked. Nobody noticed, because a preset
button that fails looks the same as one nobody clicked.

These tests are offline, so they cannot prove an id still resolves. They can
prove the shape is right and that the specific failure mode -- bare,
un-namespaced legacy ids -- has not crept back.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SPACE = ROOT / "dataset-explorer-space"


def load_presets():
    text = (SPACE / "presets.js").read_text()
    match = re.search(r"const PRESETS = (\[.*?\]);", text, re.S)
    assert match, "presets.js no longer matches `const PRESETS = [...];`"
    # JS object literal with unquoted keys -> JSON
    body = re.sub(r"(\w+):", r'"\1":', match.group(1))
    body = re.sub(r",(\s*[\]}])", r"\1", body)
    return json.loads(body)


PRESETS = load_presets()

# The renames that broke every button. Any of these reappearing bare is the
# regression this file exists to catch.
KNOWN_LEGACY_IDS = {
    "imdb",
    "squad",
    "glue",
    "wikitext",
    "ag_news",
    "yelp_review_full",
    "amazon_polarity",
    "dbpedia_14",
    "emotion",
    "financial_phrasebank",
}


def test_presets_parse_and_are_not_empty():
    assert PRESETS
    assert len(PRESETS) >= 5


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_every_preset_has_an_id_and_a_label(preset):
    assert preset.get("id", "").strip()
    assert preset.get("label", "").strip()


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_every_preset_id_is_namespaced(preset):
    """Bare ids are the exact shape that broke: the Hub namespaced them all."""
    assert "/" in preset["id"], (
        f"{preset['id']!r} is a bare dataset id. Every bare id in the original "
        f"list had been renamed and returned an error."
    )


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_no_preset_is_a_known_renamed_id(preset):
    assert preset["id"] not in KNOWN_LEGACY_IDS


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p["id"])
def test_preset_ids_look_like_hub_repo_ids(preset):
    assert re.fullmatch(r"[\w.-]+/[\w.-]+", preset["id"]), preset["id"]


def test_preset_ids_are_unique():
    ids = [p["id"] for p in PRESETS]
    assert len(ids) == len(set(ids))


def test_the_owners_own_dataset_is_offered():
    """It is the one dataset in this repo; the explorer should point at it."""
    assert any(p["id"].startswith("gr8monk3ys/") for p in PRESETS)


# --- the page ------------------------------------------------------------
def test_index_html_loads_presets_before_app():
    """app.js reads the PRESETS const, so presets.js has to come first."""
    html = (SPACE / "index.html").read_text()
    assert html.index("presets.js") < html.index("app.js")


def test_every_element_app_js_looks_up_exists_in_the_html():
    """A renamed id would break the page silently, with no console error."""
    html = (SPACE / "index.html").read_text()
    app = (SPACE / "app.js").read_text()
    for element_id in sorted(set(re.findall(r'\$\("([\w-]+)"\)', app))):
        assert f'id="{element_id}"' in html, (
            f"app.js reads #{element_id}, absent from index.html"
        )


def test_the_space_ships_no_python():
    assert not list(SPACE.glob("*.py"))


def test_the_space_declares_itself_static():
    card = (SPACE / "README.md").read_text()
    assert "sdk: static" in card
    assert "app_file: index.html" in card
