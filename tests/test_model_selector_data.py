"""Validation for the static model-selector's task data.

model-selector became a static Space, so its ranking logic now lives in JS and
this repo has no JS runner. That is a real loss of coverage and worth being
honest about. What survives is the part that rots fastest and breaks silently:
the task table itself -- model ids that stop resolving, sizes that stop parsing,
pipeline tags that stop matching the Hub's vocabulary.

So this parses tasks.js as data (no JS execution) and checks it. Nothing here
touches the network.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TASKS_JS = ROOT / "model-selector-space" / "tasks.js"


def load_data():
    """Extract the `const DATA = {...};` object from tasks.js as JSON."""
    text = TASKS_JS.read_text()
    match = re.search(r"const DATA = (\{.*\});\s*$", text, re.S)
    assert match, "tasks.js no longer matches `const DATA = {...};`"
    return json.loads(match.group(1))


DATA = load_data()
TASKS = DATA["tasks"]
SIZES = DATA["sizePreferences"]

# Mirrors parseSize() in app.js. Kept here so the sizes in the table are proven
# parseable by the same rules the page applies.
SIZE_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([BMK]?)", re.I)


def parse_size(size_str):
    if not size_str:
        return 0.0
    m = SIZE_RE.match(str(size_str).strip().upper().replace(",", ""))
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2)
    return val * 1000.0 if unit == "B" else val / 1000.0 if unit == "K" else val


# --- shape ----------------------------------------------------------------
def test_the_file_parses_and_is_not_empty():
    assert TASKS
    assert SIZES
    assert DATA["genericTemplate"]


@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_the_fields_the_page_reads(task):
    info = TASKS[task]
    for field in ("id", "description", "use_cases", "top_models"):
        assert info.get(field), f"{task} is missing {field}"


@pytest.mark.parametrize("task", TASKS)
def test_every_task_id_looks_like_a_hub_pipeline_tag(task):
    """The id goes straight into the Hub's ?filter= query."""
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", TASKS[task]["id"])


# --- the curated models ---------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_curated_models_carry_name_size_and_license(task):
    for model in TASKS[task]["top_models"]:
        for field in ("name", "size", "license"):
            assert model.get(field), f"{task}: a model is missing {field}"


@pytest.mark.parametrize("task", TASKS)
def test_every_curated_size_parses_to_something_positive(task):
    """A size that parses to 0 silently drops the model from every size filter."""
    for model in TASKS[task]["top_models"]:
        assert parse_size(model["size"]) > 0, (
            f"{task}: {model['name']} size={model['size']!r}"
        )


@pytest.mark.parametrize("task", TASKS)
def test_curated_model_names_look_like_hub_ids(task):
    """Either `org/name` or a legacy canonical name such as
    distilbert-base-cased-distilled-squad, which has no namespace."""
    for model in TASKS[task]["top_models"]:
        name = model["name"]
        assert re.fullmatch(r"[\w.-]+(/[\w.-]+)?", name), f"{task}: {name!r}"
        assert not name.startswith(("/", "http")), f"{task}: {name!r}"


def test_no_curated_model_is_the_retired_mistral_v03():
    """v0.3 is 404 on the Hub; recommending it sends people to a dead page."""
    listed = [m["name"] for t in TASKS.values() for m in t["top_models"]]
    assert "mistralai/Mistral-7B-Instruct-v0.3" not in listed


# --- size bands -----------------------------------------------------------
def test_size_bands_are_ordered_and_cover_everything():
    assert "Any size" in SIZES
    for name, band in SIZES.items():
        assert band["min"] < band["max"], name
    any_band = SIZES["Any size"]
    for name, band in SIZES.items():
        assert band["min"] >= any_band["min"] and band["max"] <= any_band["max"], name


def test_every_curated_model_falls_inside_at_least_one_band():
    """A model outside every band is unreachable through the UI."""
    for task, info in TASKS.items():
        for model in info["top_models"]:
            size = parse_size(model["size"])
            assert any(b["min"] <= size <= b["max"] for b in SIZES.values()), (
                f"{task}: {model['name']} ({model['size']}) matches no size band"
            )


# --- code templates -------------------------------------------------------
@pytest.mark.parametrize("task", TASKS)
def test_every_task_has_a_code_template_with_a_substitution_point(task):
    tpl = DATA["codeTemplates"].get(task)
    assert tpl, f"{task} has no code template"
    assert "__MODEL__" in tpl, f"{task}'s template has no __MODEL__ placeholder"


def test_the_generic_template_also_substitutes():
    assert "__MODEL__" in DATA["genericTemplate"]


# --- the page itself ------------------------------------------------------
def test_index_html_loads_both_scripts_in_order():
    """app.js reads the DATA const, so tasks.js has to come first."""
    html = (ROOT / "model-selector-space" / "index.html").read_text()
    assert html.index("tasks.js") < html.index("app.js")


def test_the_space_ships_no_python():
    """A static Space with a stray app.py is a confusing half-migration."""
    assert not list((ROOT / "model-selector-space").glob("*.py"))
