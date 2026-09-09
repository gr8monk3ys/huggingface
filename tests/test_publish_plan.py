"""Tests for the publish plan.

Only the deciding half is covered here; nothing in this file touches the Hub.
That is the point of the plan/execute split -- before it, deciding what to
upload and uploading it were the same code, so `--dry-run` could not run
without huggingface_hub installed even though the docs said it could.
"""

import pytest

from conftest import load_local_module

publish = load_local_module("publish_to_hub", "scripts/publish_to_hub.py")


@pytest.fixture
def fake_repo(tmp_path):
    """A miniature repo: one mapped Space, one opted-out folder."""
    (tmp_path / "demo-space").mkdir()
    (tmp_path / "demo-space" / "app.py").write_text("x = 1\n")
    (tmp_path / "demo-space" / "README.md").write_text("# demo\n")
    (tmp_path / "demo-space" / "notes.ipynb").write_text("{}")  # not in SCRIPT_EXTS
    (tmp_path / "demo-space" / ".venv").mkdir()
    (tmp_path / "tests").mkdir()
    return tmp_path


@pytest.fixture
def only_demo(monkeypatch):
    monkeypatch.setattr(publish, "SPACES", {"demo-space": "demo"})
    monkeypatch.setattr(publish, "MODELS", {})
    monkeypatch.setattr(publish, "DATASETS", {})


# --- reconciliation, the check that did not exist -------------------------
def test_unmapped_project_folder_fails_the_run(fake_repo, only_demo):
    """A new or renamed project used to be silently never published."""
    (fake_repo / "brand-new-space").mkdir()

    with pytest.raises(publish.UnmappedProjectError, match="brand-new-space"):
        publish.plan(fake_repo)


def test_folder_in_the_opt_out_set_is_accepted(fake_repo, only_demo):
    publish.plan(fake_repo)  # "tests" is in NOT_PUBLISHED; must not raise


def test_mapped_folder_missing_from_disk_fails_the_run(fake_repo, monkeypatch):
    monkeypatch.setattr(publish, "SPACES", {"demo-space": "demo", "gone-space": "gone"})
    monkeypatch.setattr(publish, "MODELS", {})
    monkeypatch.setattr(publish, "DATASETS", {})

    with pytest.raises(publish.UnmappedProjectError, match="gone-space"):
        publish.plan(fake_repo)


def test_hidden_directories_are_not_projects(fake_repo, only_demo):
    (fake_repo / ".github").mkdir()
    (fake_repo / ".ruff_cache").mkdir()
    publish.plan(fake_repo)  # must not raise


def test_the_real_repo_is_reconciled():
    """The live tables must match the live filesystem."""
    publish.check_reconciled()


# --- what gets uploaded ---------------------------------------------------
def test_only_allowlisted_extensions_are_uploaded(fake_repo, only_demo):
    upload = publish.plan(fake_repo).uploads[0]
    names = [p.name for p in upload.files]
    assert names == ["README.md", "app.py"]
    assert "notes.ipynb" not in names


def test_directories_are_never_uploaded_as_files(fake_repo, only_demo):
    """A stray .venv must not reach the Hub."""
    upload = publish.plan(fake_repo).uploads[0]
    assert all(p.is_file() for p in upload.files)
    assert ".venv" not in [p.name for p in upload.files]


def test_repo_id_is_namespaced(fake_repo, only_demo):
    assert publish.plan(fake_repo).uploads[0].repo_id == "gr8monk3ys/demo"


@pytest.mark.parametrize(
    "only, expected",
    [("spaces", {"space"}), ("models", {"model"}), ("datasets", {"dataset"})],
)
def test_only_filters_by_repo_type(only, expected):
    kinds = {u.repo_type for u in publish.plan(only=only).uploads}
    assert kinds == expected


def test_no_filter_plans_every_kind():
    kinds = {u.repo_type for u in publish.plan().uploads}
    assert kinds == {"space", "model", "dataset"}


# --- artifacts ------------------------------------------------------------
def test_spaces_never_upload_an_artifacts_directory():
    """Only models and datasets do; this is why a Space is a flat folder."""
    for upload in publish.plan(only="spaces").uploads:
        assert upload.artifacts is None


def test_a_missing_weights_directory_is_reported_not_fatal():
    result = publish.plan(only="models")
    for upload in result.uploads:
        if upload.artifacts_missing:
            assert any(upload.repo_id in w for w in result.warnings)


def test_dataset_artifacts_go_under_data():
    upload = publish.plan(only="datasets").uploads[0]
    assert upload.artifacts_path_in_repo == "data"


# --- rendering ------------------------------------------------------------
def test_describe_lists_every_planned_file(fake_repo, only_demo):
    text = publish.describe(publish.plan(fake_repo), fake_repo)
    assert "demo-space/app.py" in text
    assert "space:gr8monk3ys/demo" in text


def test_describe_has_no_side_effects(fake_repo, only_demo):
    before = sorted(p.name for p in fake_repo.rglob("*"))
    publish.describe(publish.plan(fake_repo), fake_repo)
    assert sorted(p.name for p in fake_repo.rglob("*")) == before
