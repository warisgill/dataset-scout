"""CLI smoke tests."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from dataset_scout import __version__
from dataset_scout.cli import app

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_root_help(runner: CliRunner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "recon" in result.stdout
    assert "inspect" in result.stdout
    assert "curate" in result.stdout
    assert "judge" in result.stdout
    assert "eval" in result.stdout


def test_judge_help(runner: CliRunner):
    result = runner.invoke(app, ["judge", "--help"], terminal_width=200)
    assert result.exit_code == 0
    out = result.output.replace("\n", " ")
    assert "--axis" in out
    assert "--judges" in out
    assert "--threshold" in out


def test_eval_help(runner: CliRunner):
    result = runner.invoke(app, ["eval", "--help"], terminal_width=200)
    assert result.exit_code == 0
    out = result.output.replace("\n", " ")
    assert "--against" in out


def test_version(runner: CliRunner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["sources", "enable", "kaggle"],
    ],
)
def test_verbs_exit_with_not_implemented_notice(runner: CliRunner, argv: list[str]):
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_cache_info_runs(runner: CliRunner, tmp_path, monkeypatch):
    """`datascout cache info` summarises an empty cache without crashing."""
    monkeypatch.setenv("DATASET_SCOUT_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    assert "Cache:" in result.output


def test_cache_prune_runs(runner: CliRunner, tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_SCOUT_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "prune"])
    assert result.exit_code == 0
    assert "Removed" in result.output


def test_cache_clear_runs(runner: CliRunner, tmp_path, monkeypatch):
    monkeypatch.setenv("DATASET_SCOUT_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "clear"])
    assert result.exit_code == 0
    assert "Removed" in result.output


def test_sources_list_runs(runner: CliRunner):
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    assert "huggingface" in result.output
    assert "kaggle" in result.output
