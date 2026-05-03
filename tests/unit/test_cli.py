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


def test_version(runner: CliRunner):
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        ["recon", "find prompt injection datasets"],
        ["inspect", "huggingface:deepset/prompt-injections"],
        ["curate", "--from", "recipe.yaml"],
        ["cache", "info"],
        ["sources", "enable", "kaggle"],
    ],
)
def test_verbs_exit_with_not_implemented_notice(runner: CliRunner, argv: list[str]):
    result = runner.invoke(app, argv)
    assert result.exit_code == 2
    assert "not implemented yet" in result.stderr


def test_sources_list_runs(runner: CliRunner):
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    assert "huggingface" in result.stderr
    assert "kaggle" in result.stderr
    assert "pwc" in result.stderr
