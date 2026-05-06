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


def test_render_help(runner: CliRunner):
    result = runner.invoke(app, ["render", "--help"], terminal_width=200)
    assert result.exit_code == 0
    out = result.output.replace("\n", " ")
    assert "results.json" in out
    assert "--html-only" in out
    assert "--md-only" in out


def test_render_regenerates_reports_from_results_json(runner, tmp_path):
    """Round-trip: write results.json from a tour result, then re-render."""
    from dataset_scout.render import write_results_json
    from dataset_scout.tour import build_tour_result

    write_results_json(build_tour_result(), tmp_path)
    assert (tmp_path / "results.json").exists()

    # Remove any pre-existing reports so we know they are produced.
    for name in ("report.html", "report.md"):
        p = tmp_path / name
        if p.exists():
            p.unlink()

    result = runner.invoke(app, ["render", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "report.md").exists()


def test_render_html_only(runner, tmp_path):
    from dataset_scout.render import write_results_json
    from dataset_scout.tour import build_tour_result

    write_results_json(build_tour_result(), tmp_path)
    result = runner.invoke(app, ["render", str(tmp_path), "--html-only"])
    assert result.exit_code == 0
    assert (tmp_path / "report.html").exists()
    assert not (tmp_path / "report.md").exists()


def test_render_errors_when_results_missing(runner, tmp_path):
    result = runner.invoke(app, ["render", str(tmp_path)])
    assert result.exit_code == 2
    assert "results.json" in result.output


def test_render_rejects_conflicting_only_flags(runner, tmp_path):
    from dataset_scout.render import write_results_json
    from dataset_scout.tour import build_tour_result

    write_results_json(build_tour_result(), tmp_path)
    result = runner.invoke(
        app, ["render", str(tmp_path), "--html-only", "--md-only"]
    )
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


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
