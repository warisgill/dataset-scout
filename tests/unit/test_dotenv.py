"""Tests for dotenv auto-loading at CLI startup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dataset_scout.cli import app

pytestmark = pytest.mark.unit


def test_dotenv_loaded_from_cwd_at_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """`.env` in the CWD is read on every CLI invocation."""
    env_file = tmp_path / ".env"
    env_file.write_text("DSCOUT_DOTENV_TEST_VAR=hello-from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSCOUT_DOTENV_TEST_VAR", raising=False)

    runner = CliRunner()
    # Use a non-eager subcommand so the root callback runs.
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    assert os.environ.get("DSCOUT_DOTENV_TEST_VAR") == "hello-from-dotenv"


def test_existing_env_vars_take_precedence_over_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Explicit shell env wins over `.env` (override=False)."""
    env_file = tmp_path / ".env"
    env_file.write_text("DSCOUT_DOTENV_PRECEDENCE_TEST=from-dotenv\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DSCOUT_DOTENV_PRECEDENCE_TEST", "from-shell")

    runner = CliRunner()
    runner.invoke(app, ["sources", "list"])
    assert os.environ.get("DSCOUT_DOTENV_PRECEDENCE_TEST") == "from-shell"


def test_no_dotenv_is_a_silent_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Absence of `.env` doesn't error or print anything."""
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(app, ["sources", "list"])
    assert result.exit_code == 0
    assert "dotenv" not in result.output.lower()
