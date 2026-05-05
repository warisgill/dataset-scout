"""Unit tests for the --review interactive decomposition flow."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from dataset_scout.cli import app
from dataset_scout.core import DecompositionDirection

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _fake_directions() -> list[DecompositionDirection]:
    return [
        DecompositionDirection(
            name="parasocial_interaction",
            rationale=(
                "Drawing on parasocial interaction studies in media psychology: "
                "one-sided emotional bonds parallel AI relationship claims."
            ),
            keywords=["parasocial", "emotional bonding"],
            threat_families=["emotional_dependence"],
            expected_finds="Datasets capturing parasocial language.",
        ),
        DecompositionDirection(
            name="clinical_psychology_dialogue",
            rationale=(
                "Drawing on clinical-psychology research: therapeutic dialogue "
                "corpora exhibit the supportive-but-bounded language pattern."
            ),
            keywords=["therapeutic dialogue", "counseling chat"],
            threat_families=[],
            expected_finds="Therapeutic conversation corpora.",
        ),
    ]


def test_review_continue_with_enter(runner, tmp_path, monkeypatch):
    """Pressing Enter at the prompt approves the LLM's directions verbatim."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    captured_directions: list = []

    def fake_run_recon(brief, **kwargs):
        captured_directions.extend(kwargs.get("directions_override") or [])
        from dataset_scout.core import Intent, ReconResult

        return ReconResult(intent=Intent(raw_brief=brief), candidates=[])

    with patch(
        "dataset_scout.decompose.decompose_intent", return_value=_fake_directions()
    ), patch("dataset_scout.pipeline.run_recon", side_effect=fake_run_recon):
        result = runner.invoke(
            app,
            [
                "recon",
                "test brief about parasocial AI",
                "--review",
                "--no-papers",
                "--out",
                str(tmp_path / "out"),
            ],
            input="\n",  # press Enter at prompt
        )
    assert result.exit_code == 0, result.output
    # Both fake directions made it into run_recon.
    assert len(captured_directions) == 2
    assert captured_directions[0].name == "parasocial_interaction"
    # decomposition.yaml was persisted for resume.
    assert (tmp_path / "out" / "decomposition.yaml").exists()


def test_review_abort_exits_cleanly(runner, tmp_path, monkeypatch):
    """Choosing 'a' aborts before paying for the full recon."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    with patch(
        "dataset_scout.decompose.decompose_intent", return_value=_fake_directions()
    ), patch("dataset_scout.pipeline.run_recon") as run_recon:
        result = runner.invoke(
            app,
            [
                "recon",
                "test brief",
                "--review",
                "--no-papers",
                "--out",
                str(tmp_path / "out"),
            ],
            input="a\n",
        )
    # Conventional Ctrl-C exit code (130).
    assert result.exit_code == 130
    # We never proceeded to the full recon.
    assert run_recon.call_count == 0
    # But we DID save the decomposition for later resume.
    assert (tmp_path / "out" / "decomposition.yaml").exists()


def test_review_decompose_failure_aborts(runner, tmp_path, monkeypatch):
    """If the LLM call fails inside --review, the user gets a clear abort."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    from dataset_scout.errors import LLMError

    with patch(
        "dataset_scout.decompose.decompose_intent",
        side_effect=LLMError("simulated network error"),
    ), patch("dataset_scout.pipeline.run_recon") as run_recon:
        result = runner.invoke(
            app,
            [
                "recon",
                "test brief",
                "--review",
                "--no-papers",
                "--out",
                str(tmp_path / "out"),
            ],
            input="\n",
        )
    assert result.exit_code == 130
    assert run_recon.call_count == 0


def test_review_skipped_when_decomposition_from_provided(
    runner, tmp_path, monkeypatch
):
    """`--review` is a no-op when the user already supplied a decomposition file."""
    from dataset_scout.decomposition_io import write_decomposition

    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://x")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    decomp_path = write_decomposition(_fake_directions(), tmp_path / "decomp.yaml")
    assert decomp_path is not None

    with patch("dataset_scout.pipeline.run_recon") as run_recon:
        from dataset_scout.core import Intent, ReconResult
        run_recon.return_value = ReconResult(
            intent=Intent(raw_brief="x"), candidates=[]
        )
        result = runner.invoke(
            app,
            [
                "recon",
                "test brief",
                "--review",
                "--decomposition-from",
                str(decomp_path),
                "--no-papers",
                "--out",
                str(tmp_path / "out"),
            ],
            input="",  # No prompt should be shown.
        )
    assert result.exit_code == 0, result.output
    # Recon was called directly with the supplied directions; no review prompt.
    assert run_recon.call_count == 1
