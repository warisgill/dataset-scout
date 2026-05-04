"""Tests for the `datascout tour` demo command (recommendation C)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_scout.tour import build_tour_result, render_tour

pytestmark = pytest.mark.unit


def test_build_tour_result_is_well_populated():
    result = build_tour_result()
    assert len(result.candidates) >= 3
    assert all(sc.strategies for sc in result.candidates)
    assert result.coverage is not None
    assert result.coverage.decomposition  # at least one direction
    assert result.coverage.semantic_gaps  # at least one gap


def test_render_tour_includes_all_sections():
    md = render_tour(out_dir=None)
    assert "# dataset-scout recon report" in md
    assert "## Decomposition" in md
    assert "## Candidates" in md
    # Strategy assessment present (any framing).
    assert "Strategy" in md or "strategy" in md
    assert "## Sourcing roadmap" in md or "Coverage gap" in md
    assert "**Strategies:**" in md


def test_render_tour_persists_artefacts(tmp_path: Path):
    md = render_tour(out_dir=tmp_path)
    assert md
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "recipe.draft.yaml").exists()
    assert (tmp_path / "decomposition.yaml").exists()
