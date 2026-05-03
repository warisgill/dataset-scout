"""End-to-end recon pipeline test using a FakeSource.

Exercises the full slice — parse -> search -> probes -> ReconResult ->
results.json + report.md — without touching the network. The CLI itself
is exercised separately in `test_cli.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    ScoutContext,
)
from dataset_scout.pipeline import run_recon
from dataset_scout.render import write_recon_report, write_results_json
from tests._fakes.fake_source import FakeSource

pytestmark = pytest.mark.unit


def _fixture_candidates() -> list[Candidate]:
    now = datetime.now(UTC)
    return [
        Candidate(
            source="fake",
            id="org/clean-permissive",
            revision="r1",
            metadata=CandidateMetadata(
                description="A neatly-licensed permissive dataset for prompt injection.",
                card_url="https://example.com/datasets/org/clean-permissive",
                license_raw="mit",
                license_spdx="MIT",
                languages_declared=["en"],
                uploaded_at=now - timedelta(days=30),
                last_modified=now - timedelta(days=10),
                rows=10_000,
                downloads=2_500,
                likes=120,
                card_fields_present=frozenset(
                    {"license", "language", "task_categories", "pretty_name", "size_categories"}
                ),
                tags=["text-classification"],
                task_categories=["text-classification"],
            ),
        ),
        Candidate(
            source="fake",
            id="org/aging-cc-share-alike",
            revision="r2",
            metadata=CandidateMetadata(
                description="An older corpus on a share-alike license.",
                card_url="https://example.com/datasets/org/aging-cc-share-alike",
                license_raw="cc-by-sa-4.0",
                license_spdx="CC-BY-SA-4.0",
                languages_declared=["en", "ja"],
                uploaded_at=now - timedelta(days=900),
                last_modified=now - timedelta(days=800),
                rows=500_000,
                downloads=42,
                card_fields_present=frozenset({"license", "language"}),
                tags=["text-classification"],
            ),
        ),
        Candidate(
            source="fake",
            id="org/silent-card",
            revision="r3",
            metadata=CandidateMetadata(
                description="A poorly-documented dataset.",
                card_url="https://example.com/datasets/org/silent-card",
            ),
        ),
    ]


def test_run_recon_end_to_end(tmp_path: Path) -> None:
    fake = FakeSource(_fixture_candidates())
    ctx = ScoutContext.from_env(env={})
    result = run_recon(
        "find labeled prompt injection datasets in english",
        ctx=ctx,
        sources=[fake],
    )

    # Three candidates surfaced; source-relevance order preserved.
    ids = [sc.candidate.id for sc in result.candidates]
    assert ids == [
        "org/clean-permissive",
        "org/aging-cc-share-alike",
        "org/silent-card",
    ]
    assert result.sources_searched == ["fake"]
    assert fake.search_calls == 1

    # Every scorecard ran the unconditional probes; conditional ones
    # (recency, freshness) only run when applicable.
    always_run = {"license", "size", "languages", "card_completeness"}
    for sc in result.candidates:
        assert always_run <= set(sc.cheap_probes)
    # Candidates with dates also got recency + freshness.
    assert {"recency", "freshness"} <= set(result.candidates[0].cheap_probes)
    # The silent-card candidate had no dates -> no recency/freshness.
    assert "freshness" not in result.candidates[2].cheap_probes

    # License probe distinguishes allow / warn-only / missing.
    assert result.candidates[0].cheap_probes["license"].value == 1.0
    assert result.candidates[1].cheap_probes["license"].value == 0.5
    assert result.candidates[2].cheap_probes["license"].status in {
        "not_applicable",
        "low_confidence",
    }

    # Renderers emit valid JSON and a Markdown report containing the
    # discovery framing language.
    json_path = write_results_json(result, tmp_path)
    md_path = write_recon_report(result, tmp_path)

    assert json_path.exists() and md_path.exists()
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["intent"]["raw_brief"].startswith("find labeled")
    assert len(parsed["candidates"]) == 3

    md = md_path.read_text(encoding="utf-8")
    assert "discovery report" in md.lower()
    assert "pre-fit metadata screening" in md.lower()
    # Discovery framing must be explicit anywhere ranking is mentioned.
    assert "not a ranking" in md.lower() or "not a fitness ranking" in md.lower()
    for cand_id in ids:
        assert cand_id in md


def test_run_recon_records_progress_events() -> None:
    fake = FakeSource(_fixture_candidates())
    ctx = ScoutContext.from_env(env={})
    events: list = []
    run_recon("brief", ctx=ctx, sources=[fake], events=events)

    stages = {e.stage for e in events}
    assert {"parse", "search", "probe"} <= stages
    # At least one CANDIDATE_FOUND per result.
    found_kinds = [e.kind for e in events if e.stage == "search"]
    assert "candidate_found" in [str(k) for k in found_kinds]


def test_run_recon_handles_failing_source() -> None:
    class BoomSource(FakeSource):
        def search(self, intent, directions, *, budget):  # type: ignore[override]
            self.search_calls += 1
            raise RuntimeError("boom")

    boom = BoomSource([])
    ctx = ScoutContext.from_env(env={})
    result = run_recon("brief", ctx=ctx, sources=[boom])
    assert result.candidates == []
    assert any("failed" in n for n in result.notices)


def test_run_recon_empty_results_emit_helpful_notice() -> None:
    fake = FakeSource([])
    ctx = ScoutContext.from_env(env={})
    result = run_recon("brief", ctx=ctx, sources=[fake])
    assert result.candidates == []
    assert any("broaden" in n.lower() for n in result.notices)
