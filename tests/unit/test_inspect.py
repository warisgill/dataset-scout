"""Unit tests for the inspect deep-dive."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    DatasetScoutError,
    Intent,
    ScoutContext,
)
from dataset_scout.inspect_ import (
    load_intent_from,
    make_intent,
    parse_target,
    run_inspect,
)
from dataset_scout.render.inspect_panel import render_inspect
from tests._fakes.fake_source import FakeSource

pytestmark = pytest.mark.unit


# ─── parse_target ────────────────────────────────────────────────────


def test_parse_target_full():
    assert parse_target("huggingface:org/x@abc123") == ("huggingface", "org/x", "abc123")


def test_parse_target_no_revision():
    assert parse_target("huggingface:org/x") == ("huggingface", "org/x", None)


def test_parse_target_defaults_to_huggingface():
    assert parse_target("org/x") == ("huggingface", "org/x", None)


def test_parse_target_rejects_empty():
    with pytest.raises(DatasetScoutError):
        parse_target(":")
    with pytest.raises(DatasetScoutError):
        parse_target("huggingface:")


# ─── intent reuse ────────────────────────────────────────────────────


def test_load_intent_from_results_json(tmp_path: Path):
    intent = Intent(raw_brief="prompt injection corpora", detection_target="prompt injection")
    payload = {"intent": intent.model_dump(mode="json"), "candidates": []}
    target = tmp_path / "results.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_intent_from(target)
    assert loaded.raw_brief == "prompt injection corpora"
    assert loaded.detection_target == "prompt injection"


def test_load_intent_from_rejects_non_results_json(tmp_path: Path):
    target = tmp_path / "junk.json"
    target.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    with pytest.raises(DatasetScoutError):
        load_intent_from(target)


def test_make_intent_priority(tmp_path: Path):
    """intent_from > brief > None."""
    target = tmp_path / "results.json"
    target.write_text(
        json.dumps({"intent": Intent(raw_brief="from-file").model_dump(mode="json")}),
        encoding="utf-8",
    )
    assert make_intent(brief=None, intent_from=None) is None
    via_brief = make_intent(brief="from-brief", intent_from=None)
    assert via_brief is not None and via_brief.raw_brief == "from-brief"
    via_file = make_intent(brief="from-brief", intent_from=target)
    assert via_file is not None and via_file.raw_brief == "from-file"


# ─── run_inspect end-to-end with FakeSource ──────────────────────────


def _fake_with_rows(rows: list[dict[str, object]]) -> FakeSource:
    cand = Candidate(
        source="fake",
        id="org/x",
        revision="r1",
        metadata=CandidateMetadata(
            description="a labeled corpus for testing",
            license_raw="mit",
            license_spdx="MIT",
            languages_declared=["en"],
            tags=["text-classification"],
        ),
    )
    return FakeSource([cand], samples={"org/x": rows})


def _ctx() -> ScoutContext:
    return ScoutContext.from_env(env={})


def _two_class_rows(n: int = 50) -> list[dict[str, object]]:
    return [
        {
            "id": i,
            "text": f"row {i} content " * (i % 3 + 1),
            "label": 1 if i % 3 == 0 else 0,
        }
        for i in range(n)
    ]


def test_run_inspect_basic_no_intent():
    fake = _fake_with_rows(_two_class_rows())
    candidate = next(iter(fake._candidates))  # type: ignore[attr-defined]
    result = run_inspect(
        "fake:org/x",
        ctx=_ctx(),
        intent=None,
        source_override=fake,
        candidate_override=candidate,
    )
    assert result.candidate.id == "org/x"
    assert result.sample_size == 50
    assert {c.name for c in result.columns} == {"id", "text", "label"}
    assert result.label_column_used == "label"
    assert len(result.label_distribution) == 2
    assert sum(b.count for b in result.label_distribution) == 50
    # Wilson CI is non-trivial.
    for b in result.label_distribution:
        assert b.ci_low <= b.fraction <= b.ci_high
    assert result.length_stats is not None
    assert result.length_stats.column == "text"
    assert result.length_stats.n == 50
    assert result.strategies == []  # no intent, no LLM


def test_run_inspect_with_intent_no_aoai_skips_assessment():
    fake = _fake_with_rows(_two_class_rows())
    candidate = next(iter(fake._candidates))  # type: ignore[attr-defined]
    result = run_inspect(
        "fake:org/x",
        ctx=_ctx(),
        intent=Intent(raw_brief="x"),
        source_override=fake,
        candidate_override=candidate,
    )
    assert result.strategies == []
    assert any("Strategy assessment skipped" in n for n in result.notices)


def test_run_inspect_renders_to_markdown():
    fake = _fake_with_rows(_two_class_rows(20))
    candidate = next(iter(fake._candidates))  # type: ignore[attr-defined]
    result = run_inspect(
        "fake:org/x",
        ctx=_ctx(),
        intent=None,
        source_override=fake,
        candidate_override=candidate,
    )
    md = render_inspect(result)
    assert "fake:org/x" in md
    assert "Schema" in md
    assert "Label distribution" in md
    assert "Sample rows" in md
    assert "MIT" in md  # license shown


def test_run_inspect_handles_missing_label_column():
    rows = [{"id": i, "text": f"row {i}"} for i in range(5)]
    fake = _fake_with_rows(rows)
    candidate = next(iter(fake._candidates))  # type: ignore[attr-defined]
    result = run_inspect(
        "fake:org/x",
        ctx=_ctx(),
        intent=None,
        source_override=fake,
        candidate_override=candidate,
    )
    assert result.label_column_used is None
    assert result.label_distribution == []
    # Length stats still populated.
    assert result.length_stats is not None


def test_inspect_result_round_trips_json():
    fake = _fake_with_rows(_two_class_rows(10))
    candidate = next(iter(fake._candidates))  # type: ignore[attr-defined]
    result = run_inspect(
        "fake:org/x",
        ctx=_ctx(),
        intent=None,
        source_override=fake,
        candidate_override=candidate,
    )
    from dataset_scout import InspectResult

    rehydrated = InspectResult.model_validate_json(result.model_dump_json())
    assert rehydrated.candidate.id == result.candidate.id
    assert len(rehydrated.label_distribution) == len(result.label_distribution)
