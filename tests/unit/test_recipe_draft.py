"""Unit tests for recipe.draft.yaml emission."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    Intent,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    TransformSpec,
)
from dataset_scout.recipe_draft import build_recipe_draft, write_recipe_draft

pytestmark = pytest.mark.unit


def _strategy(kind: StrategyKind, confidence: float = 0.8, rationale: str = "because") -> Strategy:
    return Strategy(
        kind=kind,
        confidence=confidence,
        rationale=rationale,
        transform=TransformSpec(text_column="text", label_column="label"),
    )


def _scorecard(cid: str, strategies: list[Strategy]) -> Scorecard:
    return Scorecard(
        candidate=Candidate(
            source="huggingface",
            id=cid,
            revision="r1",
            metadata=CandidateMetadata(),
        ),
        strategies=strategies,
    )


def _result(scorecards: list[Scorecard], min_confidence: float = 0.5) -> ReconResult:
    return ReconResult(
        intent=Intent(
            raw_brief="prompt injection corpora",
            detection_target="prompt injection",
            threat_families=["prompt_injection"],
            min_strategy_confidence=min_confidence,
        ),
        candidates=scorecards,
        sources_searched=["huggingface"],
    )


def test_empty_draft_for_no_strategies(tmp_path: Path):
    """write_recipe_draft returns None when no candidate has strategies."""
    result = _result([_scorecard("a", [])])
    target = write_recipe_draft(result, tmp_path)
    assert target is None
    assert not (tmp_path / "recipe.draft.yaml").exists()


def test_draft_includes_above_threshold_and_declines_below():
    s_high = _strategy(StrategyKind.DIRECT_USE, confidence=0.9)
    s_low = _strategy(StrategyKind.SIGNAL_PROXY, confidence=0.3)
    result = _result(
        [_scorecard("good", [s_high]), _scorecard("weak", [s_low])],
        min_confidence=0.5,
    )
    draft = build_recipe_draft(result)
    component_ids = {c["source_id"] for c in draft["components"]}
    declined_ids = {c["source_id"] for c in draft["declined"]}
    assert component_ids == {"good"}
    assert declined_ids == {"weak"}


def test_draft_yaml_round_trips(tmp_path: Path):
    s = _strategy(StrategyKind.DIRECT_USE, confidence=0.85)
    result = _result([_scorecard("org/x", [s])])
    target = write_recipe_draft(result, tmp_path)
    assert target is not None
    # YAML must round-trip cleanly.
    parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert parsed["intent"]["brief"] == "prompt injection corpora"
    assert parsed["min_strategy_confidence"] == 0.5
    assert len(parsed["components"]) == 1
    component = parsed["components"][0]
    assert component["source"] == "huggingface"
    assert component["source_id"] == "org/x"
    assert component["strategy"] == "direct_use"
    assert component["transform"]["text_column"] == "text"


def test_default_label_kind_for_signal_proxy():
    s = _strategy(StrategyKind.SIGNAL_PROXY, confidence=0.7)
    result = _result([_scorecard("org/p", [s])])
    draft = build_recipe_draft(result)
    component = draft["components"][0]
    # signal_proxy without an explicit label_kind_map should default to proxy.
    assert component["transform"]["label_kind_map"] == {"all": "proxy"}


def test_not_useful_is_declined():
    s = _strategy(StrategyKind.NOT_USEFUL, confidence=0.95)
    result = _result([_scorecard("org/n", [s])])
    draft = build_recipe_draft(result)
    assert draft["components"] == []
    assert draft["declined"][0]["source_id"] == "org/n"


def test_draft_intent_block_complete():
    s = _strategy(StrategyKind.DIRECT_USE, confidence=0.9)
    result = _result([_scorecard("org/x", [s])])
    draft = build_recipe_draft(result)
    intent_block = draft["intent"]
    assert intent_block["brief"] == "prompt injection corpora"
    assert intent_block["detection_target"] == "prompt injection"
    assert intent_block["threat_families"] == ["prompt_injection"]
    assert intent_block["languages"] == ["en"]
    assert "MIT" in intent_block["license_allow"]
