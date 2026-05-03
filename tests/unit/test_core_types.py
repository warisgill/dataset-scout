"""Schema and behavior tests for the core types."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from dataset_scout import (
    Candidate,
    CoverageGap,
    CoverageReport,
    DecompositionDirection,
    Evidence,
    Intent,
    LabelKind,
    LicensePolicy,
    Scorecard,
    SensitiveDomain,
    Strategy,
    StrategyKind,
    SubScore,
    TransformSpec,
)

pytestmark = pytest.mark.unit


# ─── Intent ─────────────────────────────────────────────────────────


def test_intent_minimal_construction():
    i = Intent(raw_brief="find me prompt injection datasets")
    assert i.raw_brief == "find me prompt injection datasets"
    assert i.languages == ["en"]
    assert i.sensitive_domain is SensitiveDomain.NONE
    assert 0.0 <= i.min_strategy_confidence <= 1.0


def test_intent_round_trip_json():
    i = Intent(
        raw_brief="brief",
        detection_target="prompt injection",
        threat_families=["injection", "jailbreak"],
        languages=["en", "ja"],
        min_strategy_confidence=0.7,
    )
    dumped = i.model_dump_json()
    parsed = json.loads(dumped)
    assert parsed["detection_target"] == "prompt injection"
    rehydrated = Intent.model_validate_json(dumped)
    assert rehydrated == i


def test_intent_stable_hash_is_deterministic_and_field_sensitive():
    a = Intent(raw_brief="x", detection_target="y")
    b = Intent(raw_brief="x", detection_target="y")
    c = Intent(raw_brief="x", detection_target="z")
    assert a.stable_hash() == b.stable_hash()
    assert a.stable_hash() != c.stable_hash()
    assert len(a.stable_hash()) == 64
    assert all(ch in "0123456789abcdef" for ch in a.stable_hash())


def test_intent_stable_hash_is_order_independent_for_set_fields():
    a = Intent(
        raw_brief="x",
        license_policy=LicensePolicy(allow=frozenset({"MIT", "Apache-2.0"})),
    )
    b = Intent(
        raw_brief="x",
        license_policy=LicensePolicy(allow=frozenset({"Apache-2.0", "MIT"})),
    )
    assert a.stable_hash() == b.stable_hash()


def test_intent_min_strategy_confidence_bounded():
    with pytest.raises(ValidationError):
        Intent(raw_brief="x", min_strategy_confidence=1.5)
    with pytest.raises(ValidationError):
        Intent(raw_brief="x", min_strategy_confidence=-0.1)


def test_intent_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Intent(raw_brief="x", made_up_field=True)  # type: ignore[call-arg]


def test_license_policy_accepts_list_input():
    p = LicensePolicy(allow=["MIT", "Apache-2.0"])  # type: ignore[arg-type]
    assert "MIT" in p.allow
    assert isinstance(p.allow, frozenset)


# ─── Candidate ──────────────────────────────────────────────────────


def test_candidate_minimal():
    c = Candidate(source="huggingface", id="deepset/prompt-injections", revision="abc123")
    assert c.streamable is True
    assert c.requires_auth is False
    assert c.direction is None


def test_candidate_with_direction():
    c = Candidate(
        source="huggingface",
        id="walledai/AdvBench",
        revision=None,
        direction="hard_negatives_from_jailbreaks",
    )
    assert c.direction == "hard_negatives_from_jailbreaks"


# ─── Strategy / Scorecard ───────────────────────────────────────────


def _ts(**kw) -> TransformSpec:
    return TransformSpec(**kw)


def test_strategy_confidence_bounds():
    with pytest.raises(ValidationError):
        Strategy(
            kind=StrategyKind.DIRECT_USE,
            confidence=1.2,
            rationale="x",
            transform=_ts(),
        )


def test_label_kind_values():
    assert {k.value for k in LabelKind} == {
        "ground_truth",
        "remapped",
        "proxy",
        "subset_extracted",
    }


def test_strategy_taxonomy_has_eight_kinds():
    assert len(StrategyKind) == 8


def test_scorecard_best_strategy_picks_highest_confidence():
    cand = Candidate(source="huggingface", id="x/y", revision="r")
    s_low = Strategy(
        kind=StrategyKind.SIGNAL_PROXY,
        confidence=0.3,
        rationale="proxy",
        transform=_ts(),
    )
    s_high = Strategy(
        kind=StrategyKind.DIRECT_USE,
        confidence=0.9,
        rationale="direct",
        transform=_ts(),
    )
    sc = Scorecard(candidate=cand, strategies=[s_low, s_high])
    assert sc.best_strategy is s_high


def test_scorecard_best_strategy_is_none_when_empty():
    cand = Candidate(source="huggingface", id="x/y", revision="r")
    sc = Scorecard(candidate=cand)
    assert sc.best_strategy is None


# ─── SubScore / Evidence ────────────────────────────────────────────


def test_subscore_round_trip():
    s = SubScore(
        value=0.42,
        confidence_interval=(0.3, 0.55),
        n=100,
        evidence=[Evidence(kind="x", detail="y", value=0.42)],
    )
    rehydrated = SubScore.model_validate_json(s.model_dump_json())
    assert rehydrated == s


# ─── CoverageReport ─────────────────────────────────────────────────


def test_coverage_report_notable_threshold():
    cr = CoverageReport()
    assert cr.notable is False
    cr_one = CoverageReport(
        semantic_gaps=[CoverageGap(aspect="a", description="d", suggestion="s")]
    )
    assert cr_one.notable is False
    cr_two = CoverageReport(
        semantic_gaps=[
            CoverageGap(aspect="a", description="d", suggestion="s"),
            CoverageGap(aspect="b", description="d", suggestion="s"),
        ]
    )
    assert cr_two.notable is True


# ─── DecompositionDirection ─────────────────────────────────────────


def test_decomposition_direction_minimal():
    d = DecompositionDirection(name="hard_negs", rationale="why")
    assert d.keywords == []
    assert d.threat_families == []
