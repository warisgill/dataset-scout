"""Unit tests for the two-stage shortlist."""

from __future__ import annotations

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    Evidence,
    Scorecard,
    SubScore,
)
from dataset_scout.shortlist import select_top_for_assessor

pytestmark = pytest.mark.unit


def _sc(
    cid: str,
    *,
    surfaced_by: list[str] | None = None,
    license_value: float | None = 1.0,
    cc_value: float | None = 0.5,
) -> Scorecard:
    cand = Candidate(
        source="huggingface",
        id=cid,
        revision="r",
        surfaced_by=list(surfaced_by or []),
        metadata=CandidateMetadata(),
    )
    cheap: dict[str, SubScore] = {}
    if license_value is not None:
        cheap["license"] = SubScore(
            value=license_value,
            evidence=[Evidence(kind="license_spdx", detail="MIT")],
        )
    if cc_value is not None:
        cheap["card_completeness"] = SubScore(
            value=cc_value,
            evidence=[Evidence(kind="present", detail="license")],
        )
    return Scorecard(candidate=cand, cheap_probes=cheap)


def test_empty_input_returns_empty():
    assert select_top_for_assessor([]) == []


def test_small_input_passes_through():
    cards = [_sc("a"), _sc("b"), _sc("c")]
    out = select_top_for_assessor(cards, top_per_direction=5, total_cap=20)
    assert {sc.candidate.id for sc in out} == {"a", "b", "c"}


def test_per_direction_topk_preserves_breadth():
    """Each direction contributes its first top_per_direction candidates."""
    cards = [_sc(f"a{i}", surfaced_by=["dir_a"]) for i in range(8)] + [
        _sc(f"b{i}", surfaced_by=["dir_b"]) for i in range(8)
    ]
    out = select_top_for_assessor(cards, top_per_direction=3, total_cap=20)
    ids = {sc.candidate.id for sc in out}
    # Top 3 from each group present.
    assert {"a0", "a1", "a2"} <= ids
    assert {"b0", "b1", "b2"} <= ids
    # Beyond top-3 dropped.
    assert "a5" not in ids
    assert "b5" not in ids


def test_total_cap_respected():
    cards = [_sc(f"x{i}", surfaced_by=["dir"]) for i in range(50)]
    out = select_top_for_assessor(cards, top_per_direction=20, total_cap=10)
    assert len(out) == 10


def test_multi_direction_hits_float_to_top_in_stage_2():
    """Candidates surfaced by multiple directions outrank singletons."""
    multi = _sc("multi", surfaced_by=["a", "b", "c"], license_value=1.0)
    single_a = _sc("only_a", surfaced_by=["a"], license_value=1.0)
    single_b = _sc("only_b", surfaced_by=["b"], license_value=1.0)
    out = select_top_for_assessor([single_a, single_b, multi], top_per_direction=5, total_cap=10)
    assert out[0].candidate.id == "multi"


def test_license_breaks_tie_when_directions_match():
    good_license = _sc("good", surfaced_by=["a"], license_value=1.0)
    bad_license = _sc("bad", surfaced_by=["a"], license_value=0.0)
    out = select_top_for_assessor([bad_license, good_license], top_per_direction=5, total_cap=10)
    assert out[0].candidate.id == "good"


def test_original_intent_grouping():
    """Candidates with empty surfaced_by group together under None."""
    originals = [_sc(f"orig_{i}", surfaced_by=None) for i in range(5)]
    direction = [_sc(f"d_{i}", surfaced_by=["d"]) for i in range(5)]
    out = select_top_for_assessor(originals + direction, top_per_direction=2, total_cap=20)
    ids = {sc.candidate.id for sc in out}
    # 2 from each group → 4 total.
    assert len(ids) == 4


def test_recalled_name_rescues_low_ranked_candidate():
    """A candidate matching a recalled name is force-included even if
    cheap signals + rank would have dropped it."""
    # 30 strong candidates would fill up a small cap.
    fillers = [
        _sc(f"filler_{i}", surfaced_by=["dir_x"], license_value=1.0)
        for i in range(30)
    ]
    # Buried candidate with weak signals; its id matches a recalled name.
    buried = _sc(
        "AI-companionship/INTIMA",
        surfaced_by=["dir_y"],
        license_value=0.0,
        cc_value=0.0,
    )
    out = select_top_for_assessor(
        [*fillers, buried],
        top_per_direction=5,
        total_cap=10,
        recalled_names=["INTIMA"],
    )
    ids = [sc.candidate.id for sc in out]
    assert "AI-companionship/INTIMA" in ids
    # And it lands at the front since rescues are surfaced first.
    assert ids[0] == "AI-companionship/INTIMA"


def test_recalled_name_match_is_normalised():
    """'Persona Chat' matches 'PersonaChat' and 'persona-chat' (alphanum-only)."""
    a = _sc("foo/PersonaChat", surfaced_by=["d"])
    b = _sc("foo/persona-chat", surfaced_by=["d"])
    c = _sc("foo/UnrelatedDataset", surfaced_by=["d"])
    out = select_top_for_assessor(
        [c, a, b],
        top_per_direction=5,
        total_cap=10,
        recalled_names=["Persona Chat"],
    )
    ids = [sc.candidate.id for sc in out]
    assert "foo/PersonaChat" in ids
    assert "foo/persona-chat" in ids


def test_recalled_name_rescue_no_duplicates():
    """A rescued candidate is not also re-emitted via the per-direction stage."""
    sc = _sc("foo/INTIMA", surfaced_by=["dir_y"])
    out = select_top_for_assessor([sc], total_cap=10, recalled_names=["INTIMA"])
    assert len(out) == 1


def test_default_total_cap_is_35():
    """Defaulted shortlist cap should be 35 (raised from 20)."""
    cards = [_sc(f"x{i}", surfaced_by=["dir"]) for i in range(80)]
    out = select_top_for_assessor(cards, top_per_direction=80)
    assert len(out) == 35
