"""Tests for the test-only `FakeSource` double."""

from __future__ import annotations

import pytest

from dataset_scout.core import Candidate, CandidateMetadata, Intent
from dataset_scout.sources.base import Budget, Obligation, Source
from tests._fakes.fake_source import FakeSource

pytestmark = pytest.mark.unit


def _candidates() -> list[Candidate]:
    return [
        Candidate(source="fake", id="org/alpha", revision="r1"),
        Candidate(source="fake", id="org/beta", revision="r2"),
    ]


def test_fake_source_satisfies_source_protocol():
    fake = FakeSource(candidates=_candidates())
    assert isinstance(fake, Source)
    assert fake.name == "fake"


def test_search_yields_canned_candidates_regardless_of_intent():
    cands = _candidates()
    fake = FakeSource(candidates=cands)
    intent = Intent(raw_brief="anything")
    results = list(fake.search(intent, directions=[], budget=Budget()))
    assert results == cands
    assert fake.search_calls == 1


def test_stream_sample_returns_canned_rows_and_respects_n():
    cands = _candidates()
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    fake = FakeSource(candidates=cands, samples={"org/alpha": rows})

    sampled = list(fake.stream_sample(cands[0], n=2, seed=0))
    assert sampled == rows[:2]

    # Unknown candidate id → empty stream, still counts as a call.
    other = list(fake.stream_sample(cands[1], n=5, seed=0))
    assert other == []
    assert fake.sample_calls == 2


def test_fetch_metadata_returns_dict_view_of_candidate_metadata():
    cand = Candidate(
        source="fake",
        id="org/x",
        revision="r",
        metadata=CandidateMetadata(extras={"k": 1}),
    )
    fake = FakeSource(candidates=[cand])
    md = fake.fetch_metadata(cand)
    assert md["extras"] == {"k": 1}
    md["extras"]["k"] = 2
    assert cand.metadata.extras == {"k": 1}  # caller mutation does not leak back
    assert fake.metadata_calls == 1


def test_card_url_and_terms_check_counters():
    cand = _candidates()[0]
    obligation = Obligation(source="fake", summary="agree to ToS", url="https://x/y")
    fake = FakeSource(candidates=[cand], obligations=[obligation])

    url = fake.card_url(cand)
    assert url.endswith(cand.id)
    assert fake.card_url_calls == 1

    obligations = fake.terms_check(Intent(raw_brief="x"))
    assert obligations == [obligation]
    assert fake.terms_check_calls == 1
