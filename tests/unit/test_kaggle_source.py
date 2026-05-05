"""Unit tests for `dataset_scout.sources.kaggle`."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    DecompositionDirection,
    Intent,
)
from dataset_scout.errors import SourceUnsupportedError
from dataset_scout.sources.base import Budget
from dataset_scout.sources.kaggle import (
    KaggleSource,
    _build_metadata,
    _coerce_kaggle_dt,
    _direction_queries,
    _ref_from_payload,
    kaggle_credentials,
)

pytestmark = pytest.mark.unit


# ─── translation tests ───────────────────────────────────────────────


def test_ref_from_payload_top_level():
    assert _ref_from_payload({"ref": "owner/slug"}) == "owner/slug"


def test_ref_from_payload_falls_back_to_owner_slug():
    assert (
        _ref_from_payload({"ownerName": "foo", "urlSlug": "bar"})
        == "foo/bar"
    )


def test_ref_from_payload_returns_none_when_unbuildable():
    assert _ref_from_payload({}) is None
    assert _ref_from_payload({"ref": "no-slash"}) is None


def test_coerce_kaggle_dt_iso():
    dt = _coerce_kaggle_dt("2024-01-15T10:30:00Z")
    assert dt is not None
    assert dt.year == 2024


def test_coerce_kaggle_dt_none_and_garbage():
    assert _coerce_kaggle_dt(None) is None
    assert _coerce_kaggle_dt("not-a-date") is None


def test_build_metadata_full_payload():
    payload = {
        "ref": "alice/some-dataset",
        "title": "Some Dataset",
        "subtitle": "Useful for X",
        "description": "Long-form description here.",
        "licenseName": "CC0-1.0",
        "totalBytes": 1234567,
        "downloadCount": 42,
        "voteCount": 7,
        "lastUpdated": "2024-06-01T00:00:00Z",
        "creationDate": "2024-01-01T00:00:00Z",
        "tags": [{"name": "nlp"}, {"name": "classification"}],
        "creatorName": "Alice",
        "usabilityRating": 0.85,
        "isPrivate": False,
    }
    md = _build_metadata(payload)
    assert isinstance(md, CandidateMetadata)
    assert md.description == "Long-form description here."
    assert md.license_raw == "CC0-1.0"
    assert md.license_spdx == "CC0-1.0"
    assert md.bytes == 1234567
    assert md.downloads == 42
    assert md.likes == 7
    assert md.tags == ["nlp", "classification"]
    assert md.extras["creator"] == "Alice"
    assert md.extras["usability_rating"] == pytest.approx(0.85)
    assert md.requires_auth is False
    assert md.gated is False


def test_build_metadata_handles_missing_fields():
    md = _build_metadata({"ref": "x/y"})
    assert md.description is None
    assert md.license_raw is None
    assert md.license_spdx is None
    assert md.tags == []
    assert md.bytes is None


def test_direction_queries_uses_keywords():
    d = DecompositionDirection(
        name="d1",
        rationale="r",
        keywords=["a", "b", "c", "d"],
        threat_families=[],
        expected_finds="x",
    )
    assert _direction_queries(d) == ["a", "b", "c"]


def test_direction_queries_falls_back_to_name():
    d = DecompositionDirection(
        name="my_dir",
        rationale="r",
        keywords=[],
        threat_families=[],
        expected_finds="x",
    )
    assert _direction_queries(d) == ["my dir"]


# ─── credentials tests ───────────────────────────────────────────────


def test_kaggle_credentials_from_ctx_api_keys(monkeypatch):
    """ctx.api_keys takes precedence over ~/.kaggle/kaggle.json."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    ctx = ScoutContext(api_keys={"KAGGLE_USERNAME": "alice", "KAGGLE_KEY": "k123"})
    assert kaggle_credentials(ctx) == ("alice", "k123")


def test_kaggle_credentials_from_env(monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
    monkeypatch.setenv("KAGGLE_KEY", "envkey")
    ctx = ScoutContext()  # no api_keys
    # Ensure no kaggle.json exists for the home dir to muddy the test.
    monkeypatch.setattr(Path, "home", lambda: Path("/no-such-home-for-this-test"))
    assert kaggle_credentials(ctx) == ("envuser", "envkey")


def test_kaggle_credentials_from_kaggle_json(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kdir = tmp_path / ".kaggle"
    kdir.mkdir()
    (kdir / "kaggle.json").write_text(
        json.dumps({"username": "filealice", "key": "filekey"}), encoding="utf-8"
    )
    ctx = ScoutContext()
    assert kaggle_credentials(ctx) == ("filealice", "filekey")


def test_kaggle_credentials_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ctx = ScoutContext()
    assert kaggle_credentials(ctx) is None


# ─── search / fetch tests with respx ────────────────────────────────


def _make_intent(brief: str = "labeled toxicity") -> Intent:
    return Intent(raw_brief=brief, threat_families=[])


@respx.mock
def test_search_yields_candidates_with_metadata():
    respx.get("https://www.kaggle.com/api/v1/datasets/list").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "ref": "alice/toxicity",
                    "title": "Toxicity",
                    "description": "Comments labeled toxic / non-toxic.",
                    "licenseName": "MIT",
                    "totalBytes": 999,
                    "downloadCount": 100,
                    "voteCount": 10,
                    "tags": [{"name": "nlp"}],
                    "lastUpdated": "2024-06-01T00:00:00Z",
                },
                {
                    "ref": "bob/another",
                    "title": "Another",
                    "licenseName": "Apache-2.0",
                    "downloadCount": 5,
                },
            ],
        )
    )
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    candidates = list(
        src.search(_make_intent(), directions=[], budget=Budget())
    )
    assert len(candidates) == 2
    assert candidates[0].source == "kaggle"
    assert candidates[0].id == "alice/toxicity"
    assert candidates[0].streamable is False
    assert candidates[0].metadata.license_spdx == "MIT"
    assert candidates[1].id == "bob/another"


@respx.mock
def test_search_round_robin_across_directions():
    """Multi-direction search interleaves results from each query."""
    # Each query returns a different single candidate so we can see ordering.
    route_call = {"count": 0}

    def _handler(request):
        route_call["count"] += 1
        # Two directions x 1 keyword each + 1 original brief = 3 queries.
        # Return a unique candidate per query so we can identify it.
        q = request.url.params.get("search") or "?"
        return httpx.Response(
            200,
            json=[{"ref": f"o/{q.replace(' ', '_')}-1", "title": q}],
        )

    respx.get("https://www.kaggle.com/api/v1/datasets/list").mock(side_effect=_handler)

    directions = [
        DecompositionDirection(
            name="d1",
            rationale="",
            keywords=["alpha"],
            threat_families=[],
            expected_finds="",
        ),
        DecompositionDirection(
            name="d2",
            rationale="",
            keywords=["beta"],
            threat_families=[],
            expected_finds="",
        ),
    ]
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    cands = list(src.search(_make_intent("toxicity"), directions, budget=Budget()))
    # 3 queries x 1 candidate each = 3 candidates.
    assert len(cands) == 3
    # surfaced_by reflects which query produced each.
    surfaced = [list(c.surfaced_by) for c in cands]
    # Original-brief candidate has empty surfaced_by; direction ones carry the name.
    assert [] in surfaced
    assert ["d1"] in surfaced
    assert ["d2"] in surfaced


@respx.mock
def test_search_swallows_http_errors():
    """Network failures yield zero candidates, not exceptions."""
    respx.get("https://www.kaggle.com/api/v1/datasets/list").mock(
        return_value=httpx.Response(500, text="boom")
    )
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    assert list(src.search(_make_intent(), directions=[], budget=Budget())) == []


@respx.mock
def test_fetch_metadata_returns_payload():
    respx.get(
        "https://www.kaggle.com/api/v1/datasets/view/alice/toxicity"
    ).mock(return_value=httpx.Response(200, json={"ref": "alice/toxicity", "title": "T"}))
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    cand = Candidate(
        source="kaggle",
        id="alice/toxicity",
        revision=None,
        metadata=CandidateMetadata(
            description=None,
            card_url="https://www.kaggle.com/datasets/alice/toxicity",
        ),
        streamable=False,
        surfaced_by=[],
    )
    payload = src.fetch_metadata(cand)
    assert payload["title"] == "T"


def test_stream_sample_raises_unsupported():
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    cand = Candidate(
        source="kaggle",
        id="alice/x",
        revision=None,
        metadata=CandidateMetadata(description=None, card_url="x"),
        streamable=False,
        surfaced_by=[],
    )
    with pytest.raises(SourceUnsupportedError):
        list(src.stream_sample(cand, n=5, seed=42))


def test_stream_rows_raises_unsupported():
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    cand = Candidate(
        source="kaggle",
        id="alice/x",
        revision=None,
        metadata=CandidateMetadata(description=None, card_url="x"),
        streamable=False,
        surfaced_by=[],
    )
    with pytest.raises(SourceUnsupportedError):
        list(src.stream_rows(cand))


def test_card_url():
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    cand = Candidate(
        source="kaggle",
        id="alice/x",
        revision=None,
        metadata=CandidateMetadata(description=None, card_url="x"),
        streamable=False,
        surfaced_by=[],
    )
    assert (
        src.card_url(cand) == "https://www.kaggle.com/datasets/alice/x"
    )


def test_terms_check_returns_obligation():
    src = KaggleSource(username="u", key="k", client=httpx.Client(timeout=5.0))
    obligations = src.terms_check(_make_intent())
    assert len(obligations) == 1
    assert obligations[0].source == "kaggle"


# ─── factory integration ────────────────────────────────────────────


def test_factory_skips_kaggle_without_creds(monkeypatch, tmp_path):
    """build_source_index quietly omits Kaggle when no creds present."""
    from dataset_scout.sources.factory import build_source_index

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    ctx = ScoutContext()
    index = build_source_index(ctx)
    assert "kaggle" not in index
    assert "huggingface" in index


def test_factory_includes_kaggle_with_creds(monkeypatch):
    from dataset_scout.sources.factory import build_source_index

    ctx = ScoutContext(api_keys={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"})
    index = build_source_index(ctx)
    assert "kaggle" in index
    assert index["kaggle"].name == "kaggle"
