"""Unit tests for `dataset_scout.paper_search`."""

from __future__ import annotations

import httpx
import pytest
import respx

from dataset_scout.cache import Cache
from dataset_scout.core import DecompositionDirection, Intent
from dataset_scout.paper_search import (
    DEFAULT_VENUES,
    canonical_venue,
    default_year_range,
    extract_dataset_references,
    find_papers_and_promote,
    venue_filter_value,
)

pytestmark = pytest.mark.unit


# ─── pure-function tests ───────────────────────────────────────────


def test_canonical_venue_short_form_passthrough():
    assert canonical_venue("NeurIPS") == "NeurIPS"
    assert canonical_venue("icml") == "ICML"


def test_canonical_venue_long_form_normalised():
    assert canonical_venue("Neural Information Processing Systems") == "NeurIPS"
    assert canonical_venue("International Conference on Machine Learning") == "ICML"
    assert canonical_venue(
        "International Conference on Learning Representations"
    ) == "ICLR"
    assert canonical_venue(
        "IEEE Conference on Secure and Trustworthy Machine Learning"
    ) == "SaTML"


def test_canonical_venue_unknown_returns_raw():
    assert canonical_venue("RandomVenue 2024") == "RandomVenue 2024"


def test_venue_filter_value_expands_aliases():
    """Default venues expand to multiple aliases joined with commas."""
    out = venue_filter_value(DEFAULT_VENUES)
    # Must include all four short forms.
    for short in ("NeurIPS", "ICML", "ICLR", "SaTML"):
        assert short in out
    # And at least one long form per venue.
    assert "Neural Information Processing Systems" in out
    assert "International Conference on Machine Learning" in out


def test_venue_filter_value_dedupes():
    out = venue_filter_value(["NeurIPS", "NeurIPS"])
    parts = out.split(",")
    assert parts.count("NeurIPS") == 1


def test_default_year_range_window():
    lo, hi = default_year_range(window=4)
    assert hi - lo == 3
    assert hi >= 2025  # session is in 2026


def test_default_year_range_custom_window():
    lo, hi = default_year_range(window=2)
    assert hi - lo == 1


# ─── extraction tests ──────────────────────────────────────────────


def test_extract_hf_url():
    text = "We use the dataset at https://huggingface.co/datasets/walledai/XSTest."
    refs = extract_dataset_references(text)
    assert len(refs) == 1
    assert refs[0].source == "huggingface"
    assert refs[0].identifier == "walledai/XSTest"
    assert refs[0].url == "https://huggingface.co/datasets/walledai/XSTest"
    assert refs[0].confidence == "explicit_url"


def test_extract_kaggle_url():
    text = "Released on https://www.kaggle.com/datasets/foo/bar-baz for reproducibility."
    refs = extract_dataset_references(text)
    assert len(refs) == 1
    assert refs[0].source == "kaggle"
    assert refs[0].identifier == "foo/bar-baz"


def test_extract_github_dataset_path():
    """GitHub repos that signal /data or /-dataset get extracted."""
    text = "Code and data at https://github.com/alice/example/data."
    refs = extract_dataset_references(text)
    sources = [r.source for r in refs]
    assert "github" in sources


def test_extract_github_plain_repo_skipped():
    """Bare GitHub repos (no /data hint) are NOT extracted — too generic."""
    text = "Code at https://github.com/alice/example."
    refs = extract_dataset_references(text)
    assert all(r.source != "github" for r in refs)


def test_extract_multiple_urls_deduped():
    text = (
        "Datasets: https://huggingface.co/datasets/a/b and "
        "https://huggingface.co/datasets/a/b again, plus "
        "https://www.kaggle.com/datasets/x/y."
    )
    refs = extract_dataset_references(text)
    assert len(refs) == 2
    sources = {r.source for r in refs}
    assert sources == {"huggingface", "kaggle"}


def test_extract_handles_none_or_empty():
    assert extract_dataset_references(None) == []
    assert extract_dataset_references("") == []


def test_extract_no_urls_returns_empty():
    text = "We propose a novel method evaluated on the standard test set."
    assert extract_dataset_references(text) == []


# ─── search integration with respx ─────────────────────────────────


def _make_intent(brief: str = "labeled toxicity") -> Intent:
    return Intent(raw_brief=brief, threat_families=[])


def _s2_paper(
    paper_id: str,
    title: str,
    *,
    venue: str = "NeurIPS",
    year: int = 2024,
    abstract: str | None = None,
    citations: int = 5,
    authors: list[str] | None = None,
) -> dict:
    return {
        "paperId": paper_id,
        "title": title,
        "venue": venue,
        "year": year,
        "abstract": abstract,
        "citationCount": citations,
        "authors": [{"name": a} for a in (authors or ["Alice", "Bob"])],
        "url": f"https://www.semanticscholar.org/paper/{paper_id}",
        "externalIds": {"DOI": "10.1000/foo"},
    }


@respx.mock
def test_find_papers_returns_normalised_results():
    """Happy path: a single paper with an HF citation in the abstract."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _s2_paper(
                        "p1",
                        "On Toxicity",
                        abstract=(
                            "We evaluate on https://huggingface.co/datasets/alice/tox."
                        ),
                    )
                ],
            },
        )
    )
    papers, candidates = find_papers_and_promote(
        _make_intent(),
        directions=[],
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 1
    p = papers[0]
    assert p.paper_id == "p1"
    assert p.title == "On Toxicity"
    assert p.venue == "NeurIPS"
    assert p.year == 2024
    assert len(p.referenced_datasets) == 1
    assert p.referenced_datasets[0].source == "huggingface"
    # Promoted candidate carries paper provenance.
    assert len(candidates) == 1
    assert candidates[0].source == "huggingface"
    assert candidates[0].id == "alice/tox"
    assert any("paper:" in s for s in candidates[0].surfaced_by)


@respx.mock
def test_find_papers_no_dataset_citations_yields_no_candidates():
    """Papers without HF/Kaggle URLs in abstracts produce zero promoted candidates."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    _s2_paper("p1", "Methods Paper", abstract="No dataset URLs here.")
                ]
            },
        )
    )
    papers, candidates = find_papers_and_promote(
        _make_intent(),
        directions=[],
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 1
    assert candidates == []


@respx.mock
def test_find_papers_swallows_http_errors():
    """5xx from S2 returns ([], []) without raising."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(503, text="busy")
    )
    papers, candidates = find_papers_and_promote(
        _make_intent(),
        directions=[],
        client=httpx.Client(timeout=5.0),
    )
    assert papers == []
    assert candidates == []


@respx.mock
def test_find_papers_round_robin_dedup_across_directions():
    """Same paper from two queries -> one entry, surfaced_by from first call."""
    call_count = {"n": 0}

    def _handler(request):
        call_count["n"] += 1
        # Both calls return the same paper. The merging logic should
        # collapse to a single PaperReference.
        return httpx.Response(
            200,
            json={"data": [_s2_paper("dup-1", "Shared Paper")]},
        )

    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        side_effect=_handler
    )
    directions = [
        DecompositionDirection(
            name="d1",
            rationale="r",
            keywords=["alpha"],
            threat_families=[],
            expected_finds="x",
        ),
    ]
    papers, _ = find_papers_and_promote(
        _make_intent("brief"),
        directions=directions,
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 1
    # Two queries fired (intent + 1 direction), but only one paper survived.
    assert call_count["n"] == 2


@respx.mock
def test_find_papers_uses_cache_on_second_call(tmp_path):
    call_count = {"n": 0}

    def _handler(request):
        call_count["n"] += 1
        return httpx.Response(
            200,
            json={"data": [_s2_paper("p1", "Cached Paper")]},
        )

    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        side_effect=_handler
    )
    cache = Cache(tmp_path / "cache.db")
    try:
        find_papers_and_promote(
            _make_intent(),
            directions=[],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
        first = call_count["n"]
        find_papers_and_promote(
            _make_intent(),
            directions=[],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
        second = call_count["n"]
    finally:
        cache.close()
    assert first == 1
    assert second == first  # second call hit cache


@respx.mock
def test_find_papers_skips_malformed_records():
    """S2 sometimes returns sparse / malformed entries — skip them silently."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"paperId": "ok", "title": "OK", "year": 2024, "venue": "NeurIPS"},
                    {"paperId": "no-title", "year": 2024},  # missing title
                    {"title": "no-id", "year": 2024},  # missing paperId
                    "not-a-dict",
                ]
            },
        )
    )
    papers, _ = find_papers_and_promote(
        _make_intent(),
        directions=[],
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 1
    assert papers[0].paper_id == "ok"


@respx.mock
def test_find_papers_no_query_no_calls():
    """Empty intent + empty directions -> zero queries fired."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    intent = Intent(raw_brief="", threat_families=[])
    papers, candidates = find_papers_and_promote(
        intent,
        directions=[],
        client=httpx.Client(timeout=5.0),
    )
    assert papers == []
    assert candidates == []


@respx.mock
def test_find_papers_max_papers_caps():
    """Result list is capped at max_papers."""
    respx.get("https://api.semanticscholar.org/graph/v1/paper/search/bulk").mock(
        return_value=httpx.Response(
            200,
            json={"data": [_s2_paper(f"p{i}", f"Paper {i}") for i in range(50)]},
        )
    )
    papers, _ = find_papers_and_promote(
        _make_intent(),
        directions=[],
        max_papers=5,
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 5


# ─── pipeline integration ──────────────────────────────────────────


def test_pipeline_promotes_paper_candidates_into_pool():
    """run_recon merges paper-promoted Candidates into the existing pool."""
    from dataset_scout.context import ScoutContext
    from dataset_scout.core import (
        Candidate,
        CandidateMetadata,
        ExtractedDataset,
        PaperReference,
    )
    from dataset_scout.pipeline import run_recon
    from tests._fakes.fake_source import FakeSource

    # Inject a fake paper_search_fn that returns one paper + a promoted candidate.
    promoted = [
        Candidate(
            source="huggingface",
            id="alice/from-paper",
            revision=None,
            metadata=CandidateMetadata(
                description="From paper",
                card_url="https://huggingface.co/datasets/alice/from-paper",
            ),
            streamable=True,
            surfaced_by=["paper:NeurIPS-2024-p1"],
        )
    ]
    paper = PaperReference(
        paper_id="p1",
        title="A paper",
        venue="NeurIPS",
        year=2024,
        url="https://example.com/p1",
        referenced_datasets=[
            ExtractedDataset(
                source="huggingface",
                identifier="alice/from-paper",
                url="https://huggingface.co/datasets/alice/from-paper",
            )
        ],
    )

    def fake_search(intent, directions, **kwargs):
        return [paper], list(promoted)

    fake = FakeSource([])  # HF source returns nothing
    ctx = ScoutContext.from_env(env={})
    result = run_recon(
        "labeled X",
        ctx=ctx,
        sources=[fake],
        paper_search_fn=fake_search,
    )

    # The paper-promoted candidate should now appear in result.candidates.
    ids = [sc.candidate.id for sc in result.candidates]
    assert "alice/from-paper" in ids
    # And paper is recorded.
    assert len(result.papers) == 1
    assert result.papers[0].paper_id == "p1"


def test_pipeline_paper_search_disabled_with_false():
    """paper_search_fn=False disables the stage entirely."""
    from dataset_scout.context import ScoutContext
    from dataset_scout.pipeline import run_recon
    from tests._fakes.fake_source import FakeSource

    fake = FakeSource([])
    ctx = ScoutContext.from_env(env={})
    result = run_recon(
        "anything",
        ctx=ctx,
        sources=[fake],
        paper_search_fn=False,
    )
    assert result.papers == []


def test_pipeline_merges_paper_provenance_on_existing_candidate():
    """A promoted candidate that matches an existing one merges surfaced_by."""
    from dataset_scout.context import ScoutContext
    from dataset_scout.core import (
        Candidate,
        CandidateMetadata,
        ExtractedDataset,
        PaperReference,
    )
    from dataset_scout.pipeline import run_recon
    from tests._fakes.fake_source import FakeSource

    # FakeSource yields a candidate with id "alice/x".
    existing = Candidate(
        source="huggingface",
        id="alice/x",
        revision="abc",
        metadata=CandidateMetadata(description="From HF", card_url="x"),
        streamable=True,
        surfaced_by=[],
    )
    fake = FakeSource([existing])

    # Paper-search promotes the same candidate id with a paper tag.
    promoted = Candidate(
        source="huggingface",
        id="alice/x",
        revision=None,
        metadata=CandidateMetadata(description="From paper", card_url="x"),
        streamable=True,
        surfaced_by=["paper:NeurIPS-2024-p1"],
    )
    paper = PaperReference(
        paper_id="p1",
        title="t",
        venue="NeurIPS",
        year=2024,
        url="u",
        referenced_datasets=[
            ExtractedDataset(
                source="huggingface", identifier="alice/x", url="x"
            )
        ],
    )

    def fake_search(intent, directions, **kwargs):
        return [paper], [promoted]

    ctx = ScoutContext.from_env(env={})
    result = run_recon(
        "labeled X",
        ctx=ctx,
        sources=[fake],
        paper_search_fn=fake_search,
    )

    # One candidate; surfaced_by includes the paper tag merged in.
    assert len(result.candidates) == 1
    sc = result.candidates[0]
    assert sc.candidate.id == "alice/x"
    assert any("paper:" in s for s in sc.candidate.surfaced_by)
