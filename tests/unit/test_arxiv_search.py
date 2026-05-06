"""Unit tests for `dataset_scout.arxiv_search`."""

from __future__ import annotations

import httpx
import pytest
import respx

from dataset_scout import arxiv_search
from dataset_scout.arxiv_search import search_arxiv
from dataset_scout.cache import Cache

pytestmark = pytest.mark.unit


# ─── fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_rate_gate(monkeypatch):
    """Stop the 3-second arXiv rate gate from blocking the test suite.

    The gate reads a module-level last-call timestamp; reset it before
    each test and stub `time.sleep` to a no-op so tests with multiple
    calls don't actually wait.
    """
    arxiv_search._LAST_CALL_AT[0] = 0.0
    monkeypatch.setattr("dataset_scout.arxiv_search.time.sleep", lambda _s: None)


def _atom(*entries: str) -> str:
    """Wrap entry XML fragments in a minimal Atom feed envelope."""
    body = "\n".join(entries)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>arXiv Query</title>
  {body}
</feed>"""


def _entry(
    *,
    arxiv_id: str = "2401.12345",
    title: str = "A Paper",
    abstract: str = "This is the abstract.",
    published: str = "2024-01-15T00:00:00Z",
    authors: tuple[str, ...] = ("Alice", "Bob"),
) -> str:
    authors_xml = "\n".join(f"<author><name>{a}</name></author>" for a in authors)
    return f"""<entry>
  <id>http://arxiv.org/abs/{arxiv_id}v1</id>
  <title>{title}</title>
  <summary>{abstract}</summary>
  <published>{published}</published>
  {authors_xml}
</entry>"""


# ─── happy path ────────────────────────────────────────────────────


@respx.mock
def test_search_arxiv_parses_atom_response():
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            text=_atom(
                _entry(
                    arxiv_id="2401.99999",
                    title="On Toxicity",
                    abstract=("We evaluate on https://huggingface.co/datasets/alice/tox."),
                )
            ),
        )
    )
    papers = search_arxiv(
        "INTIMA",
        year_range=(2023, 2025),
        surfaced_by=["d1"],
        client=httpx.Client(timeout=5.0),
    )
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "On Toxicity"
    assert p.venue == "arXiv"
    assert p.year == 2024
    assert p.arxiv_id == "2401.99999"  # version suffix stripped
    assert p.paper_id == "arxiv:2401.99999"
    assert p.url == "https://arxiv.org/abs/2401.99999"
    assert p.surfaced_by == ["d1"]
    # Dataset URL extracted from the abstract.
    assert len(p.referenced_datasets) == 1
    assert p.referenced_datasets[0].source == "huggingface"
    assert p.referenced_datasets[0].identifier == "alice/tox"


@respx.mock
def test_search_arxiv_year_filter_excludes_out_of_range():
    """Entries outside year_range get dropped at parse time."""
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            text=_atom(
                _entry(arxiv_id="x.1", published="2018-01-01T00:00:00Z"),
                _entry(arxiv_id="x.2", published="2024-06-01T00:00:00Z"),
                _entry(arxiv_id="x.3", published="2026-01-01T00:00:00Z"),
            ),
        )
    )
    papers = search_arxiv(
        "INTIMA",
        year_range=(2023, 2025),
        surfaced_by=[],
        client=httpx.Client(timeout=5.0),
    )
    assert [p.arxiv_id for p in papers] == ["x.2"]


# ─── error handling ────────────────────────────────────────────────


@respx.mock
def test_search_arxiv_swallows_http_errors():
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(503, text="busy")
    )
    papers = search_arxiv(
        "INTIMA",
        year_range=(2023, 2025),
        surfaced_by=[],
        client=httpx.Client(timeout=5.0),
    )
    assert papers == []


@respx.mock
def test_search_arxiv_swallows_malformed_xml():
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(200, text="<not actually xml")
    )
    papers = search_arxiv(
        "INTIMA",
        year_range=(2023, 2025),
        surfaced_by=[],
        client=httpx.Client(timeout=5.0),
    )
    assert papers == []


@respx.mock
def test_search_arxiv_skips_entries_missing_essentials():
    """Entries missing id/title/published get dropped silently."""
    respx.get("https://export.arxiv.org/api/query").mock(
        return_value=httpx.Response(
            200,
            text=_atom(
                # Good entry.
                _entry(arxiv_id="ok.1"),
                # Missing <id>.
                """<entry>
                  <title>No ID</title>
                  <summary>x</summary>
                  <published>2024-01-01T00:00:00Z</published>
                </entry>""",
                # Missing <title>.
                """<entry>
                  <id>http://arxiv.org/abs/notitle</id>
                  <summary>x</summary>
                  <published>2024-01-01T00:00:00Z</published>
                </entry>""",
                # Bad <published> — non-numeric year.
                """<entry>
                  <id>http://arxiv.org/abs/badyear</id>
                  <title>Bad Year</title>
                  <summary>x</summary>
                  <published>not-a-date</published>
                </entry>""",
            ),
        )
    )
    papers = search_arxiv(
        "INTIMA",
        year_range=(2023, 2025),
        surfaced_by=[],
        client=httpx.Client(timeout=5.0),
    )
    assert [p.arxiv_id for p in papers] == ["ok.1"]


def test_search_arxiv_empty_query_short_circuits():
    """Empty/whitespace queries skip the network entirely."""
    # No respx mock -> would raise if a request actually fired.
    papers = search_arxiv(
        "   ",
        year_range=(2023, 2025),
        surfaced_by=[],
        client=httpx.Client(timeout=5.0),
    )
    assert papers == []


# ─── caching ───────────────────────────────────────────────────────


@respx.mock
def test_search_arxiv_uses_cache_on_second_call(tmp_path):
    call_count = {"n": 0}

    def _handler(request):
        call_count["n"] += 1
        return httpx.Response(200, text=_atom(_entry(arxiv_id="cached.1")))

    respx.get("https://export.arxiv.org/api/query").mock(side_effect=_handler)
    cache = Cache(tmp_path / "cache.db")
    try:
        first = search_arxiv(
            "INTIMA",
            year_range=(2023, 2025),
            surfaced_by=["d1"],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
        first_calls = call_count["n"]
        # Second call uses a different surfaced_by — cache hit should
        # still happen, but the returned PaperReferences carry the new
        # surfaced_by tag (cache stores a stripped variant).
        second = search_arxiv(
            "INTIMA",
            year_range=(2023, 2025),
            surfaced_by=["d2"],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
        second_calls = call_count["n"]
    finally:
        cache.close()
    assert first_calls == 1
    assert second_calls == 1  # cache hit, no second network call
    assert first[0].surfaced_by == ["d1"]
    assert second[0].surfaced_by == ["d2"]


@respx.mock
def test_search_arxiv_negative_caches_empty_results(tmp_path):
    """Empty result lists also hit the cache to save the rate-gate budget."""
    call_count = {"n": 0}

    def _handler(request):
        call_count["n"] += 1
        return httpx.Response(200, text=_atom())  # empty feed

    respx.get("https://export.arxiv.org/api/query").mock(side_effect=_handler)
    cache = Cache(tmp_path / "cache.db")
    try:
        search_arxiv(
            "MissingBenchmark",
            year_range=(2023, 2025),
            surfaced_by=[],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
        search_arxiv(
            "MissingBenchmark",
            year_range=(2023, 2025),
            surfaced_by=[],
            cache=cache,
            client=httpx.Client(timeout=5.0),
        )
    finally:
        cache.close()
    assert call_count["n"] == 1
