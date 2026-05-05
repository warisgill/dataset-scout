"""Unit tests for `dataset_scout.keyword_expansion`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from dataset_scout.cache import Cache
from dataset_scout.context import ScoutContext
from dataset_scout.core import DecompositionDirection, Intent
from dataset_scout.keyword_expansion import (
    ExpansionResponse,
    _apply_expansions,
    _normalise,
    expand_dataset_keywords,
    render_expansion_prompt,
)

pytestmark = pytest.mark.unit


def _make_directions() -> list[DecompositionDirection]:
    return [
        DecompositionDirection(
            name="parasocial_interactions",
            rationale="Parasocial relationships are one-sided emotional attachments.",
            keywords=["parasocial bonds", "emotional attachment AI"],
            threat_families=[],
            expected_finds="Datasets involving user narratives of attachment to AI.",
        ),
        DecompositionDirection(
            name="emotion_generation_recognition",
            rationale="Claims of relationship development rely on emotion modeling.",
            keywords=["emotion recognition AI", "affective computing dataset"],
            threat_families=[],
            expected_finds="Datasets related to emotion detection in text.",
        ),
    ]


def _ctx() -> ScoutContext:
    return ScoutContext(
        aoai_endpoint="https://my-aoai.openai.azure.com",
        aoai_deployment="gpt-4o",
    )


def _make_completion(content: str) -> MagicMock:
    """Build a litellm.completion-shaped response object."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    return response


# ─── pure-function tests ────────────────────────────────────────────


def test_normalise_lowercase_and_dedupe():
    out = _normalise(["Mental Health Chat", "MENTAL HEALTH CHAT", "counseling dialogue"])
    assert out == ["mental health chat", "counseling dialogue"]


def test_normalise_drops_overly_long():
    long = "x" * 100
    assert _normalise([long, "ok phrase"]) == ["ok phrase"]


def test_normalise_caps_at_8():
    out = _normalise([f"phrase{i}" for i in range(20)])
    assert len(out) == 8


def test_normalise_strips_quotes_and_whitespace():
    out = _normalise(['"mental health chat"', "  counseling dialogue  "])
    assert out == ["mental health chat", "counseling dialogue"]


def test_apply_expansions_matches_by_name():
    directions = _make_directions()
    response = ExpansionResponse(
        expansions=[
            {"name": "parasocial_interactions",
             "dataset_keywords": ["companion chat", "attachment dialogue"]},
            {"name": "emotion_generation_recognition",
             "dataset_keywords": ["emotion classification", "affect dataset"]},
        ]
    )
    out = _apply_expansions(directions, response)
    assert out[0].dataset_keywords == ["companion chat", "attachment dialogue"]
    assert out[1].dataset_keywords == ["emotion classification", "affect dataset"]
    # Original directions unchanged (frozen + we used model_copy).
    assert directions[0].dataset_keywords == []


def test_apply_expansions_unmatched_direction_passes_through():
    directions = _make_directions()
    # Response only mentions one direction.
    response = ExpansionResponse(
        expansions=[
            {"name": "parasocial_interactions", "dataset_keywords": ["companion chat"]},
        ]
    )
    out = _apply_expansions(directions, response)
    assert out[0].dataset_keywords == ["companion chat"]
    # The unmatched direction keeps its empty default.
    assert out[1].dataset_keywords == []


def test_render_prompt_includes_directions_and_intent():
    intent = Intent(raw_brief="parasocial AI relationships")
    prompt = render_expansion_prompt(intent, _make_directions())
    assert "parasocial AI relationships" in prompt
    assert "parasocial_interactions" in prompt
    assert "emotion_generation_recognition" in prompt
    assert "GOOD bridges" in prompt
    assert "AVOID" in prompt


# ─── LLM integration tests ──────────────────────────────────────────


def test_expand_returns_originals_when_aoai_unconfigured():
    ctx = ScoutContext()
    directions = _make_directions()
    out = expand_dataset_keywords(Intent(raw_brief="x"), directions, ctx=ctx)
    # Same content as input (originals); no LLM call attempted.
    assert [d.name for d in out] == [d.name for d in directions]
    assert all(not d.dataset_keywords for d in out)


def test_expand_empty_directions_returns_empty():
    out = expand_dataset_keywords(Intent(raw_brief="x"), [], ctx=_ctx())
    assert out == []


def test_expand_happy_path_with_mocked_llm(tmp_path):
    directions = _make_directions()
    intent = Intent(raw_brief="parasocial AI relationships")
    cache = Cache(tmp_path / "cache.db")

    fake_content = json.dumps({
        "expansions": [
            {"name": "parasocial_interactions",
             "dataset_keywords": ["mental health chat", "companionship benchmark"]},
            {"name": "emotion_generation_recognition",
             "dataset_keywords": ["emotion classification", "affect dataset"]},
        ]
    })
    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = _make_completion(fake_content)

    with patch(
        "dataset_scout.keyword_expansion.import_litellm",
        return_value=fake_litellm,
    ):
        out = expand_dataset_keywords(intent, directions, ctx=_ctx(), cache=cache)
    cache.close()

    assert out[0].dataset_keywords == ["mental health chat", "companionship benchmark"]
    assert out[1].dataset_keywords == ["emotion classification", "affect dataset"]


def test_expand_cache_hit_skips_llm(tmp_path):
    directions = _make_directions()
    intent = Intent(raw_brief="parasocial AI relationships")
    cache = Cache(tmp_path / "cache.db")
    fake_content = json.dumps({
        "expansions": [
            {"name": "parasocial_interactions", "dataset_keywords": ["companion chat"]},
            {"name": "emotion_generation_recognition", "dataset_keywords": ["emotion data"]},
        ]
    })
    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = _make_completion(fake_content)

    with patch(
        "dataset_scout.keyword_expansion.import_litellm",
        return_value=fake_litellm,
    ):
        # First call: hits LLM, populates cache.
        expand_dataset_keywords(intent, directions, ctx=_ctx(), cache=cache)
        first_calls = fake_litellm.completion.call_count
        # Second call: same prompt → same key → cache hit, no LLM call.
        out = expand_dataset_keywords(intent, directions, ctx=_ctx(), cache=cache)
        second_calls = fake_litellm.completion.call_count
    cache.close()

    assert first_calls == 1
    assert second_calls == 1  # no second call
    assert out[0].dataset_keywords == ["companion chat"]


def test_expand_llm_failure_returns_originals():
    """Network / parse / etc. failures degrade silently to the input list."""
    directions = _make_directions()
    fake_litellm = MagicMock()
    fake_litellm.completion.side_effect = RuntimeError("boom")

    with patch(
        "dataset_scout.keyword_expansion.import_litellm",
        return_value=fake_litellm,
    ):
        out = expand_dataset_keywords(
            Intent(raw_brief="x"), directions, ctx=_ctx(), cache=None
        )
    assert [d.name for d in out] == [d.name for d in directions]
    assert all(not d.dataset_keywords for d in out)


def test_expand_invalid_json_retries_then_falls_back():
    directions = _make_directions()
    fake_litellm = MagicMock()
    # First response invalid, second response also invalid → fall back.
    fake_litellm.completion.side_effect = [
        _make_completion("not json"),
        _make_completion("still not json"),
    ]
    with patch(
        "dataset_scout.keyword_expansion.import_litellm",
        return_value=fake_litellm,
    ):
        out = expand_dataset_keywords(
            Intent(raw_brief="x"), directions, ctx=_ctx(), cache=None
        )
    assert all(not d.dataset_keywords for d in out)
    assert fake_litellm.completion.call_count == 2


# ─── HF source integration ──────────────────────────────────────────


def test_hf_direction_queries_uses_dataset_keywords_first():
    """Dataset keywords should appear first in the query list."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["parasocial bonds", "anthropomorphism"],
        dataset_keywords=["mental health chat", "counseling dialogue"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert qs[:2] == ["mental health chat", "counseling dialogue"]
    # Original keywords still appended.
    assert "parasocial bonds" in qs


def test_hf_direction_queries_caps_at_7():
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["a", "b", "c", "d"],
        dataset_keywords=["e", "f", "g", "h"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert len(qs) == 7


def test_hf_direction_queries_dedupes_case_insensitive():
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["Mental Health Chat", "counseling"],
        dataset_keywords=["mental health chat"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    # Only one mental-health entry should survive.
    assert len([q for q in qs if "mental health" in q.lower()]) == 1


def test_hf_direction_queries_falls_back_to_keywords_only():
    """Backwards compat: directions without dataset_keywords still work."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["alpha", "beta"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert qs == ["alpha", "beta"]


def test_hf_direction_queries_falls_back_to_name_when_no_keywords():
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="my_direction",
        rationale="r",
        keywords=[],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert qs == ["my direction"]


# ─── paper venue extensions ─────────────────────────────────────────


def test_default_venues_include_nlp_and_arxiv():
    from dataset_scout.paper_search import DEFAULT_VENUES

    short_names = set(DEFAULT_VENUES)
    assert "ACL" in short_names
    assert "EMNLP" in short_names
    assert "NAACL" in short_names
    assert "arXiv" in short_names
    # Original four still there.
    for v in ("NeurIPS", "ICML", "ICLR", "SaTML"):
        assert v in short_names


def test_venue_filter_value_includes_arxiv_alias():
    from dataset_scout.paper_search import DEFAULT_VENUES, venue_filter_value

    out = venue_filter_value(DEFAULT_VENUES)
    assert "arXiv.org" in out
    assert "Annual Meeting of the Association for Computational Linguistics" in out


def test_canonical_venue_arxiv():
    from dataset_scout.paper_search import canonical_venue

    assert canonical_venue("arXiv.org") == "arXiv"
    assert canonical_venue("ACL") == "ACL"
    assert (
        canonical_venue("Annual Meeting of the Association for Computational Linguistics")
        == "ACL"
    )
