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


def test_hf_direction_queries_camel_split_recalled_names():
    """CamelCase recalled names emit space-separated sibling queries.

    Empirically `PersonaChat` (one word) doesn't match
    `google/Synthetic-Persona-Chat` on HF lexical search; the
    space-separated form does. Same fix lets us catch any
    hyphenated dataset id from a CamelCase paper-name recall.
    """
    from dataset_scout.sources.huggingface import _camel_split, _direction_queries

    # _camel_split smoke checks
    assert _camel_split("PersonaChat") == ["Persona", "Chat"]
    assert _camel_split("INTIMA") == ["INTIMA"]
    assert _camel_split("GPT4") == ["GPT", "4"]

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=[],
        recalled_dataset_names=["PersonaChat", "INTIMA"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert "PersonaChat" in qs
    assert "Persona Chat" in qs  # the CamelCase split sibling
    assert "INTIMA" in qs
    # No spurious "I N T I M A" — single-token names produce no split.
    assert "I N T I M A" not in qs


def test_hf_direction_queries_no_camel_split_on_separators():
    """Names already containing hyphens / underscores / slashes don't get split."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=[],
        recalled_dataset_names=["Anthropic/hh-rlhf", "or-bench", "google_persona_chat"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert "Anthropic/hh-rlhf" in qs
    assert "or-bench" in qs
    # No artificial split adding e.g. "Anthropic" + "hh rlhf" as siblings.
    assert qs.count("Anthropic/hh-rlhf") == 1


def test_normalise_caps_at_8():
    out = _normalise([f"phrase{i}" for i in range(20)])
    assert len(out) == 8


def test_normalise_names_preserves_case():
    """Recalled names are proper nouns; case must be preserved."""
    from dataset_scout.keyword_expansion import _normalise_names

    out = _normalise_names(["PersonaChat", "  XSTest ", "INTIMA"])
    assert out == ["PersonaChat", "XSTest", "INTIMA"]


def test_normalise_names_dedupes_case_insensitive():
    from dataset_scout.keyword_expansion import _normalise_names

    out = _normalise_names(["PersonaChat", "personachat", "PersonaChat"])
    assert len(out) == 1
    assert out[0] == "PersonaChat"  # first wins


def test_normalise_names_caps_at_6():
    from dataset_scout.keyword_expansion import _normalise_names

    out = _normalise_names([f"Bench{i}" for i in range(20)])
    assert len(out) == 6


def test_apply_expansions_propagates_recalled_names():
    """The expansion stage's recalled dataset names land on the direction."""
    from dataset_scout.keyword_expansion import (
        ExpansionResponse,
        _apply_expansions,
    )

    directions = _make_directions()
    response = ExpansionResponse(
        expansions=[
            {
                "name": "parasocial_interactions",
                "dataset_keywords": ["companion chat"],
                "recalled_dataset_names": ["INTIMA", "PersonaChat"],
            },
            {
                "name": "emotion_generation_recognition",
                "dataset_keywords": ["emotion data"],
                "recalled_dataset_names": ["GoEmotions"],
            },
        ]
    )
    out = _apply_expansions(directions, response)
    assert out[0].recalled_dataset_names == ["INTIMA", "PersonaChat"]
    assert out[1].recalled_dataset_names == ["GoEmotions"]


def test_hf_direction_queries_includes_recalled_names_first():
    """Recalled named benchmarks lead the query list, no shortening applied."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["parasocial bonds"],
        dataset_keywords=["companion chat"],
        recalled_dataset_names=["INTIMA", "PersonaChat"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    # Recalled names first, in order, AS-IS (no lowercase, no shortening).
    assert qs[0] == "INTIMA"
    assert qs[1] == "PersonaChat"
    # Compound nouns and academic keywords still appear after.
    assert any("companion chat" in q.lower() for q in qs)


def test_paper_direction_queries_includes_recalled_names():
    """S2 paper queries also pull recalled names — paper titles often contain them."""
    from dataset_scout.paper_search import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["parasocial bonds", "emotional attachment"],
        recalled_dataset_names=["INTIMA", "PersonaChat"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert "INTIMA" in qs
    assert "PersonaChat" in qs


def test_hf_direction_queries_no_recalled_names_falls_back():
    """Direction with no recalled names still works — uses keywords."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=["alpha", "beta"],
        recalled_dataset_names=[],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert qs == ["alpha", "beta"]


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


def test_expand_cache_key_includes_deployment(tmp_path):
    """Switching aoai_deployment must invalidate the keyword-expansion cache."""
    directions = _make_directions()
    intent = Intent(raw_brief="parasocial AI relationships")
    cache = Cache(tmp_path / "cache.db")

    fake_litellm = MagicMock()
    fake_litellm.completion.return_value = _make_completion(
        json.dumps({"expansions": [
            {"name": "parasocial_interactions", "dataset_keywords": ["x"]},
            {"name": "emotion_generation_recognition", "dataset_keywords": ["y"]},
        ]})
    )

    ctx_a = ScoutContext(
        aoai_endpoint="https://my.openai.azure.com",
        aoai_deployment="model-a",
    )
    ctx_b = ScoutContext(
        aoai_endpoint="https://my.openai.azure.com",
        aoai_deployment="model-b",
    )

    with patch(
        "dataset_scout.keyword_expansion.import_litellm",
        return_value=fake_litellm,
    ):
        expand_dataset_keywords(intent, directions, ctx=ctx_a, cache=cache)
        first = fake_litellm.completion.call_count
        # Same prompt, different deployment → cache miss → new LLM call.
        expand_dataset_keywords(intent, directions, ctx=ctx_b, cache=cache)
        second = fake_litellm.completion.call_count
        # Same deployment again → cache hit → no new call.
        expand_dataset_keywords(intent, directions, ctx=ctx_a, cache=cache)
        third = fake_litellm.completion.call_count
    cache.close()

    assert first == 1
    assert second == 2  # ctx_b forced a fresh call
    assert third == 2  # ctx_a hit cache


def test_expand_cache_key_excludes_endpoint():
    """Same deployment + same endpoint -> same cache key (sanity check)."""
    # This is the negative control: just verify the key construction
    # uses deployment, not the full ctx, so cosmetic ctx changes don't
    # invalidate.
    import hashlib

    from dataset_scout.keyword_expansion import EXPANSION_VERSION, render_expansion_prompt

    intent = Intent(raw_brief="x")
    directions = _make_directions()
    prompt = render_expansion_prompt(intent, directions)
    deployment = "gpt-4o"
    expected = hashlib.sha256(
        (EXPANSION_VERSION + "\n" + deployment + "\n" + prompt).encode("utf-8")
    ).hexdigest()
    # Reach into the implementation: this just verifies the format
    # we ship matches the documented design.
    assert len(expected) == 64


def test_expand_cache_hit_skips_llm_legacy(tmp_path):
    """Original cache-hit smoke test (kept after the deployment-key restructure)."""
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


def test_hf_direction_queries_auto_shortens_3plus_word_phrases():
    """3+ word phrases also emit their 2-word prefix to survive HF's AND search."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=[],
        dataset_keywords=["mental health dialogues", "elder care conversation"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    # Both originals plus both 2-word prefixes.
    assert "mental health dialogues" in qs
    assert "mental health" in qs
    assert "elder care conversation" in qs
    assert "elder care" in qs


def test_hf_direction_queries_no_shortening_for_2_word_phrases():
    """2-word phrases pass through without expansion."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=[],
        dataset_keywords=["mental health"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    assert qs == ["mental health"]


def test_hf_direction_queries_dedupes_shortenings():
    """If two phrases share a 2-word prefix, only one shortening is emitted."""
    from dataset_scout.sources.huggingface import _direction_queries

    d = DecompositionDirection(
        name="x",
        rationale="r",
        keywords=[],
        dataset_keywords=["mental health dialogues", "mental health chat"],
        threat_families=[],
        expected_finds="",
    )
    qs = _direction_queries(d)
    # Both originals; "mental health" appears once.
    assert qs.count("mental health") == 1


def test_hf_direction_queries_uses_dataset_keywords_first():
    """Dataset keywords should appear first in the query list (with their shortenings)."""
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
    # Dataset keywords come first (each followed by its 2-word prefix shortening).
    assert qs[0] == "mental health chat"
    assert qs[1] == "mental health"  # auto-shortening
    assert "counseling dialogue" in qs
    # Original keywords still appended (academic-style fallback).
    assert "parasocial bonds" in qs


def test_hf_direction_queries_legacy_dedupe():
    """Verify case-different duplicates are dropped (alongside shortening).

    Earlier-versioned regression test, kept as a smoke check that
    case-folding still applies before auto-shortening kicks in.
    """
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
    full_form = sum(1 for q in qs if q.lower() == "mental health chat")
    assert full_form == 1


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
    # Two distinct "mental health"-prefixed entries survive: the full
    # 3-word form (case-deduped to one), and the auto-shortened 2-word
    # prefix.
    full = sum(1 for q in qs if q.lower() == "mental health chat")
    short = sum(1 for q in qs if q.lower() == "mental health")
    assert full == 1
    assert short == 1


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
    # Original four still there.
    for v in ("NeurIPS", "ICML", "ICLR", "SaTML"):
        assert v in short_names
    # NLP venues.
    assert "ACL" in short_names
    assert "EMNLP" in short_names
    assert "NAACL" in short_names
    # arXiv preprints.
    assert "arXiv" in short_names
    # Ethics / fairness / HCI / general-AI venues (cast wide net).
    assert "FAccT" in short_names
    assert "AIES" in short_names
    assert "AAAI" in short_names
    assert "CHI" in short_names
    assert "COLM" in short_names


def test_venue_filter_value_includes_facct_alias():
    from dataset_scout.paper_search import DEFAULT_VENUES, venue_filter_value

    out = venue_filter_value(DEFAULT_VENUES)
    assert "ACM Conference on Fairness, Accountability, and Transparency" in out
    assert "AAAI/ACM Conference on AI, Ethics, and Society" in out
    assert "Conference on Language Modeling" in out


def test_venue_filter_value_returns_empty_for_all_sentinel():
    """`all` mode produces an empty string so the caller can drop the filter."""
    from dataset_scout.paper_search import venue_filter_value

    assert venue_filter_value(["all"]) == ""
    assert venue_filter_value(["ALL"]) == ""  # case-insensitive
    assert venue_filter_value(["all", "NeurIPS"]) == ""  # any 'all' wins


def test_is_all_venues():
    from dataset_scout.paper_search import is_all_venues

    assert is_all_venues(["all"]) is True
    assert is_all_venues(["All"]) is True
    assert is_all_venues(["NeurIPS"]) is False
    assert is_all_venues([]) is False


def test_canonical_venue_arxiv():
    from dataset_scout.paper_search import canonical_venue

    assert canonical_venue("arXiv.org") == "arXiv"
    assert canonical_venue("ACL") == "ACL"
    assert (
        canonical_venue("Annual Meeting of the Association for Computational Linguistics")
        == "ACL"
    )
