"""Decomposition keyword expansion for HuggingFace-friendly search.

Problem: the decomposer produces *abstract conceptual* keywords (e.g.
"parasocial bonds", "anthropomorphic language") that match how
academic papers talk about a topic but NOT how dataset uploaders name
their corpora on HuggingFace. HF's lexical search returns zero hits
on long abstract phrases. Empirically: a brief about parasocial AI
attachment produced 7 directions x 3 keywords = 21 queries yielding
0 candidates, while two of the manually-sourced datasets
(MentalChat16K, Amod's mental_health_counseling_conversations) are
findable on HF with the simple phrase ``mental health chat`` -- a phrase
the decomposer never produced.

Fix: a second LLM call after decomposition that translates each
direction into 3-5 short HF-uploader-style compound-noun phrases
("mental health chat", "counseling dialogue", "support chatbot"). The
HF source then queries on the union of `keywords` and `dataset_keywords`
with deduplication and a per-direction cap.

Single LLM call, batched across all directions. Cached via the existing
cache module's `decompose` namespace (the expansion is conceptually a
continuation of the decomposition step). Failure is non-blocking -- if
the call fails we keep the original `keywords` and continue.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from dataset_scout.core import DecompositionDirection, Intent
from dataset_scout.errors import LLMError
from dataset_scout.llm_client import build_completion_kwargs, extract_content, import_litellm

if TYPE_CHECKING:
    from dataset_scout.cache import Cache
    from dataset_scout.context import ScoutContext


# Bumped when the prompt or response handling changes.
EXPANSION_VERSION = "3"

_MAX_KEYWORDS_PER_DIRECTION = 8


class _ExpansionEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    dataset_keywords: list[str] = Field(default_factory=list)


class ExpansionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    expansions: list[_ExpansionEntry] = Field(default_factory=list)


_PROMPT_TEMPLATE = """\
You are helping find HuggingFace datasets for an AI-safety / detection
brief. Researchers naming public datasets on HuggingFace use SHORT,
COMMON compound-noun phrases describing the data shape and domain --
typically 2-4 words. Your job is twofold:

1. Generate phrases for the brief's CORE concepts.
2. **BRIDGE TO ADJACENT DOMAINS** where data with the right content
   actually lives. Detection-target work often surfaces in
   unrelated-sounding corpora. The single most important thing this
   step adds is the cross-domain hop -- without it, queries return
   zero HF hits because the brief's literal vocabulary isn't in
   anyone's dataset id.

GOOD bridges (memorize these patterns):

  Topic: parasocial AI / emotional dependence / relationship claims
    Bridge to: mental-health, counseling, therapy, supportive dialogue
    Phrases: "mental health chat", "counseling conversations",
             "therapy dialogue", "support chatbot", "companionship",
             "mental health conversation", "emotional support dialogue"

  Topic: physical intimacy / sensual content / non-roleplay erotica
    Bridge to: romance, erotica, dating, fiction
    Phrases: "romance dialogue", "erotica dataset", "intimate scene",
             "dating conversations", "romance novel"

  Topic: over-refusal / exaggerated caution
    Bridge to: refusal benchmarks, jailbreak / safety evals
    Phrases: "refusal benchmark", "xstest", "or-bench", "safety prompts",
             "borderline benign"

  Topic: prompt injection / instruction override
    Bridge to: jailbreak datasets, prompt safety, indirect injection
    Phrases: "prompt injection", "jailbreak prompts", "indirect injection",
             "system prompt extraction"

PRINCIPLES:
  - **PREFER 1-2 WORD PHRASES**. HF's lexical search is AND across
    whitespace tokens, and 3-word phrases like "mental health
    dialogues" return ZERO hits because no dataset id contains all
    three. By contrast "mental health" (2 words) returns dozens.
    Length matters more than specificity here.
  - When you must use 3 words for clarity, prefer the form where
    the 2-word prefix is also useful ("mental health chat" -> still
    bridges to "mental health"). Avoid 3-word phrases where the
    prefix would be meaningless ("ai realism benchmark" -> "ai
    realism" is gibberish).
  - Use COMMON, EXISTING vocabulary -- terms a dataset uploader
    actually puts in an id (e.g. "mental_health_chat" → "mental health chat").
  - Bridge across AT LEAST 2 adjacent domains per direction.
  - Include shorter generalizations even if they look redundant
    ("companionship" alongside "companion chat", "mental health"
    alongside "mental health chat").
  - AVOID: invented-sounding compounds. "ai attachment logs",
    "parasocial conversations", "relationship claim corpus" -- these
    are NOT how anyone names datasets and HF returns zero results.
  - AVOID: phrases ending in "AI" or "LLM" as a suffix.
  - AVOID: long descriptive phrases like "user-AI emotional bonding samples".

INTENT
------
Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>

DIRECTIONS
----------
<<DIRECTIONS_BLOCK>>

For each direction, list 5-8 phrases that mix:
  - 2-3 phrases for the direction's CORE concepts
  - 3-5 phrases that BRIDGE to adjacent domains where data actually lives

Return JSON matching this schema:
{
  "expansions": [
    {"name": "<direction_name>", "dataset_keywords": ["...", "...", ...]},
    ...
  ]
}

Return ONE entry per direction in the same order. Empty list only if
the direction is so abstract no plausible domain bridge exists (rare).
"""


def render_expansion_prompt(
    intent: Intent, directions: list[DecompositionDirection]
) -> str:
    """Render the exact prompt sent to the model. Pure; no I/O."""
    blocks: list[str] = []
    for d in directions:
        line = f"- name: {d.name}\n  rationale: {d.rationale}"
        if d.expected_finds:
            line += f"\n  looking for: {d.expected_finds}"
        if d.keywords:
            line += f"\n  existing keywords: {', '.join(d.keywords)}"
        blocks.append(line)
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", intent.raw_brief or "(none)")
        .replace("<<DETECTION_TARGET>>", intent.detection_target or "(none)")
        .replace("<<DIRECTIONS_BLOCK>>", "\n".join(blocks) or "(no directions)")
    )


def expand_dataset_keywords(
    intent: Intent,
    directions: list[DecompositionDirection],
    *,
    ctx: ScoutContext,
    cache: Cache | None = None,
    timeout_s: float = 30.0,
) -> list[DecompositionDirection]:
    """Return new directions with `dataset_keywords` populated.

    One LLM call total, batched across all directions. Cache hits skip
    the LLM import entirely. Failures (network, parse) log a notice
    and return the original directions unchanged so the pipeline can
    continue. Output preserves direction order.

    Returns an empty list when the input is empty.
    """
    if not directions:
        return []
    if not ctx.aoai_configured:
        # No LLM available; keep originals.
        return list(directions)

    prompt = render_expansion_prompt(intent, directions)

    cache_key: str | None = None
    if cache is not None:
        cache_key = hashlib.sha256(
            (EXPANSION_VERSION + "\n" + (ctx.aoai_deployment or "") + "\n" + prompt).encode(
                "utf-8"
            )
        ).hexdigest()
        cached = cache.get_json("decompose", cache_key)
        if cached is not None:
            try:
                payload = ExpansionResponse.model_validate(cached)
                return _apply_expansions(directions, payload)
            except ValidationError:
                pass

    try:
        litellm = import_litellm()
    except LLMError:
        return list(directions)

    messages = [{"role": "user", "content": prompt}]
    completion_kwargs: dict[str, Any] = build_completion_kwargs(
        ctx,
        messages=messages,
        response_format=ExpansionResponse,
        timeout_s=timeout_s,
    )

    parsed: ExpansionResponse | None = None
    for _attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception:  # network / quota / etc.
            break
        try:
            content = extract_content(response)
            parsed = ExpansionResponse.model_validate(json.loads(content))
            break
        except (LLMError, ValidationError, json.JSONDecodeError):
            parsed = None
            continue

    if parsed is None:
        # Soft fail: pipeline continues with originals.
        return list(directions)

    if cache is not None and cache_key is not None:
        cache.set_json("decompose", cache_key, parsed.model_dump(mode="json"))

    return _apply_expansions(directions, parsed)


def _apply_expansions(
    directions: list[DecompositionDirection],
    response: ExpansionResponse,
) -> list[DecompositionDirection]:
    """Match expansion entries onto directions by name; copy with the
    new field populated. Falls through unchanged for unmatched
    directions so the result preserves the original list."""
    by_name: dict[str, list[str]] = {
        e.name: _normalise(e.dataset_keywords) for e in response.expansions
    }
    out: list[DecompositionDirection] = []
    for d in directions:
        ks = by_name.get(d.name, [])
        if not ks:
            out.append(d)
            continue
        out.append(d.model_copy(update={"dataset_keywords": ks}))
    return out


def _normalise(raw: list[str]) -> list[str]:
    """Lowercase + trim + dedupe + cap. Drops empties and overly-long phrases."""
    seen: set[str] = set()
    out: list[str] = []
    for kw in raw:
        if not isinstance(kw, str):
            continue
        norm = kw.strip().lower().strip("\"'")
        if not norm:
            continue
        # Cap each phrase length so we don't slip in essay-shaped strings.
        if len(norm) > 60:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
        if len(out) >= _MAX_KEYWORDS_PER_DIRECTION:
            break
    return out


__all__ = [
    "EXPANSION_VERSION",
    "ExpansionResponse",
    "expand_dataset_keywords",
    "render_expansion_prompt",
]
