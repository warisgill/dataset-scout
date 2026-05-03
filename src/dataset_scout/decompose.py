"""LLM-driven brief decomposition (M2a).

Given an `Intent`, ask the configured LLM for 3-7 related search
directions adjacent to the user's stated target. The pipeline uses
this output to widen the search net before scoring; reframing
(strategy assessment) lives separately in M2b.

Network-free at import time. The `litellm` import is deferred into
the call sites so unit tests don't pay for it and snapshot tests
need no provider configured.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError

from dataset_scout.core import DecompositionDirection, Intent
from dataset_scout.errors import LLMError

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext

# Bumped when the prompt or response handling changes in a way that
# would invalidate cached decomposition results. Kept here, not in
# core, so prompt iteration doesn't churn the public surface.
DECOMPOSE_VERSION = "1"

# Hard upper bound on directions returned (mirrors the prompt).
_MAX_DIRECTIONS = 7


class DecomposeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    directions: list[DecompositionDirection]


# ─── prompt rendering ───────────────────────────────────────────────


# The literal JSON-schema block contains `{` and `}`, which precludes
# str.format / f-strings. Use a sentinel-replacement instead so the
# template stays readable inline. Sentinels are unambiguous strings
# we know won't appear in the surrounding prose.
_PROMPT_TEMPLATE = """\
You are helping an AI security engineer find public datasets that are
related to their detection target. Their stated brief and target are
narrow; we want to expand the search net to find candidates that
could contribute via reframing — proxy positives, hard negatives,
benign baselines, subset extractions, or label remappings.

Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>
Threat families: <<THREAT_FAMILIES>>
Deployment context: <<DEPLOYMENT_CONTEXT>>

Propose 3-7 RELATED SEARCH DIRECTIONS adjacent to this target.
Each direction should be a distinct angle the search should explore
beyond the original brief, NOT a paraphrase of the brief itself.

Each direction's `keywords` should be SHORT, LEXICAL search terms a
keyword search engine would match (e.g., "prompt injection",
"jailbreak prompts", "indirect injection"). Avoid full sentences.

Return JSON matching this schema:
{
  "directions": [
    {
      "name": "snake_case_short_name",
      "rationale": "1-2 sentences on why this is relevant",
      "keywords": ["term", "term", "term"],
      "threat_families": ["family", ...],
      "expected_finds": "1 sentence on what useful data we'd hope to find"
    },
    ...
  ]
}

Be conservative-but-creative. Do NOT include directions you cannot
defend. Aim for 3-5 strong directions; up to 7 if genuinely warranted.
"""


def _none_or_csv(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def render_decompose_prompt(intent: Intent) -> str:
    """Render the exact prompt sent to the model. No I/O.

    Empty / None fields render as ``(none)`` so the prompt stays
    well-formed and snapshot-stable across briefs.
    """
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", intent.raw_brief or "(none)")
        .replace("<<DETECTION_TARGET>>", intent.detection_target or "(none)")
        .replace("<<THREAT_FAMILIES>>", _none_or_csv(intent.threat_families))
        .replace(
            "<<DEPLOYMENT_CONTEXT>>",
            intent.deployment_context or "(none)",
        )
    )


# ─── provider plumbing ──────────────────────────────────────────────


# Crude, narrow provider→env-var mapping. Used only when callers
# supplied keys via `ctx.api_keys` so we can forward the right one
# to litellm explicitly (overriding any process env). For unknown
# providers we fall back on litellm's own env handling.
_PROVIDER_KEY_BY_PREFIX: tuple[tuple[tuple[str, ...], str], ...] = (
    (("gpt-", "openai/", "o1", "o3", "o4"), "OPENAI_API_KEY"),
    (("claude", "anthropic/"), "ANTHROPIC_API_KEY"),
)


def _api_key_for_model(model: str, api_keys: Mapping[str, str]) -> str | None:
    for prefixes, env_name in _PROVIDER_KEY_BY_PREFIX:
        if any(model.startswith(p) for p in prefixes):
            return api_keys.get(env_name)
    return None


def llm_available(ctx: ScoutContext) -> bool:
    """Cheap, no-network probe: will a decompose call likely succeed?

    Fast path: when no plausible provider key is configured we return
    False without importing litellm. (Importing litellm costs ~10s on
    first use; users with no LLM key would otherwise pay that on every
    metadata-only `recon`.)

    When a key is present we trust the fallback. The deferred litellm
    capability check is documented but not used in v1 due to the import
    cost — re-enable behind an env var if a real configuration arises
    where the cheap check is wrong.
    """
    api_key = _api_key_for_model(ctx.llm_model, ctx.api_keys)

    # Empty-string env vars are treated as not-set (real calls would
    # fail anyway; we'd rather degrade cleanly).
    if api_key is not None and not api_key.strip():
        return False
    for prefixes, env_name in _PROVIDER_KEY_BY_PREFIX:
        if any(ctx.llm_model.startswith(p) for p in prefixes):
            env_val = os.environ.get(env_name)
            if env_val is not None and not env_val.strip():
                return False
            break

    # Cheap reject: no candidate key for this model in ctx OR env.
    return _fallback_available(ctx)


def _fallback_available(ctx: ScoutContext) -> bool:
    """True iff the relevant provider env var is set (in ctx or os.environ).

    Narrow check by design — recognises OpenAI / Anthropic / etc. via
    the model-name prefix → env-var-name mapping. Exotic providers (Azure,
    Vertex, custom proxies) won't be detected; users in those setups can
    set the relevant env var themselves and the LLM call will work.
    """
    model = ctx.llm_model
    for prefixes, env_name in _PROVIDER_KEY_BY_PREFIX:
        if any(model.startswith(p) for p in prefixes):
            if ctx.api_keys.get(env_name):
                return True
            env_val = os.environ.get(env_name)
            return bool(env_val and env_val.strip())
    return False


# ─── decomposition ──────────────────────────────────────────────────


def _extract_content(response: Any) -> str:
    """Pull the JSON string from a litellm completion response.

    OpenAI-style: `response.choices[0].message.content`. We avoid
    importing litellm response types so the module stays light and
    test mocks can be plain objects.
    """
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise LLMError(f"unexpected LLM response shape: {exc}") from exc
    if not isinstance(content, str):
        raise LLMError("LLM response message.content was not a string")
    return content


def _parse_response(content: str) -> DecomposeResponse:
    """Parse and validate. Raises ValidationError on schema mismatch,
    json.JSONDecodeError on malformed JSON; both treated as retryable.
    """
    payload = json.loads(content)
    return DecomposeResponse.model_validate(payload)


def decompose_intent(
    intent: Intent,
    *,
    ctx: ScoutContext,
    timeout_s: float = 30.0,
) -> list[DecompositionDirection]:
    """Ask the LLM for 3-7 related search directions.

    Single completion call; one retry on Pydantic validation failure
    using the same prompt. Any other litellm error (network, no
    provider, repeated validation failure) raises `LLMError` so the
    pipeline can fall back to metadata-only mode.

    Result is clipped at 7 directions; an empty list is returned
    cleanly when the model honestly reports no useful adjacencies.
    """
    try:
        import litellm
    except Exception as exc:
        raise LLMError(f"litellm not importable: {exc}") from exc

    # Suppress litellm's chatty stderr decoration around exceptions.
    # We translate failures into LLMError ourselves; users don't need
    # the framework's "Give Feedback / Get Help" boilerplate to leak.
    if hasattr(litellm, "suppress_debug_info"):
        litellm.suppress_debug_info = True

    prompt = render_decompose_prompt(intent)
    api_key = _api_key_for_model(ctx.llm_model, ctx.api_keys)
    messages = [{"role": "user", "content": prompt}]

    completion_kwargs: dict[str, Any] = {
        "model": ctx.llm_model,
        "messages": messages,
        "response_format": DecomposeResponse,
        "timeout": timeout_s,
    }
    if api_key is not None:
        completion_kwargs["api_key"] = api_key

    last_parse_error: Exception | None = None
    parsed: DecomposeResponse | None = None
    for _attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc

        content = _extract_content(response)
        try:
            parsed = _parse_response(content)
            break
        except (ValidationError, json.JSONDecodeError) as exc:
            last_parse_error = exc
            parsed = None
            continue

    if parsed is None:
        msg = f"LLM returned invalid JSON twice: {last_parse_error}"
        raise LLMError(msg) from last_parse_error

    return list(parsed.directions[:_MAX_DIRECTIONS])
