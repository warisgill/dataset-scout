"""LLM-described coverage report (M2b).

After the per-candidate strategy assessor runs, we ask the model one
final question: what aspects of the user's detection target are NOT
addressed by any candidate (even via reframing), and what concrete
next steps would close the gap?

Single litellm call. Inputs: intent, decomposition directions, and
the best (or top-2 close-confidence) strategies per shortlisted
candidate. Output: list[CoverageGap].

Same Azure OpenAI / Entra plumbing as decompose / strategy via
`llm_client`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, ValidationError

from dataset_scout.core import (
    CoverageGap,
    DecompositionDirection,
    Intent,
    Scorecard,
    Strategy,
)
from dataset_scout.errors import LLMError
from dataset_scout.llm_client import (
    build_completion_kwargs,
    extract_content,
    import_litellm,
    make_token_provider,
)

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


COVERAGE_VERSION = "1"

# Maximum candidates included in the prompt. Coverage prompt is a
# single LLM call so we cap the size of the candidate summary to keep
# context use reasonable; per-candidate detail is in the report.
_MAX_CANDIDATES_IN_PROMPT = 20

# Confidence gap below which we include the second-best strategy too —
# per duck guidance, "best-strategy-only throws away useful uncertainty
# when a candidate's second-best is still viable."
_TOP2_CLOSE_CONFIDENCE_DELTA = 0.15


class CoverageResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    gaps: list[CoverageGap]


# ─── prompt rendering ───────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are helping an AI security engineer assess whether the candidates
they've found cover their detection target adequately, or leave gaps.

USER INTENT
-----------
Brief: <<RAW_BRIEF>>
Detection target: <<DETECTION_TARGET>>
Threat families: <<THREAT_FAMILIES>>
Deployment context: <<DEPLOYMENT_CONTEXT>>

DECOMPOSITION DIRECTIONS WE EXPLORED
-------------------------------------
<<DIRECTIONS>>

CANDIDATES + BEST STRATEGY (or top-2 when close)
-------------------------------------------------
<<CANDIDATE_SUMMARY>>

YOUR TASK
---------
Identify ASPECTS of the user's detection target that NONE of the
candidates above address — even via reframing — and propose concrete
next steps. Be honest; if coverage is genuinely good, return an
empty list. Do NOT invent gaps to fill space.

Each gap should reference something specific the user said they
care about (a sub-task, a deployment surface, a language, an attack
style) that the candidate set doesn't cover. Suggestions should be
actionable: a search direction, a synthesis approach, a partner
corpus to look for.

Return JSON matching this schema:
{
  "gaps": [
    {
      "aspect": "short snake_case or hyphenated label",
      "description": "1-2 sentences on what the candidates miss",
      "suggestion": "1 sentence concrete next step"
    },
    ...
  ]
}
"""


def _none_or_csv(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _format_directions(directions: list[DecompositionDirection]) -> str:
    if not directions:
        return "(none)"
    lines: list[str] = []
    for d in directions:
        lines.append(f"- {d.name}: {d.rationale}")
    return "\n".join(lines)


def _strategies_for_coverage(scorecard: Scorecard) -> list[Strategy]:
    """Return the strongest strategy or strongest two when close."""
    if not scorecard.strategies:
        return []
    sorted_strats = sorted(scorecard.strategies, key=lambda s: s.confidence, reverse=True)
    best = sorted_strats[0]
    if len(sorted_strats) == 1:
        return [best]
    second = sorted_strats[1]
    if best.confidence - second.confidence <= _TOP2_CLOSE_CONFIDENCE_DELTA:
        return [best, second]
    return [best]


def _format_candidate_summary(scorecards: list[Scorecard]) -> str:
    if not scorecards:
        return "(none — no candidates were assessed)"
    lines: list[str] = []
    for sc in scorecards[:_MAX_CANDIDATES_IN_PROMPT]:
        cand = sc.candidate
        strats = _strategies_for_coverage(sc)
        if not strats:
            line = f"- {cand.source}:{cand.id} — (no strategy assessed)"
        else:
            parts = [f"{s.kind.value} ({s.confidence:.2f}): {s.rationale}" for s in strats]
            line = f"- {cand.source}:{cand.id} — " + "; ".join(parts)
        if cand.metadata.description:
            desc = cand.metadata.description.strip().splitlines()[0][:140]
            line += f" [card: {desc}]"
        lines.append(line)
    if len(scorecards) > _MAX_CANDIDATES_IN_PROMPT:
        lines.append(
            f"...and {len(scorecards) - _MAX_CANDIDATES_IN_PROMPT} more candidate(s) "
            "not shown here."
        )
    return "\n".join(lines)


def render_coverage_prompt(
    intent: Intent,
    directions: list[DecompositionDirection],
    scorecards: list[Scorecard],
) -> str:
    """Render the exact prompt sent to the model. No I/O."""
    return (
        _PROMPT_TEMPLATE.replace("<<RAW_BRIEF>>", intent.raw_brief or "(none)")
        .replace("<<DETECTION_TARGET>>", intent.detection_target or "(none)")
        .replace("<<THREAT_FAMILIES>>", _none_or_csv(intent.threat_families))
        .replace(
            "<<DEPLOYMENT_CONTEXT>>",
            intent.deployment_context or "(none)",
        )
        .replace("<<DIRECTIONS>>", _format_directions(directions))
        .replace("<<CANDIDATE_SUMMARY>>", _format_candidate_summary(scorecards))
    )


# ─── coverage call ──────────────────────────────────────────────────


# Re-exported so tests can stub the credential.
_make_token_provider = make_token_provider


def _parse_response(content: str) -> CoverageResponse:
    payload = json.loads(content)
    return CoverageResponse.model_validate(payload)


def build_coverage_report(
    intent: Intent,
    directions: list[DecompositionDirection],
    scorecards: list[Scorecard],
    *,
    ctx: ScoutContext,
    timeout_s: float = 30.0,
) -> list[CoverageGap]:
    """Ask the LLM to identify coverage gaps in the candidate set.

    Single completion call, one retry on Pydantic validation failure.
    Empty list returned cleanly when coverage is honestly good or when
    no scorecards were assessed.
    """
    if not ctx.aoai_configured:
        raise LLMError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT (and run `az login` for Entra "
            "auth)."
        )

    litellm = import_litellm()
    token_provider = _make_token_provider()
    prompt = render_coverage_prompt(intent, directions, scorecards)
    completion_kwargs = build_completion_kwargs(
        ctx,
        messages=[{"role": "user", "content": prompt}],
        response_format=CoverageResponse,
        timeout_s=timeout_s,
        token_provider=token_provider,
    )

    last_parse_error: Exception | None = None
    parsed: CoverageResponse | None = None
    for _attempt in range(2):
        try:
            response = litellm.completion(**completion_kwargs)
        except Exception as exc:
            raise LLMError(f"LLM call failed: {exc}") from exc
        content = extract_content(response)
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

    return list(parsed.gaps)
