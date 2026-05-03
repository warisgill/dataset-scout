"""The recon pipeline.

M2a: parse → (decompose if LLM available) → multi-direction search →
cheap probes → ReconResult.

Mode is decided once at the top by `decompose.llm_available(ctx)` and
the `explore` flag. When the LLM is available and `explore=True`, the
pipeline runs the full decomposition + multi-direction search. When
not, it runs in metadata-only mode and emits an explicit notice in the
result so the report is honest about the difference.

Strategy assessor + coverage report land in M2b.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CoverageReport,
    DecompositionDirection,
    ReconResult,
    Scorecard,
    SubScore,
)
from dataset_scout.errors import LLMError, SourceUnavailableError
from dataset_scout.events import ProgressEvent, ProgressEventKind
from dataset_scout.intent import HeuristicIntentParser
from dataset_scout.probes import cheap_probes
from dataset_scout.probes.base import Probe, ProbeRegistry
from dataset_scout.sources.base import Budget

if TYPE_CHECKING:
    from dataset_scout.core import Intent
    from dataset_scout.sources.base import Source


_DEFAULT_MAX_CANDIDATES = 50


# Single source of truth for the metadata-only-mode notice. Used by both
# the stderr emitter and the report header so wording stays in sync.
METADATA_ONLY_NOTICE = (
    "Running in metadata-only mode: Azure OpenAI is not configured, so "
    "decomposition, strategy assessment, and coverage gaps were skipped. "
    "To enable them, copy .env.example to .env, set AZURE_OPENAI_ENDPOINT "
    "and AZURE_OPENAI_DEPLOYMENT, and run `az login`."
)


def _build_sources(ctx: ScoutContext) -> list[Source]:
    """Instantiate concrete sources from the context.

    Hardcoded dispatch — only HuggingFace is wired today. Kaggle and
    PWC are deferred; when they land we'll switch to
    importlib.metadata.entry_points.
    """
    sources: list[Source] = []
    enabled = set(ctx.enabled_sources())
    if "huggingface" in enabled:
        from dataset_scout.sources.huggingface import HuggingFaceSource

        token = ctx.api_keys.get("HF_TOKEN") or ctx.api_keys.get("HUGGINGFACE_HUB_TOKEN")
        sources.append(HuggingFaceSource(token=token))
    if not sources:
        raise SourceUnavailableError(
            "No sources are enabled. Configure at least one in your "
            "ScoutContext (huggingface is the only one wired today)."
        )
    return sources


def _score_candidate(
    candidate: Candidate,
    intent: Intent,
    probes: ProbeRegistry,
) -> Scorecard:
    """Run every applicable cheap probe against a single candidate."""
    cheap: dict[str, SubScore] = {}
    for probe in probes:
        if not probe.applies(candidate, intent):
            continue
        cheap[probe.name] = probe.run(candidate, intent)
    return Scorecard(candidate=candidate, cheap_probes=cheap)


def _merge_or_register(
    pool: dict[tuple[str, str], Candidate],
    cand: Candidate,
) -> bool:
    """Insert `cand` into `pool` keyed by (source, id), or merge surfaced_by.

    Returns True if this is a newly-seen candidate, False if it merged
    into an existing one.
    """
    key = (cand.source, cand.id)
    existing = pool.get(key)
    if existing is None:
        pool[key] = cand
        return True
    merged: list[str] = list(existing.surfaced_by)
    for d in cand.surfaced_by:
        if d not in merged:
            merged.append(d)
    if merged != existing.surfaced_by:
        pool[key] = existing.model_copy(update={"surfaced_by": merged})
    return False


def run_recon(
    brief: str,
    *,
    ctx: ScoutContext,
    parser_overrides: dict[str, object] | None = None,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    events: list[ProgressEvent] | None = None,
    probes: ProbeRegistry | None = None,
    sources: list[Source] | None = None,
    explore: bool = True,
) -> ReconResult:
    """Run the discovery pipeline and return a `ReconResult`.

    Parameters
    ----------
    brief
        Natural-language brief.
    ctx
        Explicit ScoutContext (no global state).
    parser_overrides
        CLI-flag-derived overrides applied to the parsed Intent.
    max_candidates
        Upper bound on candidates surfaced in the report.
    events
        If provided, ProgressEvent entries are appended for each stage.
    probes / sources
        Dependency-injection hooks. Defaults wire the cheap probe set
        and HuggingFaceSource.
    explore
        If False, skip decomposition unconditionally (debug). The CLI
        wires this to a hidden `--no-explore` flag.
    """
    overrides = parser_overrides or {}
    notices: list[str] = []

    def _emit(kind: ProgressEventKind, stage: str, message: str = "", **data: object) -> None:
        if events is not None:
            events.append(ProgressEvent(kind=kind, stage=stage, message=message, data=dict(data)))

    started = time.monotonic()

    _emit(ProgressEventKind.STAGE_STARTED, stage="parse")
    intent = HeuristicIntentParser().parse(brief, **overrides)
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="parse",
        message="parsed brief into Intent",
        threat_families=list(intent.threat_families),
        languages=list(intent.languages),
    )

    directions: list[DecompositionDirection] = []
    use_llm = explore
    if use_llm:
        from dataset_scout.decompose import decompose_intent, llm_available

        if not llm_available(ctx):
            use_llm = False
        else:
            _emit(ProgressEventKind.STAGE_STARTED, stage="decompose")
            try:
                directions = decompose_intent(intent, ctx=ctx)
            except LLMError as exc:
                notices.append(f"decomposition skipped: {exc}")
                use_llm = False
                directions = []
            else:
                for d in directions:
                    _emit(
                        ProgressEventKind.DIRECTION_PROPOSED,
                        stage="decompose",
                        name=d.name,
                        keywords=list(d.keywords),
                    )
            _emit(
                ProgressEventKind.STAGE_FINISHED,
                stage="decompose",
                message=f"proposed {len(directions)} direction(s)",
            )

    if not use_llm:
        notices.append(METADATA_ONLY_NOTICE)

    if sources is None:
        sources = _build_sources(ctx)
    probes = probes if probes is not None else cheap_probes()
    budget = Budget()

    _emit(ProgressEventKind.STAGE_STARTED, stage="search")
    pool: dict[tuple[str, str], Candidate] = {}
    for source in sources:
        try:
            stream = source.search(intent, directions, budget=budget)
        except Exception as exc:  # defensive: misbehaving source must not kill the run
            notices.append(f"source '{source.name}' failed: {exc}")
            continue
        for cand in stream:
            is_new = _merge_or_register(pool, cand)
            if is_new:
                _emit(
                    ProgressEventKind.CANDIDATE_FOUND,
                    stage="search",
                    source=cand.source,
                    id=cand.id,
                    surfaced_by=list(cand.surfaced_by),
                )
            if len(pool) >= max_candidates:
                break
        if len(pool) >= max_candidates:
            break

    candidates = list(pool.values())[:max_candidates]
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="search",
        message=f"found {len(candidates)} unique candidate(s)",
    )

    if not candidates:
        notices.append(
            "No candidates returned. Try broadening the brief or check source connectivity."
        )

    _emit(ProgressEventKind.STAGE_STARTED, stage="probe")
    scorecards: list[Scorecard] = []
    for cand in candidates:
        sc = _score_candidate(cand, intent, probes)
        scorecards.append(sc)
        _emit(
            ProgressEventKind.CANDIDATE_SCORED,
            stage="probe",
            id=cand.id,
            probes=list(sc.cheap_probes),
        )
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="probe",
        message=f"scored {len(scorecards)} candidate(s) with {len(probes)} probes",
    )

    elapsed = time.monotonic() - started

    coverage = CoverageReport(decomposition=directions) if directions else None

    return ReconResult(
        intent=intent,
        candidates=scorecards,
        sources_searched=[s.name for s in sources],
        coverage=coverage,
        elapsed_seconds=round(elapsed, 3),
        notices=notices,
    )


__all__ = ["METADATA_ONLY_NOTICE", "Probe", "run_recon"]
