"""The recon pipeline (M1a discovery slice).

parse → search (HF only) → cheap probes → ReconResult.

No decomposition, no embedding fit, no LLM strategy assessor in this
slice — those land in M2. Candidates are returned in source/search
relevance order; probe outputs are annotations, never folded into a
single ranking score.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

from dataset_scout.context import ScoutContext
from dataset_scout.core import Candidate, ReconResult, Scorecard, SubScore
from dataset_scout.errors import SourceUnavailableError
from dataset_scout.events import ProgressEvent, ProgressEventKind
from dataset_scout.intent import HeuristicIntentParser
from dataset_scout.probes import cheap_probes
from dataset_scout.probes.base import Probe, ProbeRegistry
from dataset_scout.sources.base import Budget

if TYPE_CHECKING:
    from dataset_scout.core import Intent
    from dataset_scout.sources.base import Source


# Default upper bound on candidates we'll surface in the report. The
# discovery slice keeps this modest; M1b can grow it.
_DEFAULT_MAX_CANDIDATES = 50


def _build_sources(ctx: ScoutContext) -> list[Source]:
    """Instantiate concrete sources from the context.

    Hardcoded dispatch in M1a — only HuggingFace is wired. When Kaggle
    and PWC land we'll switch to importlib.metadata.entry_points.
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
            "ScoutContext (huggingface is the only one wired in M1a)."
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


def run_recon(
    brief: str,
    *,
    ctx: ScoutContext,
    parser_overrides: dict[str, object] | None = None,
    max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    events: list[ProgressEvent] | None = None,
    probes: ProbeRegistry | None = None,
    sources: list[Source] | None = None,
) -> ReconResult:
    """Run the M1a discovery pipeline and return a structured `ReconResult`.

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
    """
    overrides = parser_overrides or {}
    notices: list[str] = []

    def _emit(kind: ProgressEventKind, stage: str, message: str = "", **data: object) -> None:
        if events is not None:
            events.append(ProgressEvent(kind=kind, stage=stage, message=message, data=dict(data)))

    started = time.monotonic()

    # 1. Parse Intent.
    _emit(ProgressEventKind.STAGE_STARTED, stage="parse")
    intent = HeuristicIntentParser().parse(brief, **overrides)
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="parse",
        message="parsed brief into Intent",
        threat_families=list(intent.threat_families),
        languages=list(intent.languages),
    )

    # 2. Search.
    if sources is None:
        sources = _build_sources(ctx)
    probes = probes if probes is not None else cheap_probes()
    budget = Budget()

    _emit(ProgressEventKind.STAGE_STARTED, stage="search")
    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        try:
            stream = source.search(intent, [], budget=budget)
        except Exception as exc:  # defensive: a misbehaving source must not kill the run
            notices.append(f"source '{source.name}' failed: {exc}")
            continue
        for cand in _capped(stream, max_candidates - len(candidates)):
            key = (cand.source, cand.id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(cand)
            _emit(
                ProgressEventKind.CANDIDATE_FOUND,
                stage="search",
                source=cand.source,
                id=cand.id,
            )
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    _emit(
        ProgressEventKind.STAGE_FINISHED,
        stage="search",
        message=f"found {len(candidates)} candidate(s)",
    )

    if not candidates:
        notices.append(
            "No candidates returned. Try broadening the brief or check source connectivity."
        )

    # 3. Cheap probes.
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

    return ReconResult(
        intent=intent,
        candidates=scorecards,
        sources_searched=[s.name for s in sources],
        elapsed_seconds=round(elapsed, 3),
        notices=notices,
    )


def _capped(stream: Iterator[Candidate], remaining: int) -> Iterator[Candidate]:
    if remaining <= 0:
        return
    for i, cand in enumerate(stream):
        if i >= remaining:
            return
        yield cand


# Re-exported for type checkers that want to subclass / mock.
__all__ = ["Probe", "run_recon"]
