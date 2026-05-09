"""dataset-scout — reconnaissance, reframing, and curation of public datasets
for AI detection engineers, forensic analysts, and incident responders.

Public API surface (v1, M0 skeleton — most functions are stubs until later
milestones land):

    from dataset_scout import recon, inspect, curate, ScoutContext

The library is the source of truth; the CLI is a thin wrapper.
"""

from __future__ import annotations

from importlib import metadata as _metadata
from pathlib import Path

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    ColumnInfo,
    CoverageGap,
    CoverageReport,
    DecompositionDirection,
    Evidence,
    InspectResult,
    Intent,
    JudgeBlock,
    LabelBucket,
    LabelKind,
    LengthStats,
    LicensePolicy,
    LicenseSummary,
    NormalizedRecord,
    ReconResult,
    Scorecard,
    SensitiveDomain,
    Strategy,
    StrategyKind,
    SubScore,
    TransformSpec,
)
from dataset_scout.errors import (
    CompositionPairError,
    DatasetScoutError,
    LLMError,
    SourceUnavailableError,
)
from dataset_scout.events import ProgressEvent, ProgressEventKind

try:
    __version__ = _metadata.version("dataset-scout")
except _metadata.PackageNotFoundError:  # editable install before metadata exists
    __version__ = "0.0.0+local"


def recon(
    brief: str,
    *,
    ctx: ScoutContext | None = None,
    parser_overrides: dict[str, object] | None = None,
) -> ReconResult:
    """Run the discovery pipeline. M1a slice: HF only, metadata-driven."""
    from dataset_scout.context import ScoutContext as _Ctx
    from dataset_scout.pipeline import run_recon

    return run_recon(
        brief=brief,
        ctx=ctx if ctx is not None else _Ctx.from_env(),
        parser_overrides=parser_overrides or {},
    )


def inspect(
    target: str,
    *,
    ctx: ScoutContext | None = None,
    intent: object | None = None,
    intent_from: str | Path | None = None,
    brief: str | None = None,
    sample_size: int = 50,
) -> object:
    """Run a single-candidate deep-dive.

    Pass exactly one of `intent`, `intent_from`, or `brief` to drive
    LLM strategy assessment. With none of them, assessment is skipped
    and only the metadata + sample sections are produced.
    """
    from pathlib import Path as _Path

    from dataset_scout.context import ScoutContext as _Ctx
    from dataset_scout.inspect_ import make_intent, run_inspect

    resolved_intent = intent
    if resolved_intent is None and (intent_from or brief):
        resolved_intent = make_intent(
            brief=brief,
            intent_from=_Path(intent_from) if intent_from else None,
        )
    return run_inspect(
        target,
        ctx=ctx if ctx is not None else _Ctx.from_env(),
        intent=resolved_intent,  # type: ignore[arg-type]
        sample_size=sample_size,
    )


def eval(
    judged: str | Path,
    *,
    gold: str | Path,
    axis: str | None = None,
    ctx: ScoutContext | None = None,
) -> object:
    """Compare a judged corpus against a gold corpus.

    Returns an :class:`dataset_scout.eval_.EvalResult` with per-axis
    P/R/F1 + confusion matrix + coverage.
    """
    from pathlib import Path as _Path

    from dataset_scout.context import ScoutContext as _Ctx
    from dataset_scout.eval_ import run_eval

    return run_eval(
        ctx if ctx is not None else _Ctx.from_env(),
        _Path(judged),
        gold=_Path(gold),
        axis=axis,
    )


def judge(
    target: str | Path,
    *,
    axis: str,
    ctx: ScoutContext | None = None,
    rubric: str | None = None,
    judges: int = 1,
    agreement: str = "single",
    threshold: float = 0.8,
    out_dir: str | Path | None = None,
    only_unknown: bool = True,
    re_judge_all: bool = False,
    dry_run: bool = False,
) -> object:
    """Run the LLM-as-judge label-rescue pass over a corpus.

    The CLI verb is a thin wrapper over
    :func:`dataset_scout.judge.run_judge`.
    """
    from pathlib import Path as _Path
    from typing import Literal, cast

    from dataset_scout.context import ScoutContext as _Ctx
    from dataset_scout.judge import run_judge

    return run_judge(
        ctx if ctx is not None else _Ctx.from_env(),
        _Path(target),
        axis=axis,
        rubric=rubric,
        judges=judges,
        agreement=cast("Literal['single', 'majority', 'unanimous']", agreement),
        threshold=threshold,
        out_dir=_Path(out_dir) if out_dir is not None else None,
        only_unknown=only_unknown,
        re_judge_all=re_judge_all,
        dry_run=dry_run,
    )


def curate(
    recipe_path: str | Path,
    out_dir: str | Path,
    *,
    ctx: ScoutContext | None = None,
    seed: int | None = None,
    min_strategy_confidence: float | None = None,
) -> object:
    """Materialise a recipe into a corpus directory.

    M4a preview slice — produces JSONL + lockfile + manifest + report
    + fingerprint + usage. Hash-mod splits and no MinHash dedup, so
    treat output as a working artefact, not yet an audit-ready record.
    """
    from pathlib import Path as _Path

    from dataset_scout.context import ScoutContext as _Ctx
    from dataset_scout.curate import load_recipe, run_curate

    recipe = load_recipe(_Path(recipe_path))
    return run_curate(
        recipe,
        _Path(out_dir),
        ctx=ctx if ctx is not None else _Ctx.from_env(),
        seed_override=seed,
        min_strategy_confidence_override=min_strategy_confidence,
    )


__all__ = [
    "Candidate",
    "CandidateMetadata",
    "ColumnInfo",
    "CompositionPairError",
    "CoverageGap",
    "CoverageReport",
    "DatasetScoutError",
    "DecompositionDirection",
    "Evidence",
    "InspectResult",
    "Intent",
    "JudgeBlock",
    "LLMError",
    "LabelBucket",
    "LabelKind",
    "LengthStats",
    "LicensePolicy",
    "LicenseSummary",
    "NormalizedRecord",
    "ProgressEvent",
    "ProgressEventKind",
    "ReconResult",
    "Scorecard",
    "ScoutContext",
    "SensitiveDomain",
    "SourceUnavailableError",
    "Strategy",
    "StrategyKind",
    "SubScore",
    "TransformSpec",
    "__version__",
    "curate",
    "eval",
    "inspect",
    "judge",
    "recon",
]
