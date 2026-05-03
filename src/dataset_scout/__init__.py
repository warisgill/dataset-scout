"""dataset-scout — reconnaissance, reframing, and curation of public datasets
for AI detection engineers, forensic analysts, and incident responders.

Public API surface (v1, M0 skeleton — most functions are stubs until later
milestones land):

    from dataset_scout import recon, inspect, curate, ScoutContext

The library is the source of truth; the CLI is a thin wrapper.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from dataset_scout.context import ScoutContext
from dataset_scout.core import (
    Candidate,
    CoverageGap,
    CoverageReport,
    DecompositionDirection,
    Evidence,
    Intent,
    LabelKind,
    LicensePolicy,
    LicenseSummary,
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
    SourceUnavailableError,
)
from dataset_scout.events import ProgressEvent, ProgressEventKind

try:
    __version__ = _metadata.version("dataset-scout")
except _metadata.PackageNotFoundError:  # editable install before metadata exists
    __version__ = "0.0.0+local"


def recon(*args: object, **kwargs: object) -> object:
    """Run the recon pipeline. Stub until M1."""
    raise NotImplementedError("recon lands in M1")


def inspect(*args: object, **kwargs: object) -> object:
    """Inspect one candidate. Stub until M3."""
    raise NotImplementedError("inspect lands in M3")


def curate(*args: object, **kwargs: object) -> object:
    """Curate a corpus from a recipe. Stub until M4."""
    raise NotImplementedError("curate lands in M4")


__all__ = [
    "Candidate",
    "CompositionPairError",
    "CoverageGap",
    "CoverageReport",
    "DatasetScoutError",
    "DecompositionDirection",
    "Evidence",
    "Intent",
    "LabelKind",
    "LicensePolicy",
    "LicenseSummary",
    "ProgressEvent",
    "ProgressEventKind",
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
    "inspect",
    "recon",
]
