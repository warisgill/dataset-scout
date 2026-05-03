"""Public exception types for dataset-scout.

Keep this module dependency-free.
"""

from __future__ import annotations


class DatasetScoutError(Exception):
    """Base class for all dataset-scout errors."""


class SourceUnavailableError(DatasetScoutError):
    """Raised when a registered source can't be reached or auth is missing."""


class CompositionPairError(DatasetScoutError):
    """Raised when a recipe keeps one half of a composition pair without the other.

    See `TECH_DESIGN.md` §13.4. Curate must include both halves of a
    `composes_with` pair, or drop both.
    """
