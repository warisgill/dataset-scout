"""Public exception types for dataset-scout.

Keep this module dependency-free.
"""

from __future__ import annotations


class DatasetScoutError(Exception):
    """Base class for all dataset-scout errors."""


class SourceUnavailableError(DatasetScoutError):
    """Raised when a registered source can't be reached or auth is missing."""


class SourceUnsupportedError(DatasetScoutError):
    """Raised when a source intentionally does not implement an operation.

    Distinct from "no rows happened to be available": this means the
    source plugin will *never* support the requested operation for this
    candidate (e.g. Kaggle streaming, where dataset shapes are too varied
    to materialise generically). Callers should treat this as a hard
    no-op and not retry, rather than silently producing empty output.
    """


class CompositionPairError(DatasetScoutError):
    """Raised when a recipe keeps one half of a composition pair without the other.

    See `TECH_DESIGN.md` §13.4. Curate must include both halves of a
    `composes_with` pair, or drop both.
    """


class LLMError(DatasetScoutError):
    """Raised when an LLM call fails (no provider, validation, timeout, etc.).

    The pipeline catches this and falls back to metadata-only mode.
    """
