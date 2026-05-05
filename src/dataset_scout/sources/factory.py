"""Centralised source-plugin construction.

Reference: every public verb (`recon`, `curate`, `inspect`) needs to
instantiate concrete sources from a `ScoutContext`. Without a single
factory, each verb grows its own `_build_source*` helper and they
drift — the rubber-duck reviewer flagged this before Kaggle landed.

Two entry points:

- `build_sources(ctx, *, override=None)`  → ordered list, used by recon.
- `build_source_index(ctx, *, override=None)` → dict[name, Source], used
  by curate and inspect.

Both call into the same private constructor table so adding a new
source means editing exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataset_scout.context import ScoutContext
from dataset_scout.errors import DatasetScoutError, SourceUnavailableError

if TYPE_CHECKING:
    from dataset_scout.sources.base import Source


def _build_huggingface(ctx: ScoutContext) -> Source:
    from dataset_scout.sources.huggingface import HuggingFaceSource

    token = ctx.api_keys.get("HF_TOKEN") or ctx.api_keys.get("HUGGINGFACE_HUB_TOKEN")
    return HuggingFaceSource(token=token)


def _build_kaggle(ctx: ScoutContext) -> Source | None:
    """Build a Kaggle source if creds are present; otherwise return None.

    Kaggle is enabled-by-default in `ScoutContext`'s defaults, but
    requires `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`).
    Quietly skip when not configured rather than raising — the user has
    no work to do unless they want Kaggle results.
    """
    from dataset_scout.sources.kaggle import KaggleSource, kaggle_credentials

    creds = kaggle_credentials(ctx)
    if creds is None:
        return None
    username, key = creds
    return KaggleSource(username=username, key=key)


# Single source of truth: name → constructor that returns a Source or None.
# Returning None means "not configured / quietly skipped".
_BUILDERS: dict[str, Any] = {
    "huggingface": _build_huggingface,
    "kaggle": _build_kaggle,
}


def build_source_index(
    ctx: ScoutContext,
    *,
    override: list[Source] | None = None,
) -> dict[str, Source]:
    """Return enabled sources as a `{name: Source}` dict.

    `override` lets tests inject FakeSource(s) — when given, used verbatim.
    """
    if override is not None:
        return {s.name: s for s in override}

    out: dict[str, Source] = {}
    enabled = list(ctx.enabled_sources())
    for name in enabled:
        builder = _BUILDERS.get(name)
        if builder is None:
            # Source declared in ScoutContext but no builder wired —
            # development-time only; do not silently swallow.
            raise DatasetScoutError(
                f"source '{name}' is enabled in ScoutContext but no "
                f"factory entry is wired in sources.factory."
            )
        instance = builder(ctx)
        if instance is None:
            # Quietly skipped (e.g. Kaggle creds absent).
            continue
        out[name] = instance
    return out


def build_sources(
    ctx: ScoutContext,
    *,
    override: list[Source] | None = None,
) -> list[Source]:
    """Return enabled sources as an ordered list.

    Order matches `ctx.enabled_sources()` so search results from the
    first source appear first in the round-robin. Raises
    `SourceUnavailableError` if zero sources are runnable.
    """
    index = build_source_index(ctx, override=override)
    sources = list(index.values())
    if not sources:
        raise SourceUnavailableError(
            "No sources are enabled and runnable. Configure HuggingFace "
            "(default-on) or Kaggle (set KAGGLE_USERNAME and KAGGLE_KEY) "
            "in your ScoutContext."
        )
    return sources


def known_source_names() -> list[str]:
    """Names of sources the factory knows how to build, in stable order."""
    return list(_BUILDERS.keys())
