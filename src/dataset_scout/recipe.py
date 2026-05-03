"""Recipe schema — typed Pydantic models for `recipe.yaml`.

Both `recipe_draft.py` (writes recipe.draft.yaml after recon) and
`curate.py` (consumes a hand-edited recipe to materialize a corpus)
use this module as the single source of truth for the recipe shape.

The recipe is intentionally small and human-editable. Schema invariants:

- A recipe is reproducible standalone given the lockfile (M4a writes
  the lock; reviewers diff and audit it).
- min_strategy_confidence is RECIPE-AUTHORITATIVE; recon seeds it,
  curate defaults to it, the CLI's --min-strategy-confidence flag
  acts as an override and must be recorded as such.
- `filter` strings are reserved for a future minimal expression DSL.
  Curate hard-fails on any non-null filter today rather than silently
  no-op'ing — silent no-op would destroy the audit trail.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dataset_scout.core import StrategyKind, TransformSpec

# Bumped when the recipe shape changes in a backwards-incompatible way.
# The version is written into the lockfile so reviewers can detect drift.
RECIPE_VERSION = "1"


# Default split shape. Curate reads recipe.splits but uses these when
# the recipe omits the field.
DEFAULT_SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}


class RecipeIntent(BaseModel):
    """The user's stated intent, copied into the recipe verbatim.

    A trimmed-down view of the live `Intent` type — we keep what's
    user-facing in the YAML (brief, target, threats, languages,
    license allow-list) and drop machine-only state.
    """

    model_config = ConfigDict(extra="forbid")

    brief: str
    detection_target: str | None = None
    threat_families: list[str] = Field(default_factory=list)
    deployment_context: str | None = None
    languages: list[str] = Field(default_factory=lambda: ["en"])
    license_allow: list[str] = Field(default_factory=list)


class RecipeSplits(BaseModel):
    """Split proportions. Must sum to ~1.0; curate normalises silently."""

    model_config = ConfigDict(extra="forbid")

    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class RecipeTransform(BaseModel):
    """Transform spec applied per row of a component.

    Mirrors `TransformSpec` from core.py but lives in the recipe namespace
    so future recipe-only fields don't bleed into the LLM-facing type.
    """

    model_config = ConfigDict(extra="forbid")

    text_column: str | None = None
    label_column: str | None = None
    label_value_map: dict[str, Literal["positive", "benign", "hard_negative"]] = Field(
        default_factory=dict
    )
    label_kind_map: dict[str, str] = Field(default_factory=dict)
    filter: str | None = None
    take: int | Literal["all"] = "all"

    @classmethod
    def from_transform_spec(cls, t: TransformSpec) -> RecipeTransform:
        return cls(
            text_column=t.text_column,
            label_column=t.label_column,
            label_value_map=dict(t.label_value_map),
            label_kind_map=dict(t.label_kind_map),
            filter=t.filter,
            take=t.take,
        )


class RecipeComponent(BaseModel):
    """One source component to be materialised into the corpus."""

    model_config = ConfigDict(extra="forbid")

    # Local component id (used for composes_with cross-references and
    # as a stable provenance key in the manifest).
    id: str

    # Source plugin identity.
    source: str  # "huggingface" today
    source_id: str  # e.g. "deepset/prompt-injections"
    revision: str | None = None  # commit SHA when known

    # M4-recipe additions per duck guidance: many HF datasets need
    # explicit config/subset and source_split selection. Defaults
    # reflect the "load_dataset(repo) with no extra args" path.
    source_config: str | None = None
    source_split: str = "train"

    # Provenance from recon (optional but recorded when present).
    surfaced_by: list[str] = Field(default_factory=list)
    rationale: str | None = None
    caveats: list[str] = Field(default_factory=list)

    # Strategy chosen when the recipe was drafted.
    strategy: StrategyKind = StrategyKind.DIRECT_USE
    strategy_confidence: float = 1.0

    transform: RecipeTransform = Field(default_factory=RecipeTransform)

    # Reserved for future portfolio-level pass.
    composes_with: list[str] = Field(default_factory=list)


class Recipe(BaseModel):
    """Top-level recipe: intent + components + corpus shape."""

    model_config = ConfigDict(extra="forbid")

    recipe_version: str = RECIPE_VERSION
    intent: RecipeIntent
    min_strategy_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    splits: RecipeSplits = Field(default_factory=RecipeSplits)
    seed: int = 42
    leakage_keys: list[str] = Field(default_factory=lambda: ["text"])
    components: list[RecipeComponent] = Field(default_factory=list)
    declined: list[dict[str, Any]] = Field(default_factory=list)


def normalize_split_proportions(splits: RecipeSplits) -> dict[str, float]:
    """Return train/val/test as a dict normalized to sum to 1.0.

    The recipe schema doesn't enforce that proportions sum to 1.0 — the
    user might write 80/10/10 as 0.8/0.1/0.1 or 8/1/1. Normalize.
    """
    total = splits.train + splits.val + splits.test
    if total <= 0:
        raise ValueError(f"recipe splits must be positive, got {splits}")
    return {
        "train": splits.train / total,
        "val": splits.val / total,
        "test": splits.test / total,
    }
