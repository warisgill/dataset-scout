"""Compose multiple recipes into one.

Recommendation G from the threat-intel walkthrough. Multi-detection
programs need to merge recipe.draft.yaml outputs from several `recon`
runs into a single corpus.

Semantics:
- Components dedupe by (source, source_id) — the second occurrence
  wins ONLY when its strategy_confidence is strictly higher; otherwise
  the first wins. A warning is recorded for any conflict.
- Intent is taken from the FIRST input recipe with a fallback to a
  synthesized "merged" intent when briefs differ.
- Splits, seed, leakage_keys come from the first input unless a
  --override is given.
- Declined components are unioned (dedupe on (source, source_id)) so
  reviewers see why each was dropped.
- min_strategy_confidence is the MAXIMUM of the inputs (most
  conservative wins).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from dataset_scout.recipe import (
    Recipe,
    RecipeComponent,
    RecipeIntent,
)


def compose_recipes(
    recipes: list[Recipe],
    *,
    intent_override: RecipeIntent | None = None,
) -> tuple[Recipe, list[str]]:
    """Merge a list of recipes into one. Returns (merged, conflict notices).

    Behaviour:
    - dedupe components by (source, source_id); higher
      `strategy_confidence` wins on conflict; the loser becomes a notice
    - declined entries are unioned and deduped
    - intent is taken from `intent_override` if given, else from the
      first input
    - splits, seed, leakage_keys come from the first input
    - min_strategy_confidence is the MAX of the inputs (conservative)
    """
    if not recipes:
        raise ValueError("compose_recipes() requires at least one input recipe")

    base = recipes[0]
    notices: list[str] = []

    # Dedupe components, preferring higher strategy_confidence.
    by_key: OrderedDict[tuple[str, str], RecipeComponent] = OrderedDict()
    for recipe in recipes:
        for c in recipe.components:
            key = (c.source, c.source_id)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = c
                continue
            if c.strategy_confidence > existing.strategy_confidence:
                notices.append(
                    f"component {c.source}:{c.source_id} appeared in multiple "
                    f"input recipes; kept the higher-confidence variant "
                    f"(strategy={c.strategy.value} conf={c.strategy_confidence:.2f}, "
                    f"replaced strategy={existing.strategy.value} conf={existing.strategy_confidence:.2f})"
                )
                by_key[key] = c
            elif c.strategy_confidence < existing.strategy_confidence:
                notices.append(
                    f"component {c.source}:{c.source_id} duplicate dropped: "
                    f"strategy={c.strategy.value} conf={c.strategy_confidence:.2f} "
                    f"(kept higher-confidence variant strategy={existing.strategy.value} "
                    f"conf={existing.strategy_confidence:.2f})"
                )
            else:
                # Same confidence — pick the FIRST seen and warn if the
                # strategies differ (potentially conflicting recipes).
                if c.strategy != existing.strategy:
                    notices.append(
                        f"component {c.source}:{c.source_id} appears with "
                        f"different strategies in inputs (kept "
                        f"{existing.strategy.value}, dropped {c.strategy.value}); "
                        f"hand-edit if you need the other"
                    )

    # Union declined.
    declined_seen: set[tuple[str, str]] = set()
    declined: list[dict[str, Any]] = []
    for recipe in recipes:
        for entry in recipe.declined:
            key = (str(entry.get("source", "")), str(entry.get("source_id", "")))
            if key in declined_seen:
                continue
            declined_seen.add(key)
            declined.append(entry)

    # Conservative threshold: max of inputs.
    threshold = max(r.min_strategy_confidence for r in recipes)

    # Intent.
    intent = intent_override or base.intent
    # Note when intents diverge — useful for reviewers.
    if intent_override is None and len({r.intent.brief for r in recipes}) > 1:
        notices.append(
            f"input recipes had different briefs; kept the first "
            f"({base.intent.brief!r}). Pass intent_override / --intent-brief "
            "if you want a different one."
        )

    merged = Recipe(
        recipe_version=base.recipe_version,
        intent=intent,
        min_strategy_confidence=threshold,
        splits=base.splits,
        seed=base.seed,
        leakage_keys=list(base.leakage_keys),
        components=list(by_key.values()),
        declined=declined,
    )
    return merged, notices


def write_composed_recipe(merged: Recipe, out_path: Path) -> Path:
    """Dump a composed recipe to a YAML file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(
            merged.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return out_path
