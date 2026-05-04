"""recipe.draft.yaml emission (M2b).

After strategies are assessed we emit a draft recipe — a hand-editable
YAML document the user massages and feeds to `datascout curate` (M4).

The draft pulls the best strategy per candidate above
`min_strategy_confidence`, packages each as a recipe component, and
documents the input intent so future runs can diff cleanly.

`min_strategy_confidence` is recipe-authoritative: the value lands in
the YAML and `curate` uses it as the default. Recon's CLI flag seeds
that value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from dataset_scout.core import (
    Intent,
    LabelKind,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
)

# Strategies whose default label_kind we know without an explicit map.
# subset_extraction inherits from the source's existing label semantics
# (treated as ground_truth on the filtered subset). label_remapping is
# remapped by definition. signal_proxy is proxy by definition. Others
# are written through verbatim.
_DEFAULT_LABEL_KIND_FOR: dict[StrategyKind, LabelKind] = {
    StrategyKind.DIRECT_USE: LabelKind.GROUND_TRUTH,
    StrategyKind.SUBSET_EXTRACTION: LabelKind.SUBSET_EXTRACTED,
    StrategyKind.LABEL_REMAPPING: LabelKind.REMAPPED,
    StrategyKind.SIGNAL_PROXY: LabelKind.PROXY,
}


def _component_id(scorecard: Scorecard) -> str:
    """Local component id used by recipe references (composes_with etc.).

    Format: ``<source>_<id-with-/-replaced>``.
    """
    safe_id = scorecard.candidate.id.replace("/", "_").replace(":", "_")
    return f"{scorecard.candidate.source}_{safe_id}"


def _sanitize_filter(raw: str | None) -> tuple[str | None, str | None]:
    """Validate an LLM-drafted filter against the curate DSL.

    Returns ``(filter_or_None, note_or_None)``. If the filter compiles
    cleanly we keep it. Otherwise we drop it (return None) and surface
    a short caveat so the recipe edit step can put the constraint in
    prose where it belongs. This keeps ``curate`` from hard-failing on
    LLM mistakes in the *draft* — once the user has reviewed and
    saved the recipe, curate's strict validation kicks in.
    """
    if raw is None or not str(raw).strip():
        return None, None
    from dataset_scout.filter_dsl import FilterCompileError, compile_filter

    try:
        compile_filter(str(raw))
    except FilterCompileError as exc:
        return None, (
            f"LLM-drafted filter dropped (not a valid DSL expression): "
            f"{exc}. Original: {raw!r}. If you need a row-level filter, "
            f"re-express as e.g. `column == 'value'` or `len(text) > 30`."
        )
    return raw, None


def _component_dict(sc: Scorecard, strategy: Strategy) -> dict[str, Any]:
    cand = sc.candidate
    transform = strategy.transform
    sanitized_filter, filter_caveat = _sanitize_filter(transform.filter)
    component: dict[str, Any] = {
        "id": _component_id(sc),
        "source": cand.source,
        "source_id": cand.id,
        "revision": cand.revision,
        "surfaced_by": list(cand.surfaced_by),
        "strategy": strategy.kind.value,
        "strategy_confidence": round(strategy.confidence, 3),
        "rationale": strategy.rationale,
        "transform": {
            "text_column": transform.text_column,
            "label_column": transform.label_column,
            "label_value_map": dict(transform.label_value_map),
            "label_kind_map": dict(transform.label_kind_map) or _default_label_kind_map(strategy),
            "filter": sanitized_filter,
            "take": transform.take,
        },
    }
    caveats = list(strategy.caveats)
    if filter_caveat:
        caveats.append(filter_caveat)
    if caveats:
        component["caveats"] = caveats
    if strategy.composes_with:
        component["composes_with"] = list(strategy.composes_with)
    return component


def _default_label_kind_map(strategy: Strategy) -> dict[str, str]:
    """When the assessor didn't supply a label_kind_map, derive a sensible
    default from the strategy kind. The user can override during edit."""
    default = _DEFAULT_LABEL_KIND_FOR.get(strategy.kind)
    if default is None:
        return {}
    return {"all": default.value}


def build_recipe_draft(result: ReconResult) -> dict[str, Any]:
    """Build the dict we'll YAML-dump as recipe.draft.yaml.

    Selects each scorecard's best strategy whose confidence meets the
    intent's `min_strategy_confidence`. Candidates without a qualifying
    strategy are written as a commented-out section so the user can
    consider them when editing.
    """
    intent = result.intent
    threshold = intent.min_strategy_confidence

    components: list[dict[str, Any]] = []
    declined: list[dict[str, Any]] = []
    for sc in result.candidates:
        best = sc.best_strategy
        if best is None or best.kind == StrategyKind.NOT_USEFUL:
            declined.append(
                {
                    "source": sc.candidate.source,
                    "source_id": sc.candidate.id,
                    "reason": (best.rationale if best is not None else "no strategy assessed"),
                }
            )
            continue
        if best.confidence < threshold:
            declined.append(
                {
                    "source": sc.candidate.source,
                    "source_id": sc.candidate.id,
                    "reason": (
                        f"best strategy '{best.kind.value}' below confidence "
                        f"threshold ({best.confidence:.2f} < {threshold:.2f})"
                    ),
                }
            )
            continue
        components.append(_component_dict(sc, best))

    return {
        "intent": _intent_dict(intent),
        "min_strategy_confidence": threshold,
        "splits": {"train": 0.8, "val": 0.1, "test": 0.1},
        "seed": 42,
        "leakage_keys": ["text"],
        "components": components,
        "declined": declined,
    }


def _intent_dict(intent: Intent) -> dict[str, Any]:
    return {
        "brief": intent.raw_brief,
        "detection_target": intent.detection_target,
        "threat_families": list(intent.threat_families),
        "deployment_context": intent.deployment_context,
        "languages": list(intent.languages),
        "license_allow": sorted(intent.license_policy.allow),
    }


def write_recipe_draft(result: ReconResult, out_dir: Path) -> Path | None:
    """Write `<out_dir>/recipe.draft.yaml`. Returns None if no candidates
    have a strategy worth drafting (e.g., metadata-only mode)."""
    if not any(sc.best_strategy is not None for sc in result.candidates):
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "recipe.draft.yaml"
    payload = build_recipe_draft(result)
    target.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
