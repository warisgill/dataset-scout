"""Unit tests for recipe composition (recommendation G)."""

from __future__ import annotations

import pytest

from dataset_scout import StrategyKind
from dataset_scout.recipe import (
    Recipe,
    RecipeComponent,
    RecipeIntent,
    RecipeSplits,
    RecipeTransform,
)
from dataset_scout.recipe_compose import compose_recipes

pytestmark = pytest.mark.unit


def _component(
    cid: str,
    *,
    source_id: str = "org/x",
    strategy: StrategyKind = StrategyKind.DIRECT_USE,
    confidence: float = 0.8,
) -> RecipeComponent:
    return RecipeComponent(
        id=cid,
        source="huggingface",
        source_id=source_id,
        strategy=strategy,
        strategy_confidence=confidence,
        transform=RecipeTransform(text_column="text", label_column="label"),
    )


def _recipe(
    components: list[RecipeComponent],
    *,
    brief: str = "test brief",
    threshold: float = 0.5,
) -> Recipe:
    return Recipe(
        intent=RecipeIntent(brief=brief),
        min_strategy_confidence=threshold,
        splits=RecipeSplits(),
        components=components,
    )


def test_compose_requires_inputs():
    with pytest.raises(ValueError):
        compose_recipes([])


def test_compose_unique_components_unioned():
    a = _recipe([_component("a", source_id="org/a")])
    b = _recipe([_component("b", source_id="org/b")])
    merged, notices = compose_recipes([a, b])
    ids = {c.source_id for c in merged.components}
    assert ids == {"org/a", "org/b"}
    assert notices == []


def test_compose_higher_confidence_wins_on_dup():
    a = _recipe([_component("low", source_id="org/x", confidence=0.4)])
    b = _recipe([_component("high", source_id="org/x", confidence=0.9)])
    merged, notices = compose_recipes([a, b])
    assert len(merged.components) == 1
    assert merged.components[0].strategy_confidence == 0.9
    assert any("higher-confidence" in n for n in notices)


def test_compose_keeps_first_on_equal_confidence():
    a = _recipe([_component("first", source_id="org/x", strategy=StrategyKind.DIRECT_USE)])
    b = _recipe([_component("second", source_id="org/x", strategy=StrategyKind.SIGNAL_PROXY)])
    merged, notices = compose_recipes([a, b])
    assert merged.components[0].strategy == StrategyKind.DIRECT_USE
    # Different strategies but equal confidence triggers a warning.
    assert any("different strategies" in n for n in notices)


def test_compose_threshold_is_max_of_inputs():
    a = _recipe([_component("a")], threshold=0.4)
    b = _recipe([_component("b", source_id="org/b")], threshold=0.7)
    merged, _ = compose_recipes([a, b])
    assert merged.min_strategy_confidence == 0.7


def test_compose_intent_override():
    a = _recipe([_component("a")], brief="brief A")
    b = _recipe([_component("b", source_id="org/b")], brief="brief B")
    merged, notices = compose_recipes([a, b])
    # Different briefs without override -> first wins, with notice.
    assert merged.intent.brief == "brief A"
    assert any("different briefs" in n for n in notices)


def test_compose_with_explicit_intent_override():
    a = _recipe([_component("a")], brief="brief A")
    b = _recipe([_component("b", source_id="org/b")], brief="brief B")
    merged, notices = compose_recipes([a, b], intent_override=RecipeIntent(brief="merged brief"))
    assert merged.intent.brief == "merged brief"
    assert not any("different briefs" in n for n in notices)


def test_compose_unions_declined_components():
    a = _recipe([_component("a")])
    a = a.model_copy(update={"declined": [{"source": "huggingface", "source_id": "org/junk"}]})
    b = _recipe([_component("b", source_id="org/b")])
    b = b.model_copy(
        update={
            "declined": [
                {"source": "huggingface", "source_id": "org/junk"},  # duplicate
                {"source": "huggingface", "source_id": "org/other-junk"},
            ]
        }
    )
    merged, _ = compose_recipes([a, b])
    declined_ids = {d["source_id"] for d in merged.declined}
    assert declined_ids == {"org/junk", "org/other-junk"}
