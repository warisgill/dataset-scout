"""End-to-end curate tests using FakeSource."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    DatasetScoutError,
    NormalizedRecord,
    ScoutContext,
    StrategyKind,
)
from dataset_scout.curate import load_recipe, run_curate
from dataset_scout.recipe import (
    Recipe,
    RecipeComponent,
    RecipeIntent,
    RecipeSplits,
    RecipeTransform,
)
from tests._fakes.fake_source import FakeSource

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


def _two_class_rows(n: int = 100) -> list[dict[str, object]]:
    """Half positive, half benign with stable text + integer id."""
    rows: list[dict[str, object]] = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "text": f"prompt content row {i}",
                "label": 1 if i % 2 == 0 else 0,
            }
        )
    return rows


def _make_recipe(
    *,
    components: list[RecipeComponent],
    splits: RecipeSplits | None = None,
    seed: int = 42,
    threshold: float = 0.5,
) -> Recipe:
    return Recipe(
        intent=RecipeIntent(
            brief="prompt injection corpora",
            detection_target="prompt injection",
            threat_families=["prompt_injection"],
        ),
        min_strategy_confidence=threshold,
        seed=seed,
        splits=splits or RecipeSplits(),
        components=components,
    )


def _component(
    cid: str = "fake_org_x",
    *,
    strategy: StrategyKind = StrategyKind.DIRECT_USE,
    confidence: float = 0.9,
    take: int | str = "all",
    text_column: str = "text",
    label_column: str | None = "label",
    label_value_map: dict[str, str] | None = None,
    label_kind_map: dict[str, str] | None = None,
    filter_str: str | None = None,
) -> RecipeComponent:
    return RecipeComponent(
        id=cid,
        source="fake",
        source_id="org/x",
        revision="r1",
        source_split="train",
        strategy=strategy,
        strategy_confidence=confidence,
        transform=RecipeTransform(
            text_column=text_column,
            label_column=label_column,
            label_value_map=(
                label_value_map if label_value_map is not None else {"1": "positive", "0": "benign"}
            ),
            label_kind_map=label_kind_map or {},
            filter=filter_str,
            take=take,
        ),
    )


def _ctx() -> ScoutContext:
    return ScoutContext.from_env(env={})


def _fake(rows: list[dict[str, object]]) -> FakeSource:
    cand = Candidate(
        source="fake",
        id="org/x",
        revision="r1",
        metadata=CandidateMetadata(),
    )
    return FakeSource([cand], samples={"org/x": rows})


# ─── load_recipe ────────────────────────────────────────────────────


def test_load_recipe_round_trip(tmp_path: Path):
    recipe = _make_recipe(components=[_component()])
    target = tmp_path / "recipe.yaml"
    target.write_text(yaml.safe_dump(recipe.model_dump(mode="json")), encoding="utf-8")
    loaded = load_recipe(target)
    assert loaded.intent.brief == "prompt injection corpora"
    assert len(loaded.components) == 1
    assert loaded.components[0].source_id == "org/x"


# ─── full materialisation ───────────────────────────────────────────


def test_run_curate_writes_all_artefacts(tmp_path: Path):
    rows = _two_class_rows(n=80)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    out = tmp_path / "corpus"

    result = run_curate(recipe, out, ctx=_ctx(), sources_override=[fake])

    assert result.total_rows == 80
    assert result.components_kept == 1
    # All expected files written.
    for fname in (
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "recipe.yaml",
        "recipe.lock.yaml",
        "manifest.json",
        "report.md",
        "fingerprint.txt",
        "usage.md",
    ):
        assert (out / fname).exists(), f"missing {fname}"

    # Leakage-aware MinHash splits give wider variance on small N
    # because each cluster is assigned wholesale; relax the band but
    # keep the total invariant.
    train_lines = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (out / "val.jsonl").read_text(encoding="utf-8").splitlines()
    test_lines = (out / "test.jsonl").read_text(encoding="utf-8").splitlines()
    assert 40 <= len(train_lines) <= 80
    assert 0 <= len(val_lines) <= 30
    assert 0 <= len(test_lines) <= 30
    assert len(train_lines) + len(val_lines) + len(test_lines) == 80


def test_run_curate_records_are_valid_normalized(tmp_path: Path):
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])

    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    # Every line in train.jsonl validates as NormalizedRecord.
    for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines():
        rec = NormalizedRecord.model_validate_json(line)
        assert rec.source == "fake:org/x"
        assert rec.label in {"positive", "benign", "hard_negative"}
        assert rec.strategy == StrategyKind.DIRECT_USE


def test_run_curate_label_kind_propagates(tmp_path: Path):
    rows = _two_class_rows(n=20)
    fake = _fake(rows)
    component = _component(
        strategy=StrategyKind.SIGNAL_PROXY,
        label_kind_map={"1": "proxy", "0": "ground_truth"},
    )
    recipe = _make_recipe(components=[component])

    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    seen_kinds: set[str] = set()
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for line in (tmp_path / split).read_text(encoding="utf-8").splitlines():
            seen_kinds.add(NormalizedRecord.model_validate_json(line).label_kind.value)
    assert "proxy" in seen_kinds


# ─── filter hard-fail (M4a) ─────────────────────────────────────────


def test_run_curate_applies_filter_expression(tmp_path: Path):
    """Non-null filter is now compiled and applied (M4b filter DSL)."""
    rows = _two_class_rows(n=10)
    fake = _fake(rows)
    # Filter to label == 1 only — half the rows
    component = _component(filter_str="label == 1")
    recipe = _make_recipe(components=[component])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    total = 0
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total += len((tmp_path / split).read_text(encoding="utf-8").splitlines())
    # Of 10 rows, 5 had label==1 (i % 2 == 0 → labels 1,0,1,0,...). Hmm
    # actually our _two_class_rows uses i % 2 == 0 → label 1, so 5 pass.
    assert total == 5


def test_run_curate_rejects_invalid_filter_syntax(tmp_path: Path):
    """Compile-time errors are surfaced clearly to the user."""
    fake = _fake(_two_class_rows(n=10))
    component = _component(filter_str="this is not valid python")
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="invalid filter expression"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


def test_run_curate_rejects_disallowed_filter_function(tmp_path: Path):
    """Whitelist enforcement: __import__, attribute access, etc. blocked."""
    fake = _fake(_two_class_rows(n=10))
    component = _component(filter_str="__import__('os').system('rm -rf /')")
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="invalid filter expression"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


# ─── min_strategy_confidence ─────────────────────────────────────────


def test_run_curate_filters_below_threshold(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    weak = _component(cid="weak", confidence=0.3)
    strong = _component(cid="strong", confidence=0.8)
    recipe = _make_recipe(
        components=[strong, weak],
        threshold=0.5,
    )
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])

    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    kept_ids = {c["id"] for c in lock["components"]}
    declined_ids = {c["id"] for c in lock["declined_components"]}
    assert kept_ids == {"strong"}
    assert "weak" in declined_ids


def test_run_curate_min_confidence_override_recorded(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    component = _component(confidence=0.7)
    recipe = _make_recipe(components=[component], threshold=0.5)
    # Override threshold to 0.8 — component now drops.
    with pytest.raises(DatasetScoutError, match="No components remain"):
        run_curate(
            recipe,
            tmp_path,
            ctx=_ctx(),
            sources_override=[fake],
            min_strategy_confidence_override=0.8,
        )


def test_run_curate_seed_override_recorded(tmp_path: Path):
    fake = _fake(_two_class_rows(n=20))
    recipe = _make_recipe(components=[_component()], seed=42)
    run_curate(
        recipe,
        tmp_path,
        ctx=_ctx(),
        sources_override=[fake],
        seed_override=99,
    )
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert lock["seed"]["recipe"] == 42
    assert lock["seed"]["effective"] == 99
    assert lock["seed"]["overridden_by_cli"] is True


# ─── lockfile / manifest / report shape ─────────────────────────────


def test_lockfile_has_audit_readiness_flag(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    lock = yaml.safe_load((tmp_path / "recipe.lock.yaml").read_text(encoding="utf-8"))
    assert lock["audit_readiness"] == "ready"
    assert any(
        "leakage-aware" in n.lower() or "minhash" in n.lower()
        for n in lock["audit_readiness_notes"]
    )
    # Splits block records the dedup parameters.
    assert lock["splits"]["method"] == "minhash_lsh"
    assert "num_perm" in lock["splits"]
    assert "clusters_total" in lock["splits"]


def test_report_carries_audit_ready_banner(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "audit-ready" in report.lower()
    assert "leakage-aware" in report.lower()


def test_manifest_is_json_round_trip(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["audit_readiness"] == "ready"
    assert manifest["recipe_version"] == "1"


def test_fingerprint_is_deterministic_for_same_seed(tmp_path: Path):
    """Same recipe + same seed -> identical fingerprint across runs."""
    rows = _two_class_rows(n=30)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    recipe = _make_recipe(components=[_component()], seed=7)

    run_curate(recipe, out_a, ctx=_ctx(), sources_override=[_fake(rows)])
    run_curate(recipe, out_b, ctx=_ctx(), sources_override=[_fake(rows)])

    fp_a = (out_a / "fingerprint.txt").read_text(encoding="utf-8").strip()
    fp_b = (out_b / "fingerprint.txt").read_text(encoding="utf-8").strip()
    assert fp_a == fp_b


# ─── composition reference checks ───────────────────────────────────


def test_run_curate_rejects_composition_only_strategy(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    component = _component(
        strategy=StrategyKind.COMPOSITION_ONLY,
        confidence=0.9,
    )
    recipe = _make_recipe(components=[component])
    with pytest.raises(DatasetScoutError, match="composition_only"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


def test_run_curate_rejects_unknown_composes_with(tmp_path: Path):
    fake = _fake(_two_class_rows(n=10))
    a = _component(cid="a")
    a = a.model_copy(update={"composes_with": ["b"]})
    recipe = _make_recipe(components=[a])
    with pytest.raises(DatasetScoutError, match="references unknown id"):
        run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])


# ─── unmapped labels are silently dropped ───────────────────────────


def test_unmapped_label_rows_dropped(tmp_path: Path):
    """Rows whose source label isn't in label_value_map are skipped, not crashed."""
    rows: list[dict[str, object]] = [
        {"id": 0, "text": "a", "label": 1},  # mapped
        {"id": 1, "text": "b", "label": 999},  # unmapped — dropped
        {"id": 2, "text": "c", "label": 0},  # mapped
    ]
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    total = 0
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        total += len((tmp_path / split).read_text(encoding="utf-8").splitlines())
    assert total == 2


# ─── multimodal coercion ────────────────────────────────────────────


def test_extras_coerce_bytes_values(tmp_path: Path):
    rows: list[dict[str, object]] = [
        {"id": 0, "text": "hello", "label": 1, "image_bytes": b"\\x89PNG..."},
    ]
    fake = _fake(rows)
    recipe = _make_recipe(components=[_component()])
    run_curate(recipe, tmp_path, ctx=_ctx(), sources_override=[fake])
    for split in ("train.jsonl", "val.jsonl", "test.jsonl"):
        for line in (tmp_path / split).read_text(encoding="utf-8").splitlines():
            rec = NormalizedRecord.model_validate_json(line)
            assert rec.extras_coercion is True
            assert isinstance(rec.extras["image_bytes"], str)
