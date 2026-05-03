"""Curate orchestrator (M4a — preview slice).

Recipe → materialised JSONL corpus + lockfile + manifest + report +
fingerprint + usage. Per duck guidance the slice is **preview**
quality: hash-mod splits, no MinHash dedup, filter DSL hard-fails. The
output report carries a prominent "not audit-ready" banner; M4b adds
the leakage-aware splitter, dedup, and filter DSL.

Public API:

    from dataset_scout.curate import run_curate
    result = run_curate(recipe, out_dir, ctx=ctx)

The CLI verb is a thin wrapper.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml  # type: ignore[import-untyped]

from dataset_scout.context import ScoutContext
from dataset_scout.core import LabelKind, NormalizedRecord, StrategyKind
from dataset_scout.errors import DatasetScoutError
from dataset_scout.recipe import (
    DEFAULT_SPLITS,
    Recipe,
    RecipeComponent,
    normalize_split_proportions,
)

if TYPE_CHECKING:
    from dataset_scout.sources.base import Source


# Bumped when the curate output schema changes in a breaking way.
CURATE_VERSION = "1"

# Heuristic order for picking a row-id column when the source row has
# multiple plausible candidates. First match wins.
_ID_COLUMN_PREFERENCE: tuple[str, ...] = (
    "id",
    "uuid",
    "row_id",
    "idx",
    "index",
)


# Output filenames written under <out>/.
JSONL_FILENAMES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
}


# ─── public types ────────────────────────────────────────────────────


class CurateOverrides:
    """Effective-value record: what the recipe said vs what curate ran with.

    Both go into the lockfile so a reviewer reading the lock can tell
    whether a CLI flag deviated from the source-of-truth recipe.
    """

    __slots__ = ("min_conf_effective", "min_conf_recipe", "seed_effective", "seed_recipe")

    def __init__(
        self,
        *,
        seed_recipe: int,
        seed_effective: int,
        min_conf_recipe: float,
        min_conf_effective: float,
    ) -> None:
        self.seed_recipe = seed_recipe
        self.seed_effective = seed_effective
        self.min_conf_recipe = min_conf_recipe
        self.min_conf_effective = min_conf_effective

    @property
    def seed_overridden(self) -> bool:
        return self.seed_recipe != self.seed_effective

    @property
    def min_conf_overridden(self) -> bool:
        return self.min_conf_recipe != self.min_conf_effective


class CurateResult:
    """Summary of a curate run; mostly used by the CLI to print a tidy line."""

    def __init__(
        self,
        *,
        out_dir: Path,
        components_kept: int,
        components_skipped: int,
        rows_per_split: dict[str, int],
        fingerprint: str,
        elapsed_seconds: float,
    ) -> None:
        self.out_dir = out_dir
        self.components_kept = components_kept
        self.components_skipped = components_skipped
        self.rows_per_split = rows_per_split
        self.fingerprint = fingerprint
        self.elapsed_seconds = elapsed_seconds

    @property
    def total_rows(self) -> int:
        return sum(self.rows_per_split.values())


# ─── recipe loading ──────────────────────────────────────────────────


def load_recipe(path: Path) -> Recipe:
    """Parse a recipe YAML file into a typed `Recipe`."""
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    return Recipe.model_validate(data)


# ─── filter handling (M4a: hard-fail) ───────────────────────────────


def _validate_no_filters(components: list[RecipeComponent]) -> None:
    """Per duck guidance: silent no-op on filters destroys audit trail.

    Until the filter DSL lands (M4b), curate hard-fails on any non-null
    filter rather than ignoring it.
    """
    offenders: list[str] = []
    for c in components:
        if c.transform.filter:
            offenders.append(f"{c.id} ({c.transform.filter!r})")
    if offenders:
        raise DatasetScoutError(
            "Recipe contains components with non-null `filter` strings, "
            "but the filter DSL is not implemented yet. Either drop the "
            "filter or wait for M4b. Offending components: " + ", ".join(offenders)
        )


# ─── composition cross-references ────────────────────────────────────


def _validate_composition_references(components: list[RecipeComponent]) -> None:
    """Cheap referential integrity. Full portfolio semantics deferred."""
    ids = {c.id for c in components}
    for c in components:
        if c.id in c.composes_with:
            raise DatasetScoutError(f"Component {c.id} composes_with includes itself.")
        for partner in c.composes_with:
            if partner not in ids:
                raise DatasetScoutError(
                    f"Component {c.id} composes_with references unknown id '{partner}'."
                )
        if c.strategy == StrategyKind.COMPOSITION_ONLY:
            raise DatasetScoutError(
                f"Component {c.id} uses strategy 'composition_only', "
                "which is reserved for a future portfolio pass."
            )


# ─── filter components by min_strategy_confidence ────────────────────


def _filter_by_confidence(
    components: list[RecipeComponent], threshold: float
) -> tuple[list[RecipeComponent], list[RecipeComponent]]:
    kept: list[RecipeComponent] = []
    dropped: list[RecipeComponent] = []
    for c in components:
        if c.strategy == StrategyKind.NOT_USEFUL:
            dropped.append(c)
            continue
        if c.strategy_confidence < threshold:
            dropped.append(c)
            continue
        kept.append(c)
    return kept, dropped


# ─── row identity ────────────────────────────────────────────────────


def _row_id(row: dict[str, Any], fallback_index: int) -> tuple[str, str]:
    """Return (row_id, identity_method).

    Identity methods, in order of preference:
      - `column:<name>` — value of an upstream id column
      - `sha256:row` — sha256 of canonical-JSON of the row
      - `index` — last-resort fallback (when JSON canonicalisation fails)
    """
    for col in _ID_COLUMN_PREFERENCE:
        if col in row and row[col] is not None:
            return str(row[col]), f"column:{col}"
    try:
        canonical = json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        return f"idx:{fallback_index}", "index"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16], "sha256:row"


# ─── multimodal coercion ─────────────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """Coerce a value into something JSON can serialize.

    Returns the value unchanged when already JSON-friendly; stringifies
    bytes / arrays / nested feature objects otherwise. This is a pragmatic
    fallback so multimodal HF datasets don't crash the writer.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def _coerce_extras(row: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (jsonable_row, was_coerced)."""
    coerced = False
    out: dict[str, Any] = {}
    for k, v in row.items():
        new_v = _jsonable(v)
        if new_v is not v and not isinstance(v, (dict, list, tuple)):
            coerced = True
        out[str(k)] = new_v
    return out, coerced


# ─── source dispatch (mirrors pipeline.py) ───────────────────────────


def _build_source_index(
    ctx: ScoutContext,
    *,
    sources_override: list[Source] | None = None,
) -> dict[str, Source]:
    """Index sources by `name`. M4a wires HF only; sources_override
    lets tests inject FakeSource."""
    if sources_override is not None:
        return {s.name: s for s in sources_override}

    sources: dict[str, Source] = {}
    enabled = set(ctx.enabled_sources())
    if "huggingface" in enabled:
        from dataset_scout.sources.huggingface import HuggingFaceSource

        token = ctx.api_keys.get("HF_TOKEN") or ctx.api_keys.get("HUGGINGFACE_HUB_TOKEN")
        sources["huggingface"] = HuggingFaceSource(token=token)
    return sources


# ─── component materialisation ───────────────────────────────────────


def _materialize_component(
    component: RecipeComponent,
    *,
    source: Source,
    threat_family: str | None,
) -> Iterator[NormalizedRecord]:
    """Stream rows for one component and yield NormalizedRecord per row.

    The `take` parameter on the recipe transform is honored. Rows whose
    label_value_map mapping is missing for the source-side label are
    silently dropped (the user can add to the map and re-run).
    """
    candidate = _component_to_candidate(component)
    take = component.transform.take
    take_int = None if take == "all" else int(take)
    rows = source.stream_rows(
        candidate,
        config=component.source_config,
        split=component.source_split,
        take=take_int,
    )
    for i, row in enumerate(rows):
        rec = _row_to_record(component, row, i, threat_family)
        if rec is not None:
            yield rec


def _component_to_candidate(component: RecipeComponent) -> Any:
    """Build a minimal Candidate for the source's stream_rows call.

    We only need source/id/revision; the rest of the metadata envelope
    isn't read by HuggingFaceSource.stream_rows.
    """
    from dataset_scout.core import Candidate, CandidateMetadata

    return Candidate(
        source=component.source,
        id=component.source_id,
        revision=component.revision,
        metadata=CandidateMetadata(),
    )


def _row_to_record(
    component: RecipeComponent,
    row: dict[str, Any],
    index: int,
    threat_family: str | None,
) -> NormalizedRecord | None:
    """Apply transform to one row. Returns None if the row should be skipped."""
    transform = component.transform

    text_col = transform.text_column
    if text_col is None or text_col not in row:
        return None
    text_val = row[text_col]
    if text_val is None:
        return None
    text = str(text_val)

    label_col = transform.label_column
    label_kind: LabelKind = LabelKind.GROUND_TRUTH
    if label_col is None:
        # No label column: derive from label_kind_map["all"] and
        # label_value_map["all"] when present, else default to benign + ground_truth.
        label_value = transform.label_value_map.get("all", "benign")
        label_kind = _resolve_label_kind(transform.label_kind_map.get("all"))
    else:
        if label_col not in row or row[label_col] is None:
            return None
        raw_label = str(row[label_col])
        if raw_label not in transform.label_value_map:
            return None  # row label not mapped — skip honestly
        label_value = transform.label_value_map[raw_label]
        label_kind = _resolve_label_kind(
            transform.label_kind_map.get(raw_label) or transform.label_kind_map.get("all")
        )

    extras, coerced = _coerce_extras(row)
    row_id, _ = _row_id(row, index)

    source_str = f"{component.source}:{component.source_id}"

    return NormalizedRecord(
        text=text,
        label=label_value,
        label_kind=label_kind,
        strategy=component.strategy,
        strategy_confidence=component.strategy_confidence,
        source=source_str,
        source_row_id=row_id,
        source_revision=component.revision,
        source_config=component.source_config,
        source_split=component.source_split,
        threat_family=threat_family,
        extras=extras,
        extras_coercion=coerced,
    )


def _resolve_label_kind(value: str | None) -> LabelKind:
    if value is None:
        return LabelKind.GROUND_TRUTH
    try:
        return LabelKind(value)
    except ValueError:
        return LabelKind.GROUND_TRUTH


# ─── splitter (M4a: hash-mod) ────────────────────────────────────────


def _split_records(
    records: list[NormalizedRecord],
    proportions: dict[str, float],
    *,
    seed: int,
    leakage_keys: list[str],
) -> dict[str, list[NormalizedRecord]]:
    """Hash-mod split — deterministic, NOT leakage-aware.

    `recipe.lock.yaml` will record `split_method: "hash_mod"` so the
    audit trail is honest. M4b's MinHash + group-aware splitter
    replaces this without breaking the recipe shape.
    """
    splits: dict[str, list[NormalizedRecord]] = {k: [] for k in proportions}
    # Build cumulative thresholds in canonical order.
    cumulative: list[tuple[str, float]] = []
    running = 0.0
    for name in ("train", "val", "test"):
        running += proportions[name]
        cumulative.append((name, running))
    for rec in records:
        bucket = _bucket_for(rec, seed=seed, cumulative=cumulative)
        splits[bucket].append(rec)
    return splits


def _bucket_for(
    rec: NormalizedRecord,
    *,
    seed: int,
    cumulative: list[tuple[str, float]],
) -> str:
    h = hashlib.sha256()
    h.update(str(seed).encode())
    h.update(b":")
    h.update(rec.source.encode())
    h.update(b":")
    h.update(rec.source_row_id.encode())
    # Map first 16 hex chars to [0, 1).
    fraction = int(h.hexdigest()[:16], 16) / 16**16
    for name, threshold in cumulative:
        if fraction < threshold:
            return name
    return cumulative[-1][0]


# ─── output writers ──────────────────────────────────────────────────


def _write_jsonl(path: Path, records: list[NormalizedRecord]) -> str:
    """Write records as JSONL and return the sha256 of the file's bytes."""
    h = hashlib.sha256()
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            line = rec.model_dump_json() + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def _write_lockfile(
    path: Path,
    *,
    recipe: Recipe,
    kept: list[RecipeComponent],
    dropped: list[RecipeComponent],
    realized: dict[str, dict[str, Any]],
    splits: dict[str, list[NormalizedRecord]],
    overrides: CurateOverrides,
    fingerprint: str,
    started_iso: str,
    elapsed_s: float,
) -> None:
    payload = {
        "recipe_version": recipe.recipe_version,
        "curate_version": CURATE_VERSION,
        "audit_readiness": "preview",
        "audit_readiness_notes": [
            "Hash-mod split is deterministic but NOT leakage-aware.",
            "MinHash dedup is deferred to M4b; near-duplicate rows may cross splits.",
            "Filter DSL is deferred to M4b; recipes with non-null `filter` are rejected.",
        ],
        "intent": recipe.intent.model_dump(mode="json"),
        "min_strategy_confidence": {
            "recipe": overrides.min_conf_recipe,
            "effective": overrides.min_conf_effective,
            "overridden_by_cli": overrides.min_conf_overridden,
        },
        "seed": {
            "recipe": overrides.seed_recipe,
            "effective": overrides.seed_effective,
            "overridden_by_cli": overrides.seed_overridden,
        },
        "splits": {
            "method": "hash_mod",
            "proportions_recipe": recipe.splits.model_dump(mode="json"),
            "realized": {name: len(rows) for name, rows in splits.items()},
        },
        "leakage_keys": list(recipe.leakage_keys),
        "components": [_component_lock_entry(c, realized.get(c.id, {})) for c in kept],
        "declined_components": [
            {
                "id": c.id,
                "source": c.source,
                "source_id": c.source_id,
                "strategy": c.strategy.value,
                "strategy_confidence": c.strategy_confidence,
                "reason": (
                    "below min_strategy_confidence"
                    if c.strategy_confidence < overrides.min_conf_effective
                    and c.strategy != StrategyKind.NOT_USEFUL
                    else c.strategy.value
                ),
            }
            for c in dropped
        ],
        "fingerprint": fingerprint,
        "started_at": started_iso,
        "elapsed_seconds": round(elapsed_s, 3),
        "scout_version": _scout_version(),
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _component_lock_entry(c: RecipeComponent, realized: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": c.id,
        "source": c.source,
        "source_id": c.source_id,
        "revision": c.revision,
        "source_config": c.source_config,
        "source_split": c.source_split,
        "strategy": c.strategy.value,
        "strategy_confidence": c.strategy_confidence,
        "rationale": c.rationale,
        "caveats": list(c.caveats),
        "transform": c.transform.model_dump(mode="json"),
        "composes_with": list(c.composes_with),
        "realized": realized,
    }


def _write_manifest(path: Path, lock_path: Path) -> None:
    """Machine-readable equivalent of the lockfile (just JSON of the YAML)."""
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def _write_report(
    path: Path,
    *,
    recipe: Recipe,
    splits: dict[str, list[NormalizedRecord]],
    realized: dict[str, dict[str, Any]],
    overrides: CurateOverrides,
    fingerprint: str,
    elapsed_s: float,
) -> None:
    total = sum(len(v) for v in splits.values())
    label_counts = _aggregate_label_counts(splits)
    licenses = _aggregate_licenses(realized)

    lines: list[str] = [
        f"# Curated corpus — {recipe.intent.brief.strip().splitlines()[0][:80]}",
        "",
        "> ⚠️ **Preview build — not audit-ready.**  ",
        "> This corpus uses a hash-mod split (deterministic but not\n"
        "> leakage-aware) and skips MinHash dedup. The filter DSL is\n"
        "> deferred to M4b. Treat this as a working artefact, not a\n"
        "> defensible record.",
        "",
        "## Stats",
        "",
        f"- **{splits['train'].__len__():,} train** · "
        f"**{splits['val'].__len__():,} val** · "
        f"**{splits['test'].__len__():,} test** "
        f"({total:,} rows total)",
        f"- **Labels:** "
        f"{label_counts['positive']} positive · "
        f"{label_counts['benign']} benign · "
        f"{label_counts['hard_negative']} hard_negative",
        f"- **Label kinds:** "
        f"{label_counts['ground_truth']} ground_truth · "
        f"{label_counts['proxy']} proxy · "
        f"{label_counts['remapped']} remapped · "
        f"{label_counts['subset_extracted']} subset_extracted",
        f"- **Licenses:** {licenses or '(unknown)'}",
        f"- **Fingerprint:** `{fingerprint[:16]}…`",
        f"- **Wall-clock:** {elapsed_s:.2f}s",
        "",
    ]
    if overrides.min_conf_overridden or overrides.seed_overridden:
        lines += [
            "## CLI overrides applied",
            "",
        ]
        if overrides.min_conf_overridden:
            lines.append(
                f"- `--min-strategy-confidence` "
                f"({overrides.min_conf_recipe} → {overrides.min_conf_effective})"
            )
        if overrides.seed_overridden:
            lines.append(f"- `--seed` ({overrides.seed_recipe} → {overrides.seed_effective})")
        lines.append("")
    lines += [
        "## Components",
        "",
    ]
    for cid, info in realized.items():
        rows = info.get("rows_taken", 0)
        license_raw = info.get("license_raw") or "(unknown)"
        lines.append(f"- `{cid}` — {rows:,} rows · license `{license_raw}`")
    lines += [
        "",
        "---",
        "",
        "_License signals are an SPDX-best-effort guess. Always read the "
        "upstream card before redistributing. This is not legal advice._",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_label_counts(
    splits: dict[str, list[NormalizedRecord]],
) -> dict[str, int]:
    out: dict[str, int] = {
        "positive": 0,
        "benign": 0,
        "hard_negative": 0,
        "ground_truth": 0,
        "proxy": 0,
        "remapped": 0,
        "subset_extracted": 0,
    }
    for recs in splits.values():
        for rec in recs:
            out[rec.label] += 1
            out[rec.label_kind.value] += 1
    return out


def _aggregate_licenses(realized: dict[str, dict[str, Any]]) -> str:
    from collections import Counter

    counter: Counter[str] = Counter()
    for info in realized.values():
        spdx = info.get("license_spdx") or info.get("license_raw") or "(unknown)"
        counter[spdx] += 1
    if not counter:
        return ""
    return ", ".join(f"{count} {name}" for name, count in counter.most_common())


def _write_fingerprint(path: Path, fingerprint: str) -> None:
    path.write_text(fingerprint + "\n", encoding="utf-8")


def _write_usage(path: Path, out_dir: Path) -> None:
    name = out_dir.name
    path.write_text(
        (
            "# Usage snippets\n"
            "\n"
            "```python\n"
            "# huggingface_hub datasets\n"
            "from datasets import load_dataset\n"
            f"ds = load_dataset('json', data_files={{'train': '{name}/train.jsonl', 'val': '{name}/val.jsonl', 'test': '{name}/test.jsonl'}})\n"
            "```\n"
            "\n"
            "```python\n"
            "# pandas\n"
            "import pandas as pd\n"
            f"train = pd.read_json('{name}/train.jsonl', lines=True)\n"
            "```\n"
            "\n"
            "```python\n"
            "# raw jsonl\n"
            "import json\n"
            f"rows = [json.loads(l) for l in open('{name}/train.jsonl')]\n"
            "```\n"
        ),
        encoding="utf-8",
    )


def _scout_version() -> str:
    from dataset_scout import __version__

    return __version__


def _compute_fingerprint(jsonl_hashes: list[str]) -> str:
    h = hashlib.sha256()
    for hh in jsonl_hashes:
        h.update(hh.encode("ascii"))
    return h.hexdigest()


# ─── orchestrator ────────────────────────────────────────────────────


def run_curate(
    recipe: Recipe,
    out_dir: Path,
    *,
    ctx: ScoutContext,
    sources_override: list[Source] | None = None,
    seed_override: int | None = None,
    min_strategy_confidence_override: float | None = None,
) -> CurateResult:
    """Materialise a recipe into a corpus directory.

    Raises `DatasetScoutError` on validation problems (filter strings,
    composition references, missing components). Source-side errors
    propagate; M4b adds a more granular error policy.
    """
    import time

    started = time.monotonic()
    started_iso = datetime.now(UTC).isoformat()

    # ── effective values (recipe-authoritative, CLI as override) ──
    overrides = CurateOverrides(
        seed_recipe=recipe.seed,
        seed_effective=seed_override if seed_override is not None else recipe.seed,
        min_conf_recipe=recipe.min_strategy_confidence,
        min_conf_effective=(
            min_strategy_confidence_override
            if min_strategy_confidence_override is not None
            else recipe.min_strategy_confidence
        ),
    )

    # ── validation ──
    _validate_no_filters(recipe.components)
    _validate_composition_references(recipe.components)
    kept, dropped = _filter_by_confidence(recipe.components, overrides.min_conf_effective)

    if not kept:
        raise DatasetScoutError(
            "No components remain after applying min_strategy_confidence "
            f"({overrides.min_conf_effective:.2f}). Lower the threshold "
            "or hand-edit the recipe."
        )

    # ── source dispatch ──
    source_index = _build_source_index(ctx, sources_override=sources_override)
    for c in kept:
        if c.source not in source_index:
            raise DatasetScoutError(
                f"Component {c.id} uses unknown source '{c.source}'. "
                f"Enabled sources: {list(source_index)}."
            )

    # ── materialize ──
    threat_family = recipe.intent.threat_families[0] if recipe.intent.threat_families else None
    all_records: list[NormalizedRecord] = []
    realized: dict[str, dict[str, Any]] = {}
    for c in kept:
        component_records = list(
            _materialize_component(c, source=source_index[c.source], threat_family=threat_family)
        )
        all_records.extend(component_records)
        realized[c.id] = {
            "rows_taken": len(component_records),
            "label_kind_counts": _label_kind_counts(component_records),
            "license_raw": _component_license_raw(component_records),
            "license_spdx": _component_license_spdx(component_records),
            "row_identity_method": _component_identity_method(component_records),
        }

    # ── split ──
    proportions = (
        normalize_split_proportions(recipe.splits)
        if any((recipe.splits.train, recipe.splits.val, recipe.splits.test))
        else dict(DEFAULT_SPLITS)
    )
    splits = _split_records(
        all_records,
        proportions,
        seed=overrides.seed_effective,
        leakage_keys=list(recipe.leakage_keys),
    )

    # ── write ──
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_hashes: list[str] = []
    for split_name, filename in JSONL_FILENAMES.items():
        h = _write_jsonl(out_dir / filename, splits[split_name])
        jsonl_hashes.append(h)
    fingerprint = _compute_fingerprint(jsonl_hashes)

    # Recipe is dumped via Pydantic; we don't preserve the original
    # YAML formatting verbatim today (the dump is canonical-equivalent).
    (out_dir / "recipe.yaml").write_text(
        yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    elapsed_s = time.monotonic() - started
    lock_path = out_dir / "recipe.lock.yaml"
    _write_lockfile(
        lock_path,
        recipe=recipe,
        kept=kept,
        dropped=dropped,
        realized=realized,
        splits=splits,
        overrides=overrides,
        fingerprint=fingerprint,
        started_iso=started_iso,
        elapsed_s=elapsed_s,
    )
    _write_manifest(out_dir / "manifest.json", lock_path)
    _write_report(
        out_dir / "report.md",
        recipe=recipe,
        splits=splits,
        realized=realized,
        overrides=overrides,
        fingerprint=fingerprint,
        elapsed_s=elapsed_s,
    )
    _write_fingerprint(out_dir / "fingerprint.txt", fingerprint)
    _write_usage(out_dir / "usage.md", out_dir)

    return CurateResult(
        out_dir=out_dir,
        components_kept=len(kept),
        components_skipped=len(dropped),
        rows_per_split={k: len(v) for k, v in splits.items()},
        fingerprint=fingerprint,
        elapsed_seconds=round(elapsed_s, 3),
    )


def _label_kind_counts(records: list[NormalizedRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        out[rec.label_kind.value] = out.get(rec.label_kind.value, 0) + 1
    return out


def _component_license_raw(records: list[NormalizedRecord]) -> str | None:
    # Curate doesn't yet thread license info from the source — placeholder
    # that the lockfile records "(unknown)". M1b's cache will carry license.
    return None


def _component_license_spdx(records: list[NormalizedRecord]) -> str | None:
    return None


def _component_identity_method(records: list[NormalizedRecord]) -> str | None:
    if not records:
        return None
    # All records in a component use the same row-identity method (the
    # method depends on the row's columns, which are the same per source).
    rec = records[0]
    # Re-derive method from the row id format.
    if rec.source_row_id.startswith("idx:"):
        return "index"
    if len(rec.source_row_id) == 16 and all(c in "0123456789abcdef" for c in rec.source_row_id):
        return "sha256:row"
    return "column"


__all__ = [
    "CurateOverrides",
    "CurateResult",
    "load_recipe",
    "run_curate",
]
