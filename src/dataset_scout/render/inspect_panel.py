"""Pretty-prints an `InspectResult` to stdout.

stdout is the inspect output channel (rather than stderr, which is
where progress / notices go for `recon`/`curate`). This keeps `inspect`
useful in pipes: `datascout inspect ... > inspect.md`.
"""

from __future__ import annotations

from io import StringIO

from dataset_scout.core import (
    InspectResult,
    Strategy,
    StrategyKind,
)


def render_inspect(result: InspectResult) -> str:
    """Render an `InspectResult` to a Markdown-flavoured string.

    Plain markdown — no ANSI/rich codes — so the output works whether
    piped to a file or rendered in a terminal that highlights md.
    """
    buf = StringIO()
    cand = result.candidate
    meta = cand.metadata

    buf.write(f"# {cand.source}:{cand.id}\n\n")
    if cand.revision:
        buf.write(f"_Revision:_ `{cand.revision[:12]}`\n\n")

    if meta.card_url:
        buf.write(f"- 🔗 **Card:** {meta.card_url}\n")
    if meta.description:
        first_line = meta.description.strip().splitlines()[0][:200]
        buf.write(f"- 📝 **Description:** {first_line}\n")
    if cand.requires_auth:
        buf.write("- 🔒 **Access:** gated / requires authentication\n")

    buf.write("\n## License\n\n")
    if result.license_summary:
        ls = result.license_summary
        buf.write(f"- raw: `{ls.raw_string or '(none)'}`\n")
        buf.write(f"- SPDX best-effort: `{ls.spdx_guess or '(unknown)'}`\n")
    else:
        buf.write("_No license declared on the dataset card._\n")

    buf.write("\n## Card-declared metadata\n\n")
    declared: list[tuple[str, str]] = []
    if meta.languages_declared:
        declared.append(("languages", ", ".join(meta.languages_declared)))
    if meta.task_categories:
        declared.append(("task categories", ", ".join(meta.task_categories)))
    if meta.tags:
        declared.append(("tags", ", ".join(meta.tags[:10]) + ("…" if len(meta.tags) > 10 else "")))
    if meta.uploaded_at:
        declared.append(("uploaded", meta.uploaded_at.isoformat()))
    if meta.last_modified:
        declared.append(("last_modified", meta.last_modified.isoformat()))
    if meta.downloads is not None:
        declared.append(("downloads", str(meta.downloads)))
    if meta.likes is not None:
        declared.append(("likes", str(meta.likes)))
    if not declared:
        buf.write("_No card-declared metadata captured._\n")
    else:
        for k, v in declared:
            buf.write(f"- **{k}:** {v}\n")

    buf.write(f"\n## Schema (from {result.sample_size}-row sample)\n\n")
    if not result.columns:
        buf.write("_No rows could be sampled._\n")
    else:
        buf.write("| Column | Inferred type |\n|---|---|\n")
        for col in result.columns:
            buf.write(f"| `{col.name}` | `{col.dtype or '?'}` |\n")

    if result.label_distribution and result.label_column_used:
        buf.write(f"\n## Label distribution (column `{result.label_column_used}`)\n\n")
        buf.write("| value | count | fraction | 95% CI |\n|---|---:|---:|---|\n")
        for b in result.label_distribution:
            buf.write(
                f"| `{b.raw_value}` | {b.count} | {b.fraction * 100:.1f}% | "
                f"{b.ci_low * 100:.0f}-{b.ci_high * 100:.0f}% |\n"
            )

    if result.length_stats:
        lens = result.length_stats
        buf.write(f"\n## Text length (column `{lens.column}`, characters)\n\n")
        buf.write(f"- n: {lens.n}\n")
        buf.write(f"- min: {lens.min}\n")
        buf.write(f"- median: {lens.median}\n")
        buf.write(f"- max: {lens.max}\n")

    if result.sample_rows:
        buf.write("\n## Sample rows\n\n")
        for i, row in enumerate(result.sample_rows, start=1):
            preview = _row_preview(row)
            buf.write(f"**[{i}]** {preview}\n\n")

    if result.intent_used and result.strategies:
        buf.write("\n## Strategy assessment\n\n")
        buf.write(
            f"_Assessed against intent:_ "
            f"{result.intent_used.detection_target or result.intent_used.raw_brief}\n\n"
        )
        for s in result.strategies:
            _render_strategy(buf, s)
    elif result.intent_used:
        buf.write(
            "\n## Strategy assessment\n\n_No strategies returned (or assessment was skipped)._\n"
        )

    if result.notices:
        buf.write("\n## Notices\n\n")
        for n in result.notices:
            buf.write(f"- {n}\n")

    buf.write(f"\n_Wall-clock: {result.elapsed_seconds:.2f}s_\n")
    return buf.getvalue()


# ─── helpers ────────────────────────────────────────────────────────


def _strategy_badge(kind: StrategyKind) -> str:
    return {
        StrategyKind.DIRECT_USE: "✅ direct use",
        StrategyKind.SUBSET_EXTRACTION: "🔎 subset extraction",
        StrategyKind.LABEL_REMAPPING: "🔁 label remapping",
        StrategyKind.CROSS_CLASS_REPURPOSING: "🔀 cross-class repurposing",
        StrategyKind.SIGNAL_PROXY: "📡 signal proxy",
        StrategyKind.BENIGN_BASELINE: "🧊 benign baseline",
        StrategyKind.COMPOSITION_ONLY: "🧩 composition-only",
        StrategyKind.NOT_USEFUL: "❌ not useful",
    }.get(kind, kind.value)


def _render_strategy(buf: StringIO, s: Strategy) -> None:
    badge = _strategy_badge(s.kind)
    buf.write(f"- {badge} — confidence {s.confidence:.2f}\n")
    buf.write(f"  - {s.rationale}\n")
    if s.caveats:
        for c in s.caveats:
            buf.write(f"  - ⚠ {c}\n")
    t = s.transform
    bits: list[str] = []
    if t.text_column:
        bits.append(f"text=`{t.text_column}`")
    if t.label_column:
        bits.append(f"label=`{t.label_column}`")
    if t.label_value_map:
        bits.append(
            "values={" + ", ".join(f"`{k}`→{v}" for k, v in t.label_value_map.items()) + "}"
        )
    if t.take != "all":
        bits.append(f"take={t.take}")
    if bits:
        buf.write("  - transform: " + " · ".join(bits) + "\n")


def _row_preview(row: dict[str, object]) -> str:
    """One-line preview of a sample row for the report."""
    parts: list[str] = []
    for k, v in row.items():
        if isinstance(v, str):
            value = v.strip().replace("\n", " ")
            if len(value) > 80:
                value = value[:77] + "..."
            parts.append(f'`{k}`="{value}"')
        else:
            preview = repr(v)
            if len(preview) > 60:
                preview = preview[:57] + "..."
            parts.append(f"`{k}`={preview}")
        if sum(len(p) for p in parts) > 200:
            parts.append("…")
            break
    return " · ".join(parts)
