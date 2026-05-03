"""Markdown report renderer.

Three framings depending on what ran:

- **Metadata-only.** No LLM configured. Discovery-only language.
- **Discovery + decomposition.** LLM ran the decomposer; candidates
  surfaced from multiple directions but no per-candidate strategies.
- **Strategy-assessed.** Per-candidate strategies + (optional)
  coverage gaps. Report leads with the strongest fits.

Receipts everywhere — every claim links back to a card URL or
specific evidence.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from dataset_scout.core import (
    Evidence,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    SubScore,
)


def write_recon_report(result: ReconResult, out_dir: Path) -> Path:
    """Render and write `<out_dir>/report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.md"
    target.write_text(render_recon_report(result), encoding="utf-8")
    return target


def _has_strategies(result: ReconResult) -> bool:
    return any(sc.strategies for sc in result.candidates)


def render_recon_report(result: ReconResult) -> str:
    """Render a ReconResult to a Markdown string."""
    buf = StringIO()
    intent = result.intent
    metadata_only = any("Azure OpenAI is not configured" in n for n in result.notices)
    llm_runtime_error = any("Azure OpenAI was configured but" in n for n in result.notices)
    has_strategies = _has_strategies(result)
    has_decomposition = bool(result.coverage and result.coverage.decomposition)
    notable_gaps = bool(result.coverage and len(result.coverage.semantic_gaps) >= 2)

    buf.write("# dataset-scout recon report\n\n")
    if metadata_only:
        buf.write(
            "> ⚠️ **Metadata-only mode.**  \n"
            "> Azure OpenAI is not configured, so decomposition, strategy\n"
            "> assessment, and coverage gaps were skipped. To enable them,\n"
            "> copy `.env.example` to `.env`, set `AZURE_OPENAI_ENDPOINT`\n"
            "> and `AZURE_OPENAI_DEPLOYMENT`, and run `az login`.\n\n"
        )
    elif llm_runtime_error:
        buf.write(
            "> ⚠️ **LLM call failed — running in metadata-only mode.**  \n"
            "> Azure OpenAI was configured but a call failed at runtime\n"
            "> (deployment name, token, network, or quota — see notices\n"
            "> below). Decomposition, strategy assessment, and coverage\n"
            "> gaps were skipped.\n\n"
        )
    elif has_strategies:
        n_dirs = len(result.coverage.decomposition) if result.coverage else 0
        buf.write(
            "> **Strategy-assessed.**  \n"
            "> Search expanded across the original brief and "
            f"{n_dirs} related direction(s); shortlisted candidates were "
            "assessed for direct fits and reframings. Listed here in\n"
            "> best-strategy order; review the strategy + caveats before "
            "committing.\n\n"
        )
    elif has_decomposition and result.coverage is not None:
        buf.write(
            "> **Discovery + decomposition.**  \n"
            "> Search expanded across the original brief and "
            f"{len(result.coverage.decomposition)} related direction(s) "
            "proposed by the LLM.\n\n"
        )
    else:
        buf.write(
            "> **Pre-fit metadata screening.**  \n"
            "> Candidates are returned in source/search relevance order.\n"
            "> Probe outputs are annotations, not a ranking score.\n\n"
        )

    buf.write("## Brief\n\n")
    buf.write(f"**Raw brief:** {intent.raw_brief}\n\n")
    if intent.detection_target:
        buf.write(f"**Detection target:** {intent.detection_target}\n\n")
    if intent.threat_families:
        buf.write(f"**Threat families:** {', '.join(intent.threat_families)}\n\n")
    buf.write(f"**Languages requested:** {', '.join(intent.languages)}\n\n")

    # ─── Coverage gaps (lead when notable) ─────────────────────
    if notable_gaps and result.coverage:
        buf.write("## Coverage gaps\n\n")
        buf.write(
            "The candidates above don't fully cover what you described. Concrete next steps:\n\n"
        )
        for gap in result.coverage.semantic_gaps:
            buf.write(f"- **{gap.aspect}** — {gap.description}\n")
            buf.write(f"  - *Suggestion:* {gap.suggestion}\n")
        buf.write("\n")

    # ─── Decomposition audit ────────────────────────────────────
    if has_decomposition and result.coverage:
        buf.write("## Decomposition\n\n")
        buf.write("The LLM proposed these search directions in addition to the original brief:\n\n")
        for d in result.coverage.decomposition:
            buf.write(f"- **{d.name}** — {d.rationale}\n")
            if d.keywords:
                buf.write(f"  - keywords: `{', '.join(d.keywords)}`\n")
            if d.expected_finds:
                buf.write(f"  - expected: {d.expected_finds}\n")
        buf.write("\n")

    # ─── Run summary ────────────────────────────────────────────
    buf.write("## Run summary\n\n")
    buf.write(f"- Sources searched: {', '.join(result.sources_searched) or '(none)'}\n")
    buf.write(f"- Candidates returned: **{len(result.candidates)}**\n")
    if has_strategies:
        assessed = sum(1 for sc in result.candidates if sc.strategies)
        buf.write(f"- Strategy-assessed: **{assessed}**\n")
    buf.write(f"- Wall-clock: {result.elapsed_seconds:.2f}s\n")
    buf.write(f"- dataset-scout version: {result.scout_version}\n")
    if result.notices:
        buf.write("\n### Notices\n\n")
        for n in result.notices:
            buf.write(f"- {n}\n")
    buf.write("\n")

    if not result.candidates:
        buf.write("No candidates were returned. Try broadening the brief.\n")
        return buf.getvalue()

    # ─── Coverage gaps (non-leading; below candidates section) ──
    # When gaps aren't notable, we render them below candidates instead
    # of leading with them.

    buf.write("## Candidates\n\n")
    if has_strategies:
        buf.write(
            "Listed in **best-strategy order.** Each candidate's "
            "primary strategy and confidence are shown; review caveats\n"
            "before committing.\n\n"
        )
    elif has_decomposition:
        buf.write(
            "Listed in search-relevance order, deduped across the "
            "original brief and decomposition directions. The "
            "`surfaced by` annotation\n"
            "on each candidate shows which direction(s) found it.\n\n"
        )
    else:
        buf.write(
            "Listed in **search-relevance order from the source.** "
            "This is not a fitness ranking — embedding fit and the "
            "strategy assessor land in a follow-up milestone.\n\n"
        )
    for i, sc in enumerate(result.candidates, start=1):
        _render_candidate(buf, i, sc)

    if result.coverage and result.coverage.semantic_gaps and not notable_gaps:
        buf.write("## Coverage gaps\n\n")
        buf.write("Aspects worth augmenting:\n\n")
        for gap in result.coverage.semantic_gaps:
            buf.write(f"- **{gap.aspect}** — {gap.description}\n")
            buf.write(f"  - *Suggestion:* {gap.suggestion}\n")
        buf.write("\n")

    buf.write("\n---\n\n")
    buf.write(
        "_License signals are an SPDX-best-effort guess. Always read the "
        "upstream card before redistributing. This is not legal advice._\n"
    )
    return buf.getvalue()


def _render_candidate(buf: StringIO, index: int, sc: Scorecard) -> None:
    cand = sc.candidate
    meta = cand.metadata
    title = f"{index}. `{cand.source}:{cand.id}`"
    buf.write(f"### {title}\n\n")

    if meta.card_url:
        buf.write(f"- 🔗 **Card:** {meta.card_url}\n")
    if cand.revision:
        buf.write(f"- 📌 **Revision:** `{cand.revision[:12]}`\n")
    if meta.description:
        desc = meta.description.strip().splitlines()[0][:180]
        buf.write(f"- 📝 **Description:** {desc}\n")

    if cand.surfaced_by:
        buf.write(f"- 🧭 **Surfaced by:** {', '.join(cand.surfaced_by)}\n")

    badges = list(_render_badges(sc))
    if badges:
        buf.write("- 🏷️ **Badges:** " + " · ".join(badges) + "\n")

    if cand.requires_auth:
        buf.write("- 🔒 **Access:** gated / requires authentication\n")

    if sc.strategies:
        _render_strategies(buf, sc.strategies)

    buf.write("\n")
    _render_probe_signals(buf, sc)
    buf.write("\n")


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


def _render_strategies(buf: StringIO, strategies: list[Strategy]) -> None:
    buf.write("\n**Strategies:**\n\n")
    for s in strategies:
        badge = _strategy_badge(s.kind)
        buf.write(f"- {badge} — confidence {s.confidence:.2f}  \n  {s.rationale}\n")
        if s.caveats:
            for cav in s.caveats:
                buf.write(f"  - ⚠ {cav}\n")
        t = s.transform
        transform_bits: list[str] = []
        if t.text_column:
            transform_bits.append(f"text=`{t.text_column}`")
        if t.label_column:
            transform_bits.append(f"label=`{t.label_column}`")
        if t.label_value_map:
            transform_bits.append(
                "values={" + ", ".join(f"`{k}`→{v}" for k, v in t.label_value_map.items()) + "}"
            )
        if t.filter:
            transform_bits.append(f"filter=`{t.filter}`")
        if t.take != "all":
            transform_bits.append(f"take={t.take}")
        if transform_bits:
            buf.write("  - transform: " + " · ".join(transform_bits) + "\n")


def _render_badges(sc: Scorecard) -> list[str]:
    """Compact per-candidate badges. Order is informative, not scored."""
    badges: list[str] = []
    license_sub = sc.cheap_probes.get("license")
    if license_sub is not None:
        match = _evidence_detail(license_sub, "policy_match")
        spdx = _evidence_detail(license_sub, "license_spdx")
        if spdx and match == "allow":
            badges.append(f"license: {spdx} ✅")
        elif spdx and match == "warn_only":
            badges.append(f"license: {spdx} ⚠️")
        elif spdx:
            badges.append(f"license: {spdx} ❗")
        elif license_sub.status == "low_confidence":
            raw = _evidence_detail(license_sub, "license_raw") or "?"
            badges.append(f"license: {raw} (?)")
        else:
            badges.append("license: missing")

    fresh_sub = sc.cheap_probes.get("freshness")
    if fresh_sub is not None and fresh_sub.status == "ok":
        bucket = _evidence_detail(fresh_sub, "bucket") or "?"
        badges.append(f"freshness: {bucket}")

    size_sub = sc.cheap_probes.get("size")
    if size_sub is not None and size_sub.status == "ok":
        rows = _evidence_detail(size_sub, "rows")
        bytes_ = _evidence_detail(size_sub, "bytes")
        downloads = _evidence_detail(size_sub, "downloads")
        if rows:
            badges.append(f"rows: {rows}")
        elif bytes_:
            badges.append(f"bytes: {bytes_}")
        if downloads:
            badges.append(f"downloads: {downloads}")

    lang_sub = sc.cheap_probes.get("languages")
    if lang_sub is not None and lang_sub.status == "ok":
        declared = _evidence_detail(lang_sub, "declared") or "?"
        badges.append(f"languages: {declared}")

    return badges


def _render_probe_signals(buf: StringIO, sc: Scorecard) -> None:
    """Per-probe signal block in a Markdown table."""
    rows: list[tuple[str, str, str]] = []
    for name, sub in sc.cheap_probes.items():
        rows.append(
            (
                name,
                sub.status,
                _format_subscore_evidence(sub),
            )
        )
    if not rows:
        return
    buf.write("| Probe | Status | Signal |\n|---|---|---|\n")
    for name, status, sig in rows:
        buf.write(f"| {name} | {status} | {sig} |\n")


def _format_subscore_evidence(sub: SubScore) -> str:
    """One-line summary of a SubScore's evidence."""
    if sub.status != "ok":
        if sub.evidence:
            return _safe_md(sub.evidence[0].detail)
        return "—"
    parts: list[str] = []
    for ev in sub.evidence:
        parts.append(f"{ev.kind}={_safe_md(ev.detail)}")
    if sub.value is not None:
        parts.append(f"value={sub.value:.2f}")
    return "; ".join(parts) or "ok"


def _evidence_detail(sub: SubScore, kind: str) -> str | None:
    for ev in sub.evidence:
        if ev.kind == kind:
            return ev.detail
    return None


def _safe_md(text: str | Evidence) -> str:
    """Trim and escape pipes/newlines so the markdown table stays well-formed."""
    s = text if isinstance(text, str) else str(text)
    s = s.replace("|", "\\|").replace("\n", " ")
    if len(s) > 80:
        s = s[:77] + "..."
    return s
