"""Markdown report renderer.

Discovery-slice framing: explicit "pre-fit metadata screening" header,
no ranking language. Per-candidate annotations come from the cheap
probes; everything links back to a card URL.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from dataset_scout.core import (
    Evidence,
    ReconResult,
    Scorecard,
    SubScore,
)


def write_recon_report(result: ReconResult, out_dir: Path) -> Path:
    """Render and write `<out_dir>/report.md`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.md"
    target.write_text(render_recon_report(result), encoding="utf-8")
    return target


def render_recon_report(result: ReconResult) -> str:
    """Render a ReconResult to a Markdown string."""
    buf = StringIO()
    intent = result.intent
    buf.write("# dataset-scout recon report\n\n")
    buf.write(
        "> **This is a discovery report — pre-fit metadata screening.**  \n"
        "> Candidates are returned in source/search relevance order.  \n"
        "> Probe outputs are annotations, not a ranking score. Semantic\n"
        "> fit (embedding + LLM strategy assessor) lands in a later\n"
        "> milestone.\n\n"
    )

    # ─── Brief ──────────────────────────────────────────────────
    buf.write("## Brief\n\n")
    buf.write(f"**Raw brief:** {intent.raw_brief}\n\n")
    if intent.detection_target:
        buf.write(f"**Detection target:** {intent.detection_target}\n\n")
    if intent.threat_families:
        buf.write(f"**Threat families:** {', '.join(intent.threat_families)}\n\n")
    buf.write(f"**Languages requested:** {', '.join(intent.languages)}\n\n")

    # ─── Run summary ────────────────────────────────────────────
    buf.write("## Run summary\n\n")
    buf.write(f"- Sources searched: {', '.join(result.sources_searched) or '(none)'}\n")
    buf.write(f"- Candidates returned: **{len(result.candidates)}**\n")
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

    # ─── Candidates ─────────────────────────────────────────────
    buf.write("## Candidates\n\n")
    buf.write("Listed in **search-relevance order from the source.** ")
    buf.write("This is *not* a fitness ranking.\n\n")
    for i, sc in enumerate(result.candidates, start=1):
        _render_candidate(buf, i, sc)

    # ─── Footer ─────────────────────────────────────────────────
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

    badges = list(_render_badges(sc))
    if badges:
        buf.write("- 🏷️ **Badges:** " + " · ".join(badges) + "\n")

    if cand.requires_auth:
        buf.write("- 🔒 **Access:** gated / requires authentication\n")

    buf.write("\n")
    _render_probe_signals(buf, sc)
    buf.write("\n")


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
